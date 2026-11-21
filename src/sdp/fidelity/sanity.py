"""Sanity checks for saliency-style explanations (Adebayo et al., 2018).

a) **Model-parameter randomisation** (MLP): the ``nn.Linear`` layers are re-initialised
   with PyTorch's default initialisation (``reset_parameters``, seeded), either
   cumulatively from the output layer towards the input ("cascading") or one layer at a
   time ("independent"). Every MLP method is recomputed on all test instances (seed 0)
   and compared with the original explanation.
b) **Data randomisation** (all models): the training labels (fitting and validation
   rows of the training release) are permuted (seed 0), the model is refitted with its
   persisted hyperparameters, preprocessing and imbalance handling, and re-explained.

Similarity per instance: Spearman correlation of |attribution| and feature agreement@5
(step 4 metrics). A method is flagged when its mean Spearman stays above 0.5 after full
randomisation.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sdp.analysis.explanations import rank_correlations, topk_agreement
from sdp.config import Config
from sdp.data.clean import LABEL_COL
from sdp.eval.metrics import roc_auc
from sdp.eval.preprocess import feature_frame
from sdp.experiments.persistence import artifact_dir
from sdp.explain.runner import EXPLAINERS, ExplainSettings
from sdp.explain.base import subset_context
from sdp.fidelity.common import ABSOLUTE, EVALUATED, refit, store_attributions_by_id
from sdp.robustness.common import seed_context

logger = logging.getLogger(__name__)

FLAG_THRESHOLD = 0.5


def linear_positions(model) -> list[int]:
    """Positions of the ``nn.Linear`` modules in the MLP's layer list (input → output)."""
    return [i for i, m in enumerate(model.net_.layers) if isinstance(m, torch.nn.Linear)]


def randomised_model(model, positions: list[int], seed: int):
    """A deep copy of the MLP with the given Linear layers re-initialised (seeded)."""
    new = copy.deepcopy(model)
    torch.manual_seed(seed)
    for i in sorted(positions, reverse=True):  # output side first, so a stage's draws are reproducible
        new.net_.layers[i].reset_parameters()
    new.net_.eval()
    return new


def cascade_stages(model) -> list[tuple[str, list[int]]]:
    """Cumulative stages from the output layer towards the input, e.g. ``output``,
    ``output+hidden2``, ..., ``all``."""
    pos = linear_positions(model)
    names = [f"hidden{j + 1}" for j in range(len(pos) - 1)] + ["output"]
    stages = []
    for n in range(1, len(pos) + 1):
        label = "all" if n == len(pos) else "+".join(names[-n:])
        stages.append((label, pos[-n:]))
    return stages


def independent_stages(model) -> list[tuple[str, list[int]]]:
    pos = linear_positions(model)
    names = [f"hidden{j + 1}" for j in range(len(pos) - 1)] + ["output"]
    return [(f"only {n}", [p]) for n, p in zip(names, pos)]


def permuted_labels(y, seed: int = 0) -> np.ndarray:
    """A seeded permutation of the labels (class counts unchanged)."""
    return np.random.default_rng(seed).permutation(np.asarray(y))


def similarity(original: np.ndarray, other: np.ndarray) -> dict[str, np.ndarray]:
    """Per-instance Spearman of |attribution| and feature agreement@5."""
    rc = rank_correlations(original, other)
    return {"spearman": rc["spearman"], "fa5": topk_agreement(original, other, 5)["feature_agreement@5"]}


def _explain(es: ExplainSettings, ctx, method: str) -> np.ndarray:
    a = EXPLAINERS[method](**es.method_params(method)).explain(ctx).attributions.to_numpy()
    return np.abs(a) if method in ABSOLUTE else a


def cascade_job(cfg: Config, es: ExplainSettings, split, df: pd.DataFrame, seed: int = 0) -> dict:
    """Cascading and independent randomisation of the seed-0 MLP of one split."""
    t0 = time.perf_counter()
    ctx = seed_context(cfg, es, split, {split.project: df}, "mlp", seed)
    rows = []
    stages = [("cascading", s) for s in cascade_stages(ctx.model)] + [("independent", s) for s in independent_stages(ctx.model)]
    for k, (kind, (stage, positions)) in enumerate(stages):
        rctx = dataclasses.replace(ctx, model=randomised_model(ctx.model, positions, seed=1000 * seed + k))
        for method in EVALUATED["mlp"]:
            ids, orig = store_attributions_by_id(es.root(cfg), split.split_id, "mlp", method, seed)
            sim = similarity(orig, _explain(es, subset_context(rctx, ids), method))
            rows.append({"split_id": split.split_id, "project": split.project, "model": "mlp", "method": method,
                         "kind": kind, "stage": stage, "n_layers": len(positions), "n_linear": len(linear_positions(ctx.model)),
                         "spearman": float(np.nanmean(sim["spearman"])), "fa5": float(np.nanmean(sim["fa5"])),
                         "n_instances": len(orig)})  # fmt: skip
    return {"rows": rows, "timing": {"split_id": split.split_id, "study": "cascade", "time_s": time.perf_counter() - t0}}


def data_randomisation_job(cfg: Config, es: ExplainSettings, split, df: pd.DataFrame, model: str, seed: int = 0) -> dict:
    """Refit with permuted training labels and compare the explanations."""
    t0 = time.perf_counter()
    ctx = seed_context(cfg, es, split, {split.project: df}, model, seed)
    models_root = Path(cfg.paths.artifacts_dir) / "models" / es.experiment
    meta = json.loads((artifact_dir(models_root, "", split.split_id, model, es.models[model], seed) / "meta.json").read_text())
    feats = list(meta["feature_order"])
    X = feature_frame(df)[feats]
    y = df[LABEL_COL].to_numpy(dtype=int)
    train = np.concatenate([split.fit_idx, split.val_idx])
    y_perm = y.copy()
    y_perm[train] = permuted_labels(y[train], seed)
    fitted = refit(model, meta["hyperparameters"], meta["preprocessing"], meta["imbalance"], seed,
                   X.iloc[split.fit_idx], y_perm[split.fit_idx], X.iloc[split.val_idx], y_perm[split.val_idx])  # fmt: skip
    rctx = dataclasses.replace(ctx, model=fitted.model, preprocessor=fitted.preprocessor)
    auc = roc_auc(ctx.y_true, rctx.y_prob)
    rows = []
    for method in EVALUATED[model]:
        ids, orig = store_attributions_by_id(es.root(cfg), split.split_id, model, method, seed)
        sim = similarity(orig, _explain(es, subset_context(rctx, ids), method))
        rows.append({"split_id": split.split_id, "project": split.project, "model": model, "method": method,
                     "kind": "data", "stage": "permuted labels", "spearman": float(np.nanmean(sim["spearman"])),
                     "fa5": float(np.nanmean(sim["fa5"])), "permuted_auc": auc, "n_instances": len(orig)})  # fmt: skip
    return {"rows": rows, "timing": {"split_id": split.split_id, "study": f"data_{model}", "time_s": time.perf_counter() - t0}}
