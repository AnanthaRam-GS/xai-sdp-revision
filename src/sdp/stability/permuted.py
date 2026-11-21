"""Persisted permuted-label models and their explanations (data-only reference).

Step 6 refitted each seed-0 model with permuted training labels for its data-randomisation
check but kept only summary similarities. This module repeats exactly that refit with the
step 6 building blocks — the same seeded permutation (:func:`sdp.fidelity.sanity.permuted_labels`
over the fitting and validation rows), the persisted hyperparameters, preprocessing and
imbalance handling, no re-tuning (:func:`sdp.fidelity.common.refit`) — and additionally

* saves each model under ``<artifacts>/models/permuted_labels/<split>/<model>/<prep>/seed_0/``;
* stores its test-set explanations for every evaluated method in the attribution store
  under ``<artifacts>/attributions/_permuted/`` with ``subset="permuted_labels"``;
* recomputes the step 6 similarity and test AUC so that the reproduction can be checked
  against ``<artifacts>/fidelity/sanity/*__data__<model>.parquet``.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.config import Config
from sdp.data.clean import LABEL_COL
from sdp.eval.metrics import roc_auc
from sdp.eval.preprocess import feature_frame
from sdp.experiments.persistence import artifact_dir, safe_split_id, save_artifact
from sdp.explain.base import partition_path, to_long, write_long
from sdp.explain.runner import EXPLAINERS, ExplainSettings
from sdp.fidelity.common import ABSOLUTE, EVALUATED, refit, store_attributions
from sdp.fidelity.sanity import permuted_labels, similarity
from sdp.robustness.common import seed_context

EXPERIMENT = "permuted_labels"
SUBSET = "permuted_labels"


def permuted_root(es_root: Path) -> Path:
    return Path(es_root) / "_permuted"


def permuted_job(cfg: Config, es: ExplainSettings, split, df: pd.DataFrame, model: str, seed: int = 0) -> dict:
    t0 = time.perf_counter()
    ctx = seed_context(cfg, es, split, {split.project: df}, model, seed)
    models_root = Path(cfg.paths.artifacts_dir) / "models"
    meta = json.loads((artifact_dir(models_root / es.experiment, "", split.split_id, model, es.models[model], seed)
                       / "meta.json").read_text(encoding="utf-8"))  # fmt: skip
    feats = list(meta["feature_order"])
    X = feature_frame(df)[feats]
    y = df[LABEL_COL].to_numpy(dtype=int)
    train = np.concatenate([split.fit_idx, split.val_idx])
    y_perm = y.copy()
    y_perm[train] = permuted_labels(y[train], seed)
    fitted = refit(model, meta["hyperparameters"], meta["preprocessing"], meta["imbalance"], seed,
                   X.iloc[split.fit_idx], y_perm[split.fit_idx], X.iloc[split.val_idx], y_perm[split.val_idx])  # fmt: skip
    rctx = dataclasses.replace(ctx, model=fitted.model, preprocessor=fitted.preprocessor, threshold=fitted.threshold)
    auc = roc_auc(ctx.y_true, rctx.y_prob)
    save_artifact(artifact_dir(models_root / EXPERIMENT, "", split.split_id, model, meta["preprocessing"], seed),
                  fitted.preprocessor, fitted.model,
                  meta | {"experiment": EXPERIMENT, "threshold_val_mcc": fitted.threshold, "permutation_seed": seed,
                          "permuted": "fitting and validation labels of the training release", "test_auc": auc})  # fmt: skip
    root = permuted_root(es.root(cfg))
    rows = []
    for method in EVALUATED[model]:
        res = EXPLAINERS[method](**es.method_params(method)).explain(rctx)
        write_long(to_long(rctx, res).assign(subset=SUBSET),
                   partition_path(root, model, method, safe_split_id(split.split_id), seed))  # fmt: skip
        new = res.attributions.to_numpy()
        new = np.abs(new) if method in ABSOLUTE else new
        sim = similarity(store_attributions(es.root(cfg), split.split_id, model, method, seed), new)
        rows.append({"split_id": split.split_id, "model": model, "method": method, "permuted_auc": auc,
                     "spearman": float(np.nanmean(sim["spearman"])), "fa5": float(np.nanmean(sim["fa5"]))})  # fmt: skip
    return {"rows": rows, "timing": {"split_id": split.split_id, "model": model, "time_s": time.perf_counter() - t0}}


def step6_rows(fidelity_root: Path) -> pd.DataFrame:
    files = sorted((Path(fidelity_root) / "sanity").glob("*__data__*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def reproduction_check(new: pd.DataFrame, step6: pd.DataFrame) -> pd.DataFrame:
    """Absolute differences of AUC, Spearman and FA@5 between this refit and step 6."""
    m = new.merge(step6[["split_id", "model", "method", "permuted_auc", "spearman", "fa5"]],
                  on=["split_id", "model", "method"], suffixes=("", "_step6"))  # fmt: skip
    for c in ("permuted_auc", "spearman", "fa5"):
        m[f"diff_{c}"] = (m[c] - m[f"{c}_step6"]).abs()
    return m
