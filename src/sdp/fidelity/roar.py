"""ROAR and KAR retraining benchmark (Hooker et al., 2019).

1. **Training-set attributions** (no test data): for each split and seed, the methods
   below explain the model's *fitting rows* (training part without the validation
   hold-out); LIME uses a stratified, seeded sample of 200 fitting rows. They are stored
   under ``<artifacts>/attributions/_train/`` with ``subset="train"``.
2. **Global ranking** per (split, seed, model, method): features ordered by the mean
   L1-normalised |attribution| over those rows (ties by feature order).
3. **Reference rankings**: 5 seeded random orderings; "size-first" (loc, rfc, wmc, npm,
   then the other metrics in random ordering 0); the RF-TreeSHAP ranking of the same
   split and seed applied to every other model.
4. **Retraining**: ROAR drops the top-k features, KAR keeps only the top-k
   (k ∈ {1, 2, 3, 5, 8, 10, 15}); the same model type is refitted with its persisted
   step 3 hyperparameters (no re-tuning), preprocessing, imbalance handling and early
   stopping, and evaluated on the test release (AUC, MCC@tuned, Popt). Identical
   feature sets are fitted once per (split, model, seed).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sdp.analysis.explanations import normalised_abs
from sdp.config import Config
from sdp.data.clean import ID_COL, LABEL_COL
from sdp.eval.preprocess import EFFORT_COL, FEATURES, feature_frame
from sdp.experiments.persistence import safe_split_id
from sdp.explain.base import partition_path, subset_context, to_long, write_long
from sdp.explain.runner import EXPLAINERS, ExplainSettings
from sdp.fidelity.common import refit, test_metrics
from sdp.robustness.common import seed_context

logger = logging.getLogger(__name__)

TRAIN_METHODS = {
    "mlp": ["DeepLiftShap", "IntegratedGradients", "Saliency", "LIME"],
    "lr": ["LinearSHAP"],
    "rf": ["TreeSHAP"],
    "xgb": ["TreeSHAP"],
}
SIZE_FIRST = ["loc", "rfc", "wmc", "npm"]
KS = (1, 2, 3, 5, 8, 10, 15)
N_RANDOM = 5


def train_root(es_root: Path) -> Path:
    return Path(es_root) / "_train"


def stratified_positions(y: np.ndarray, size: int, seed: int) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    if len(y) <= size:
        return np.arange(len(y))
    strat = y if np.bincount(y, minlength=2).min() >= 2 else None
    idx, _ = train_test_split(np.arange(len(y)), train_size=size, random_state=seed, stratify=strat)
    return np.sort(idx)


def train_attributions_job(cfg: Config, es: ExplainSettings, split, df: pd.DataFrame, model: str, seed: int,
                           out_root: str, lime_rows: int = 200) -> dict:  # fmt: skip
    """Explain the fitting rows of one (split, model, seed) and store them with subset="train"."""
    t0 = time.perf_counter()
    ctx = seed_context(cfg, es, split, {split.project: df}, model, seed)
    fit = split.fit_idx
    tctx = dataclasses.replace(
        ctx, X_test=ctx.X_train.reset_index(drop=True),
        X_test_raw=feature_frame(df)[list(ctx.X_train.columns)].iloc[fit].reset_index(drop=True),
        class_name=df[ID_COL].astype(str).to_numpy()[fit], y_true=df[LABEL_COL].to_numpy(dtype=int)[fit],
        loc=df[EFFORT_COL].to_numpy(dtype=float)[fit],
    )  # fmt: skip
    times = {}
    for method in TRAIN_METHODS[model]:
        use = tctx
        if method == "LIME":
            use = subset_context(tctx, stratified_positions(tctx.y_true, lime_rows, seed))
        res = EXPLAINERS[method](**es.method_params(method)).explain(use)
        frame = to_long(use, res).assign(subset="train")
        write_long(frame, partition_path(out_root, model, method, safe_split_id(split.split_id), seed))
        times[method] = res.time_s
    return {"timing": {"split_id": split.split_id, "model": model, "seed": seed, "n_fit": len(fit),
                       "time_s": time.perf_counter() - t0, **{f"t_{m}": t for m, t in times.items()}}}  # fmt: skip


def global_ranking(root: Path, split_id: str, model: str, method: str, seed: int) -> list[str]:
    """Features by mean L1-normalised |attribution| over the stored *training* rows."""
    frame = pd.read_parquet(partition_path(root, model, method, safe_split_id(split_id), seed))
    if "subset" not in frame.columns or not (frame["subset"] == "train").all():
        raise ValueError(f"{split_id} {model} {method}: ROAR/KAR rankings must come from training rows only")
    wide = frame.pivot(index="instance_id", columns="feature", values="attribution")[FEATURES]
    imp = np.nanmean(normalised_abs(wide.to_numpy()), axis=0)
    order = np.argsort(-imp, kind="stable")
    return [FEATURES[i] for i in order]


def _split_seed(split_id: str, seed: int) -> int:
    return zlib.crc32(f"{split_id}|{seed}".encode()) % (2**31)


def reference_rankings(split_id: str, seed: int, n_random: int = N_RANDOM) -> dict[str, list[str]]:
    rng = np.random.default_rng(_split_seed(split_id, seed))
    out = {f"random_{r}": [FEATURES[i] for i in rng.permutation(len(FEATURES))] for r in range(n_random)}
    out["size_first"] = SIZE_FIRST + [f for f in out["random_0"] if f not in SIZE_FIRST]
    return out


def rankings_for(root: Path, split_id: str, model: str, seed: int, n_random: int = N_RANDOM) -> dict[str, list[str]]:
    out = {m: global_ranking(root, split_id, model, m, seed) for m in TRAIN_METHODS[model]}
    out |= reference_rankings(split_id, seed, n_random)
    if model != "rf":
        out["rf_TreeSHAP"] = global_ranking(root, split_id, "rf", "TreeSHAP", seed)
    return out


def feature_sets(rank: list[str], k: int) -> dict[str, list[str]]:
    """ROAR (drop the top-k) and KAR (keep the top-k) metric sets, in canonical order."""
    top = set(rank[:k])
    return {"ROAR": [f for f in FEATURES if f not in top], "KAR": [f for f in FEATURES if f in top]}


def roar_job(cfg: Config, split, df: pd.DataFrame, model: str, seed: int, artifact_dir: str, rank_root: str,
             ks=KS, n_random: int = N_RANDOM) -> dict:  # fmt: skip
    """All ROAR/KAR refits of one (split, model, seed)."""
    t0 = time.perf_counter()
    meta = json.loads((Path(artifact_dir) / "meta.json").read_text(encoding="utf-8"))
    params, prep, imb = meta["hyperparameters"], meta["preprocessing"], meta["imbalance"]
    X = feature_frame(df)
    y = df[LABEL_COL].to_numpy(dtype=int)
    loc = df[EFFORT_COL].to_numpy(dtype=float)
    fit, val, test = split.fit_idx, split.val_idx, split.test_idx
    memo: dict[tuple, dict] = {}

    def run(feats: list[str]) -> dict:
        key = tuple(feats)
        if key not in memo:
            r = refit(model, params, prep, imb, seed, X.iloc[fit][feats], y[fit], X.iloc[val][feats], y[val])
            memo[key] = test_metrics(r, X.iloc[test][feats], y[test], loc[test])
        return memo[key]

    rows = []
    full = run(list(FEATURES))
    base = {"split_id": split.split_id, "project": split.project, "model": model, "seed": seed}
    for name, rank in rankings_for(Path(rank_root), split.split_id, model, seed, n_random).items():
        rows.append(base | {"ranking": name, "test": "ROAR", "k": 0, "features": ",".join(FEATURES)} | full)
        rows.append(base | {"ranking": name, "test": "KAR", "k": len(FEATURES), "features": ",".join(FEATURES)} | full)
        for k in ks:
            for test_name, feats in feature_sets(rank, k).items():
                rows.append(base | {"ranking": name, "test": test_name, "k": k, "features": ",".join(feats)} | run(feats))
    return {"rows": rows, "timing": {**base, "n_fits": len(memo), "time_s": time.perf_counter() - t0}}


def ranking_family(name: str) -> str:
    return "random" if name.startswith("random_") else name


def curve_area(frame: pd.DataFrame, test: str, ks=KS, metric: str = "auc") -> float:
    """Area under the metric-vs-k curve over the evaluated k (normalised to a mean level)."""
    g = frame[(frame["test"] == test) & frame["k"].isin(ks)].groupby("k")[metric].mean().reindex(sorted(ks))
    x = np.asarray(sorted(ks), dtype=float)
    return float(np.trapezoid(g.to_numpy(), x=x) / (x[-1] - x[0]))


def area_scores(rows: pd.DataFrame, ks=KS, metric: str = "auc") -> pd.DataFrame:
    """Per (split, model, seed, ranking family, test): area score (random averaged over orderings)."""
    out = []
    for key, g in rows.groupby(["split_id", "project", "model", "seed", "ranking", "test"]):
        out.append(dict(zip(["split_id", "project", "model", "seed", "ranking", "test"], key))
                   | {"area": curve_area(g, key[-1], ks, metric)})  # fmt: skip
    t = pd.DataFrame(out)
    t["family"] = t["ranking"].map(ranking_family)
    return t.groupby(["split_id", "project", "model", "seed", "family", "test"], as_index=False)["area"].mean()
