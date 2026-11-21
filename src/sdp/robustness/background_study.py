"""Background-set sensitivity of DeepLiftShap, GradientShap (MLP) and TreeSHAP (RF, XGBoost).

All test instances of every split are explained with the seed-0 models under each
background variant; the step 4 explainers are reused unchanged, only the context's
background set is swapped. Variants (all drawn from the fitting part of the training
data, in the model's input space):

* ``size=<n>``: stratified random sample of n rows (``size=all``: every fitting row);
  ``size=100`` with draw seed 0 is the main (step 4) background;
* ``scheme=random``: simple random sample of 100 rows;
* ``scheme=kmeans``: ``shap.kmeans`` with 100 centres; its weights are the cluster
  sizes, so each centre is repeated as often as its cluster has rows. Interventional
  SHAP and DeepLiftShap average over the background rows, so this is exactly the
  weighted background (GradientShap draws its baselines from it, so the centres are
  drawn in proportion to their weights);
* ``scheme=defective`` / ``scheme=clean``: 100 rows of one class (all if fewer);
* ``draw=<d>``: stratified samples of 100 rows with draw seed d = 0..4 (d = 0 is main).

Attributions are written to ``<artifacts>/robustness/background/<model>/<method>/<variant>/<split>.parquet``.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.analysis.explanations import Explanations, normalised_abs, rank_correlations, topk_agreement
from sdp.config import Config
from sdp.eval.preprocess import FEATURES
from sdp.explain.base import ExplainContext, sample_background
from sdp.explain.runner import EXPLAINERS, ExplainSettings
from sdp.robustness.common import output_root, run_parallel, seed_context, split_dir

logger = logging.getLogger(__name__)

MAIN = "size=100"


def variants(bs: dict) -> list[str]:
    """Variant names; ``size=100`` (= stratified draw 0) is the main background."""
    out = [f"size={s}" for s in bs["sizes"]]
    out += [f"scheme={s}" for s in bs["schemes"] if s != "stratified"]
    out += [f"draw={d}" for d in bs["draws"] if d != 0]
    return out


def make_background(X: pd.DataFrame, y: np.ndarray, variant: str, seed: int = 0, size: int = 100) -> pd.DataFrame:
    """The background rows (model input space) of one variant."""
    kind, value = variant.split("=")
    y = np.asarray(y, dtype=int)
    if kind == "size":
        return X.copy() if value == "all" else sample_background(X, y, int(value), seed)
    if kind == "draw":
        return sample_background(X, y, size, int(value))
    if value == "random":
        return X.copy() if len(X) <= size else X.sample(n=size, random_state=seed)
    if value in ("defective", "clean"):
        Xc = X[y == (1 if value == "defective" else 0)]
        return Xc.copy() if len(Xc) <= size else Xc.sample(n=size, random_state=seed)
    if value == "kmeans":
        return kmeans_background(X, size)
    raise ValueError(f"unknown background variant {variant!r}")


def kmeans_background(X: pd.DataFrame, k: int = 100) -> pd.DataFrame:
    """``shap.kmeans`` centres, each repeated by its cluster size (= the shap weights)."""
    import shap

    if len(X) <= k:
        return X.copy()
    summary = shap.kmeans(X.to_numpy(dtype="float64"), k)
    counts = np.asarray(summary.weights, dtype=float)
    counts = counts * len(X) / counts.sum()  # shap normalises the cluster sizes to sum to 1
    reps = np.rint(counts).astype(int)
    if not np.allclose(reps, counts, atol=1e-6) or reps.sum() != len(X):
        raise ValueError("unexpected k-means weights: expected integer cluster sizes")
    rows = np.repeat(summary.data, reps, axis=0)
    return pd.DataFrame(rows, columns=X.columns)


def background_job(cfg: Config, es: ExplainSettings, split, df: pd.DataFrame, model: str, methods: list[str],
                   variant: str, out: str, seed: int = 0) -> dict:  # fmt: skip
    t0 = time.perf_counter()
    ctx: ExplainContext = seed_context(cfg, es, split, {split.project: df}, model, seed)
    bg = make_background(ctx.X_train, ctx.y_train, variant, seed)
    vctx = dataclasses.replace(ctx, background=bg.reset_index(drop=True))
    times = {}
    for method in methods:
        res = EXPLAINERS[method](**es.method_params(method)).explain(vctx)
        frame = res.attributions.assign(split_id=split.split_id, project=split.project, model=model, method=method,
                                        variant=variant, seed=seed, instance_id=np.arange(len(vctx.X_test)),
                                        outcome=vctx.outcome, y_true=vctx.y_true)  # fmt: skip
        path = Path(out) / model / method / variant.replace("=", "_") / f"{split_dir(split)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        times[method] = res.time_s
    return {"timing": {"split_id": split.split_id, "model": model, "variant": variant, "n_instances": len(vctx.X_test),
                       "n_background": len(bg), "time_s": time.perf_counter() - t0,
                       **{f"t_{m}": t for m, t in times.items()}}}  # fmt: skip


def run_background_study(cfg: Config, es: ExplainSettings, rs: dict, splits, frames, n_jobs: int,
                         out_dir: Path | None = None, only_variants=None) -> list[dict]:  # fmt: skip
    bs = rs["background"]
    out = out_dir or (output_root(cfg, rs) / "background")
    tasks = []
    for split in splits:
        for model, methods in bs["models"].items():
            for v in only_variants or variants(bs):
                tasks.append((cfg, es, split, frames[split.project], model, list(methods), v, str(out),
                              rs.get("model_seed", 0)))  # fmt: skip
    weight = {"rf": 6.0, "xgb": 1.0, "mlp": 1.5}
    big = {"size=all": 6.0, "scheme=kmeans": 6.0, "size=200": 2.0}
    tasks.sort(key=lambda t: -(len(t[2].test_idx) * weight[t[4]] * big.get(t[6], 1.0)))
    return run_parallel(background_job, tasks, n_jobs, "Background study")


def load_background_runs(path: Path) -> Explanations:
    """All variant attributions as one wide :class:`Explanations` (``variant`` in meta)."""
    files = sorted(Path(path).glob("*/*/*/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    meta = df.drop(columns=FEATURES).reset_index(drop=True)
    return Explanations(meta, df[FEATURES].to_numpy(dtype="float64"), np.full((len(df), len(FEATURES)), np.nan))


# ------------------------------------------------------------------------- analysis


def _align(ex: Explanations, a: str, b: str):
    key = ["model", "method", "split_id", "instance_id"]
    m = ex.meta.reset_index().rename(columns={"index": "_row"})
    left = m[m["variant"] == a][key + ["project", "_row"]]
    right = m[m["variant"] == b][key + ["_row"]]
    j = left.merge(right, on=key, suffixes=("_a", "_b"))
    return j, ex.attr[j["_row_a"].to_numpy()], ex.attr[j["_row_b"].to_numpy()]


def compare(A: np.ndarray, B: np.ndarray) -> dict[str, np.ndarray]:
    rc = rank_correlations(A, B)
    top = topk_agreement(A, B, 5)
    return {"spearman": rc["spearman"], "fa5": top["feature_agreement@5"], "sa5": top["sign_agreement@5"],
            "mad": np.nanmean(np.abs(normalised_abs(A) - normalised_abs(B)), axis=1)}  # fmt: skip


def agreement_with_main(ex: Explanations) -> pd.DataFrame:
    """Per instance: every variant vs the main background."""
    parts = []
    for v in sorted(set(ex.meta["variant"]) - {MAIN}):
        j, A, B = _align(ex, MAIN, v)
        parts.append(j[["model", "method", "project", "split_id", "instance_id"]].assign(variant=v, **compare(A, B)))
    return pd.concat(parts, ignore_index=True)


def draw_noise(ex: Explanations, draws=("size=100", "draw=1", "draw=2", "draw=3", "draw=4")) -> pd.DataFrame:
    """Noise floor from the 5 repeated stratified draws: per instance, the mean pairwise
    agreement between draws and the mean (over features) variance of the L1-normalised
    attributions across draws."""
    rows = []
    for a, b in combinations(draws, 2):
        j, A, B = _align(ex, a, b)
        rows.append(j[["model", "method", "project", "split_id", "instance_id"]].assign(pair=f"{a}|{b}", **compare(A, B)))
    pairs = pd.concat(rows, ignore_index=True)
    key = ["model", "method", "project", "split_id", "instance_id"]
    per_inst = pairs.groupby(key)[["spearman", "fa5", "sa5", "mad"]].mean()
    stack = []
    base = ex.meta[ex.meta["variant"].isin(draws)].reset_index()
    for k, g in base.groupby(key):
        stack.append({**dict(zip(key, k)), "variance": float(np.nanmean(np.var(normalised_abs(ex.attr[g["index"]]), axis=0, ddof=1)))})
    return per_inst.join(pd.DataFrame(stack).set_index(key)).reset_index()


def summarise(per_instance: pd.DataFrame, by=("model", "method", "variant")) -> pd.DataFrame:
    """Mean over instances within a split, then over splits (each split weighted equally)."""
    cols = [c for c in ("spearman", "fa5", "sa5", "mad", "variance") if c in per_instance.columns]
    split_means = per_instance.groupby(list(by) + ["split_id"])[cols].mean()
    return split_means.groupby(level=list(by)).mean().reset_index()
