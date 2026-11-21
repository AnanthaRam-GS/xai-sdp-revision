"""Per (split, model) evaluation of deletion/insertion curves and faithfulness.

One job loads the seed-0 model and its step 4 context (``robustness.common.seed_context``),
reads the stored v2 attributions of every evaluated method and computes, on the same
instances, with the same 50 background rows:

* deletion and insertion curves (marginal removal) for each method and for 10 random
  orderings; deletion curves with mean removal as a sensitivity result;
* the faithfulness correlation over 50 random subsets of size 5 (shared by all methods).

Results go to ``<artifacts>/fidelity/curves/<split>__<model>.parquet`` (per instance) and
``..._curves.parquet`` (mean curves).
"""

from __future__ import annotations

import logging
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.config import Config
from sdp.experiments.persistence import safe_split_id
from sdp.explain.base import OUTCOMES
from sdp.explain.runner import ExplainSettings
from sdp.fidelity.common import EVALUATED, store_attributions
from sdp.fidelity.curves import evaluate_orders
from sdp.fidelity.faithfulness import faithfulness, prediction_drops, random_subsets
from sdp.fidelity.removal import MarginalRemover, MeanRemover, select_background
from sdp.robustness.common import seed_context

logger = logging.getLogger(__name__)


def job_seed(split_id: str, model: str = "") -> int:
    return zlib.crc32(f"{split_id}|{model}".encode()) % (2**31)


def sample_by_outcome(outcome: np.ndarray, size: int | None, seed: int) -> np.ndarray:
    """All positions, or a sample of up to ``size`` stratified by outcome group (seeded)."""
    n = len(outcome)
    if size is None or n <= size:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    chosen = []
    for g in OUTCOMES:
        pos = np.flatnonzero(outcome == g)
        take = int(round(size * len(pos) / n))
        chosen += list(rng.permutation(pos)[:take])
    return np.sort(np.asarray(chosen[:size], dtype=int))


def curves_job(cfg: Config, es: ExplainSettings, split, df: pd.DataFrame, model: str, out_dir: str, fs: dict,
               seed: int = 0) -> dict:  # fmt: skip
    t0 = time.perf_counter()
    ctx = seed_context(cfg, es, split, {split.project: df}, model, seed)
    outcome = ctx.outcome
    idx = sample_by_outcome(outcome, fs.get("sample_per_split"), job_seed(split.split_id))
    X = ctx.X_test.to_numpy(dtype="float64")[idx]
    p = ctx.y_prob[idx]
    pred = p >= ctx.threshold
    attrs = {m: store_attributions(es.root(cfg), split.split_id, model, m, seed)[idx] for m in EVALUATED[model]}
    B = select_background(ctx.background, fs["n_background"], seed)
    marginal = MarginalRemover(ctx.predict, B)
    rs = job_seed(split.split_id, model)
    res = evaluate_orders(marginal, X, attrs, pred, fs["n_random"], seed=rs)
    t_curves = time.perf_counter() - t0
    mean_res = evaluate_orders(MeanRemover(ctx.predict, B), X, attrs, pred, fs["n_random"], seed=rs, kinds=("deletion",))
    subsets = random_subsets(len(idx), X.shape[1], fs["faith_subsets"], fs["faith_size"], seed=rs + 1)
    drops = prediction_drops(marginal, X, p, subsets)
    base = pd.DataFrame({"split_id": split.split_id, "project": split.project, "model": model,
                         "instance_id": idx, "outcome": outcome[idx], "y_prob": p})  # fmt: skip
    rows = []
    for m in EVALUATED[model]:
        a = res["auc"]
        rows.append(base.assign(
            method=m, deletion_auc=a["deletion"][m], insertion_auc=a["insertion"][m],
            deletion_random=a["deletion"]["random"], insertion_random=a["insertion"]["random"],
            deletion_gain=a["deletion"]["random"] - a["deletion"][m], insertion_gain=a["insertion"][m] - a["insertion"]["random"],
            deletion_auc_mean=mean_res["auc"]["deletion"][m], deletion_random_mean=mean_res["auc"]["deletion"]["random"],
            deletion_gain_mean=mean_res["auc"]["deletion"]["random"] - mean_res["auc"]["deletion"][m],
            faithfulness=faithfulness(attrs[m], subsets, drops),
        ))  # fmt: skip
    per_inst = pd.concat(rows, ignore_index=True)
    crv = []
    for removal, r in (("marginal", res), ("mean", mean_res)):
        for kind, d in r["curves"].items():
            for name, c in d.items():
                crv.append(pd.DataFrame({"split_id": split.split_id, "project": split.project, "model": model,
                                         "method": name, "kind": kind, "removal": removal, "k": np.arange(len(c)),
                                         "value": c}))  # fmt: skip
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_split_id(split.split_id)}__{model}"
    per_inst.to_parquet(out / f"{stem}.parquet", index=False)
    pd.concat(crv, ignore_index=True).to_parquet(out / f"{stem}_curves.parquet", index=False)
    return {"timing": {"split_id": split.split_id, "model": model, "n_instances": len(idx), "n_background": len(B),
                       "time_s": time.perf_counter() - t0, "t_curves": t_curves}}  # fmt: skip


def load_instances(out_dir: Path) -> pd.DataFrame:
    files = [f for f in sorted(Path(out_dir).glob("*.parquet")) if not f.stem.endswith("_curves")]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_curves(out_dir: Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f) for f in sorted(Path(out_dir).glob("*_curves.parquet"))], ignore_index=True)
