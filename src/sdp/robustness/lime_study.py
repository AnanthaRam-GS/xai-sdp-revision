"""LIME stability and settings sensitivity (step 5).

* **Instances**: for each time-aware split, up to ``instances_per_group`` (10) test
  instances from each outcome group TP/FP/FN/TN, drawn with a seeded generator. The
  outcome groups are those of the seed-0 MLP; the same instances are explained for all
  four models (whose own outcome groups are recorded too).
* **Run-to-run variability**: the main LIME setting is run ``runs`` (10) times per
  instance with ``random_state`` 0..9.
* **Settings grid**: one run (``random_state=0``) per setting, varying one factor at a
  time from the main setting: ``num_samples``, ``kernel_width`` (× sqrt(20)) and the
  sampling scheme (mean sampling = step 4 v1, instance sampling = main, quartile
  discretisation = LIME's default). Each is compared with the main setting's run 0.
* The local surrogate's R² (``explanation.score``) is kept for every explanation.

Explanations are written to ``<artifacts>/robustness/lime/<split>__<model>.parquet``.
"""

from __future__ import annotations

import logging
import math
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.analysis.explanations import rank_correlations, topk_agreement
from sdp.config import Config
from sdp.eval.preprocess import FEATURES
from sdp.explain.base import OUTCOMES, partition_path, subset_context
from sdp.explain.other_methods import LimeExplainer
from sdp.explain.runner import ExplainSettings
from sdp.robustness.common import output_root, run_parallel, seed_context, split_dir

logger = logging.getLogger(__name__)

SAMPLING = {
    "mean": {"sample_around_instance": False, "discretize_continuous": False},
    "instance": {"sample_around_instance": True, "discretize_continuous": False},
    "quartile": {"sample_around_instance": False, "discretize_continuous": True},
}
MAIN = "main"


def grid_settings(ls: dict) -> list[tuple[str, dict]]:
    """``[(name, LIME params)]``: the main setting first, then one-factor-at-a-time variants."""
    main = dict(ls["main"])
    n_feat = len(FEATURES)
    out = [(MAIN, main)]
    for n in ls["grid"]["num_samples"]:
        if n != main["num_samples"]:
            out.append((f"num_samples={n}", main | {"num_samples": int(n)}))
    for f in ls["grid"]["kernel_width_factor"]:
        if not math.isclose(f, 0.75):
            out.append((f"kernel_width={f}×√20", main | {"kernel_width": float(f) * math.sqrt(n_feat)}))
    for s in ls["grid"]["sampling"]:
        if s != "instance":
            out.append((f"sampling={s}", main | SAMPLING[s]))
    return out


def sample_instances(outcome: np.ndarray, per_group: int, seed: int, total: int | None = None) -> np.ndarray:
    """Positions of up to ``per_group`` instances per outcome group (sorted). With
    ``total``, the sample is then cut to ``total`` instances keeping the groups balanced."""
    rng = np.random.default_rng(seed)
    picks = {}
    for g in OUTCOMES:
        pos = np.flatnonzero(np.asarray(outcome) == g)
        picks[g] = list(rng.permutation(pos)[:per_group])
    if total is not None:
        chosen = []
        while len(chosen) < total and any(picks.values()):
            for g in OUTCOMES:
                if picks[g] and len(chosen) < total:
                    chosen.append(picks[g].pop(0))
        return np.sort(np.asarray(chosen, dtype=int))
    return np.sort(np.asarray([i for g in OUTCOMES for i in picks[g]], dtype=int))


def instance_samples(cfg: Config, es: ExplainSettings, splits, ls: dict, seed: int = 0, reduced: bool = False) -> pd.DataFrame:
    """The instance sample of every split, from the stored seed-0 MLP outcome groups."""
    rows = []
    for i, split in enumerate(splits):
        f = partition_path(es.root(cfg), "mlp", "DeepLift", split_dir(split), seed)
        inst = pd.read_parquet(f, columns=["instance_id", "class_name", "outcome"]).drop_duplicates("instance_id")
        inst = inst.sort_values("instance_id").reset_index(drop=True)
        total = ls["reduced_instances_per_split"] if reduced else None
        idx = sample_instances(inst["outcome"].to_numpy(), ls["instances_per_group"], seed=1000 * seed + i, total=total)
        rows.append(inst.iloc[idx].assign(split_id=split.split_id, project=split.project, reference_outcome=lambda t: t["outcome"]))
    return pd.concat(rows, ignore_index=True)[["split_id", "project", "instance_id", "class_name", "reference_outcome"]]


def lime_job(cfg: Config, es: ExplainSettings, split, df: pd.DataFrame, model: str, idx: np.ndarray, ls: dict,
             out: str, seed: int = 0) -> dict:  # fmt: skip
    """All LIME runs and grid settings for one (split, model) on the sampled instances."""
    t0 = time.perf_counter()
    frames = {split.project: df}
    ctx = subset_context(seed_context(cfg, es, split, frames, model, seed), idx)
    rows, timing = [], {}
    settings = grid_settings(ls)
    tasks = [(MAIN, settings[0][1], r) for r in range(int(ls["runs"]))] + [(n, p, 0) for n, p in settings[1:]]
    for name, params, run in tasks:
        exp = LimeExplainer(**(params | {"random_state": run}))
        res = exp.explain(ctx)
        timing[name] = timing.get(name, 0.0) + res.time_s
        rows.append(pd.DataFrame(res.attributions.to_numpy(), columns=list(res.attributions.columns)).assign(
            split_id=split.split_id, project=split.project, model=model, setting=name, run=run,
            instance_id=idx, class_name=ctx.class_name, outcome=ctx.outcome, y_true=ctx.y_true, score=exp.scores_,
        ))  # fmt: skip
    frame = pd.concat(rows, ignore_index=True)
    path = Path(out) / f"{split_dir(split)}__{model}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return {"path": str(path), "timing": {"split_id": split.split_id, "model": model, "n_instances": len(idx),
                                          "time_s": time.perf_counter() - t0, **{f"t_{k}": v for k, v in timing.items()}}}  # fmt: skip


def run_lime_study(cfg: Config, es: ExplainSettings, rs: dict, splits, frames, sample: pd.DataFrame,
                   n_jobs: int, out_dir: Path | None = None) -> list[dict]:  # fmt: skip
    ls = rs["lime"]
    out = out_dir or (output_root(cfg, rs) / "lime")
    tasks = []
    for split in splits:
        idx = sample.loc[sample["split_id"] == split.split_id, "instance_id"].to_numpy()
        for model in ls["models"]:
            tasks.append((cfg, es, split, frames[split.project], model, idx, ls, str(out), rs.get("model_seed", 0)))
    tasks.sort(key=lambda t: -(len(t[5]) * (8 if t[4] == "rf" else 1)))
    return run_parallel(lime_job, tasks, n_jobs, "LIME study")


def load_lime_runs(path: Path) -> pd.DataFrame:
    files = sorted(Path(path).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ------------------------------------------------------------------------- analysis


def run_to_run(df: pd.DataFrame) -> pd.DataFrame:
    """Per instance and model: agreement between the 10 main runs, the coefficient of
    variation of each feature's weight across runs and whether the top-1 feature changes."""
    main = df[df["setting"] == MAIN]
    rows = []
    for (model, split, inst), g in main.groupby(["model", "split_id", "instance_id"], sort=False):
        W = g.sort_values("run")[FEATURES].to_numpy()
        pairs = list(combinations(range(len(W)), 2))
        A, B = W[[i for i, _ in pairs]], W[[j for _, j in pairs]]
        rc = rank_correlations(A, B)
        fa = topk_agreement(A, B, 5)["feature_agreement@5"]
        mean = W.mean(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cv = W.std(axis=0, ddof=1) / np.abs(mean)
        cv = cv[np.isfinite(cv)]
        top1 = np.argmax(np.abs(W), axis=1)
        rows.append({"model": model, "split_id": split, "project": g["project"].iloc[0], "instance_id": inst,
                     "outcome": g["outcome"].iloc[0], "spearman": float(np.nanmean(rc["spearman"])),
                     "fa5": float(fa.mean()), "cv_mean": float(cv.mean()), "cv_median": float(np.median(cv)),
                     "top1_changes": bool(len(set(top1)) > 1), "r2": float(g["score"].mean())})  # fmt: skip
    return pd.DataFrame(rows)


def grid_vs_main(df: pd.DataFrame) -> pd.DataFrame:
    """Per instance: each grid setting vs the main setting's run 0 (Spearman, FA@5, R²)."""
    key = ["model", "split_id", "instance_id"]
    ref = df[(df["setting"] == MAIN) & (df["run"] == 0)].set_index(key).sort_index()
    rows = []
    for name, g in df[df["setting"] != MAIN].groupby("setting", sort=False):
        g = g.set_index(key).sort_index()
        r = ref.loc[g.index]
        A, B = r[FEATURES].to_numpy(), g[FEATURES].to_numpy()
        rc = rank_correlations(A, B)
        rows.append(pd.DataFrame({"setting": name, "model": g.index.get_level_values(0), "split_id": g.index.get_level_values(1),
                                  "instance_id": g.index.get_level_values(2), "spearman": rc["spearman"],
                                  "fa5": topk_agreement(A, B, 5)["feature_agreement@5"], "r2": g["score"].to_numpy()}))  # fmt: skip
    main = ref.reset_index()
    rows.append(pd.DataFrame({"setting": MAIN, "model": main["model"], "split_id": main["split_id"],
                              "instance_id": main["instance_id"], "spearman": 1.0, "fa5": 1.0, "r2": main["score"]}))  # fmt: skip
    return pd.concat(rows, ignore_index=True)


def summarise_run_to_run(rr: pd.DataFrame) -> pd.DataFrame:
    return rr.groupby("model").agg(
        instances=("instance_id", "size"), spearman=("spearman", "mean"), fa5=("fa5", "mean"),
        cv_mean=("cv_mean", "mean"), cv_median=("cv_median", "median"),
        top1_change_pct=("top1_changes", lambda s: 100 * s.mean()), r2=("r2", "mean"),
    ).reset_index()  # fmt: skip


def summarise_grid(gv: pd.DataFrame) -> pd.DataFrame:
    return gv.groupby(["setting", "model"], sort=False).agg(
        instances=("instance_id", "size"), spearman=("spearman", "mean"), fa5=("fa5", "mean"), r2=("r2", "mean"),
    ).reset_index()  # fmt: skip
