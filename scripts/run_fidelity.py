"""Run the step 6 fidelity tests.

Usage::

    python -m scripts.run_fidelity [--parts curves train roar sanity] [--smoke] [--estimate-only] [--n-jobs 10]

Parts: deletion/insertion curves and faithfulness (``curves``); training-set
attributions for ROAR/KAR (``train``); ROAR/KAR retraining (``roar``); model- and
data-randomisation sanity checks (``sanity``). ``--smoke`` first runs everything on
one split (seed 0), estimates the full runtime and applies the reductions of
``configs/fidelity.yaml`` in order if the estimate exceeds ``max_hours``: (1) curves and
faithfulness on up to 200 instances per split, (2) ROAR/KAR k ∈ {1, 3, 5, 10},
(3) ROAR/KAR seeds 0–1. Finished jobs (existing output files) are skipped on reruns.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from sdp.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from sdp.experiments.persistence import artifact_dir, safe_split_id
from sdp.explain.base import partition_path
from sdp.explain.runner import experiment_splits, load_explain_settings
from sdp.fidelity import roar as R
from sdp.fidelity import sanity as S
from sdp.fidelity.evaluation import curves_job
from sdp.log import setup_logging
from sdp.robustness.common import run_parallel
from sdp.seed import set_seed

logger = logging.getLogger("run_fidelity")
PARTS = ("curves", "train", "roar", "sanity")
MODELS = ("mlp", "lr", "rf", "xgb")


def run_parts(cfg, es, fs, splits, frames, parts, n_jobs, root: Path, label: str) -> dict:
    """Run the requested parts; returns wall times and timing frames."""
    out, timings = {}, {}
    seeds = fs["roar"]["seeds"]
    if "curves" in parts:
        t0 = time.perf_counter()
        tasks = [(cfg, es, s, frames[s.project], m, str(root / "curves"), fs["curves"]) for s in splits for m in MODELS
                 if not (root / "curves" / f"{safe_split_id(s.split_id)}__{m}.parquet").exists()]  # fmt: skip
        tasks.sort(key=lambda t: -len(t[2].test_idx) * (6 if t[4] == "rf" else 1))
        res = run_parallel(curves_job, tasks, n_jobs, f"{label} curves")
        out["curves"], timings["curves"] = time.perf_counter() - t0, pd.DataFrame([r["timing"] for r in res])
    train_root = R.train_root(root / "attributions")
    if "train" in parts:
        t0 = time.perf_counter()
        tasks = [(cfg, es, s, frames[s.project], m, seed, str(train_root), fs["roar"]["lime_rows"])
                 for s in splits for m in MODELS for seed in seeds
                 if not all(partition_path(train_root, m, meth, safe_split_id(s.split_id), seed).exists()
                            for meth in R.TRAIN_METHODS[m])]  # fmt: skip
        tasks.sort(key=lambda t: -len(t[2].fit_idx) * (5 if t[4] == "mlp" else 1))
        res = run_parallel(R.train_attributions_job, tasks, n_jobs, f"{label} training attributions")
        out["train"], timings["train"] = time.perf_counter() - t0, pd.DataFrame([r["timing"] for r in res])
    if "roar" in parts:
        t0 = time.perf_counter()
        roar_dir = root / "roar"
        roar_dir.mkdir(parents=True, exist_ok=True)
        models_root = Path(cfg.paths.artifacts_dir) / "models" / es.experiment
        tasks = []
        for s in splits:
            for m in MODELS:
                for seed in seeds:
                    if (roar_dir / f"{safe_split_id(s.split_id)}__{m}__seed_{seed}.parquet").exists():
                        continue
                    adir = artifact_dir(models_root, "", s.split_id, m, es.models[m], seed)
                    tasks.append((cfg, s, frames[s.project], m, seed, str(adir), str(train_root),
                                  tuple(fs["roar"]["ks"]), fs["roar"]["n_random"]))  # fmt: skip
        tasks.sort(key=lambda t: -len(t[1].fit_idx) * {"mlp": 4, "rf": 3, "xgb": 1, "lr": 0.2}[t[3]])
        res = []
        for r in run_parallel(_roar_and_save, [(str(roar_dir),) + t for t in tasks], n_jobs, f"{label} ROAR/KAR"):
            res.append(r)
        out["roar"], timings["roar"] = time.perf_counter() - t0, pd.DataFrame([r["timing"] for r in res])
    if "sanity" in parts:
        t0 = time.perf_counter()
        sdir = root / "sanity"
        sdir.mkdir(parents=True, exist_ok=True)
        tasks = [("cascade", str(sdir), cfg, es, s, frames[s.project], None) for s in splits]
        tasks += [("data", str(sdir), cfg, es, s, frames[s.project], m) for s in splits for m in MODELS]
        tasks = [t for t in tasks if not _sanity_path(t).exists()]
        res = run_parallel(_sanity_and_save, tasks, n_jobs, f"{label} sanity")
        out["sanity"], timings["sanity"] = time.perf_counter() - t0, pd.DataFrame([r["timing"] for r in res])
    return {"wall_s": out, "timings": timings}


def _roar_and_save(out_dir, *args):
    r = R.roar_job(*args)
    split, model, seed = args[1], args[3], args[4]
    pd.DataFrame(r["rows"]).to_parquet(Path(out_dir) / f"{safe_split_id(split.split_id)}__{model}__seed_{seed}.parquet", index=False)
    return {"timing": r["timing"]}


def _sanity_path(task) -> Path:
    kind, out_dir, _, _, split, _, model = task
    return Path(out_dir) / f"{safe_split_id(split.split_id)}__{kind}{'' if model is None else '__' + model}.parquet"


def _sanity_and_save(kind, out_dir, cfg, es, split, df, model):
    r = S.cascade_job(cfg, es, split, df) if kind == "cascade" else S.data_randomisation_job(cfg, es, split, df, model)
    pd.DataFrame(r["rows"]).to_parquet(_sanity_path((kind, out_dir, cfg, es, split, df, model)), index=False)
    return {"timing": r["timing"]}


def estimate(smoke: dict, splits, smoke_split, fs: dict, n_jobs: int) -> dict:
    """Scale the smoke-test CPU time per part to all splits (by test or fitting rows)."""
    n_test = {s.split_id: len(s.test_idx) for s in splits}
    n_fit = {s.split_id: len(s.fit_idx) for s in splits}
    t = smoke["timings"]
    cpu = {}
    if "curves" in t:
        per = t["curves"]["time_s"].sum() / n_test[smoke_split.split_id]
        cpu["curves"] = per * sum(n_test.values())
        cpu["curves_reduced"] = per * sum(min(v, fs["curves"]["reduced_sample_per_split"]) for v in n_test.values())
    seeds = len(fs["roar"]["seeds"])
    for part in ("train", "roar"):
        if part in t:
            per = t[part]["time_s"].sum() / n_fit[smoke_split.split_id]  # one seed in the smoke test
            cpu[part] = per * sum(n_fit.values()) * seeds
    if "roar" in cpu:
        cpu["roar_reduced_ks"] = cpu["roar"] * len(fs["roar"]["reduced_ks"]) / len(fs["roar"]["ks"])
    if "sanity" in t:
        cpu["sanity"] = t["sanity"]["time_s"].sum() / n_test[smoke_split.split_id] * sum(n_test.values())
    main_parts = [p for p in ("curves", "train", "roar", "sanity") if p in cpu]
    total = sum(cpu[p] for p in main_parts)
    eff = 0.7
    return {"cpu_s": cpu, "cpu_hours": total / 3600, "wall_hours": total / (n_jobs * eff) / 3600, "efficiency": eff}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    parser.add_argument("--settings", default=str(REPO_ROOT / "configs" / "fidelity.yaml"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    es = load_explain_settings(REPO_ROOT / "configs" / "explain.yaml")
    fs = yaml.safe_load(Path(args.settings).read_text(encoding="utf-8"))
    n_jobs = args.n_jobs or fs.get("n_jobs", 10)
    splits, frames = experiment_splits(cfg, es)
    root = Path(cfg.paths.artifacts_dir) / fs.get("output_dir", "fidelity")
    res_dir = cfg.paths.results_dir / "fidelity"
    res_dir.mkdir(parents=True, exist_ok=True)
    reductions = []

    if args.smoke:
        by_size = sorted(splits, key=lambda s: len(s.test_idx))
        smoke_split = by_size[len(by_size) // 2]
        sfs = copy.deepcopy(fs)
        sfs["roar"]["seeds"] = [0]
        sroot = root / "_smoke"
        smoke = run_parts(cfg, es, sfs, [smoke_split], frames, args.parts, n_jobs, sroot, "smoke")
        est = estimate(smoke, splits, smoke_split, fs, n_jobs)
        wall = est["wall_hours"]
        to_h = lambda s: s / (n_jobs * est["efficiency"]) / 3600  # noqa: E731
        c = est["cpu_s"]
        if wall > fs["max_hours"] and "curves" in c:
            wall -= to_h(c["curves"] - c["curves_reduced"])
            fs["curves"]["sample_per_split"] = fs["curves"]["reduced_sample_per_split"]
            reductions.append(f"curves and faithfulness on up to {fs['curves']['sample_per_split']} instances per split")
        if wall > fs["max_hours"] and "roar" in c:
            wall -= to_h(c["roar"] - c["roar_reduced_ks"])
            fs["roar"]["ks"] = fs["roar"]["reduced_ks"]
            reductions.append(f"ROAR/KAR k in {fs['roar']['ks']}")
        if wall > fs["max_hours"] and "roar" in c:
            frac = len(fs["roar"]["reduced_seeds"]) / len(fs["roar"]["seeds"])
            wall -= to_h((c.get("roar_reduced_ks", c["roar"]) + c.get("train", 0)) * (1 - frac))
            fs["roar"]["seeds"] = fs["roar"]["reduced_seeds"]
            reductions.append(f"ROAR/KAR seeds {fs['roar']['seeds']}")
        for r in reductions:
            logger.warning("Estimate above %s h: reduced %s.", fs["max_hours"], r)
        info = {"smoke_split": smoke_split.split_id, "smoke_wall_s": smoke["wall_s"],
                "smoke_timings": {k: v.to_dict(orient="records") for k, v in smoke["timings"].items()},
                "estimate": est, "wall_hours_after_reductions": wall, "reductions": reductions, "n_jobs": n_jobs}  # fmt: skip
        (res_dir / "runtime_estimate.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        logger.info("Estimated full run: %.2f CPU h, ~%.2f h wall; reductions: %s.", est["cpu_hours"], wall,
                    reductions or "none")  # fmt: skip
        if args.estimate_only:
            return 0

    full = run_parts(cfg, es, fs, splits, frames, args.parts, n_jobs, root, "full")
    for part, t in full["timings"].items():
        t.to_csv(res_dir / f"timings_{part}.csv", index=False)
    info_path = res_dir / "run_info.json"
    prev = json.loads(info_path.read_text()) if info_path.exists() else {}
    prev.setdefault("wall_time_s", {}).update(full["wall_s"])
    prev["settings"] = fs
    if args.smoke:
        prev["reductions"] = reductions
    info_path.write_text(json.dumps(prev, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
