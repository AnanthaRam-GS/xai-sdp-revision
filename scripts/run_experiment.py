"""Run step 3 experiments, with an optional smoke test and runtime estimate first.

Usage::

    python -m scripts.run_experiment configs/experiments/main_time_aware.yaml [...]
        [--n-jobs 10] [--smoke] [--smoke-project ant] [--estimate-only]
        [--n-trials-time-aware 25]

``--smoke`` first runs the first experiment on one project with 2 seeds and 5 tuning
trials, logs its timing and uses it to estimate the runtime of all given experiments
(jobs already in the cache and jobs shared between experiments are counted once).
Requires the processed data (``python -m scripts.prepare_data``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time

import pandas as pd

from sdp.config import DEFAULT_CONFIG, load_config
from sdp.data.study import StudyDataset, build_study
from sdp.eval.splits import project_frame
from sdp.experiments.config import load_experiment
from sdp.experiments.engine import build_jobs, run_experiment
from sdp.log import setup_logging
from sdp.models.registry import is_learned
from sdp.seed import set_seed

logger = logging.getLogger("run_experiment")


def estimate(cfg, experiments, smoke_timings: pd.DataFrame, smoke_wall: float, n_jobs: int) -> dict:
    """Estimate wall time from smoke-test costs per tuning trial and per fit, per row."""
    t = smoke_timings.copy()
    learned = t[t["model"].map(is_learned)]
    tun = learned[learned["tuning_time_s"] > 0]
    n_trials_smoke = 5
    trial_rate = (tun["tuning_time_s"] / (n_trials_smoke * tun["n_fit"])).groupby(tun["model"]).mean()
    fit_rate = (t["fit_time_s"] / t["n_fit"]).groupby(t["model"]).mean()
    cpu_smoke = t["tuning_time_s"].sum() + t["fit_time_s"].sum()
    efficiency = min(1.0, cpu_smoke / (smoke_wall * n_jobs)) if smoke_wall > 0 else 1.0

    study = build_study(cfg)
    frames = {p: project_frame(cfg, study, p) for p in study.projects}
    cache = cfg.paths.results_dir / "_cache"
    seen, per_exp = set(), {}
    for exp in experiments:
        cpu = 0.0
        for job in build_jobs(cfg, exp, study, frames):
            key = job.cache_key(cfg)
            if key in seen or (cache / f"{key}.pkl").exists():
                continue
            seen.add(key)
            n_fit = len(job.splits[0].fit_idx)
            cpu += job.n_trials * trial_rate.get(job.model, 0.0) * n_fit
            cpu += sum(fit_rate.get(job.model, 0.0) * len(s.fit_idx) for s in job.splits)
        per_exp[exp.name] = cpu
    total_cpu = sum(per_exp.values())
    # Wall time is bounded below by the parallel efficiency seen in the smoke test.
    eff = max(efficiency, 0.3)
    return {
        "trial_seconds_per_row": trial_rate.to_dict(),
        "fit_seconds_per_row": fit_rate.to_dict(),
        "smoke_parallel_efficiency": efficiency,
        "cpu_hours_per_experiment": {k: v / 3600 for k, v in per_exp.items()},
        "total_cpu_hours": total_cpu / 3600,
        "estimated_wall_hours": total_cpu / (n_jobs * eff) / 3600,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="+", help="experiment YAML files")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="run a smoke test and a runtime estimate first")
    parser.add_argument("--smoke-project", default="ant")
    parser.add_argument("--estimate-only", action="store_true", help="stop after the estimate")
    parser.add_argument("--n-trials-time-aware", type=int, default=None, help="override the time-aware trials")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    if args.n_trials_time_aware is not None:
        cfg = dataclasses.replace(cfg, tuning=dataclasses.replace(cfg.tuning, n_trials_time_aware=args.n_trials_time_aware))
        logger.info("Time-aware tuning trials overridden: %d.", args.n_trials_time_aware)
    set_seed(cfg.seed)
    experiments = [load_experiment(p) for p in args.experiments]
    n_jobs = args.n_jobs or experiments[0].n_jobs

    if args.smoke:
        smoke = experiments[0].smoke(args.smoke_project)
        t0 = time.perf_counter()
        res = run_experiment(cfg, smoke, n_jobs=n_jobs)
        wall = time.perf_counter() - t0
        logger.info("Smoke test (%s: project %s, %d seeds, %d trials, %d jobs) took %.1f s.",
                    smoke.name, args.smoke_project, len(smoke.seeds), smoke.n_trials, res.info["jobs"], wall)  # fmt: skip
        est = estimate(cfg, experiments, res.timings, wall, n_jobs)
        out = cfg.paths.results_dir / "runtime_estimate.json"
        out.write_text(json.dumps(est | {"smoke_wall_s": wall, "n_jobs": n_jobs}, indent=2), encoding="utf-8")
        for name, h in est["cpu_hours_per_experiment"].items():
            logger.info("Estimate %-28s %7.2f CPU h", name, h)
        logger.info("Estimated total: %.2f CPU h, ~%.2f h wall time with n_jobs=%d (smoke efficiency %.2f).",
                    est["total_cpu_hours"], est["estimated_wall_hours"], n_jobs, est["smoke_parallel_efficiency"])  # fmt: skip
        if args.estimate_only:
            return 0

    for exp in experiments:
        run_experiment(cfg, exp, n_jobs=n_jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
