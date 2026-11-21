"""Compute explanations for the persisted time-aware models (step 4).

Usage::

    python -m scripts.run_explanations [--settings configs/explain.yaml] [--n-jobs 10]
        [--smoke] [--smoke-split ta-consecutive:ant:1.3>1.4] [--estimate-only] [--reduced]

``--smoke`` first explains one split with seed 0 and every method, logs the timings
and estimates the full runtime. If the estimate exceeds ``max_hours`` (8 h), LIME and
KernelSHAP are switched to their reduced sample sizes (``reduced_params``) and the
reduction is logged. Requires the processed data and the step 3 artefacts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys

from sdp.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from sdp.explain.base import rename_method
from sdp.explain.runner import (
    MissingArtifactError,
    build_jobs,
    estimate_runtime,
    experiment_splits,
    load_explain_settings,
    missing_artifacts,
    run_explanations,
)
from sdp.log import setup_logging
from sdp.seed import set_seed

logger = logging.getLogger("run_explanations")
DEFAULT_SETTINGS = REPO_ROOT / "configs" / "explain.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", default=str(DEFAULT_SETTINGS))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="run a smoke test and a runtime estimate first")
    parser.add_argument("--smoke-split", default=None, help="split id for the smoke test (default: median size)")
    parser.add_argument("--estimate-only", action="store_true", help="stop after the estimate")
    parser.add_argument("--reduced", action="store_true", help="use the reduced LIME/KernelSHAP sample sizes")
    parser.add_argument("--keep-method-as", nargs=2, metavar=("OLD", "NEW"), default=None,
                        help="before running, keep the stored results of method OLD under the name NEW")  # fmt: skip
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    settings = load_explain_settings(args.settings)
    if args.reduced:
        settings = settings.with_reduction()
    n_jobs = args.n_jobs or settings.n_jobs
    if args.keep_method_as:
        moved = rename_method(settings.root(cfg), *args.keep_method_as)
        logger.info("Kept %d stored %s file(s) as %s.", len(moved), *args.keep_method_as)
    splits, frames = experiment_splits(cfg, settings)
    jobs = build_jobs(cfg, settings, frames, splits)
    missing = missing_artifacts(jobs)
    if missing:
        for path in missing:
            logger.error("Missing artefact: %s", path)
        raise MissingArtifactError(f"{len(missing)} persisted model(s) missing; nothing was retrained")
    logger.info("%d splits, %d jobs; all %d persisted models found.", len(splits), len(jobs), len(jobs))

    if args.smoke:
        by_size = sorted(splits, key=lambda s: len(s.test_idx))
        smoke_split = next((s for s in splits if s.split_id == args.smoke_split), by_size[len(by_size) // 2])
        smoke_settings = dataclasses.replace(settings, expensive_seeds=[0])
        smoke_jobs = build_jobs(cfg, smoke_settings, frames, [smoke_split], seeds=[0])
        info = run_explanations(cfg, smoke_settings, smoke_jobs, frames, n_jobs=n_jobs, tag="smoke")
        timings = info.pop("timings")
        for row in timings.itertuples():
            logger.info("Smoke %-4s %-20s %4d instances: %7.2f s (%.4f s/instance)", row.model, row.method,
                        row.n_instances, row.time_s, row.time_s / row.n_instances)  # fmt: skip
        eff = info["cpu_time_s"] / max(info["wall_time_s"] * min(n_jobs, len(smoke_jobs)), 1e-9)
        est = estimate_runtime(timings, jobs, n_jobs, eff)
        reduced_est = None
        if est["wall_hours"] > settings.max_hours and not settings.reduced:
            scale = {"LIME": settings.reduced_params["LIME"]["num_samples"] / settings.params["LIME"]["num_samples"],
                     "KernelSHAP": settings.reduced_params["KernelSHAP"]["nsamples"]
                     / settings.params["KernelSHAP"]["nsamples"]}  # fmt: skip
            reduced_est = estimate_runtime(timings, jobs, n_jobs, eff, scale)
            logger.warning("Estimated %.1f h > %.0f h: reducing LIME num_samples to %d and KernelSHAP nsamples "
                           "to %d (new estimate %.1f h).", est["wall_hours"], settings.max_hours,
                           settings.reduced_params["LIME"]["num_samples"],
                           settings.reduced_params["KernelSHAP"]["nsamples"], reduced_est["wall_hours"])  # fmt: skip
            settings = settings.with_reduction()
            jobs = build_jobs(cfg, settings, frames, splits)
        logger.info("Estimated full run: %.2f CPU h, ~%.2f h wall with n_jobs=%d (efficiency %.2f).",
                    est["cpu_hours"], est["wall_hours"], n_jobs, est["parallel_efficiency"])  # fmt: skip
        out = cfg.paths.results_dir / "explanations" / "runtime_estimate.json"
        out.write_text(json.dumps({"smoke_split": smoke_split.split_id, "smoke": info, "estimate": est,
                                   "reduced_estimate": reduced_est, "reduced": settings.reduced, "n_jobs": n_jobs},
                                  indent=2, default=str), encoding="utf-8")  # fmt: skip
        if args.estimate_only:
            return 0

    run_explanations(cfg, settings, jobs, frames, n_jobs=n_jobs, tag="full")
    return 0


if __name__ == "__main__":
    sys.exit(main())
