"""Step 8 closing analyses: NECM cost-ratio sweep and calibration.

Usage::

    python -m scripts.run_closing_analyses [--n-jobs 10]

Derives the validation-split predictions of the persisted ``main_time_aware`` models by
inference only (``results/main_time_aware/validation_predictions.csv``, with a check that
they reproduce the stored thresholds and test predictions), then writes
``reports/step8_cost_ratio.md`` and ``reports/step8_calibration.md``.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from sdp.analysis import cost_calibration as CC
from sdp.analysis import step8
from sdp.analysis.validation_predictions import split_validation_predictions, verify
from sdp.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from sdp.explain.runner import experiment_splits, load_explain_settings
from sdp.log import setup_logging
from sdp.robustness.common import run_parallel

logger = logging.getLogger("run_closing_analyses")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--explain-settings", default=str(REPO_ROOT / "configs" / "explain.yaml"))
    parser.add_argument("--n-jobs", type=int, default=10)
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    es = load_explain_settings(args.explain_settings)
    exp_dir = cfg.paths.results_dir / "main_time_aware"
    pred = pd.read_csv(exp_dir / "predictions.csv", dtype={"class_name": str})
    seeds = sorted(pred.loc[pred["model"].isin(CC.LEARNED), "seed"].unique())
    splits, frames = experiment_splits(cfg, es)
    res = run_parallel(split_validation_predictions, [(cfg, s, frames[s.project], seeds) for s in splits], args.n_jobs,
                       "Validation predictions")  # fmt: skip
    val = pd.concat([r["rows"] for r in res], ignore_index=True)
    check = verify([c for r in res for c in r["checks"]], pred[["split_id", "model", "seed", "y_prob"]])
    val.to_csv(exp_dir / "validation_predictions.csv", index=False)
    check.to_csv(exp_dir / "validation_predictions_check.csv", index=False)
    if check["threshold_diff"].max() > 1e-12 or check["test_prob_max_diff"].max() > 1e-9:
        raise RuntimeError("persisted models do not reproduce the stored step 3 thresholds or predictions")

    project = {s.split_id: s.project for s in splits}
    pred["project"] = pred["split_id"].map(project)
    val["project"] = val["split_id"].map(project)
    metrics = pd.read_csv(exp_dir / "metrics.csv")
    thresholds = metrics[(metrics["metric"] == "threshold") & (metrics["threshold_mode"] == "val_mcc")]
    sweep = CC.necm_sweep(pred, thresholds, val)
    ps = CC.per_split(sweep)
    step8.cost_report(cfg, ps, CC.sweep_summary(ps), CC.break_even(ps), CC.sweep_tests(ps), check)
    cal, rel = CC.calibration(pred, val)
    step8.calibration_report(cfg, cal, rel, CC.calibration_tests(cal))
    return 0


if __name__ == "__main__":
    sys.exit(main())
