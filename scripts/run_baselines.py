"""Run the trivial baselines on the study dataset and write the summary report.

Usage::

    python -m scripts.run_baselines [--config configs/default.yaml]
                                    [--protocols time_aware within_release]
                                    [--preprocessing standard]

Requires the processed data (``python -m scripts.prepare_data``).
"""

from __future__ import annotations

import argparse
import logging
import sys

from sdp.config import DEFAULT_CONFIG, load_config
from sdp.data.study import build_study
from sdp.eval.splits import PROTOCOLS, project_frame
from sdp.experiments.baselines import BASELINES
from sdp.experiments.runner import ExperimentSpec, run_experiment
from sdp.experiments.summary import write_baselines_summary
from sdp.log import setup_logging
from sdp.seed import set_seed

logger = logging.getLogger("run_baselines")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    parser.add_argument("--preprocessing", default=None, help="override the configured preprocessing mode")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    study = build_study(cfg)
    frames = {p: project_frame(cfg, study, p) for p in study.projects}
    mode = args.preprocessing or cfg.evaluation.preprocessing

    results = {}
    for protocol in args.protocols:
        suffix = f"time_aware_{cfg.evaluation.time_aware_variant}" if protocol == "time_aware" else protocol
        spec = ExperimentSpec(name=f"baselines_{suffix}", models=BASELINES, preprocessing=mode, protocol=protocol)
        results[spec.name], _ = run_experiment(cfg, study, spec, frames)

    write_baselines_summary(cfg, results, cfg.paths.reports_dir / "baselines_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
