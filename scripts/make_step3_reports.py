"""Generate the step 3 reports from ``results/<experiment>/``.

Usage::

    python -m scripts.make_step3_reports [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys

from sdp.analysis import step3
from sdp.config import DEFAULT_CONFIG, load_config
from sdp.log import setup_logging

EXPERIMENTS = [
    "main_time_aware",
    "main_within_release",
    "ablation_preprocessing",
    "ablation_imbalance",
    "main_time_aware_cumulative",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)

    step3.model_report(cfg, "main_time_aware", "step3_main_time_aware", "Main results: time-aware (consecutive)")
    step3.model_report(cfg, "main_within_release", "step3_within_release", "Within-release (10 x 5-fold CV)")
    step3.ablation_report(cfg, "ablation_preprocessing", "step3_ablation_preprocessing",
                          "Ablation: preprocessing", "preprocessing")  # fmt: skip
    step3.ablation_report(cfg, "ablation_imbalance", "step3_ablation_imbalance",
                          "Ablation: class-imbalance handling", "imbalance")  # fmt: skip
    step3.cumulative_report(cfg)
    step3.runtime_report(cfg, EXPERIMENTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
