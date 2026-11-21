"""Generate the step 7 reports and figures.

Usage::

    python -m scripts.make_step7_reports

Requires ``python -m scripts.refit_permuted`` and ``python -m scripts.run_stability``.
"""

from __future__ import annotations

import argparse
import sys

from sdp.analysis import step7
from sdp.config import DEFAULT_CONFIG, load_config
from sdp.log import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    r = step7.load(cfg)
    cv = step7.cross_version_report(cfg, r)
    step7.topk_report(cfg, r)
    step7.unchanged_report(cfg, r)
    step7.seed_report(cfg, r, cv)
    proj = step7.cross_project_report(cfg, r)
    step7.drift_report(cfg, r, cv)
    r["summary_points"] = step7.summary_points(r, cv, proj)
    step7.summary_report(cfg, r, cv, proj)
    step7.runtime_report(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
