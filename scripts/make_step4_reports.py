"""Generate the step 4 reports and draft figures from the attribution store.

Usage::

    python -m scripts.make_step4_reports [--settings configs/explain.yaml]

Writes ``reports/step4_*.md`` / ``.csv`` and ``reports/figures/step4/*.png``.
Requires ``python -m scripts.run_explanations`` to have finished.
"""

from __future__ import annotations

import argparse
import json
import sys

from sdp.analysis import explanations as E
from sdp.analysis import step4
from sdp.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from sdp.explain.runner import load_explain_settings
from sdp.log import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", default=str(REPO_ROOT / "configs" / "explain.yaml"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    settings = load_explain_settings(args.settings)
    root = settings.root(cfg)
    info = json.loads((cfg.paths.results_dir / "explanations" / "run_info_full.json").read_text(encoding="utf-8"))

    ex = E.load_explanations(root)
    imp = step4.importance_report(cfg, ex)
    step4.direction_report(cfg, ex, imp)
    step4.size_share_report(cfg, ex)
    step4.agreement_report(cfg, ex)
    step4.sanity_report(cfg, root, settings.completeness_tolerance, info)
    step4.runtime_report(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
