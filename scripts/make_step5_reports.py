"""Generate the step 5 reports and figures.

Usage::

    python -m scripts.make_step5_reports

Requires ``python -m scripts.run_robustness`` (and the regenerated step 4 store).
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from sdp.analysis import explanations as E
from sdp.analysis import step5
from sdp.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from sdp.explain.runner import experiment_splits, load_explain_settings
from sdp.log import setup_logging
from sdp.robustness.common import load_robustness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--only", nargs="*", default=None,
                        choices=["lime", "background", "preprocessing", "collinearity", "runtime"])  # fmt: skip
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    es = load_explain_settings(REPO_ROOT / "configs" / "explain.yaml")
    rs = load_robustness()
    info_path = cfg.paths.results_dir / "robustness" / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    want = set(args.only or ["lime", "background", "preprocessing", "collinearity", "runtime"])

    if want & {"lime", "collinearity"}:
        full = E.load_explanations(es.root(cfg))
    if "lime" in want:
        lime = full.subset(full.meta["method"].isin(["LIME", "LIME_mean_sampling"]).to_numpy())
        sample = pd.read_csv(cfg.paths.results_dir / "robustness" / "lime_instance_sample.csv")
        step5.lime_report(cfg, rs, E.instance_cv(lime), sample, info)
    if "background" in want:
        step5.background_report(cfg, rs, info)
    if "preprocessing" in want:
        step5.preprocessing_report(cfg, es, rs)
    if "collinearity" in want:
        splits, frames = experiment_splits(cfg, es)
        ex_main = full.subset((full.meta["method"] != "LIME_mean_sampling").to_numpy())
        step5.collinearity_report(cfg, es, rs, splits, frames, ex_main)
    if "runtime" in want:
        step5.runtime_report(cfg, rs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
