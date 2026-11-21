"""Generate the step 6 reports and figures.

Usage::

    python -m scripts.make_step6_reports

Requires ``python -m scripts.run_fidelity``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from sdp.analysis import step6
from sdp.config import CONFIG_DIR, DEFAULT_CONFIG, REPO_ROOT, load_config
from sdp.log import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--settings", default=str(CONFIG_DIR / "fidelity.yaml"))
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    fs = yaml.safe_load(Path(args.settings).read_text(encoding="utf-8"))
    root = Path(cfg.paths.artifacts_dir) / fs.get("output_dir", "fidelity")
    info_path = cfg.paths.results_dir / "fidelity" / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    inst = step6.deletion_insertion_report(cfg, root, info)
    step6.faithfulness_report(cfg, inst)
    roar_split = step6.roar_report(cfg, root, info)
    verdict = step6.sanity_report(cfg, root, fs["sanity"]["flag_threshold"])
    step6.summary_report(cfg, inst, roar_split, verdict)
    step6.runtime_report(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
