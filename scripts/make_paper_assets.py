"""Regenerate the paper assets: figures, LaTeX tables and the numbers file.

Usage::

    python -m scripts.make_paper_assets [--collect] [--reports reports] [--data paper/data] [--out paper]

``--collect`` first refreshes ``paper/data/`` from ``results/`` and ``artifacts/`` (only
possible where the experiments were run); otherwise the committed ``paper/data/`` and
``reports/`` are used, so this also works in a fresh clone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from sdp.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from sdp.log import setup_logging
from sdp.paper import collect, figures, numbers, tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--reports", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    reports = Path(args.reports) if args.reports else Path(cfg.paths.reports_dir)
    out = Path(args.out) if args.out else REPO_ROOT / "paper"
    data = Path(args.data) if args.data else out / "data"
    if args.collect:
        fs = yaml.safe_load((REPO_ROOT / "configs" / "fidelity.yaml").read_text(encoding="utf-8"))
        collect.collect(cfg, data, Path(cfg.paths.artifacts_dir) / fs.get("output_dir", "fidelity"))
    figures.make_all(reports, data, out / "figures")
    tables.make_all(reports, data, out / "tables")
    root = REPO_ROOT if reports.resolve().parent == REPO_ROOT.resolve() else reports.resolve().parent
    numbers.write_numbers(root, out / "numbers.tex")
    problems = numbers.check_numbers(root, out / "numbers.tex")
    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
