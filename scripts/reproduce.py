"""Reproduce the study end to end (steps 1–8), quickly, or only the paper assets.

Usage::

    python -m scripts.reproduce --mode full|quick|paper-assets [--steps 3 4] [--dry-run]

``full`` runs every step from the raw data (downloads only if a raw file is missing) and
prints the expected runtime of each step; ``quick`` runs a reduced version (ant and
synapse, two seeds, five Optuna trials, LIME/KernelSHAP on 20 instances per split) that
writes only to ``results_quick/``, ``reports_quick/`` and ``artifacts_quick/``;
``paper-assets`` regenerates ``paper/`` from the existing results and checks the numbers.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from sdp.config import REPO_ROOT
from sdp.repro import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["full", "quick", "paper-assets"], required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=None, help="run only these steps (1-8)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    args = parser.parse_args(argv)
    if args.mode == "paper-assets":
        code = subprocess.run([sys.executable, "-m", "scripts.make_paper_assets", "--collect"], cwd=REPO_ROOT).returncode
        return code or subprocess.run([sys.executable, "-m", "scripts.check_numbers"], cwd=REPO_ROOT).returncode
    return run(args.mode, dry_run=args.dry_run, steps=set(args.steps) if args.steps else None)


if __name__ == "__main__":
    sys.exit(main())
