"""Verify that every macro in paper/numbers.tex matches its source report or CSV.

Usage::

    python -m scripts.check_numbers [--numbers paper/numbers.tex] [--root .]

Re-reads every source, recomputes each value and compares it with the macro as an exact
string after rounding. Exits with status 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sdp.config import REPO_ROOT
from sdp.paper.numbers import check_numbers, registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--numbers", default=str(REPO_ROOT / "paper" / "numbers.tex"))
    parser.add_argument("--root", default=str(REPO_ROOT))
    args = parser.parse_args(argv)
    problems = check_numbers(Path(args.root), Path(args.numbers))
    n = len(registry())
    if problems:
        print(f"check_numbers: {len(problems)} problem(s) in {n} macros:")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"check_numbers: all {n} macros in {args.numbers} match their sources.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
