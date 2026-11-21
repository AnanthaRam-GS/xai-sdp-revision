"""Fresh-clone check: clone the repository, build the environment from the lock file, run
``make quick`` and the test suite there, and record the outcome.

Usage::

    python -m scripts.fresh_clone_check [--workdir /tmp/...] [--keep]

The clone is made from the current commit (``HEAD``) of this working copy into a new
temporary directory; the working copy and its results are not touched. The outcome and
durations are written to ``reports/step8_reproduction_check.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from sdp.config import REPO_ROOT


def sh(cmd: list[str], cwd: Path, log: Path) -> tuple[int, float]:
    t0 = time.perf_counter()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        code = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT).returncode
    return code, time.perf_counter() - t0


def tail(log: Path, n: int = 3) -> str:
    lines = [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    return " / ".join(lines[-n:])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--keep", action="store_true", help="keep the temporary clone")
    args = parser.parse_args(argv)
    work = Path(args.workdir or tempfile.mkdtemp(prefix="sdp-fresh-clone-"))
    clone = work / "repo"
    log = work / "fresh_clone.log"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    rows = []
    steps = [
        ("clone", ["git", "clone", "--quiet", "--no-hardlinks", str(REPO_ROOT), str(clone)], work),
        ("checkout", ["git", "checkout", "--quiet", head], clone),
        ("environment", ["uv", "venv", "--quiet", "--python", "3.11", ".venv"], clone),
        ("install (lock file)", ["uv", "pip", "install", "--quiet", "--python", ".venv/bin/python", "--require-hashes",
                                 "-r", "requirements.lock"], clone),
        ("install (package)", ["uv", "pip", "install", "--quiet", "--python", ".venv/bin/python", "--no-deps", "-e", "."], clone),
        ("make quick", ["make", "quick", "PYTHON=.venv/bin/python"], clone),
        ("tests", [".venv/bin/python", "-m", "pytest", "-q", "-p", "no:cacheprovider"], clone),
    ]  # fmt: skip
    ok = True
    for name, cmd, cwd in steps:
        code, secs = sh(cmd, cwd, log)
        rows.append((name, code, secs, tail(log, 2) if name in ("make quick", "tests") else ""))
        if code != 0:
            ok = False
            break
    quick_reports = sorted(p.name for p in (clone / "reports_quick").glob("*.md")) if (clone / "reports_quick").exists() else []
    real_untouched = not (clone / "results").exists() and not (clone / "artifacts").exists()
    md = [
        "# Step 8: fresh-clone reproduction check",
        "",
        f"Date (UTC): {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M')}. Commit: `{head}`. "
        f"Machine: {platform.platform()}, Python for the check environment: 3.11 (uv). The repository was "
        "cloned into a new temporary directory, the environment was created from `requirements.lock` (hashes "
        "required), then `make quick` and the test suite were run there. The working copy and its results were "
        "not used.",
        "",
        "| step | exit code | duration (min) | last output |",
        "|---|---|---|---|",
    ] + [f"| {n} | {c} | {s / 60:.1f} | {t.replace('|', '/')[:220]} |" for n, c, s, t in rows] + [
        "",
        f"Outcome: **{'passed' if ok else 'failed'}**. Total: {sum(s for _, _, s, _ in rows) / 60:.1f} min.",
        "",
        f"Quick-mode reports written in the clone: {len(quick_reports)} (`reports_quick/`); the clone's `results/` "
        f"and `artifacts/` {'were not created' if real_untouched else 'EXIST (unexpected)'}.",
        "",
    ]
    out = REPO_ROOT / "reports" / "step8_reproduction_check.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    if not args.keep and ok:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"Clone kept at {work}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
