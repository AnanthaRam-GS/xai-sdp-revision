"""Build the replication archive dist/replication_package_v<version>.zip.

Usage::

    python -m scripts.build_archive [--version 1.0.0]

Contents: every file committed at ``HEAD`` (source code, configurations, raw data and
``MANIFEST.csv``, reports, paper assets, lock file, ``REPLICATION.md``) plus the result
files needed to rerun the closing analyses and ``make paper-assets``: for every
experiment its metrics, hyperparameters, timings and run information; the test and
validation predictions of ``main_time_aware``; and the step 4–7 summary files. The
per-instance attribution store (``artifacts/``, about 300 MB) and the fitted models are
not included; ``REPLICATION.md`` explains how to regenerate them. Prints the size and
SHA-256 and writes them to ``dist/<archive>.sha256``.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

from sdp.config import REPO_ROOT

RESULT_PATTERNS = [
    "*/metrics.csv", "*/hyperparameters.csv", "*/timings.csv", "*/run_info.json",
    "main_time_aware/predictions.csv", "main_time_aware/validation_predictions.csv",
    "main_time_aware/validation_predictions_check.csv", "runtime_estimate.json",
    "explanations/*.json", "explanations/*.csv", "explanations/v1/*.json", "explanations/v1/*.csv",
    "robustness/*.json", "robustness/timings_*.csv", "robustness/lime_instance_sample.csv",
    "fidelity/*.json", "fidelity/timings_*.csv",
    "stability/*.csv", "stability/*.json", "stability/cross_version.parquet", "stability/topk_frequency.parquet",
    "stability/directions.parquet", "stability/seed_pairs.parquet", "stability/unchanged.parquet",
]  # fmt: skip


def committed_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args(argv)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, capture_output=True, text=True).stdout
    if dirty.strip():
        print("Refusing to build: the working tree has uncommitted changes to tracked files.")
        return 1
    name = f"replication_package_v{args.version}"
    dist = REPO_ROOT / "dist"
    dist.mkdir(exist_ok=True)
    zpath = dist / f"{name}.zip"
    results = sorted({p for pat in RESULT_PATTERNS for p in (REPO_ROOT / "results").glob(pat)
                      if "_cache" not in p.parts and "smoke" not in p.parent.name})  # fmt: skip
    n = 0
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in committed_files():
            # the committed blob at HEAD, not the working copy
            blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO_ROOT, capture_output=True, check=True).stdout
            info = zipfile.ZipInfo(f"{name}/{rel}", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, blob)
            n += 1
        for p in results:
            info = zipfile.ZipInfo(f"{name}/{p.relative_to(REPO_ROOT).as_posix()}", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
            n += 1
        note = (f"Built from commit {head} with scripts/build_archive.py.\n"
                f"{len(committed_files())} committed files and {len(results)} result files.\n")  # fmt: skip
        z.writestr(zipfile.ZipInfo(f"{name}/ARCHIVE_INFO.txt", date_time=(2026, 1, 1, 0, 0, 0)), note)
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    size = zpath.stat().st_size
    (dist / f"{name}.zip.sha256").write_text(f"{sha}  {zpath.name}\n", encoding="utf-8")
    print(f"{zpath.relative_to(REPO_ROOT)}: {n + 1} files, {size / 2**20:.1f} MiB, SHA-256 {sha}, commit {head}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
