"""Comparison of locally held (legacy) release CSVs with the Zenodo downloads.

The legacy files used in the original submission live in ``data/legacy/<project>/``
and are not redistributed. For every legacy release we check byte identity (md5), row
counts and per-row equality after sorting by class name.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sdp.config import version_sort_key
from sdp.data.download import md5_of, read_raw_csv

logger = logging.getLogger(__name__)

CLASS_COL = "name.1"


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    # Sort by class name, then by every column so that tied class names are ordered too.
    cols = [CLASS_COL] + [c for c in df.columns if c != CLASS_COL]
    return df.sort_values(cols, kind="mergesort").reset_index(drop=True)


def compare_release(legacy_path: Path, zenodo_path: Path) -> dict:
    """Compare one legacy CSV with its Zenodo counterpart."""
    legacy = read_raw_csv(legacy_path)
    zenodo = read_raw_csv(zenodo_path)
    result = {
        "legacy_rows": len(legacy),
        "zenodo_rows": len(zenodo),
        "same_columns": list(legacy.columns) == list(zenodo.columns),
        "byte_identical": md5_of(legacy_path) == md5_of(zenodo_path),
        "differing_rows": None,
        "identical": False,
    }
    if result["same_columns"] and len(legacy) == len(zenodo):
        a, b = _sorted(legacy), _sorted(zenodo)
        diff = ~((a == b) | (a.isna() & b.isna())).all(axis=1)
        result["differing_rows"] = int(diff.sum())
        result["identical"] = result["differing_rows"] == 0
    return result


def compare_legacy(legacy_dir: Path, raw_dir: Path, project: str) -> pd.DataFrame:
    """Compare every legacy release of ``project`` with the Zenodo file, if any."""
    folder = legacy_dir / project
    if not folder.is_dir():
        logger.info("No legacy folder at %s; skipping the legacy comparison.", folder)
        return pd.DataFrame()

    records = []
    prefix = f"{project}-"
    files = sorted(folder.glob(f"{prefix}*.csv"), key=lambda p: version_sort_key(p.stem[len(prefix):]))
    for path in files:
        version = path.stem[len(prefix):]
        zenodo_path = raw_dir / path.name
        row = {"project": project, "version": version, "legacy_file": path.name}
        if not zenodo_path.exists():
            row.update(status="no Zenodo counterpart", identical=None)
            logger.warning(
                "Legacy %s %s: no Zenodo counterpart, so it cannot be compared.", project, version
            )
        else:
            res = compare_release(path, zenodo_path)
            row.update(res)
            row["status"] = "identical" if res["identical"] else "DIFFERENT"
            logger.log(
                logging.INFO if res["identical"] else logging.WARNING,
                "Legacy %s %s vs Zenodo: %s (rows %d vs %d, byte-identical=%s, differing rows=%s).",
                project,
                version,
                row["status"],
                res["legacy_rows"],
                res["zenodo_rows"],
                res["byte_identical"],
                res["differing_rows"],
            )
        records.append(row)
    return pd.DataFrame(records)
