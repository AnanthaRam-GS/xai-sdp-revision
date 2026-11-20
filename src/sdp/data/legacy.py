"""Row-level comparison of release CSVs from different sources.

Used for (a) the locally held legacy files in ``data/legacy/<project>/`` (not
redistributed) and (b) the GitHub mirror versus Zenodo.

Two raw layouts exist. Zenodo files have ``name`` (project), ``version`` and
``name.1`` (class); the GitHub mirror has only ``name`` (class). Both are reduced to
the same *content* columns (class name, the 20 metrics and the bug count) before
comparison. Rows are compared after sorting by class name, so line endings, column
order of the identifying fields and row order do not matter.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sdp.config import version_sort_key
from sdp.data.download import md5_of, read_raw_csv

logger = logging.getLogger(__name__)

CLASS = "class_name"


def content_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Reduce either raw layout to class name + metrics + bug count, sorted by class."""
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "name.1" in df.columns:
        df = df.drop(columns=[c for c in ("name", "version") if c in df.columns])
        df = df.rename(columns={"name.1": CLASS})
    else:
        df = df.drop(columns=[c for c in ("version",) if c in df.columns]).rename(columns={"name": CLASS})
    cols = [CLASS] + [c for c in df.columns if c != CLASS]
    return df[cols].sort_values(cols, kind="mergesort").reset_index(drop=True)


def _bytes(src: Path | bytes) -> bytes:
    return src if isinstance(src, bytes) else Path(src).read_bytes()


def compare_release(a_src: Path | bytes, b_src: Path | bytes) -> dict:
    """Compare two release CSVs (paths or raw bytes) row by row."""
    a_bytes, b_bytes = _bytes(a_src), _bytes(b_src)
    a, b = content_frame(read_raw_csv(a_bytes)), content_frame(read_raw_csv(b_bytes))
    result = {
        "rows_a": len(a),
        "rows_b": len(b),
        "same_columns": list(a.columns) == list(b.columns),
        "byte_identical": md5_of(a_bytes) == md5_of(b_bytes),
        # Zenodo serves CRLF line endings; other copies are often LF.
        "identical_ignoring_line_endings": a_bytes.replace(b"\r\n", b"\n") == b_bytes.replace(b"\r\n", b"\n"),
        "differing_rows": None,
        "identical": False,
    }
    if result["same_columns"] and len(a) == len(b):
        diff = ~((a == b) | (a.isna() & b.isna())).all(axis=1)
        result["differing_rows"] = int(diff.sum())
        result["identical"] = result["differing_rows"] == 0
    return result


def compare_legacy(legacy_dir: Path, raw_dir: Path, project: str) -> pd.DataFrame:
    """Compare every legacy release of ``project`` with the file in ``data/raw``, if any."""
    folder = legacy_dir / project
    if not folder.is_dir():
        logger.info("No legacy folder at %s; skipping the legacy comparison.", folder)
        return pd.DataFrame()

    records = []
    prefix = f"{project}-"
    files = sorted(folder.glob(f"{prefix}*.csv"), key=lambda p: version_sort_key(p.stem[len(prefix):]))
    for path in files:
        version = path.stem[len(prefix):]
        raw_path = raw_dir / path.name
        row = {"project": project, "version": version, "legacy_file": path.name}
        if not raw_path.exists():
            row.update(status="no counterpart in data/raw", identical=None)
            logger.warning("Legacy %s %s: no counterpart in data/raw; cannot compare.", project, version)
        else:
            res = compare_release(path, raw_path)
            row.update(
                legacy_rows=res["rows_a"],
                raw_rows=res["rows_b"],
                byte_identical=res["byte_identical"],
                identical_ignoring_line_endings=res["identical_ignoring_line_endings"],
                differing_rows=res["differing_rows"],
                identical=res["identical"],
                status="identical" if res["identical"] else "DIFFERENT",
            )
            logger.log(
                logging.INFO if res["identical"] else logging.WARNING,
                "Legacy %s %s vs data/raw: %s (rows %d vs %d, differing rows=%s).",
                project, version, row["status"], res["rows_a"], res["rows_b"], res["differing_rows"],
            )  # fmt: skip
        records.append(row)
    out = pd.DataFrame(records)
    # Keep counts as integers even when some releases have no counterpart (missing values).
    for col in ("legacy_rows", "raw_rows", "differing_rows"):
        if col in out.columns:
            out[col] = out[col].astype("Int64")
    return out
