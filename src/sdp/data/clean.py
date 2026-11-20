"""Per-release cleaning of the Jureczko CK-metric datasets.

Each release is cleaned on its own, in this order:

1. Standardise column names (lower case); rename ``name.1`` to ``class_name`` and
   ``bug`` to ``bug_count``. Fail loudly if any of the 20 metrics is missing.
2. Report missing or non-numeric values and drop the affected rows (if any).
3. Binarise the label: ``defective = 1 if bug_count > 0 else 0``.
4. Following Sun et al. (2018, arXiv:1805.10787):
   a) remove ALL rows of *inconsistent* groups (identical on the 20 metrics but with
      different ``defective`` labels), then
   b) remove *duplicates* (identical on the 20 metrics with the same label), keeping
      one row per group.

No scaling, normalisation or feature selection happens here; that is done inside the
training pipeline to avoid leakage. ``class_name`` is kept only to track classes
across releases and must never be used as a feature (see :func:`feature_columns`).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from sdp.data.download import read_raw_csv

logger = logging.getLogger(__name__)

METRICS: tuple[str, ...] = (
    "wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", "lcom3",
    "loc", "dam", "moa", "mfa", "cam", "ic", "cbm", "amc", "max_cc", "avg_cc",
)  # fmt: skip
ID_COL = "class_name"
COUNT_COL = "bug_count"
LABEL_COL = "defective"
OUTPUT_COLUMNS: list[str] = [ID_COL, *METRICS, COUNT_COL, LABEL_COL]

_RENAMES = {"name.1": ID_COL, "bug": COUNT_COL, "bugs": COUNT_COL}


class MissingMetricsError(ValueError):
    """Raised when a release lacks one or more of the 20 expected metrics."""


@dataclass
class CleaningStats:
    raw_rows: int
    invalid_rows: int
    inconsistent_rows: int
    inconsistent_groups: int
    duplicate_rows: int
    final_rows: int

    def as_dict(self) -> dict:
        return asdict(self)


def feature_columns(df: pd.DataFrame | None = None) -> list[str]:
    """Columns that may be used as model features: the 20 metrics, never ``class_name``."""
    cols = list(METRICS)
    if df is not None:
        cols = [c for c in cols if c in df.columns]
    assert ID_COL not in cols and COUNT_COL not in cols and LABEL_COL not in cols
    return cols


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case column names, apply the canonical renames and check the metrics."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    if "name.1" not in out.columns and "name" in out.columns:
        # GitHub-mirror layout: a single 'name' column holds the class name.
        out = out.rename(columns={"name": ID_COL})
    out = out.rename(columns=_RENAMES)

    required = [*METRICS, COUNT_COL, ID_COL]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise MissingMetricsError(f"Missing required column(s): {', '.join(missing)}")
    return out


def drop_invalid(df: pd.DataFrame, label: str = "") -> tuple[pd.DataFrame, int]:
    """Coerce metrics and bug counts to numbers; drop rows with missing/invalid values."""
    out = df.copy()
    numeric = [*METRICS, COUNT_COL]
    bad = pd.Series(False, index=out.index)
    for col in numeric:
        converted = pd.to_numeric(out[col], errors="coerce")
        col_bad = converted.isna()
        if col_bad.any():
            logger.warning("%s: %d missing/non-numeric value(s) in '%s'.", label, int(col_bad.sum()), col)
        out[col] = converted
        bad |= col_bad
    id_bad = out[ID_COL].isna() | (out[ID_COL].astype(str).str.strip() == "")
    if id_bad.any():
        logger.warning("%s: %d row(s) without a class name.", label, int(id_bad.sum()))
    bad |= id_bad
    neg = out[COUNT_COL] < 0
    if neg.any():
        logger.warning("%s: %d row(s) with a negative bug count.", label, int(neg.sum()))
    bad |= neg.fillna(False)

    n_bad = int(bad.sum())
    if n_bad:
        logger.warning("%s: dropping %d row(s) with missing or invalid values.", label, n_bad)
    else:
        logger.info("%s: no missing or invalid values.", label)
    out = out.loc[~bad].copy()
    out[COUNT_COL] = out[COUNT_COL].astype("int64")
    return out, n_bad


def binarise_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``defective`` = 1 if ``bug_count`` > 0 else 0."""
    out = df.copy()
    out[LABEL_COL] = (out[COUNT_COL] > 0).astype("int64")
    return out


def remove_inconsistent(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Remove every row of groups with identical metrics but conflicting labels.

    Returns ``(clean_df, rows_removed, groups_removed)``.
    """
    n_labels = df.groupby(list(METRICS), sort=False, dropna=False)[LABEL_COL].transform("nunique")
    conflict = n_labels > 1
    n_groups = int(df.loc[conflict, list(METRICS)].drop_duplicates().shape[0])
    return df.loc[~conflict].copy(), int(conflict.sum()), n_groups


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep one row per group of identical metrics and label (first occurrence)."""
    dup = df.duplicated(subset=[*METRICS, LABEL_COL], keep="first")
    return df.loc[~dup].copy(), int(dup.sum())


def clean_release(raw: pd.DataFrame, label: str = "") -> tuple[pd.DataFrame, CleaningStats]:
    """Apply the full cleaning procedure to one release."""
    raw_rows = len(raw)
    df = standardise_columns(raw)
    df, n_invalid = drop_invalid(df, label)
    df = binarise_label(df)
    df, n_incons, n_groups = remove_inconsistent(df)
    df, n_dup = remove_duplicates(df)
    df = df[OUTPUT_COLUMNS].reset_index(drop=True)

    stats = CleaningStats(raw_rows, n_invalid, n_incons, n_groups, n_dup, len(df))
    logger.info(
        "%s: raw=%d, invalid=%d, inconsistent removed=%d (%d group(s)), duplicates removed=%d, final=%d.",
        label, raw_rows, n_invalid, n_incons, n_groups, n_dup, len(df),
    )  # fmt: skip
    dupe_ids = int(df[ID_COL].duplicated().sum())
    if dupe_ids:
        logger.warning("%s: %d class name(s) occur more than once after cleaning.", label, dupe_ids)
    return df, stats


def clean_file(raw_path: Path, out_path: Path, label: str = "") -> CleaningStats:
    """Clean one raw CSV and write the processed CSV."""
    df, stats = clean_release(read_raw_csv(raw_path), label or raw_path.stem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return stats
