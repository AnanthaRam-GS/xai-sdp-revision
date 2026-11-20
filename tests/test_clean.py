"""Tests for the per-release cleaning rules, using small synthetic DataFrames."""

from __future__ import annotations

import pandas as pd
import pytest

from sdp.data.clean import (
    COUNT_COL,
    ID_COL,
    LABEL_COL,
    METRICS,
    OUTPUT_COLUMNS,
    MissingMetricsError,
    binarise_label,
    clean_release,
    feature_columns,
    remove_duplicates,
    remove_inconsistent,
    standardise_columns,
)


def make_raw(metric_rows: list[list[float]], bugs: list[int], classes: list[str] | None = None) -> pd.DataFrame:
    """Build a DataFrame laid out like a raw Jureczko CSV (``name``, ``version``, ``name.1``, ...)."""
    n = len(metric_rows)
    classes = classes or [f"pkg.C{i}" for i in range(n)]
    df = pd.DataFrame(metric_rows, columns=[m.upper() for m in METRICS])  # upper case on purpose
    df.insert(0, "name.1", classes)
    df.insert(0, "version", ["1.0"] * n)
    df.insert(0, "name", ["proj"] * n)
    df["bug"] = bugs
    return df


def row(v: float) -> list[float]:
    return [v] * len(METRICS)


# ------------------------------------------------------------------ label binarisation


def test_binarise_label_maps_counts_to_zero_one():
    df = pd.DataFrame({COUNT_COL: [0, 1, 2, 7, 0]})
    out = binarise_label(df)
    assert out[LABEL_COL].tolist() == [0, 1, 1, 1, 0]
    assert out[COUNT_COL].tolist() == [0, 1, 2, 7, 0]  # original count preserved


# ------------------------------------------------------------- inconsistent removal


def test_inconsistent_groups_are_removed_entirely():
    raw = make_raw([row(1), row(1), row(1), row(2), row(3)], bugs=[0, 3, 0, 1, 0])
    clean, stats = clean_release(raw)
    # All three rows with metrics == 1 conflict (labels 0, 1, 0) and must all go.
    assert stats.inconsistent_rows == 3
    assert stats.inconsistent_groups == 1
    assert (clean[list(METRICS)] == 1).all(axis=1).sum() == 0
    assert len(clean) == 2


def test_remove_inconsistent_keeps_consistent_groups():
    df = binarise_label(standardise_columns(make_raw([row(1), row(1), row(2)], bugs=[1, 4, 0])))
    out, n_rows, n_groups = remove_inconsistent(df)
    assert (n_rows, n_groups) == (0, 0)
    assert len(out) == 3


# ------------------------------------------------------------------ de-duplication


def test_duplicates_keep_one_row_per_group():
    df = binarise_label(standardise_columns(make_raw([row(1), row(1), row(1), row(2)], bugs=[0, 0, 0, 2])))
    out, n_dup = remove_duplicates(df)
    assert n_dup == 2
    assert len(out) == 2
    assert out[ID_COL].tolist() == ["pkg.C0", "pkg.C3"]  # first occurrence kept


def test_same_label_different_bug_counts_count_as_duplicates():
    raw = make_raw([row(5), row(5)], bugs=[1, 3])
    clean, stats = clean_release(raw)
    assert stats.duplicate_rows == 1 and stats.inconsistent_rows == 0
    assert len(clean) == 1


def test_inconsistent_removed_before_duplicates():
    # Group A: 2x label 0 and 1x label 1 -> inconsistent, all 3 removed (none counted as duplicates).
    # Group B: 2x label 1 -> 1 duplicate removed.
    raw = make_raw([row(1), row(1), row(1), row(2), row(2)], bugs=[0, 0, 1, 1, 1])
    clean, stats = clean_release(raw)
    assert stats.inconsistent_rows == 3
    assert stats.duplicate_rows == 1
    assert stats.final_rows == len(clean) == 1


# ------------------------------------------------------------- missing metrics / values


def test_missing_metric_raises():
    raw = make_raw([row(1)], bugs=[0]).drop(columns=["CAM"])
    with pytest.raises(MissingMetricsError, match="cam"):
        clean_release(raw)


def test_invalid_values_are_dropped_and_counted():
    raw = make_raw([row(1), row(2), row(3)], bugs=[0, 1, 0])
    raw["WMC"] = raw["WMC"].astype(object)
    raw.loc[1, "WMC"] = "?"
    raw.loc[2, "LOC"] = None
    clean, stats = clean_release(raw)
    assert stats.invalid_rows == 2
    assert len(clean) == 1


# ------------------------------------------------------------------- feature columns


def test_class_name_is_never_a_feature():
    raw = make_raw([row(1), row(2)], bugs=[0, 1])
    clean, _ = clean_release(raw)
    feats = feature_columns(clean)
    assert ID_COL not in feats
    assert COUNT_COL not in feats and LABEL_COL not in feats
    assert feats == list(METRICS)
    assert ID_COL not in feature_columns()


def test_output_schema_and_lower_case_names():
    clean, _ = clean_release(make_raw([row(1), row(2)], bugs=[0, 1]))
    assert list(clean.columns) == OUTPUT_COLUMNS
    assert "name" not in clean.columns and "version" not in clean.columns
