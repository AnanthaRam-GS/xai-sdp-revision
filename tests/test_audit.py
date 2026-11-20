"""Tests for the cross-release overlap and flagging logic of the audit."""

from __future__ import annotations

import pandas as pd

from sdp.data.audit import flag_projects, flag_releases, pair_overlap
from sdp.data.clean import ID_COL, LABEL_COL, METRICS


def release(classes: list[str], values: list[float], labels: list[int]) -> pd.DataFrame:
    df = pd.DataFrame({m: values for m in METRICS})
    df.insert(0, ID_COL, classes)
    df[LABEL_COL] = labels
    return df


def test_pair_overlap_counts_shared_classes_and_carry_over():
    prev = release(["A", "B", "C"], [1, 2, 3], [0, 1, 0])
    # A unchanged; B same metrics but label flipped (not carry-over);
    # D is a new class whose metrics and label equal C's (counts as carry-over).
    nxt = release(["A", "B", "D", "E"], [1, 2, 3, 9], [0, 0, 0, 1])
    res = pair_overlap(prev, nxt)
    assert res["shared_classes"] == 2
    assert res["shared_pct_of_n1"] == 50.0
    assert res["unchanged_rows"] == 2
    assert res["unchanged_pct_of_n1"] == 50.0


def test_flags():
    summary = pd.DataFrame(
        {
            "project": ["p", "p", "q"],
            "version": ["1", "2", "1"],
            "final_rows": [100, 100, 50],
            "defective_ratio_pct": [4.0, 30.0, 80.0],
        }
    )
    flagged = flag_releases(summary, 5, 75)
    assert list(zip(flagged["project"], flagged["version"])) == [("p", "1"), ("q", "1")]
    assert flag_projects(summary, ["p", "q", "r"], 3) == {"p": 2, "q": 1, "r": 0}
