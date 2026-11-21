"""Tests for the Scott-Knott ESD cross-check helpers (no R needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sdp.analysis import skesd_crosscheck as X


def test_identical_treatments_are_detected():
    w = pd.DataFrame({"a": [0.5, 0.5], "b": [0.6, 0.7], "c": [0.5, 0.5]})
    assert X.duplicate_columns(w) == {"c": "a"}


def test_model_input_matches_the_transform_of_our_implementation():
    pos = pd.DataFrame({"a": [0.5, 0.7], "b": [0.1, 0.2]})
    np.testing.assert_allclose(X.model_input(pos).to_numpy(), np.log1p(pos.to_numpy()))
    neg = pd.DataFrame({"a": [-0.5, 0.7], "b": [0.1, 0.2]})
    np.testing.assert_allclose(X.model_input(neg).to_numpy(), neg.to_numpy())


def test_compare_partitions():
    a = pd.Series({"x": 1, "y": 1, "z": 2})
    same = X.compare(a, pd.Series({"x": 1, "y": 1, "z": 2}))
    assert same["same_partition"] and same["same_group_numbers"] and same["differences"] == "none"
    other = X.compare(a, pd.Series({"x": 1, "y": 2, "z": 3}))
    assert not other["same_partition"] and "y: ours 1, R 2" in other["differences"]
