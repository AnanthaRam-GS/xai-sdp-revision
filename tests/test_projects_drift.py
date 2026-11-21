"""Tests for cross-project stability and data drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdp.eval.preprocess import FEATURES
from sdp.stability import drift as D
from sdp.stability import projects as Pj


def releases(shift: float, n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = pd.DataFrame(rng.gamma(2, 5, size=(n, 20)), columns=FEATURES).assign(defective=rng.random(n) < 0.2)
    B = pd.DataFrame(rng.gamma(2, 5, size=(n, 20)) + shift, columns=FEATURES).assign(defective=rng.random(n) < 0.3)
    return A, B


def test_classifier_two_sample_test_is_chance_on_identical_distributions():
    A, B = releases(0.0)
    d = D.drift_pair(A, B, seed=0)
    assert d["c2st_auc"] == pytest.approx(0.5, abs=0.07)
    assert d["mean_ks"] < 0.15
    A, B = releases(8.0)
    d2 = D.drift_pair(A, B, seed=0)
    assert d2["c2st_auc"] > 0.9 and d2["mean_ks"] > d["mean_ks"]


def test_correlate_with_holm():
    f = pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0) ** 2, "z": np.random.default_rng(0).normal(size=10)})
    res = D.correlate(f, ["x"], ["y", "z"]).set_index("y")
    assert res.loc["y", "rho"] == pytest.approx(1.0) and res.loc["y", "p_holm"] >= res.loc["y", "p_value"]


def test_project_matrix_and_common_features():
    v = {"a": np.linspace(1, 2, 20), "b": np.linspace(1, 2, 20), "c": np.linspace(2, 1, 20)}
    m = Pj.project_matrix(v, "spearman")
    assert m.loc["a", "b"] == pytest.approx(1.0) and m.loc["a", "c"] == pytest.approx(-1.0)
    mean, lo, hi = Pj.off_diagonal(m)
    assert mean == pytest.approx((1 - 1 - 1) / 3)
    top = Pj.common_top_features(v, FEATURES, k=5, min_projects=2)
    assert set(top["feature"]) == set(FEATURES[15:])


def test_classifier_two_sample_test_groups_versions_of_a_class():
    """Two versions of the same classes, slightly changed: grouping by class name keeps
    both versions in one fold, so the test stays near chance."""
    rng = np.random.default_rng(3)
    A = pd.DataFrame(rng.gamma(2, 5, size=(200, 20)), columns=FEATURES)
    B = A * (1 + 0.02 * rng.normal(size=A.shape))
    names = [f"C{i}" for i in range(200)]
    grouped, _ = D.classifier_two_sample(A, B, seed=0, names_a=names, names_b=names)
    assert grouped == pytest.approx(0.5, abs=0.12)
