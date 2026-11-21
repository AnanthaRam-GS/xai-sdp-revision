"""Tests for the stability metrics, null references and bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from sdp.stability import reference as Rf


def ranks_to_scores(order: list[int], p: int) -> np.ndarray:
    """Importance vector whose ranking is ``order`` (first = most important)."""
    g = np.zeros(p)
    for pos, item in enumerate(order):
        g[item] = p - pos
    return g


def test_rbo_known_examples():
    a = ranks_to_scores([0, 1], 2)
    b = ranks_to_scores([1, 0], 2)
    assert Rf.rbo(a, a)[0] == pytest.approx(1.0)
    assert Rf.rbo(a, b)[0] == pytest.approx(0.81 + 0.1 / 0.9 * 0.81)  # = 0.9
    x = ranks_to_scores([0, 1, 2], 3)
    y = ranks_to_scores([0, 2, 1], 3)
    p = 0.9
    expected = p**3 + (1 - p) / p * (1 * p + 0.5 * p**2 + 1 * p**3)
    assert Rf.rbo(x, y)[0] == pytest.approx(expected)
    rev = ranks_to_scores(list(range(20))[::-1], 20)
    fwd = ranks_to_scores(list(range(20)), 20)
    assert Rf.rbo(fwd, rev)[0] < Rf.rbo(fwd, fwd)[0] and 0 < Rf.rbo(fwd, rev)[0] < 1


def test_jaccard_and_metrics():
    a = ranks_to_scores(list(range(20)), 20)
    b = ranks_to_scores([0, 1, 2, 3, 19] + list(range(4, 19)), 20)
    m = Rf.stability_metrics(a, b)
    assert m["jaccard3"][0] == 1.0 and m["jaccard5"][0] == pytest.approx(4 / 6)
    same = Rf.stability_metrics(a, a)
    assert all(same[k][0] == pytest.approx(1.0) for k in Rf.METRICS)


def test_random_null_properties():
    null = Rf.random_null(20, 10_000, seed=0)
    assert abs(null["spearman"].mean()) < 0.01 and abs(null["kendall"].mean()) < 0.01
    # expected top-5 overlap of two random 5-subsets of 20 items is 1.25
    inter = null["jaccard5"] * 10 / (1 + null["jaccard5"])
    assert inter.mean() == pytest.approx(1.25, abs=0.03)
    assert 0 < null["rbo"].mean() < 0.5
    s = Rf.null_summary(null).set_index("metric")
    assert s.loc["spearman", "null_p95"] > 0.3


def test_bootstrap_reproduces_point_on_degenerate_sample():
    row = np.random.default_rng(0).dirichlet(np.ones(20))
    a = np.tile(row, (30, 1))
    b = np.tile(np.roll(row, 1), (25, 1))
    ci = Rf.comparison_ci(a, b, n_boot=200, seed=1).set_index("metric")
    for m in Rf.METRICS:
        assert ci.loc[m, "ci_low"] == pytest.approx(ci.loc[m, "value"]) == pytest.approx(ci.loc[m, "ci_high"])


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(2)
    a = rng.dirichlet(np.linspace(1, 5, 20), size=200)
    alpha = np.linspace(1, 5, 20)
    b = rng.dirichlet(np.r_[alpha[:10][::-1], alpha[10:]], size=150)  # partly reordered: moderate agreement
    ci = Rf.comparison_ci(a, b, n_boot=300, seed=0).set_index("metric")
    # continuous metrics: the percentile interval brackets the point estimate (discrete top-k
    # Jaccard values need not, which is a property of the percentile bootstrap)
    for m in ("spearman", "kendall"):
        assert ci.loc[m, "ci_low"] <= ci.loc[m, "value"] <= ci.loc[m, "ci_high"]
    assert ci.loc["spearman", "ci_high"] - ci.loc["spearman", "ci_low"] > 0
