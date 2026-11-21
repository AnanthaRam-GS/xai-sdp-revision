"""Tests for the multicollinearity diagnostics, AutoSpearman and grouped attributions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdp.analysis.explanations import Explanations, global_importance
from sdp.eval.preprocess import FEATURES
from sdp.robustness import collinearity as C


def known_example(n=400, seed=0) -> pd.DataFrame:
    """a~b (rho~1), c~d (rho~1), e independent, f = a+c+e+g (moderate rho, high VIF), g independent."""
    rng = np.random.default_rng(seed)
    a, c, e, g = (rng.uniform(1, 50, n) for _ in range(4))
    b = a + rng.normal(0, 0.5, n)
    d = 2 * c + rng.normal(0, 0.5, n)
    f = a + c + e + g + rng.normal(0, 0.5, n)
    h = np.full(n, 3.0)  # constant
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "g": g, "h": h})


def test_autospearman_on_known_example():
    X = known_example()
    selected, log = C.autospearman(X, 0.7, 5.0)
    steps = {r["removed"]: r["step"] for r in log}
    assert steps["h"] == "constant"
    assert len({"a", "b"} & set(selected)) == 1 and len({"c", "d"} & set(selected)) == 1
    assert sum(s == "spearman" for s in steps.values()) == 2
    assert {"e", "g"} <= set(selected)
    assert steps.get("f") == "vif"  # f is a sum of other metrics: removed by the VIF step
    rho = C.spearman_matrix(X[selected]).abs().to_numpy(copy=True)
    np.fill_diagonal(rho, 0)
    assert rho.max() < 0.7 and C.vif(X[selected]).max() < 5


def test_autospearman_keeps_metric_with_lower_mean_correlation():
    rng = np.random.default_rng(1)
    a = rng.uniform(1, 50, 300)
    o = a + rng.normal(0, 25, 300)  # moderately correlated with a
    b = 0.85 * a + 0.15 * o  # near-copy of a that leans more towards o
    X = pd.DataFrame({"a": a, "b": b, "o": o, "z": rng.uniform(1, 50, 300)})
    selected, _ = C.autospearman(X, 0.7, 1e9)  # correlation step only
    assert "a" in selected and "b" not in selected


def test_clusters_and_stable_clusters():
    X = known_example()
    clusters = C.correlation_clusters(X)
    assert ["a", "b"] in clusters and ["c", "d"] in clusters and ["h"] in clusters
    feats = list("abcdefgh")
    lists = [[["a", "b"], ["c", "d"], ["e"], ["f"], ["g"], ["h"]]] * 3 + [[["a", "b", "e"], ["c"], ["d"], ["f"], ["g"], ["h"]]]
    stable = C.stable_clusters(lists, 0.75, feats)
    assert ["a", "b"] in stable and ["c", "d"] in stable and ["e"] in stable
    co = C.co_clustering(lists, feats)
    assert co.loc["a", "e"] == pytest.approx(0.25) and co.loc["c", "d"] == pytest.approx(0.75)


def test_grouping_sums_attributions():
    rng = np.random.default_rng(0)
    attr = rng.normal(size=(4, 20))
    meta = pd.DataFrame({"split_id": "s", "project": "p", "model": "rf", "method": "TreeSHAP", "seed": 0,
                         "instance_id": range(4), "y_true": 0, "outcome": "TN"})
    ex = Explanations(meta, attr, np.zeros((4, 20)))
    clusters = [["wmc", "rfc", "loc"], ["cbo", "ce"]] + [[f] for f in FEATURES if f not in {"wmc", "rfc", "loc", "cbo", "ce"}]
    clusters = sorted([sorted(g, key=FEATURES.index) for g in clusters], key=lambda g: FEATURES.index(g[0]))
    g = C.group_explanations(ex, clusters)
    j = g.features.index("wmc+rfc+loc")
    np.testing.assert_allclose(g.attr[:, j], attr[:, [FEATURES.index(f) for f in ("wmc", "rfc", "loc")]].sum(axis=1))
    np.testing.assert_allclose(g.attr.sum(axis=1), attr.sum(axis=1))
    assert len(g.features) == 17 and C.loc_cluster(clusters) == ["wmc", "rfc", "loc"]
    imp = global_importance(g, groups=("all",))
    assert set(imp["feature"]) == set(g.features)
    with pytest.raises(ValueError):
        C.group_explanations(ex, [["wmc"]])
