"""Tests for cross-version stability, top-k frequency and direction stability."""

from __future__ import annotations

import numpy as np
import pytest

from sdp.eval.preprocess import FEATURES
from sdp.stability import versions as V
from tests.stability_helpers import FakeSplit, fake_store

BASE = np.linspace(1, 2, 20)
S = [FakeSplit("p:1>2", "p", ("1",), "2"), FakeSplit("p:2>3", "p", ("2",), "3"), FakeSplit("q:1>2", "q", ("1",), "2"),
     FakeSplit("q:2>3", "q", ("2",), "3")]


def test_consecutive_pairs_within_projects():
    pairs = V.consecutive_pairs(S)
    assert [(a.split_id, b.split_id) for a, b in pairs] == [("p:1>2", "p:2>3"), ("q:1>2", "q:2>3")]
    assert V.pair_id(*pairs[0]) == "p 1>2 vs 2>3"


def test_cross_version_identical_vectors_are_perfectly_stable():
    store = fake_store({(s.split_id, s.project, "rf", "TreeSHAP"): BASE for s in S}, noise=0.05)
    perm = fake_store({(s.split_id, s.project, "rf", "TreeSHAP"): BASE[::-1] for s in S}, seeds=(0,))
    res = V.cross_version(store, perm, V.consecutive_pairs(S), [("rf", "TreeSHAP")], n_boot=50).set_index(["comparison", "metric"])
    r = res.loc[("p 1>2 vs 2>3", "spearman")]
    assert r["value"] > 0.95 and r["ci_low"] <= r["value"] <= r["ci_high"] and r["n_seeds"] == 2
    assert r["data_only"] == pytest.approx(1.0)  # permuted vectors identical across versions too


def test_topk_frequency():
    a = np.zeros(20)
    a[[0, 1, 2, 3, 4]] = [5, 4, 3, 2, 1]
    b = np.zeros(20)
    b[[0, 1, 2, 3, 19]] = [5, 4, 3, 2, 1]
    store = fake_store({("p:1>2", "p", "rf", "TreeSHAP"): a + 1e-3, ("p:2>3", "p", "rf", "TreeSHAP"): b + 1e-3})
    f = V.topk_frequency(store, S[:2], [("rf", "TreeSHAP")]).set_index(["scope", "feature"])["frequency"]
    assert f[("p", FEATURES[0])] == 1.0 and f[("p", FEATURES[4])] == 0.5 and f[("p", FEATURES[19])] == 0.5
    assert f[("ALL", FEATURES[10])] == 0.0


def test_direction_flips_are_listed():
    import pandas as pd

    d = pd.DataFrame({"project": "p", "explanation": "RF-TreeSHAP", "feature": ["loc", "loc", "cbo", "cbo"],
                      "split_id": ["a", "b", "a", "b"], "rho": [0.3, 0.2, 0.4, -0.1], "sign": [1.0, 1.0, 1.0, -1.0]})
    summ, flips = V.direction_stability(d)
    assert summ.iloc[0]["same_sign_share"] == 0.5
    assert list(flips["feature"]) == ["cbo"] and flips.iloc[0]["signs"] == "+ −"
