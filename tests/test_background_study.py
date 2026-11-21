"""Tests for the background-set variants and their comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sdp.analysis.explanations import Explanations
from sdp.eval.preprocess import FEATURES
from sdp.explain.base import sample_background
from sdp.robustness import background_study as B
from sdp.robustness.common import load_robustness
from tests.explain_helpers import synthetic


def test_variant_names():
    v = B.variants(load_robustness()["background"])
    assert "size=100" in v and "size=all" in v and "scheme=kmeans" in v and "draw=4" in v
    assert "scheme=stratified" not in v and "draw=0" not in v and len(v) == 6 + 4 + 4


def test_sizes_and_stratification():
    X, y = synthetic(400, seed=2)
    for n in (10, 25, 50, 100, 200):
        bg = B.make_background(X, y, f"size={n}")
        assert len(bg) == n and abs(y[bg.index].mean() - y.mean()) < 0.06
    assert len(B.make_background(X, y, "size=all")) == 400
    pd.testing.assert_frame_equal(B.make_background(X, y, "size=100"), sample_background(X, y, 100, 0))
    pd.testing.assert_frame_equal(B.make_background(X, y, "draw=3"), sample_background(X, y, 100, 3))
    assert not B.make_background(X, y, "draw=1").index.equals(B.make_background(X, y, "draw=2").index)
    assert (y[B.make_background(X, y, "scheme=defective").index] == 1).all()
    clean = B.make_background(X, y, "scheme=clean")
    assert len(clean) == 100 and (y[clean.index] == 0).all()
    assert len(B.make_background(X, y, "scheme=random")) == 100


def test_kmeans_background_repeats_centres_by_cluster_size():
    X, y = synthetic(300, seed=3)
    bg = B.make_background(X, y, "scheme=kmeans")
    assert len(bg) == 300 and list(bg.columns) == FEATURES
    centres, counts = np.unique(bg.to_numpy(), axis=0, return_counts=True)
    assert len(centres) <= 100 and counts.sum() == 300
    import shap

    summary = shap.kmeans(X.to_numpy(), 100)
    np.testing.assert_allclose(np.sort(summary.weights / summary.weights.sum() * 300), np.sort(counts))


def test_agreement_with_main_and_noise_floor():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(5, 20))
    rows, attrs = [], []
    for v, a in (("size=100", base), ("size=10", -base[:, ::-1]), ("draw=1", base), ("draw=2", base),
                 ("draw=3", base), ("draw=4", base)):
        for i in range(5):
            rows.append({"model": "rf", "method": "TreeSHAP", "variant": v, "split_id": "s", "project": "p",
                         "seed": 0, "instance_id": i, "outcome": "TN", "y_true": 0})
            attrs.append(a[i])
    ex = Explanations(pd.DataFrame(rows), np.array(attrs), np.zeros((30, 20)))
    agr = B.summarise(B.agreement_with_main(ex)).set_index("variant")
    assert agr.loc["draw=1", "spearman"] == 1.0 and agr.loc["draw=1", "mad"] == 0.0
    assert agr.loc["size=10", "spearman"] < 1.0
    noise = B.draw_noise(ex)
    assert np.allclose(noise["spearman"], 1.0) and np.allclose(noise["variance"], 0.0)
