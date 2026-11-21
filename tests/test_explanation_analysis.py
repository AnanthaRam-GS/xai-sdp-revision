"""Tests for the explanation analysis: agreement metrics, aggregation, figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau, spearmanr

from sdp.analysis import explanations as E
from sdp.analysis import figures as F
from sdp.eval.preprocess import FEATURES


def vec(**values) -> np.ndarray:
    v = np.zeros(len(FEATURES))
    for k, x in values.items():
        v[FEATURES.index(k)] = x
    return v


# ---------------------------------------------------------------- agreement metrics


def test_topk_agreement_hand_made():
    A = vec(loc=5, wmc=4, rfc=3, cbo=-2, ce=1)
    B = vec(loc=5, rfc=4, wmc=3, cbo=2, ca=1)
    m = E.topk_agreement(A, B, 3)
    assert m["feature_agreement@3"][0] == pytest.approx(1.0)  # {loc, wmc, rfc} both
    assert m["rank_agreement@3"][0] == pytest.approx(1 / 3)  # only loc keeps its position
    assert m["sign_agreement@3"][0] == pytest.approx(1.0)
    m5 = E.topk_agreement(A, B, 5)
    assert m5["feature_agreement@5"][0] == pytest.approx(4 / 5)  # ce vs ca differ
    assert m5["sign_agreement@5"][0] == pytest.approx(3 / 5)  # cbo has opposite signs
    assert m5["rank_agreement@5"][0] == pytest.approx(2 / 5)  # loc and cbo keep positions
    assert m5["signed_rank_agreement@5"][0] == pytest.approx(1 / 5)  # cbo's sign differs


def test_topk_ties_are_broken_by_feature_order():
    # wmc (index 0) and dit (index 1) tie; the stable order puts wmc first.
    A = vec(wmc=1, dit=1)
    B = vec(dit=1, wmc=1)
    m = E.topk_agreement(A, B, 1)
    assert m["feature_agreement@1"][0] == 1.0 and m["rank_agreement@1"][0] == 1.0
    C = vec(dit=1.0, wmc=0.5)
    assert E.topk_agreement(A, C, 1)["feature_agreement@1"][0] == 0.0


def test_rank_correlations_match_scipy_with_ties():
    rng = np.random.default_rng(0)
    A = rng.integers(0, 4, size=(6, 20)).astype(float) * rng.choice([-1, 1], size=(6, 20))
    B = rng.integers(0, 4, size=(6, 20)).astype(float)
    r = E.rank_correlations(A, B)
    for i in range(6):
        assert r["spearman"][i] == pytest.approx(spearmanr(np.abs(A[i]), np.abs(B[i]))[0])
        assert r["kendall"][i] == pytest.approx(kendalltau(np.abs(A[i]), np.abs(B[i]))[0])
    same = E.rank_correlations(A, -A)
    np.testing.assert_allclose(same["spearman"], 1.0)
    assert np.isnan(E.rank_correlations(np.zeros((1, 20)), A[:1])["spearman"][0])


# ------------------------------------------------------------------ aggregation


def fake_explanations() -> E.Explanations:
    """Two splits of one project with unequal sizes; split s1 has 1 instance, s2 has 3."""
    rows, attrs = [], []
    spec = [("s1", 0, vec(loc=1.0)), ("s2", 0, vec(wmc=1.0)), ("s2", 1, vec(wmc=1.0)), ("s2", 2, vec(wmc=1.0))]
    for split, inst, a in spec:
        rows.append({"split_id": split, "project": "p", "model": "rf", "method": "TreeSHAP", "seed": 0,
                     "instance_id": inst, "class_name": f"c{inst}", "outcome": "TP", "y_true": 1, "y_prob": 0.9,
                     "loc": 10.0, "output_scale": "probability"})  # fmt: skip
        attrs.append(a)
    return E.Explanations(pd.DataFrame(rows), np.array(attrs), np.ones((4, 20)))


def test_importance_weights_splits_equally_and_ranks():
    imp = E.global_importance(fake_explanations(), groups=("all",))
    allr = imp[imp["scope"] == "ALL"].set_index("feature")
    assert allr.loc["loc", "importance"] == pytest.approx(0.5)  # not 0.25 (instance-weighted)
    assert allr.loc["wmc", "importance"] == pytest.approx(0.5)
    assert allr.loc["loc", "rank"] == 1 and allr.loc["cbo", "rank"] == 3


def test_size_share():
    s = E.size_share(fake_explanations()).set_index("scope").loc["ALL"]
    assert s["loc_share"] == pytest.approx(0.5) and s["size_group_share"] == pytest.approx(1.0)
    assert s["loc_top1_pct"] == pytest.approx(50.0)


def test_from_long_matches_by_feature_name():
    rows = []
    for j, f in enumerate(reversed(FEATURES)):  # deliberately reversed order
        rows.append({"split_id": "s", "project": "p", "model": "lr", "method": "LIME", "seed": 0, "instance_id": 0,
                     "class_name": "c", "outcome": "TN", "y_true": 0, "y_prob": 0.1, "loc": 3.0, "feature": f,
                     "feature_value_raw": float(FEATURES.index(f)), "attribution": 10.0 + FEATURES.index(f),
                     "output_scale": "probability"})  # fmt: skip
    ex = E.from_long(pd.DataFrame(rows))
    np.testing.assert_array_equal(ex.attr[0], 10.0 + np.arange(20))
    np.testing.assert_array_equal(ex.raw[0], np.arange(20))


def test_direction_sign():
    rng = np.random.default_rng(1)
    raw = rng.random((50, 20))
    meta = pd.DataFrame({"split_id": "s", "project": "p", "model": "lr", "method": "LIME", "seed": 0,
                         "instance_id": range(50), "y_true": 0, "outcome": "TN"})  # fmt: skip
    attr = raw.copy()
    attr[:, FEATURES.index("cbo")] *= -1
    d = E.direction(E.Explanations(meta, attr, raw))
    d = d[d["scope"] == "ALL"].set_index("feature")
    assert d.loc["loc", "direction"] == "+" and d.loc["cbo", "direction"] == "−"
    assert d.loc["loc", "spearman"] == pytest.approx(1.0)


# ------------------------------------------------------------------------ figures


def test_heatmap_labels_come_from_feature_names():
    mat = pd.DataFrame({"MLP-LIME": [0.1, 0.7, 0.2], "RF-TreeSHAP": [0.3, 0.05, 0.65]}, index=["cbo", "loc", "wmc"])
    shuffled = mat.loc[["wmc", "cbo", "loc"]]
    fig, ax = F.heatmap(shuffled, "t", "importance")
    ylabels = [t.get_text() for t in ax.get_yticklabels()]
    assert ylabels == ["wmc", "cbo", "loc"]
    data = ax.images[0].get_array()
    i = ylabels.index("loc")
    j = [t.get_text() for t in ax.get_xticklabels()].index("MLP-LIME")
    assert data[i, j] == pytest.approx(0.7)  # the cell labelled loc / MLP-LIME holds loc's value
    cell_texts = {(round(t.get_position()[1]), round(t.get_position()[0])): t.get_text() for t in ax.texts}
    assert cell_texts[(i, j)] == "0.70"


def test_bar_labels_come_from_index():
    table = pd.DataFrame({"loc": [0.4, 0.1], "size group": [0.6, 0.3]}, index=["MLP-LIME", "RF-TreeSHAP"])
    fig, ax = F.size_share_bars(table, "t")
    labels = [t.get_text() for t in ax.get_yticklabels()]
    ticks = ax.get_yticks()
    loc_bars = ax.containers[0]
    assert loc_bars.get_label() == "loc"
    y_rf = ticks[labels.index("RF-TreeSHAP")]
    bar = min(loc_bars, key=lambda b: abs(b.get_y() + b.get_height() / 2 - y_rf))
    assert bar.get_width() == pytest.approx(0.1)


def test_step5_figure_labels_come_from_names():
    s = pd.Series({"loc": 0.9, "wmc": 0.2, "cbo": 0.5})
    fig, ax = F.single_bars(s, "t", "x")
    labels = [t.get_text() for t in ax.get_yticklabels()]
    bars = ax.patches
    assert bars[labels.index("loc")].get_width() == pytest.approx(0.9)
    table = pd.DataFrame({"MLP-DeepLiftShap": [0.5, 0.9], "RF-TreeSHAP": [0.4, 0.8]}, index=["10", "all"])
    fig, ax = F.line_chart(table, "t", "x", "y")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["10", "all"]
    line = next(l for l in ax.get_lines() if l.get_label() == "RF-TreeSHAP")
    assert list(line.get_ydata()) == [0.4, 0.8]
    fig, axes = F.box_panels({"MLP": {"main": np.array([0.1, 0.9]), "sampling=mean": np.array([0.5])}}, "t", "y")
    assert [t.get_text() for t in axes[0].get_xticklabels()] == ["main", "sampling=mean"]


def test_line_panels_label_lines_by_column_name():
    t = pd.DataFrame({"TreeSHAP": [0.9, 0.5, 0.2], "random": [0.9, 0.7, 0.5]}, index=[0, 1, 2])
    fig, axes = F.line_panels({"RF": t, "XGB": t[["random"]]}, "t", "k", "p")
    ax = axes[0][0]
    assert ax.get_title() == "RF"
    tree = next(l for l in ax.get_lines() if l.get_label() == "TreeSHAP")
    assert list(tree.get_ydata()) == [0.9, 0.5, 0.2] and tree.get_linestyle() == "-"
    assert next(l for l in ax.get_lines() if l.get_label() == "random").get_linestyle() == "--"
