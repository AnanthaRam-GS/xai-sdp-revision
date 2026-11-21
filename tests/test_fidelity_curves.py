"""Tests for the removal operators and the deletion/insertion curves."""

from __future__ import annotations

import numpy as np
import pytest

from sdp.fidelity import curves as C
from sdp.fidelity.common import random_orders, ranking, trapezoid_area
from sdp.fidelity.removal import MarginalRemover, MeanRemover, select_background


def test_marginal_removal_matches_analytic_expectation():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(40, 3))
    f = lambda X: X[:, 0] ** 2 + 2.0 * X[:, 1] * X[:, 2]  # noqa: E731
    x = np.array([[1.5, -0.5, 2.0]])
    marg, mean = MarginalRemover(f, B), MeanRemover(f, B)
    m0 = np.array([[True, False, False]])
    assert marg(x, m0)[0] == pytest.approx(np.mean(B[:, 0] ** 2) + 2 * (-0.5) * 2.0)
    assert mean(x, m0)[0] == pytest.approx(B[:, 0].mean() ** 2 + 2 * (-0.5) * 2.0)
    m12 = np.array([[False, True, True]])
    assert marg(x, m12)[0] == pytest.approx(1.5**2 + 2 * np.mean(B[:, 1] * B[:, 2]))
    none = np.zeros((1, 3), bool)
    assert marg(x, none)[0] == pytest.approx(f(x)[0])
    full = np.ones((1, 3), bool)
    assert marg(x, full)[0] == pytest.approx(f(B).mean())
    # (n, K, p) masks give (n, K)
    stacked = np.stack([m0, m12, none], axis=1)
    np.testing.assert_allclose(marg(x, stacked)[0], [marg(x, m0)[0], marg(x, m12)[0], f(x)[0]])


def test_background_selection_is_seeded_and_capped():
    B = np.arange(200, dtype=float).reshape(100, 2)
    a, b = select_background(B, 50, 0), select_background(B, 50, 0)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 50 and len(select_background(B[:30], 50, 0)) == 30


def test_masks_for_deletion_and_insertion():
    order = np.array([[2, 0, 1]])
    d = C.removal_masks(order, "deletion")[0]
    assert d.shape == (4, 3)
    assert not d[0].any() and d[3].all()
    np.testing.assert_array_equal(d[1], [False, False, True])  # top-1 = feature 2 removed
    np.testing.assert_array_equal(d[2], [True, False, True])
    ins = C.removal_masks(order, "insertion")[0]
    np.testing.assert_array_equal(ins, ~d)


def test_auc_on_hand_made_curves():
    flat = np.ones(21)
    linear = np.linspace(1, 0, 21)
    step = np.r_[1.0, np.zeros(20)]
    assert trapezoid_area(flat) == pytest.approx(1.0)
    assert trapezoid_area(linear) == pytest.approx(0.5)
    assert trapezoid_area(step) == pytest.approx(0.025)


def test_confidence_tracks_own_prediction():
    p = np.array([[0.9, 0.2], [0.3, 0.6]])
    np.testing.assert_allclose(C.confidence(p, np.array([True, False])), [[0.9, 0.2], [0.7, 0.4]])


def test_ranking_ties_and_random_orders():
    np.testing.assert_array_equal(ranking(np.array([[0.1, -0.5, 0.5, 0.0]])), [[1, 2, 0, 3]])
    r = random_orders(4, 5, 3, seed=1)
    assert r.shape == (3, 4, 5) and all(sorted(row) == list(range(5)) for row in r.reshape(-1, 5))
    np.testing.assert_array_equal(r, random_orders(4, 5, 3, seed=1))


def test_good_explanation_beats_random_on_a_linear_model():
    rng = np.random.default_rng(3)
    w = np.array([5.0, 3.0, 1.0, 0.1, 0.0])
    f = lambda X: 1 / (1 + np.exp(-(X @ w)))  # noqa: E731
    X = rng.normal(1.0, 1.0, size=(30, 5))
    B = rng.normal(0.0, 1.0, size=(20, 5))
    attr = (X - B.mean(axis=0)) * w  # exact contributions of a linear logit
    pred = f(X) >= 0.5
    res = C.evaluate_orders(MarginalRemover(f, B), X, {"exact": attr}, pred, n_random=10, seed=0)
    assert res["auc"]["deletion"]["exact"].mean() < res["auc"]["deletion"]["random"].mean()
    assert res["auc"]["insertion"]["exact"].mean() > res["auc"]["insertion"]["random"].mean()
    assert res["curves"]["deletion"]["exact"].shape == (6,)


def test_outcome_sample_is_stratified_and_complete_when_small():
    from sdp.fidelity.evaluation import sample_by_outcome

    outcome = np.array(["TP"] * 50 + ["FP"] * 150 + ["FN"] * 100 + ["TN"] * 700)
    idx = sample_by_outcome(outcome, 200, seed=0)
    assert len(idx) == 200 and len(set(idx)) == 200
    counts = {g: int((outcome[idx] == g).sum()) for g in ("TP", "FP", "FN", "TN")}
    assert counts == {"TP": 10, "FP": 30, "FN": 20, "TN": 140}
    np.testing.assert_array_equal(sample_by_outcome(outcome[:120], 200, seed=0), np.arange(120))
    np.testing.assert_array_equal(sample_by_outcome(outcome, None, seed=0), np.arange(1000))
