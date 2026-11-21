"""Tests for the faithfulness correlation."""

from __future__ import annotations

import numpy as np
import pytest

from sdp.fidelity import faithfulness as Fa
from sdp.fidelity.removal import MarginalRemover


def test_subsets_have_fixed_size_and_are_seeded():
    s = Fa.random_subsets(3, 20, 50, 5, seed=2)
    assert s.shape == (3, 50, 20) and (s.sum(axis=2) == 5).all()
    np.testing.assert_array_equal(s, Fa.random_subsets(3, 20, 50, 5, seed=2))


def test_faithfulness_is_one_for_exact_linear_attributions():
    rng = np.random.default_rng(0)
    w = rng.normal(size=20)
    f = lambda X: X @ w  # noqa: E731
    X = rng.normal(size=(4, 20))
    B = rng.normal(size=(30, 20))
    attr = (X - B.mean(axis=0)) * w  # exact Shapley values of a linear model
    subsets = Fa.random_subsets(4, 20, 50, 5, seed=0)
    drops = Fa.prediction_drops(MarginalRemover(f, B), X, f(X), subsets)
    np.testing.assert_allclose(Fa.faithfulness(attr, subsets, drops), 1.0, atol=1e-10)
    assert (Fa.faithfulness(-attr, subsets, drops) == pytest.approx(-1.0))


def test_constant_attributions_give_nan():
    subsets = Fa.random_subsets(2, 20, 10, 5, seed=0)
    drops = np.random.default_rng(1).normal(size=(2, 10))
    assert np.isnan(Fa.faithfulness(np.zeros((2, 20)), subsets, drops)).all()
