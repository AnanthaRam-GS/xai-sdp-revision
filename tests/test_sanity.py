"""Tests for the model- and data-randomisation sanity checks."""

from __future__ import annotations

import numpy as np
import torch

from sdp.fidelity import sanity as S
from tests.explain_helpers import make_context


def _weights(model):
    return [(m.weight.detach().clone(), m.bias.detach().clone()) for m in model.net_.layers if isinstance(m, torch.nn.Linear)]


def test_cascading_randomisation_changes_only_the_intended_layers():
    ctx = make_context("mlp", n_test=5)
    model = ctx.model
    pos = S.linear_positions(model)
    assert len(pos) == 3  # two hidden layers + output
    stages = S.cascade_stages(model)
    assert [s for s, _ in stages] == ["output", "hidden2+output", "all"]
    assert [p for _, p in stages] == [pos[-1:], pos[-2:], pos]
    before = _weights(model)
    for _, positions in stages:
        new = S.randomised_model(model, positions, seed=1)
        after = _weights(new)
        for j, p in enumerate(pos):
            changed = not torch.equal(before[j][0], after[j][0])
            assert changed == (p in positions)
    np.testing.assert_array_equal([w[0].numpy() for w in _weights(model)][0], before[0][0].numpy())  # original untouched
    ind = S.independent_stages(model)
    assert [s for s, _ in ind] == ["only hidden1", "only hidden2", "only output"]
    a = S.randomised_model(model, pos[-1:], seed=5)
    b = S.randomised_model(model, pos[-1:], seed=5)
    assert torch.equal(a.net_.layers[pos[-1]].weight, b.net_.layers[pos[-1]].weight)  # seeded


def test_label_permutation_keeps_class_counts():
    y = np.array([0] * 70 + [1] * 30)
    p = S.permuted_labels(y, seed=0)
    assert np.bincount(p).tolist() == [70, 30] and not np.array_equal(p, y)
    np.testing.assert_array_equal(p, S.permuted_labels(y, seed=0))


def test_similarity_of_identical_and_unrelated_explanations():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 20))
    same = S.similarity(A, 2 * A)
    assert np.allclose(same["spearman"], 1.0) and np.allclose(same["fa5"], 1.0)
    other = S.similarity(A, rng.normal(size=(30, 20)))
    assert abs(np.nanmean(other["spearman"])) < 0.2
