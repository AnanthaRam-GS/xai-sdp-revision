"""Tests for the PyTorch MLP: architecture constraints, Captum completeness,
determinism and save/load."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from sdp.models.mlp import MLPClassifierTorch, MLPNet, ProbabilityModel
from sdp.seed import set_seed


def toy_data(n: int = 120, d: int = 6, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n) > 0.3).astype(int)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y


@pytest.fixture(scope="module")
def trained():
    set_seed(0, quiet=True)
    X, y = toy_data()
    m = MLPClassifierTorch(hidden_layers=2, units=16, dropout=0.1, max_epochs=30, patience=5, random_state=0)
    return m.fit(X.iloc[:90], y[:90], X.iloc[90:], y[90:]), X, y


def test_architecture_uses_only_allowed_modules():
    net = MLPNet(5, hidden_layers=3, units=8, dropout=0.3)
    allowed = (nn.Linear, nn.ReLU, nn.Dropout)
    leaves = [m for m in net.modules() if not list(m.children())]
    assert all(isinstance(m, allowed) for m in leaves)
    assert all(not m.inplace for m in leaves if isinstance(m, nn.ReLU))
    assert len({id(m) for m in net.layers}) == len(net.layers)  # no module reused
    assert net(torch.zeros(4, 5)).shape == (4, 1)


def test_deeplift_completeness(trained):
    from captum.attr import DeepLift

    model, X, _ = trained
    net = model.get_torch_module()
    x = torch.tensor(X.iloc[:10].to_numpy(), dtype=torch.float32)
    baseline = torch.tensor(X.iloc[:90].mean().to_numpy(), dtype=torch.float32).repeat(10, 1)
    attr = DeepLift(net).attribute(x, baselines=baseline, target=0)
    with torch.no_grad():
        expected = (net(x) - net(baseline)).squeeze(1)
    np.testing.assert_allclose(attr.sum(dim=1).detach().numpy(), expected.numpy(), atol=1e-4)


def test_probability_model_matches_predict_proba(trained):
    model, X, _ = trained
    pm = model.get_probability_module()
    assert isinstance(pm, ProbabilityModel) and isinstance(pm.sigmoid, nn.Sigmoid)
    with torch.no_grad():
        p = pm(torch.tensor(X.to_numpy(), dtype=torch.float32)).squeeze(1).numpy()
    np.testing.assert_allclose(p, model.predict_proba(X)[:, 1], atol=1e-6)


def test_two_fits_with_same_seed_are_identical():
    X, y = toy_data()
    kw = dict(hidden_layers=2, units=16, dropout=0.3, max_epochs=25, patience=5, random_state=7)
    p1 = MLPClassifierTorch(**kw).fit(X, y).predict_proba(X)
    p2 = MLPClassifierTorch(**kw).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)
    p3 = MLPClassifierTorch(**(kw | {"random_state": 8})).fit(X, y).predict_proba(X)
    assert not np.array_equal(p1, p3)


def test_save_load_gives_identical_probabilities(trained, tmp_path):
    model, X, _ = trained
    model.save(tmp_path / "m")
    loaded = MLPClassifierTorch.load(tmp_path / "m")
    np.testing.assert_array_equal(loaded.predict_proba(X), model.predict_proba(X))
    assert list(loaded.feature_names_in_) == list(X.columns)


def test_early_stopping_restores_best_weights():
    X, y = toy_data(200)
    m = MLPClassifierTorch(units=32, max_epochs=300, patience=10, random_state=0).fit(X, y)
    assert m.n_epochs_ < 300 and m.best_epoch_ == m.n_epochs_ - 10
    assert m.internal_validation_
