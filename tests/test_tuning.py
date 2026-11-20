"""Tests for Optuna tuning: search spaces, determinism and no leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdp import tuning
from sdp.config import load_config

SPACES = load_config().tuning.spaces


def data(n=150, seed=0, offset=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.gamma(2, 10, size=(n, 3)), columns=["loc", "a", "b"], index=np.arange(offset, offset + n))
    y = (rng.random(n) < 0.2 + 0.5 * (X["a"] > X["a"].median())).astype(int)
    return X, y


class Spy:
    """Model that records the row labels of every frame it is given."""

    seen: list[tuple[str, set]] = []

    def __init__(self, **kw):
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, X_val=None, y_val=None):
        Spy.seen.append(("fit", set(X.index)))
        if X_val is not None:
            Spy.seen.append(("fit_val", set(X_val.index)))
        return self

    def predict_proba(self, X):
        Spy.seen.append(("predict", set(X.index)))
        p = np.linspace(0.1, 0.9, len(X))
        return np.column_stack([1 - p, p])


@pytest.mark.parametrize("supports_val", [True, False])
def test_tuning_only_touches_the_fitting_part(monkeypatch, supports_val):
    X, y = data(300)
    fit_idx = np.arange(0, 200)  # validation (200-249) and test (250-299) must never be seen
    Spy.seen = []
    cls = Spy if supports_val else type("SpyNoVal", (Spy,), {"fit": lambda self, X, y: Spy.fit(self, X, y)})
    monkeypatch.setattr(tuning, "make_model", lambda *a, **k: cls())
    res = tuning.tune("spy", X.iloc[fit_idx], y[fit_idx], {"c": {"type": "float", "low": 0, "high": 1}},
                      n_trials=3, preprocessing="standard", imbalance="class_weight", seed=0)  # fmt: skip
    assert res.n_trials == 3 and Spy.seen
    touched = set().union(*(s for _, s in Spy.seen))
    assert touched <= set(fit_idx)
    assert not touched & set(range(200, 300))


def test_tuning_is_deterministic_and_within_space():
    X, y = data()
    kw = dict(space=SPACES["lr"], n_trials=5, preprocessing="log_standard", imbalance="class_weight", seed=3)
    a = tuning.tune("lr", X, y, **kw)
    b = tuning.tune("lr", X, y, **kw)
    assert a.params == b.params and a.best_value == b.best_value
    assert 1e-3 <= a.params["C"] <= 1e2
    assert len(a.trials) == 5


def test_suggest_covers_all_spaces():
    import optuna

    for name, space in SPACES.items():
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        params = tuning.suggest(study.ask(), space)
        assert set(params) == set(space)
    assert set(SPACES) == {"mlp", "lr", "rf", "xgb"}


def test_mlp_and_xgb_tuning_run():
    X, y = data(120)
    small_mlp = {k: v for k, v in SPACES["mlp"].items() if k != "units"} | {"units": {"type": "categorical", "choices": [8]}}
    res = tuning.tune("mlp", X, y, small_mlp, n_trials=2, preprocessing="standard", imbalance="smote", seed=0)
    assert 0 <= res.best_value <= 1
    res = tuning.tune("xgb", X, y, SPACES["xgb"], n_trials=2, preprocessing="none", imbalance="class_weight", seed=0)
    assert set(res.params) == set(SPACES["xgb"])
