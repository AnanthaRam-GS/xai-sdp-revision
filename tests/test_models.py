"""Tests for LR/RF/XGBoost, the imbalance options and the ManualUp/ManualDown baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdp.models.baselines import ManualDown, ManualUp, Majority
from sdp.models.imbalance import apply_smote, fit_model, model_imbalance
from sdp.models.registry import LEARNED_MODELS, make_model


def data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.gamma(2, 10, size=(n, 4)), columns=["loc", "a", "b", "c"])
    y = (rng.random(n) < 0.2 + 0.5 * (X["a"] > X["a"].median())).astype(int)
    return X, y


# ------------------------------------------------------------------ baselines


def test_manual_down_risk_is_loc():
    m = ManualDown().fit(pd.DataFrame({"loc": [10.0, 20.0, 30.0]}))
    p = m.predict_proba(pd.DataFrame({"loc": [5.0, 20.0, 40.0]}))[:, 1]
    np.testing.assert_allclose(p, [0.0, 0.5, 1.0])


def test_manual_up_risk_is_inverse_loc():
    # Training risk 1/max(LOC,1) = [1, 0.5, 0.25] -> min 0.25, max 1.
    m = ManualUp().fit(pd.DataFrame({"loc": [1.0, 2.0, 4.0]}))
    p = m.predict_proba(pd.DataFrame({"loc": [2.0, 0.0, 8.0, 4.0]}))[:, 1]
    np.testing.assert_allclose(p, [(0.5 - 0.25) / 0.75, 1.0, 0.0, 0.0])
    # Smaller classes are riskier.
    assert np.all(np.diff(m.predict_proba(pd.DataFrame({"loc": [1.0, 2.0, 3.0, 4.0]}))[:, 1]) < 0)


def test_baselines_require_raw_features():
    for m in (ManualUp(), ManualDown(), Majority()):
        assert m.requires_raw_features


# --------------------------------------------------------------- learned models


@pytest.mark.parametrize("name", LEARNED_MODELS)
@pytest.mark.parametrize("imbalance", ["none", "class_weight"])
def test_learned_models_fit_and_predict(name, imbalance):
    X, y = data()
    params = {"max_epochs": 20, "patience": 5} if name == "mlp" else {}
    if name == "rf":
        params = {"n_estimators": 20}
    m = make_model(name, params, imbalance=imbalance, seed=0)
    fit_model(m, X.iloc[:150], y[:150], X.iloc[150:], y[150:])
    p = m.predict_proba(X)[:, 1]
    assert p.shape == (len(X),) and np.all((p >= 0) & (p <= 1))


def test_class_weight_settings():
    assert make_model("lr", imbalance="class_weight").class_weight == "balanced"
    assert make_model("rf", imbalance="none").class_weight is None
    assert make_model("rf", imbalance="smote").class_weight is None  # SMOTE replaces weighting
    assert make_model("mlp", imbalance="smote").imbalance == "none"
    X, y = data()
    xgb = make_model("xgb", imbalance="class_weight").fit(X, y, X, y)
    assert xgb.model_.get_params()["scale_pos_weight"] == pytest.approx((y == 0).sum() / (y == 1).sum())
    with pytest.raises(ValueError):
        model_imbalance("undersample")


def test_xgb_uses_early_stopping():
    X, y = data(300)
    m = make_model("xgb", {"n_estimators": 600, "learning_rate": 0.3}, seed=0).fit(X.iloc[:200], y[:200], X.iloc[200:], y[200:])
    assert m.best_iteration_ < 599


# ------------------------------------------------------------------------ SMOTE


class Recorder:
    """Records what it is fitted on."""

    def fit(self, X, y, X_val=None, y_val=None):
        self.n_fit, self.y_fit = len(X), np.asarray(y)
        self.n_val = None if X_val is None else len(X_val)
        self.y_val = None if y_val is None else np.asarray(y_val)
        return self


def test_smote_is_applied_only_to_the_fitting_part():
    X, y = data(200)
    X_fit, y_fit, X_val, y_val = X.iloc[:160], y[:160], X.iloc[160:], y[160:]
    rec = fit_model(Recorder(), X_fit, y_fit, X_val, y_val, imbalance="smote", seed=0)
    assert rec.n_fit > len(X_fit)  # fitting part oversampled
    assert np.bincount(rec.y_fit)[0] == np.bincount(rec.y_fit)[1]
    assert rec.n_val == len(X_val)  # validation part untouched
    np.testing.assert_array_equal(rec.y_val, y_val)
    rec = fit_model(Recorder(), X_fit, y_fit, X_val, y_val, imbalance="class_weight", seed=0)
    assert rec.n_fit == len(X_fit)


def test_smote_is_seeded():
    X, y = data()
    a, _ = apply_smote(X, y, 3)
    b, _ = apply_smote(X, y, 3)
    pd.testing.assert_frame_equal(a, b)
