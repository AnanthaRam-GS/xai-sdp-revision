"""Small synthetic contexts shared by the explainer tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sdp.eval.preprocess import FEATURES, make_preprocessor
from sdp.explain.base import ExplainContext, sample_background
from sdp.models.imbalance import fit_model
from sdp.models.registry import make_model

FAST_PARAMS = {
    "mlp": {"hidden_layers": 2, "units": 8, "max_epochs": 30, "patience": 5},
    "lr": {},
    "rf": {"n_estimators": 15, "max_depth": 4},
    "xgb": {"n_estimators": 30, "max_depth": 3},
}


def synthetic(n: int = 160, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Non-negative metric-like data; defects depend on loc and cbo."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.gamma(2.0, 5.0, size=(n, len(FEATURES))), columns=FEATURES)
    score = 0.08 * X["loc"] + 0.1 * X["cbo"] - 2.0
    y = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)
    return X, y


def make_context(model_name: str = "lr", preprocessing: str = "log_standard", n_test: int = 12, seed: int = 0,
                 background_size: int = 30) -> ExplainContext:  # fmt: skip
    X, y = synthetic(160 + n_test, seed)
    X_tr, y_tr, X_te, y_te = X.iloc[:160], y[:160], X.iloc[160:].reset_index(drop=True), y[160:]
    pre = make_preprocessor(preprocessing).fit(X_tr)
    model = make_model(model_name, FAST_PARAMS[model_name], "class_weight", seed)
    Xt = pre.transform(X_tr)
    fit_model(model, Xt.iloc[:130], y_tr[:130], Xt.iloc[130:], y_tr[130:], imbalance="class_weight", seed=seed)
    return ExplainContext(
        split_id="ta-consecutive:toy:1>2", project="toy", train_release="1", test_release="2",
        model_name=model_name, seed=seed, model=model, preprocessor=pre, threshold=0.5,
        X_train=Xt, y_train=y_tr, X_test=pre.transform(X_te), X_test_raw=X_te,
        background=sample_background(Xt, y_tr, background_size, seed),
        class_name=np.array([f"org.toy.C{i}" for i in range(n_test)], dtype=object),
        y_true=y_te, loc=X_te["loc"].to_numpy(),
    )  # fmt: skip
