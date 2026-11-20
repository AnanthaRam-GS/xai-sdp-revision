"""Class-imbalance handling and model fitting with validation data.

SMOTE is applied ONLY to the data a model is fitted on (the 80% fitting part, or an
inner-CV training fold during tuning) and never to validation or test data.
"""

from __future__ import annotations

import inspect
import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

IMBALANCE_OPTIONS = ("none", "class_weight", "smote")


def model_imbalance(imbalance: str) -> str:
    """Weighting passed to the model: SMOTE replaces weighting, so models are unweighted."""
    if imbalance not in IMBALANCE_OPTIONS:
        raise ValueError(f"unknown imbalance option {imbalance!r}; choose from {IMBALANCE_OPTIONS}")
    return "none" if imbalance == "smote" else imbalance


def apply_smote(X: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Oversample the minority class with SMOTE (k = min(5, n_minority - 1))."""
    y = np.asarray(y, dtype=int)
    n_min = int(np.bincount(y, minlength=2).min())
    if n_min < 2:
        logger.warning("SMOTE skipped: only %d minority instance(s).", n_min)
        return X, y
    sm = SMOTE(k_neighbors=min(5, n_min - 1), random_state=seed)
    Xr, yr = sm.fit_resample(X, y)
    return Xr, np.asarray(yr, dtype=int)


def supports_validation(model) -> bool:
    return "X_val" in inspect.signature(model.fit).parameters


def fit_model(model, X_fit, y_fit, X_val=None, y_val=None, imbalance: str = "none", seed: int = 0):
    """Fit ``model`` on the fitting data (SMOTE-resampled if requested).

    Validation data are passed to models that support early stopping and are never
    resampled.
    """
    if imbalance == "smote":
        X_fit, y_fit = apply_smote(X_fit, y_fit, seed)
    if supports_validation(model) and X_val is not None:
        return model.fit(X_fit, y_fit, X_val=X_val, y_val=y_val)
    return model.fit(X_fit, y_fit)
