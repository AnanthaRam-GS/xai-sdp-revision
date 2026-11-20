"""Hyperparameter tuning with Optuna (TPE sampler, seeded).

The objective is the mean ROC-AUC of a stratified inner k-fold cross-validation
(default k = 3) on the *fitting part* of the training data only. Callers pass exactly
that data; validation and test data are never available to the tuner.

Inside each inner fold:

* the preprocessor is fitted on the inner training fold only;
* for models with early stopping (MLP, XGBoost), a stratified 20% of the inner training
  fold is held out as early-stopping data *before* any resampling;
* SMOTE (if requested) is applied to the remaining inner fitting data only;
* the AUC is measured on the inner validation fold.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from sdp.eval.metrics import roc_auc
from sdp.eval.preprocess import make_preprocessor
from sdp.models.imbalance import fit_model, supports_validation
from sdp.models.registry import make_model, positive_proba

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

EARLY_STOPPING_FRACTION = 0.2


@dataclass
class TuningResult:
    params: dict
    best_value: float
    n_trials: int
    time_s: float
    trials: list[dict] = field(default_factory=list)


def suggest(trial: optuna.Trial, space: dict) -> dict:
    """Sample one configuration from a search space defined in the configuration."""
    params = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        elif kind == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log")))
        elif kind == "float":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log")))
        else:
            raise ValueError(f"unknown parameter type {kind!r} for {name}")
    return params


def fit_on_training_data(
    model_name: str,
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    preprocessing: str,
    imbalance: str,
    seed: int,
):
    """Fit preprocessor + model on training data only (early-stopping hold-out carved
    out of ``X`` when the model supports it). Returns ``(preprocessor, model)``."""
    model = make_model(model_name, params, imbalance, seed)
    pre = make_preprocessor(preprocessing).fit(X)
    Xt = pre.transform(X)
    if supports_validation(model):
        strat = y if np.bincount(y, minlength=2).min() >= 2 else None
        X_fit, X_es, y_fit, y_es = train_test_split(
            Xt, y, test_size=EARLY_STOPPING_FRACTION, random_state=seed, stratify=strat
        )
        fit_model(model, X_fit, y_fit, X_es, y_es, imbalance=imbalance, seed=seed)
    else:
        fit_model(model, Xt, y, imbalance=imbalance, seed=seed)
    return pre, model


def inner_cv_auc(
    model_name: str,
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    preprocessing: str,
    imbalance: str,
    seed: int,
    inner_folds: int = 3,
) -> float:
    """Mean ROC-AUC over a stratified inner k-fold CV of (X, y)."""
    y = np.asarray(y, dtype=int)
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in skf.split(X, y):
        pre, model = fit_on_training_data(model_name, params, X.iloc[tr], y[tr], preprocessing, imbalance, seed)
        aucs.append(roc_auc(y[va], positive_proba(model, pre.transform(X.iloc[va]))))
    aucs = np.asarray(aucs, dtype=float)
    return float(np.nanmean(aucs)) if np.isfinite(aucs).any() else 0.0


def tune(
    model_name: str,
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    space: dict,
    n_trials: int,
    preprocessing: str,
    imbalance: str,
    seed: int,
    inner_folds: int = 3,
) -> TuningResult:
    """Tune ``model_name`` on the fitting data with Optuna TPE (seeded with ``seed``)."""
    t0 = time.perf_counter()
    if not space or n_trials <= 0:
        return TuningResult({}, float("nan"), 0, 0.0)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, space)
        return inner_cv_auc(model_name, params, X_fit, y_fit, preprocessing, imbalance, seed, inner_folds)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trials = [{"number": t.number, "value": t.value, **t.params} for t in study.trials]
    return TuningResult(dict(study.best_params), float(study.best_value), n_trials, time.perf_counter() - t0, trials)
