"""Logistic Regression, Random Forest and XGBoost with the project's imbalance options.

``imbalance`` is ``"none"``, ``"class_weight"`` or ``"smote"``. For ``class_weight``,
LR and RF use ``class_weight="balanced"`` and XGBoost uses
``scale_pos_weight = n_neg / n_pos``. For ``smote`` the models are built *unweighted*;
the resampling itself is done outside the model (:mod:`sdp.models.imbalance`) and only
on the fitting part of the training data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def _balanced(imbalance: str) -> str | None:
    return "balanced" if imbalance == "class_weight" else None


def make_lr(C: float = 1.0, imbalance: str = "class_weight", random_state: int = 0) -> LogisticRegression:
    """L2-regularised logistic regression (lbfgs)."""
    return LogisticRegression(C=C, solver="lbfgs", max_iter=5000, class_weight=_balanced(imbalance),
                              random_state=random_state)  # fmt: skip


def make_rf(
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    max_features: str | float = "sqrt",
    imbalance: str = "class_weight",
    random_state: int = 0,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=_balanced(imbalance),
        n_jobs=1,
        random_state=random_state,
    )


class XGBClassifierES(ClassifierMixin, BaseEstimator):
    """XGBoost (``tree_method="hist"``) with early stopping on validation data.

    ``n_estimators`` is the maximum number of boosting rounds; training stops after
    ``early_stopping_rounds`` rounds without improvement of the validation log loss and
    predictions use the best iteration. Without validation data, a stratified
    ``internal_val_fraction`` of the training data is held out for early stopping.
    """

    def __init__(
        self,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        n_estimators: int = 300,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: float = 1.0,
        imbalance: str = "class_weight",
        early_stopping_rounds: int = 50,
        internal_val_fraction: float = 0.2,
        random_state: int = 0,
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.imbalance = imbalance
        self.early_stopping_rounds = early_stopping_rounds
        self.internal_val_fraction = internal_val_fraction
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y, dtype=int)
        if X_val is None:
            strat = y if np.bincount(y, minlength=2).min() >= 2 else None
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=self.internal_val_fraction, random_state=self.random_state, stratify=strat
            )
        n_pos = float(np.sum(y))
        spw = (len(y) - n_pos) / n_pos if (self.imbalance == "class_weight" and n_pos > 0) else 1.0
        self.model_ = XGBClassifier(
            tree_method="hist",
            max_depth=int(self.max_depth),
            learning_rate=self.learning_rate,
            n_estimators=int(self.n_estimators),
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            scale_pos_weight=spw,
            eval_metric="logloss",
            early_stopping_rounds=self.early_stopping_rounds,
            n_jobs=1,
            random_state=self.random_state,
        )
        self.model_.fit(X, y, eval_set=[(X_val, np.asarray(y_val, dtype=int))], verbose=False)
        self.classes_ = self.model_.classes_
        self.best_iteration_ = int(self.model_.best_iteration)
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
