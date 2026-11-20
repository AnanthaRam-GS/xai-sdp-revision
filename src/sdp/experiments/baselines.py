"""Trivial baselines used to test the evaluation framework end to end.

* ``all_defective``: predicts every class as defective (probability 1.0). With the
  risk-density ranking of the effort-aware metrics this inspects the smallest classes
  first.
* ``loc_ranking``: risk = LOC; probability = LOC min-max normalised with the minimum
  and maximum of the training data (clipped to [0, 1]). This is the "ManualDown"-style
  effort baseline: larger classes are considered riskier.
* ``majority``: :class:`~sklearn.dummy.DummyClassifier` with ``strategy="prior"``: it
  predicts the majority class and outputs the training prevalence as the probability.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted


class AllDefective(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.zeros(n), np.ones(n)])

    def predict(self, X):
        return np.ones(len(X), dtype=int)


class LocRanking(ClassifierMixin, BaseEstimator):
    """Probability = min-max normalised LOC (fitted on training data).

    It reads the ``loc`` column of its input. Min-max normalisation is invariant under
    the affine ``standard`` scaling, so the output equals min-max normalised raw LOC;
    under ``log_standard`` the ranking is unchanged but the probabilities differ.
    """

    def __init__(self, loc_column: str = "loc"):
        self.loc_column = loc_column

    def fit(self, X, y=None):
        v = pd.DataFrame(X)[self.loc_column].to_numpy(dtype="float64")
        self.classes_ = np.array([0, 1])
        self.min_, self.max_ = float(v.min()), float(v.max())
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "min_")
        v = pd.DataFrame(X)[self.loc_column].to_numpy(dtype="float64")
        span = self.max_ - self.min_
        p = np.full(len(v), 0.5) if span == 0 else np.clip((v - self.min_) / span, 0.0, 1.0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


BASELINES: dict[str, Callable[[], BaseEstimator]] = {
    "all_defective": AllDefective,
    "loc_ranking": LocRanking,
    "majority": lambda: DummyClassifier(strategy="prior"),
}
