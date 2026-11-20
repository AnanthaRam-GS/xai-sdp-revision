"""Rule-based and trivial baselines.

* ``manual_down``: risk = LOC (larger classes first; Zhou et al., 2018 "ManualDown").
* ``manual_up``: risk = 1 / max(LOC, 1) (smaller classes first; "ManualUp").
  For both, the probability is the risk min-max normalised with the minimum and maximum
  of the *training* data, clipped to [0, 1].
* ``all_defective``: probability 1.0 for every class.
* ``majority``: :class:`~sklearn.dummy.DummyClassifier` (``strategy="prior"``):
  predicts the majority class with the training prevalence as probability.

ManualUp/ManualDown read the ``loc`` column and must see *raw* LOC (1 / LOC is not
invariant to scaling), so they declare ``requires_raw_features = True`` and the runner
does not preprocess their input.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted


class AllDefective(ClassifierMixin, BaseEstimator):
    requires_raw_features = True

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.zeros(n), np.ones(n)])

    def predict(self, X):
        return np.ones(len(X), dtype=int)


class _LocRisk(ClassifierMixin, BaseEstimator):
    requires_raw_features = True

    def __init__(self, loc_column: str = "loc"):
        self.loc_column = loc_column

    def risk(self, X) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, X, y=None):
        r = self.risk(X)
        self.classes_ = np.array([0, 1])
        self.min_, self.max_ = float(r.min()), float(r.max())
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "min_")
        r = self.risk(X)
        span = self.max_ - self.min_
        p = np.full(len(r), 0.5) if span == 0 else np.clip((r - self.min_) / span, 0.0, 1.0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def _loc(self, X) -> np.ndarray:
        return pd.DataFrame(X)[self.loc_column].to_numpy(dtype="float64")


class ManualDown(_LocRisk):
    """Risk = LOC."""

    def risk(self, X) -> np.ndarray:
        return self._loc(X)


class ManualUp(_LocRisk):
    """Risk = 1 / max(LOC, 1)."""

    def risk(self, X) -> np.ndarray:
        return 1.0 / np.maximum(self._loc(X), 1.0)


class Majority(DummyClassifier):
    requires_raw_features = True

    def __init__(self):
        super().__init__(strategy="prior")


BASELINES: dict[str, Callable[[], BaseEstimator]] = {
    "manual_up": ManualUp,
    "manual_down": ManualDown,
    "all_defective": AllDefective,
    "majority": Majority,
}
