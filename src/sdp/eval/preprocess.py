"""Feature extraction and preprocessing.

The feature set is exactly the 20 CK metrics. ``class_name``, ``bug_count`` and
``defective`` are never features. LOC is additionally kept separately, unscaled and
with its original values, for the effort-aware metrics.

Preprocessors are scikit-learn transformers placed *inside* the model pipeline, so
they are only ever fitted on training data:

* ``none``: identity;
* ``standard``: :class:`~sklearn.preprocessing.StandardScaler`;
* ``log_standard``: ``log1p`` on metrics that are non-negative in the training data,
  then :class:`~sklearn.preprocessing.StandardScaler`.

All transformers output pandas DataFrames with the metric names, so that downstream
steps (and later the explainers) can address features by name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

from sdp.data.clean import COUNT_COL, ID_COL, LABEL_COL, METRICS

logger = logging.getLogger(__name__)

FEATURES: list[str] = list(METRICS)
NON_FEATURES: tuple[str, ...] = (ID_COL, COUNT_COL, LABEL_COL)
EFFORT_COL = "loc"
PREPROCESSING_MODES = ("none", "standard", "log_standard")


@dataclass
class XY:
    """Model inputs for one set of instances."""

    X: pd.DataFrame  # the 20 metrics only
    y: np.ndarray  # binary label
    loc: np.ndarray  # original, unscaled LOC (effort)
    class_name: np.ndarray  # identifiers, never a feature


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The 20 metrics, as floats, in canonical order."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature column(s): {missing}")
    X = df[FEATURES].astype("float64")
    assert not set(NON_FEATURES) & set(X.columns)
    return X


def split_xy(df: pd.DataFrame) -> XY:
    return XY(
        X=feature_frame(df),
        y=df[LABEL_COL].to_numpy(dtype="int64"),
        loc=df[EFFORT_COL].to_numpy(dtype="float64").copy(),
        class_name=df[ID_COL].astype(str).to_numpy(),
    )


class Log1pNonNegative(TransformerMixin, BaseEstimator):
    """Apply ``log1p`` to columns that are non-negative in the training data.

    Columns are chosen at ``fit`` time. At ``transform`` time, negative values in a
    logged column (unseen in training) are clipped to 0 before ``log1p``, with a warning.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.log_columns_ = [c for c in X.columns if (X[c].min() >= 0)]
        return self

    def transform(self, X):
        check_is_fitted(self, "log_columns_")
        X = pd.DataFrame(X, columns=self.feature_names_in_).astype("float64").copy()
        cols = self.log_columns_
        neg = (X[cols] < 0).to_numpy().sum()
        if neg:
            logger.warning("log1p: clipping %d negative value(s) unseen in training to 0.", int(neg))
        X[cols] = np.log1p(X[cols].clip(lower=0))
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


def make_preprocessor(mode: str):
    """Return an unfitted, scikit-learn compatible preprocessor for ``mode``."""
    if mode == "none":
        pre = FunctionTransformer(feature_names_out="one-to-one")
    elif mode == "standard":
        pre = StandardScaler()
    elif mode == "log_standard":
        pre = Pipeline([("log1p", Log1pNonNegative()), ("scale", StandardScaler())])
    else:
        raise ValueError(f"unknown preprocessing mode {mode!r}; choose from {PREPROCESSING_MODES}")
    return pre.set_output(transform="pandas")


def make_pipeline(mode: str, model) -> Pipeline:
    """Preprocessor + model, so that preprocessing is fitted on training data only."""
    return Pipeline([("preprocess", make_preprocessor(mode)), ("model", model)])
