"""Shared helpers for the step 6 fidelity tests.

* Explanations are read from the (v2) step 4 attribution store and put in wide form
  with :func:`sdp.analysis.explanations.from_long` (columns matched by feature name).
* Feature rankings are by |attribution|, ties broken by the canonical feature order
  (a stable sort), exactly as in step 4.
* Saliency is used as |gradient| (its ranking is the same either way).
* :func:`refit` retrains a model type with given hyperparameters (no tuning) using the
  step 3 building blocks (preprocessor, model registry, imbalance handling, early
  stopping on the validation part, MCC-optimal threshold).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.analysis.explanations import from_long
from sdp.eval.metrics import best_mcc_threshold, evaluate
from sdp.eval.preprocess import FEATURES, make_preprocessor
from sdp.experiments.persistence import safe_split_id
from sdp.explain.base import partition_path
from sdp.models.imbalance import fit_model
from sdp.models.registry import make_model, positive_proba

logger = logging.getLogger(__name__)

EVALUATED = {
    "mlp": ["DeepLift", "DeepLiftShap", "GradientShap", "IntegratedGradients", "Saliency", "InputXGradient", "LIME",
            "KernelSHAP"],
    "lr": ["LinearSHAP", "LIME"],
    "rf": ["TreeSHAP", "LIME"],
    "xgb": ["TreeSHAP", "LIME"],
}  # fmt: skip
ABSOLUTE = {"Saliency"}  # evaluated as |attribution|


def store_attributions_by_id(root: Path, split_id: str, model: str, method: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(instance ids, attributions)`` from the store, sorted by instance id (the ids are
    positions in the split's test set; a method may cover only a subset of them)."""
    path = partition_path(root, model, method, safe_split_id(split_id), seed)
    ex = from_long(pd.read_parquet(path))
    ids = ex.meta["instance_id"].to_numpy()
    order = np.argsort(ids, kind="stable")
    attr = ex.attr[order]
    return ids[order], (np.abs(attr) if method in ABSOLUTE else attr)


def store_attributions(root: Path, split_id: str, model: str, method: str, seed: int = 0) -> np.ndarray:
    """Attributions (n_test × 20, canonical feature order, instance order) from the store;
    the method must cover every test instance."""
    ids, attr = store_attributions_by_id(root, split_id, model, method, seed)
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError(f"{split_id} {model} {method}: explanations cover only part of the test set")
    return attr


def ranking(attr: np.ndarray) -> np.ndarray:
    """Feature order per row, most important first (|attribution|, ties by feature order)."""
    return np.argsort(-np.abs(np.atleast_2d(attr)), axis=1, kind="stable")


def random_orders(n: int, p: int, n_orders: int, seed: int) -> np.ndarray:
    """``(n_orders, n, p)`` random feature orderings (seeded)."""
    rng = np.random.default_rng(seed)
    return np.stack([np.argsort(rng.random((n, p)), axis=1) for _ in range(n_orders)])


def trapezoid_area(curves: np.ndarray) -> np.ndarray:
    """Normalised area under curves sampled at k/K, k = 0..K (last axis), trapezoid rule."""
    curves = np.asarray(curves, dtype=float)
    K = curves.shape[-1] - 1
    return np.trapezoid(curves, dx=1.0 / K, axis=-1)


@dataclass
class Refit:
    preprocessor: object
    model: object
    threshold: float


def refit(model_name: str, params: dict, preprocessing: str, imbalance: str, seed: int,
          X_fit: pd.DataFrame, y_fit, X_val: pd.DataFrame, y_val) -> Refit:  # fmt: skip
    """Fit ``model_name`` with fixed hyperparameters on (X_fit, y_fit), early stopping and
    threshold on (X_val, y_val) — as in step 3, without tuning."""
    from sdp.seed import set_seed

    set_seed(seed, torch_threads=1, quiet=True)
    model = make_model(model_name, params, imbalance, seed)
    pre = make_preprocessor(preprocessing).fit(X_fit)
    Xf, Xv = pre.transform(X_fit), pre.transform(X_val)
    fit_model(model, Xf, np.asarray(y_fit), Xv, np.asarray(y_val), imbalance=imbalance, seed=seed)
    threshold = best_mcc_threshold(np.asarray(y_val), positive_proba(model, Xv))
    return Refit(pre, model, threshold)


def test_metrics(fitted: Refit, X_test: pd.DataFrame, y_test, loc_test) -> dict[str, float]:
    """AUC, MCC at the validation-tuned threshold and Popt on the test release."""
    p = positive_proba(fitted.model, fitted.preprocessor.transform(X_test))
    rows = {(m, k): v for m, k, v in evaluate(np.asarray(y_test), p, np.asarray(loc_test), fitted.threshold)}
    return {"auc": rows[("none", "roc_auc")], "mcc_tuned": rows[("val_mcc", "mcc")], "popt": rows[("none", "popt")]}


def feature_index(names) -> list[int]:
    return [FEATURES.index(f) for f in names]
