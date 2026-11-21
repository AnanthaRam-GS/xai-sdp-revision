"""Deletion and insertion curves.

For an explanation's per-instance ranking (|attribution|, ties by feature order):

* deletion: p_k after removing the top-k features, k = 0..20;
* insertion: p_k starting from all features removed and adding the top-k back.

The curve tracks the confidence in the model's own prediction at its validation-tuned
threshold: p for instances predicted defective, 1 − p for instances predicted clean.
Areas use the trapezoid rule over k/20 (deletion: lower is better; insertion: higher is
better). The random reference is the mean over 10 seeded random orderings per instance;
gains are AUC_random − AUC_method (deletion) and AUC_method − AUC_random (insertion).
"""

from __future__ import annotations

import numpy as np

from sdp.fidelity.common import random_orders, ranking, trapezoid_area

KINDS = ("deletion", "insertion")


def removal_masks(order: np.ndarray, kind: str) -> np.ndarray:
    """``(n, p + 1, p)`` masks (True = removed) for k = 0..p."""
    n, p = order.shape
    rank = np.empty_like(order)
    rank[np.arange(n)[:, None], order] = np.arange(p)[None, :]  # position of each feature
    k = np.arange(p + 1)[None, :, None]
    in_top = rank[:, None, :] < k  # (n, p+1, p): feature among the top-k
    if kind == "deletion":
        return in_top
    if kind == "insertion":
        return ~in_top
    raise ValueError(f"unknown curve kind {kind!r}")


def confidence(prob: np.ndarray, predicted_defective: np.ndarray) -> np.ndarray:
    """p where the model predicts defective, 1 − p otherwise (broadcast over k)."""
    pd_ = np.asarray(predicted_defective, dtype=bool)
    pd_ = pd_.reshape(pd_.shape + (1,) * (np.ndim(prob) - pd_.ndim))
    return np.where(pd_, prob, 1.0 - prob)


def curves(remover, X: np.ndarray, order: np.ndarray, kind: str, predicted_defective: np.ndarray) -> np.ndarray:
    """Confidence curves ``(n, p + 1)``."""
    return confidence(remover(X, removal_masks(order, kind)), predicted_defective)


def evaluate_orders(remover, X: np.ndarray, attributions: dict[str, np.ndarray], predicted_defective: np.ndarray,
                    n_random: int = 10, seed: int = 0, kinds=KINDS) -> dict:  # fmt: skip
    """Curves and areas for every explanation and for the random reference.

    Returns ``{"auc": {kind: {name: (n,)}}, "curves": {kind: {name: (p+1,) mean curve}}}``;
    the random entry (``"random"``) is the mean over ``n_random`` orderings."""
    n, p = X.shape
    out = {"auc": {k: {} for k in kinds}, "curves": {k: {} for k in kinds}}
    rand = random_orders(n, p, n_random, seed)
    for kind in kinds:
        for name, attr in attributions.items():
            c = curves(remover, X, ranking(attr), kind, predicted_defective)
            out["auc"][kind][name] = trapezoid_area(c)
            out["curves"][kind][name] = c.mean(axis=0)
        rc = np.stack([curves(remover, X, r, kind, predicted_defective) for r in rand])  # (R, n, p+1)
        out["auc"][kind]["random"] = trapezoid_area(rc).mean(axis=0)
        out["curves"][kind]["random"] = rc.mean(axis=(0, 1))
    return out
