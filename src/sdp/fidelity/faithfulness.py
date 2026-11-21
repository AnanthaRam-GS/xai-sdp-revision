"""Faithfulness correlation (Bhatt, Weller and Moura, 2020).

For each instance, 50 random feature subsets of size 5 are drawn (seeded; the same
subsets for every explanation of that instance). For each subset S:

* the sum of the explanation's attributions over S, and
* the prediction drop f(x) − f(x with S removed), with marginal removal and f = P(defective).

Faithfulness is the Pearson correlation between the two over the 50 subsets (NaN when
either is constant). Saliency is used as |gradient|.
"""

from __future__ import annotations

import numpy as np


def random_subsets(n: int, p: int, n_subsets: int = 50, size: int = 5, seed: int = 0) -> np.ndarray:
    """``(n, n_subsets, p)`` boolean masks with exactly ``size`` features each."""
    rng = np.random.default_rng(seed)
    keys = rng.random((n, n_subsets, p))
    chosen = np.argsort(keys, axis=2)[:, :, :size]
    masks = np.zeros((n, n_subsets, p), dtype=bool)
    np.put_along_axis(masks, chosen, True, axis=2)
    return masks


def prediction_drops(remover, X: np.ndarray, f_x: np.ndarray, subsets: np.ndarray) -> np.ndarray:
    """``(n, n_subsets)`` drops f(x) − f(x without S)."""
    return np.asarray(f_x, dtype=float)[:, None] - remover(X, subsets)


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-15, (a * b).sum(axis=1) / den, np.nan)


def faithfulness(attr: np.ndarray, subsets: np.ndarray, drops: np.ndarray) -> np.ndarray:
    """Per-instance Pearson correlation between subset attribution sums and drops."""
    sums = np.einsum("np,nsp->ns", np.asarray(attr, dtype=float), subsets.astype(float))
    return pearson_rows(sums, drops)
