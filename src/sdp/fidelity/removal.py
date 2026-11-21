"""Feature-removal operators.

To *remove* a feature set S from an instance x (model input space):

* **marginal removal** (main): replace x[S] by the values of each of the first 50 rows of
  the instance's step 4 background set (a seeded selection of the 100 rows) and average
  the model's P(defective) over those rows, i.e. E_b[f(x_{¬S}, b_S)];
* **mean removal** (sensitivity): replace x[S] by the background mean.

Masks are boolean arrays with ``True`` = removed.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

BATCH_ROWS = 250_000


def select_background(background: pd.DataFrame | np.ndarray, n: int = 50, seed: int = 0) -> np.ndarray:
    """``n`` rows of the background set chosen with a seeded permutation (all if fewer)."""
    B = np.asarray(background, dtype="float64")
    idx = np.random.default_rng(seed).permutation(len(B))[: min(n, len(B))]
    return B[np.sort(idx)]


class MarginalRemover:
    """P(defective) with the masked features replaced by background rows, averaged."""

    name = "marginal"

    def __init__(self, predict: Callable[[np.ndarray], np.ndarray], background: np.ndarray):
        self.predict = predict
        self.B = np.asarray(background, dtype="float64")

    def __call__(self, X: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """``X``: (n, p); ``masks``: (n, K, p) or (n, p). Returns (n, K) or (n,)."""
        X = np.asarray(X, dtype="float64")
        squeeze = masks.ndim == 2
        M = masks[:, None, :] if squeeze else masks
        n, K, p = M.shape
        m = len(self.B)
        rows = X[:, None, :].repeat(K, axis=1).reshape(n * K, p)
        flat = M.reshape(n * K, p)
        out = np.empty(n * K)
        step = max(1, BATCH_ROWS // m)
        for s in range(0, n * K, step):
            x, mk = rows[s : s + step], flat[s : s + step]
            big = np.where(mk[:, None, :], self.B[None, :, :], x[:, None, :]).reshape(-1, p)
            out[s : s + step] = self.predict(big).reshape(len(x), m).mean(axis=1)
        out = out.reshape(n, K)
        return out[:, 0] if squeeze else out


class MeanRemover:
    """P(defective) with the masked features replaced by the background mean."""

    name = "mean"

    def __init__(self, predict: Callable[[np.ndarray], np.ndarray], background: np.ndarray):
        self.predict = predict
        self.mean = np.asarray(background, dtype="float64").mean(axis=0)

    def __call__(self, X: np.ndarray, masks: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype="float64")
        squeeze = masks.ndim == 2
        M = masks[:, None, :] if squeeze else masks
        n, K, p = M.shape
        rows = np.where(M, self.mean[None, None, :], X[:, None, :]).reshape(n * K, p)
        out = self.predict(rows).reshape(n, K)
        return out[:, 0] if squeeze else out
