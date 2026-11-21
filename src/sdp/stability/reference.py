"""Stability metrics between global importance vectors, null references and uncertainty.

* **Global importance** g(split, model, method, seed): mean L1-normalised |attribution|
  over all test instances (step 4 convention); per split, g is averaged over the
  available seeds unless stated otherwise.
* **Metrics** between two importance vectors: Spearman ρ and Kendall τ-b (the step 4
  :func:`~sdp.analysis.explanations.rank_correlations`), top-5 and top-3 Jaccard, and
  rank-biased overlap (RBO, p = 0.9; Webber, Moffat and Zobel, 2010, extrapolated form
  for two complete rankings of the same items). Rankings break ties by feature order.
* **Random null**: the metrics between two independent uniformly random rankings
  (10,000 seeded draws).
* **Instance bootstrap**: resample the test instances of each split (1,000 resamples,
  seeded), recompute g and the metric; the percentile interval is the 95% CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sdp.analysis.explanations import normalised_abs, rank_correlations

METRICS = ("spearman", "kendall", "jaccard5", "jaccard3", "rbo")
RBO_P = 0.9


def order(g: np.ndarray) -> np.ndarray:
    """Feature order (most important first; ties by feature order), row-wise."""
    return np.argsort(-np.atleast_2d(g), axis=1, kind="stable")


def jaccard_topk(A: np.ndarray, B: np.ndarray, k: int) -> np.ndarray:
    oa, ob = order(A)[:, :k], order(B)[:, :k]
    p = np.atleast_2d(A).shape[1]
    ma = np.zeros((len(oa), p), bool)
    mb = np.zeros((len(ob), p), bool)
    np.put_along_axis(ma, oa, True, axis=1)
    np.put_along_axis(mb, ob, True, axis=1)
    return (ma & mb).sum(axis=1) / (ma | mb).sum(axis=1)


def rbo(A: np.ndarray, B: np.ndarray, p: float = RBO_P) -> np.ndarray:
    """Extrapolated rank-biased overlap of the two rankings induced by A and B (row-wise).

    For complete rankings of the same k items: RBO = (X_k / k) p^k + (1 − p)/p Σ_{d=1..k} (X_d / d) p^d,
    with X_d the overlap of the top-d sets; identical rankings give 1."""
    oa, ob = order(A), order(B)
    n, k = oa.shape
    rank_b = np.empty_like(ob)
    rank_b[np.arange(n)[:, None], ob] = np.arange(k)[None, :]
    rank_a = np.empty_like(oa)
    rank_a[np.arange(n)[:, None], oa] = np.arange(k)[None, :]
    # X_d = number of items with max(rank_a, rank_b) < d
    deepest = np.maximum(rank_a, rank_b)  # (n, k): depth at which an item is in both prefixes
    d = np.arange(1, k + 1)
    X = (deepest[:, None, :] < d[None, :, None]).sum(axis=2)  # (n, k)
    agree = X / d[None, :]
    weights = p ** d
    return agree[:, -1] * p**k + (1 - p) / p * (agree * weights[None, :]).sum(axis=1)


def stability_metrics(A: np.ndarray, B: np.ndarray) -> dict[str, np.ndarray]:
    """All metrics, row-wise, between importance vectors A and B (n × p)."""
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    rc = rank_correlations(A, B)
    return {"spearman": rc["spearman"], "kendall": rc["kendall"], "jaccard5": jaccard_topk(A, B, 5),
            "jaccard3": jaccard_topk(A, B, 3), "rbo": rbo(A, B)}  # fmt: skip


def random_null(n_features: int = 20, n_draws: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """Metrics between pairs of independent uniformly random rankings."""
    rng = np.random.default_rng(seed)
    A = np.argsort(rng.random((n_draws, n_features)), axis=1).astype(float)
    B = np.argsort(rng.random((n_draws, n_features)), axis=1).astype(float)
    return pd.DataFrame(stability_metrics(A, B))


def null_summary(null: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"metric": list(null.columns), "null_mean": null.mean().to_numpy(),
                         "null_p95": null.quantile(0.95).to_numpy()})  # fmt: skip


def global_importance(attr: np.ndarray) -> np.ndarray:
    """g = mean L1-normalised |attribution| over instances (rows of ``attr``)."""
    return np.nanmean(normalised_abs(attr), axis=0)


def bootstrap_weights(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """``(n_boot, n)`` resampling weights (counts / n) of a nonparametric bootstrap."""
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot) / n


def bootstrap_g(norm: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap replicates of g from per-instance normalised importances (n × p);
    rows with NaN (all-zero attributions) are dropped first."""
    norm = norm[~np.isnan(norm).any(axis=1)]
    return bootstrap_weights(len(norm), n_boot, rng) @ norm


def comparison_ci(norm_a: np.ndarray, norm_b: np.ndarray, n_boot: int = 1000, seed: int = 0,
                  confidence: float = 0.95) -> pd.DataFrame:  # fmt: skip
    """Point estimate and percentile CI of every metric between g_a and g_b, resampling
    the instances of both sides independently."""
    rng = np.random.default_rng(seed)
    point = stability_metrics(np.nanmean(norm_a, axis=0), np.nanmean(norm_b, axis=0))
    boot = stability_metrics(bootstrap_g(norm_a, n_boot, rng), bootstrap_g(norm_b, n_boot, rng))
    a = (1 - confidence) / 2
    return pd.DataFrame([{"metric": m, "value": float(point[m][0]), "ci_low": float(np.nanquantile(boot[m], a)),
                          "ci_high": float(np.nanquantile(boot[m], 1 - a))} for m in METRICS])  # fmt: skip
