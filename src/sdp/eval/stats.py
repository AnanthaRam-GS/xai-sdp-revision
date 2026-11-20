"""Statistical analysis: bootstrap confidence intervals, paired tests, effect sizes and
the Scott-Knott ESD ranking.

References
----------
* Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65-70.
* Romano, J., Kromrey, J. D., Coraggio, J., & Skowronek, J. (2006). Appropriate
  statistics for ordinal level data. *Annual Meeting of the Florida Association of
  Institutional Research*. (Cliff's delta thresholds 0.147 / 0.33 / 0.474.)
* Scott, A. J., & Knott, M. (1974). A cluster analysis method for grouping means in the
  analysis of variance. *Biometrics*, 30(3), 507-512.
* Tantithamthavorn, C., McIntosh, S., Hassan, A. E., & Matsumoto, K. (2017). An
  empirical comparison of model validation techniques for defect prediction models.
  *IEEE TSE*, 43(1), 1-18. (Scott-Knott ESD.)

No maintained Python implementation of Scott-Knott ESD exists on PyPI (the reference
implementation is the R package ``ScottKnottESD``), so it is implemented here.
"""

from __future__ import annotations

import logging
import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as st

logger = logging.getLogger(__name__)

NAN = float("nan")


# ------------------------------------------------------------------- bootstrap CI


def bootstrap_ci(
    values, n_resamples: int = 2000, confidence: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap confidence interval of the mean.

    NaN values are dropped. Returns ``(mean, lower, upper)``; the interval is NaN when
    fewer than two values remain.
    """
    x = np.asarray(values, dtype="float64")
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return NAN, NAN, NAN
    mean = float(x.mean())
    if len(x) < 2:
        logger.warning("bootstrap_ci: only %d value(s); interval undefined.", len(x))
        return mean, NAN, NAN
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_resamples, len(x)))].mean(axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return mean, float(lo), float(hi)


# ----------------------------------------------------------------- effect sizes


def cliffs_delta(x, y) -> float:
    """Cliff's delta: P(X > Y) - P(X < Y)."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if not len(x) or not len(y):
        return NAN
    diff = x[:, None] - y[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / diff.size)


def cliffs_magnitude(d: float) -> str:
    """Magnitude label following Romano et al. (2006)."""
    if math.isnan(d):
        return "n/a"
    a = abs(d)
    return "negligible" if a < 0.147 else "small" if a < 0.33 else "medium" if a < 0.474 else "large"


def cohens_d(x, y) -> float:
    """Cohen's d with pooled standard deviation."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return NAN
    pooled = math.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    diff = x.mean() - y.mean()
    if pooled == 0:
        return 0.0 if diff == 0 else math.copysign(math.inf, diff)
    return float(diff / pooled)


# ------------------------------------------------------------- paired comparisons


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    p = np.asarray(p_values, dtype="float64")
    out = np.full_like(p, NAN)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv, kind="mergesort")
    adj = np.minimum(1.0, np.maximum.accumulate(pv[order] * (m - np.arange(m))))
    res = np.empty(m)
    res[order] = adj
    out[ok] = res
    return out


def pairwise_wilcoxon(
    df: pd.DataFrame,
    block: str = "split_id",
    group: str = "model",
    value: str = "value",
) -> pd.DataFrame:
    """Two-sided Wilcoxon signed-rank tests for every pair of ``group`` values, paired
    on ``block`` (e.g. the split), with Holm correction across all pairs and Cliff's
    delta (on the paired samples) as the effect size.

    Pairs are compared on the blocks where both values are present. If all paired
    differences are zero, p = 1.
    """
    wide = df.pivot_table(index=block, columns=group, values=value, aggfunc="mean")
    rows = []
    for a, b in combinations(wide.columns, 2):
        pair = wide[[a, b]].dropna()
        n = len(pair)
        if n == 0:
            stat, p = NAN, NAN
        elif np.allclose(pair[a].to_numpy(), pair[b].to_numpy()):
            stat, p = 0.0, 1.0
        else:
            res = st.wilcoxon(pair[a], pair[b], zero_method="wilcox", alternative="two-sided")
            stat, p = float(res.statistic), float(res.pvalue)
        d = cliffs_delta(pair[a], pair[b]) if n else NAN
        rows.append({
            "model_a": a, "model_b": b, "n": n,
            "mean_a": float(pair[a].mean()) if n else NAN,
            "mean_b": float(pair[b].mean()) if n else NAN,
            "statistic": stat, "p_value": p,
            "cliffs_delta": d, "magnitude": cliffs_magnitude(d),
        })  # fmt: skip
    out = pd.DataFrame(rows)
    if len(out):
        out.insert(out.columns.get_loc("p_value") + 1, "p_holm", holm(out["p_value"]))
    return out


# --------------------------------------------------------------- Scott-Knott ESD


def _sk_best_split(means: np.ndarray) -> tuple[int, float]:
    """Best bipartition of sorted means: (index of first element of group 2, B0)."""
    k = len(means)
    total = means.sum()
    best_j, best_b = 1, -np.inf
    for j in range(1, k):
        t1, t2 = means[:j].sum(), means[j:].sum()
        b0 = t1**2 / j + t2**2 / (k - j) - total**2 / k
        if b0 > best_b + 1e-15:
            best_j, best_b = j, b0
    return best_j, float(best_b)


def sk_lambda(means: np.ndarray, s2_mean: float, nu: int) -> tuple[int, float]:
    """Scott-Knott likelihood-ratio statistic for one group of sorted means.

    ``s2_mean`` is the variance of a treatment mean (MSE / r) and ``nu`` the error
    degrees of freedom. Returns (split index, lambda).
    """
    k = len(means)
    j, b0 = _sk_best_split(means)
    sigma2 = (((means - means.mean()) ** 2).sum() + nu * s2_mean) / (k + nu)
    if sigma2 <= 0:
        return j, (math.inf if b0 > 0 else 0.0)
    return j, float(math.pi / (2 * (math.pi - 2)) * b0 / sigma2)


def scott_knott_esd(
    data: pd.DataFrame | dict,
    alpha: float = 0.05,
    transform: str = "auto",
    negligible_d: float = 0.2,
) -> pd.DataFrame:
    """Scott-Knott ESD ranking (Tantithamthavorn et al., 2017).

    ``data`` is wide: one column (or dict entry) per treatment, one row per observation
    (e.g. per split). Steps:

    1. Correct skewness with a log transformation: ``transform="auto"`` applies
       ``log1p`` when every value is non-negative, otherwise leaves the data as they
       are; ``"log1p"`` forces it, ``"none"`` disables it.
    2. Scott-Knott: sort treatments by mean (descending), split recursively at the
       partition maximising the between-group sum of squares while the likelihood-ratio
       statistic is significant (chi-squared with k/(pi-2) d.f.; MSE and error d.f.
       from a one-way ANOVA over all treatments).
    3. Effect-size correction: adjacent groups (by mean) whose pooled observations have
       a negligible Cohen's |d| (< ``negligible_d``) are merged, sequentially from the
       top, re-checking each merged group against the next one.

    Returns a DataFrame with ``treatment``, ``mean`` (original scale) and ``rank``
    (1 = best, i.e. highest mean), sorted by rank then mean.
    """
    wide = pd.DataFrame({k: pd.Series(v, dtype="float64") for k, v in dict(data).items()}) \
        if isinstance(data, dict) else data.astype("float64")  # fmt: skip
    obs = {c: wide[c].dropna().to_numpy() for c in wide.columns}
    obs = {c: v for c, v in obs.items() if len(v)}
    if not obs:
        return pd.DataFrame(columns=["treatment", "mean", "rank"])
    raw_means = {c: float(v.mean()) for c, v in obs.items()}

    allv = np.concatenate(list(obs.values()))
    if transform == "log1p" or (transform == "auto" and (allv >= 0).all()):
        obs = {c: np.log1p(v) for c, v in obs.items()}

    names = sorted(obs, key=lambda c: obs[c].mean(), reverse=True)
    means = np.array([obs[c].mean() for c in names])
    counts = np.array([len(obs[c]) for c in names])
    n_total, k = counts.sum(), len(names)
    nu = int(n_total - k)
    sse = sum(((obs[c] - obs[c].mean()) ** 2).sum() for c in names)
    mse = sse / nu if nu > 0 else 0.0
    r = len(counts) / (1.0 / counts).sum()  # harmonic mean of replications
    s2_mean = mse / r

    groups: list[list[int]] = []

    def split(lo: int, hi: int) -> None:  # treatments names[lo:hi]
        if hi - lo < 2:
            groups.append(list(range(lo, hi)))
            return
        j, lam = sk_lambda(means[lo:hi], s2_mean, nu)
        crit = st.chi2.ppf(1 - alpha, (hi - lo) / (math.pi - 2))
        if lam > crit:
            split(lo, lo + j)
            split(lo + j, hi)
        else:
            groups.append(list(range(lo, hi)))

    split(0, k)

    # Effect-size-aware merging of adjacent groups.
    merged: list[list[int]] = [groups[0]]
    for g in groups[1:]:
        a = np.concatenate([obs[names[i]] for i in merged[-1]])
        b = np.concatenate([obs[names[i]] for i in g])
        d = cohens_d(a, b)
        if not math.isnan(d) and abs(d) < negligible_d:
            merged[-1] = merged[-1] + g
        else:
            merged.append(g)

    rows = [
        {"treatment": names[i], "mean": raw_means[names[i]], "rank": rank}
        for rank, g in enumerate(merged, start=1)
        for i in g
    ]
    return pd.DataFrame(rows).sort_values(["rank", "mean"], ascending=[True, False]).reset_index(drop=True)
