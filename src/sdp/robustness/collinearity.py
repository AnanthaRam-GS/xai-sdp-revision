"""Multicollinearity of the CK metrics: diagnostics, AutoSpearman and grouped attributions.

* **Diagnostics** per training release (the fitting rows of each time-aware split, raw
  metrics): Spearman correlation matrix; hierarchical clustering (average linkage) on the
  distance 1 − |ρ|, cut at |ρ| = 0.7 (distance 0.3); VIF on log1p-transformed metrics
  (from the inverse correlation matrix). Constant metrics are left out of the
  correlation analysis and form their own clusters.
* **Stable clusters**: two metrics are linked when they fall in the same cluster in at
  least 75% of the training releases; the stable clusters are the connected components
  of these links (singletons included).
* **AutoSpearman** (Jiarpakdee, Tantithamthavorn and Treude, 2018): (1) while some
  pair has |ρ| ≥ 0.7, take the most strongly correlated pair and remove the metric with
  the higher mean |ρ| to the other remaining metrics; (2) while the largest VIF is ≥ 5,
  remove that metric. VIF is computed on log1p-transformed metrics, as in the
  diagnostics. Constant metrics are removed first.
* **Grouped attributions**: attributions are summed within each stable cluster per
  instance; the step 4 analysis functions then run at the cluster level.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from sdp.analysis.explanations import Explanations
from sdp.eval.preprocess import FEATURES

logger = logging.getLogger(__name__)


def _non_constant(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if X[c].nunique() > 1]


def spearman_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Spearman ρ between the metrics (NaN rows/columns for constant metrics)."""
    return X.astype(float).corr(method="spearman")


def correlation_linkage(X: pd.DataFrame, method: str = "average"):
    """``(linkage matrix, metric names)`` on the distance 1 − |ρ| (non-constant metrics)."""
    cols = _non_constant(X)
    rho = spearman_matrix(X[cols]).abs().to_numpy()
    dist = np.clip(1.0 - rho, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method=method), cols


def correlation_clusters(X: pd.DataFrame, threshold: float = 0.7, method: str = "average") -> list[list[str]]:
    """Clusters of metrics (canonical order) from the dendrogram cut at |ρ| = ``threshold``."""
    Z, cols = correlation_linkage(X, method)
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance") if len(cols) > 1 else np.ones(len(cols), int)
    groups: dict[int, list[str]] = {}
    for c, lab in zip(cols, labels):
        groups.setdefault(int(lab), []).append(c)
    clusters = list(groups.values()) + [[c] for c in X.columns if c not in cols]
    return sorted((sorted(g, key=list(X.columns).index) for g in clusters), key=lambda g: list(X.columns).index(g[0]))


def log1p_frame(X: pd.DataFrame) -> pd.DataFrame:
    """log1p of the metrics that are non-negative (all CK metrics are)."""
    X = X.astype(float).copy()
    cols = [c for c in X.columns if X[c].min() >= 0]
    X[cols] = np.log1p(X[cols])
    return X


def vif(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factors on log1p-transformed, non-constant metrics."""
    cols = _non_constant(X)
    Z = log1p_frame(X[cols])
    corr = np.corrcoef(Z.to_numpy(), rowvar=False)
    inv = np.linalg.pinv(np.atleast_2d(corr))
    return pd.Series(np.diag(inv), index=cols, name="vif")


def autospearman(X: pd.DataFrame, threshold: float = 0.7, vif_threshold: float = 5.0) -> tuple[list[str], list[dict]]:
    """AutoSpearman metric selection. Returns ``(selected metrics, removal log)``."""
    order = list(X.columns)
    log = [{"step": "constant", "removed": c, "reason": "constant"} for c in order if X[c].nunique() <= 1]
    keep = _non_constant(X)
    while len(keep) > 1:
        rho = spearman_matrix(X[keep]).abs()
        values = rho.to_numpy(copy=True)
        np.fill_diagonal(values, 0.0)
        i, j = np.unravel_index(np.nanargmax(values), values.shape)
        if values[i, j] < threshold:
            break
        a, b = keep[i], keep[j]
        mean_corr = values.sum(axis=1) / (len(keep) - 1)  # mean |rho| to the other remaining metrics
        drop = a if mean_corr[i] > mean_corr[j] or (mean_corr[i] == mean_corr[j] and order.index(a) > order.index(b)) else b
        log.append({"step": "spearman", "removed": drop, "reason": f"|rho|={values[i, j]:.3f} with {b if drop == a else a}"})
        keep.remove(drop)
    while len(keep) > 1:
        v = vif(X[keep])
        top = v.idxmax()
        if v[top] < vif_threshold:
            break
        log.append({"step": "vif", "removed": top, "reason": f"VIF={v[top]:.2f}"})
        keep.remove(top)
    return [c for c in order if c in keep], log


# ------------------------------------------------------------------ stable clusters


def co_clustering(cluster_lists: list[list[list[str]]], features=FEATURES) -> pd.DataFrame:
    """Share of releases in which each pair of metrics is in the same cluster."""
    m = pd.DataFrame(0.0, index=list(features), columns=list(features))
    for clusters in cluster_lists:
        for g in clusters:
            for a, b in combinations(g, 2):
                m.loc[a, b] += 1
                m.loc[b, a] += 1
    m /= max(len(cluster_lists), 1)
    for f in features:
        m.loc[f, f] = 1.0
    return m


def stable_clusters(cluster_lists: list[list[list[str]]], share: float = 0.75, features=FEATURES) -> list[list[str]]:
    """Connected components of the pairs clustered together in ≥ ``share`` of releases."""
    co = co_clustering(cluster_lists, features)
    parent = {f: f for f in features}

    def find(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    for a, b in combinations(features, 2):
        if co.loc[a, b] >= share:
            parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for f in features:
        groups.setdefault(find(f), []).append(f)
    return sorted(groups.values(), key=lambda g: list(features).index(g[0]))


def cluster_name(group: list[str]) -> str:
    return "+".join(group)


def group_explanations(ex: Explanations, clusters: list[list[str]]) -> Explanations:
    """Sum the (signed) attributions within each cluster; features become cluster names."""
    covered = [f for g in clusters for f in g]
    if sorted(covered) != sorted(ex.features) or len(covered) != len(set(covered)):
        raise ValueError("clusters must partition the features exactly")
    idx = [[ex.features.index(f) for f in g] for g in clusters]
    attr = np.column_stack([ex.attr[:, i].sum(axis=1) for i in idx])
    raw = np.full_like(attr, np.nan)
    return Explanations(ex.meta.copy(), attr, raw, [cluster_name(g) for g in clusters])


def loc_cluster(clusters: list[list[str]]) -> list[str]:
    return next(g for g in clusters if "loc" in g)


# ------------------------------------------------------------------------- driver


def release_diagnostics(splits, frames, threshold: float = 0.7, vif_threshold: float = 5.0, method: str = "average"):
    """Per training release: clusters, VIF, AutoSpearman selection (fitting rows)."""
    from sdp.eval.preprocess import feature_frame

    out = []
    for split in splits:
        X = feature_frame(frames[split.project]).iloc[split.fit_idx]
        selected, log = autospearman(X, threshold, vif_threshold)
        out.append({
            "split_id": split.split_id, "project": split.project, "train_release": "+".join(split.train_releases),
            "n_fit": len(X), "clusters": correlation_clusters(X, threshold, method), "vif": vif(X),
            "selected": selected, "log": log, "rho": spearman_matrix(X),
        })  # fmt: skip
    return out
