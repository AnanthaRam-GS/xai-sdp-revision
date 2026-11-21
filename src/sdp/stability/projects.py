"""Cross-project stability (step 7, section 5).

A project-level importance vector is the mean of the seed-averaged global importance
vectors over the project's splits. Between the 10 projects, Spearman and RBO matrices
are computed; the mean off-diagonal value gets a bootstrap CI over project pairs and is
shown next to the random null and the data-only (permuted-label) reference.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from sdp.eval.stats import bootstrap_ci
from sdp.stability.data import ImportanceStore
from sdp.stability.reference import order, stability_metrics


def project_vectors(store: ImportanceStore, splits, model: str, method: str, seeds=None) -> dict[str, np.ndarray]:
    by: dict[str, list] = {}
    for s in splits:
        if store.has(s.split_id, model, method):
            g = store.g(s.split_id, model, method) if seeds is None else np.nanmean(
                [store.g(s.split_id, model, method, sd) for sd in seeds], axis=0)  # fmt: skip
            by.setdefault(s.project, []).append(g)
    return {p: np.mean(v, axis=0) for p, v in by.items()}


def project_matrix(vectors: dict[str, np.ndarray], metric: str) -> pd.DataFrame:
    names = sorted(vectors)
    m = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for a, b in combinations(names, 2):
        v = float(stability_metrics(vectors[a], vectors[b])[metric][0])
        m.loc[a, b] = m.loc[b, a] = v
    return m


def off_diagonal(matrix: pd.DataFrame, seed: int = 0) -> tuple[float, float, float]:
    """Mean of the off-diagonal values with a bootstrap CI over the project pairs."""
    values = matrix.to_numpy()[np.triu_indices(len(matrix), k=1)]
    return bootstrap_ci(values, 2000, 0.95, seed)


def common_top_features(vectors: dict[str, np.ndarray], features: list[str], k: int = 5, min_projects: int = 7) -> pd.DataFrame:
    counts = np.zeros(len(features), dtype=int)
    for v in vectors.values():
        counts[order(v)[0, :k]] += 1
    t = pd.DataFrame({"feature": features, "projects_in_top5": counts})
    return t[t["projects_in_top5"] >= min_projects].sort_values("projects_in_top5", ascending=False).reset_index(drop=True)
