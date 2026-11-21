"""Cross-version stability, top-k frequency and direction stability (step 7, section 2).

* Comparisons: consecutive time-aware splits of a project (e.g. ant 1.3→1.4 vs ant
  1.4→1.5); 14 comparisons over the 10 projects.
* For each comparison and explanation: every stability metric between the seed-averaged
  global importance vectors, with a 95% CI from resampling the test instances of both
  splits, the between-seed spread (the metric between same-seed vectors), the
  data-only reference (the same metric between the permuted-label models' vectors,
  seed 0) and the random null.
* Top-k frequency: the share of a project's splits (or of all splits) in whose global
  top-5 a feature appears (the corrected version of the original Table 2).
* Direction stability: the sign of the step 4 direction statistic (Spearman of the raw
  value vs the signed attribution, seeds averaged) per split.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sdp.analysis.explanations import _spearman_columns, label
from sdp.stability.data import ImportanceStore
from sdp.stability.reference import METRICS, comparison_ci, order, stability_metrics

logger = logging.getLogger(__name__)


def consecutive_pairs(splits) -> list[tuple]:
    """(earlier split, later split) for consecutive time-aware splits of each project."""
    by_project: dict[str, list] = {}
    for s in splits:
        by_project.setdefault(s.project, []).append(s)
    return [(a, b) for ss in by_project.values() for a, b in zip(ss, ss[1:])]


def pair_id(a, b) -> str:
    return f"{a.project} {'+'.join(a.train_releases)}>{a.test_release} | {'+'.join(b.train_releases)}>{b.test_release}"


def cross_version(store: ImportanceStore, perm: ImportanceStore | None, pairs, explanations, n_boot: int = 1000,
                  seed: int = 0) -> pd.DataFrame:  # fmt: skip
    rows = []
    for i, (a, b) in enumerate(pairs):
        for model, method in explanations:
            if not (store.has(a.split_id, model, method) and store.has(b.split_id, model, method)):
                continue
            ci = comparison_ci(store.instance_norm(a.split_id, model, method), store.instance_norm(b.split_id, model, method),
                               n_boot, seed=seed + 1000 * i).set_index("metric")  # fmt: skip
            common = sorted(set(store.seeds(a.split_id, model, method)) & set(store.seeds(b.split_id, model, method)))
            per_seed = stability_metrics(np.array([store.g(a.split_id, model, method, s) for s in common]),
                                         np.array([store.g(b.split_id, model, method, s) for s in common]))  # fmt: skip
            data_only = {}
            if perm is not None and perm.has(a.split_id, model, method) and perm.has(b.split_id, model, method):
                data_only = stability_metrics(perm.g(a.split_id, model, method, 0), perm.g(b.split_id, model, method, 0))
            for m in METRICS:
                rows.append({"project": a.project, "comparison": pair_id(a, b), "split_a": a.split_id, "split_b": b.split_id,
                             "model": model, "method": method, "explanation": label(model, method), "metric": m,
                             "value": ci.loc[m, "value"], "ci_low": ci.loc[m, "ci_low"], "ci_high": ci.loc[m, "ci_high"],
                             "seed_mean": float(np.nanmean(per_seed[m])), "seed_sd": float(np.nanstd(per_seed[m], ddof=1)) if len(common) > 1 else np.nan,
                             "n_seeds": len(common), "data_only": float(data_only[m][0]) if data_only else np.nan})  # fmt: skip
    return pd.DataFrame(rows)


def topk_frequency(store: ImportanceStore, splits, explanations, k: int = 5) -> pd.DataFrame:
    """Share of splits (per project and pooled as ``ALL``) whose top-k contains each feature."""
    rows = []
    feats = store.ex.features
    for model, method in explanations:
        tops = {}
        for s in splits:
            if store.has(s.split_id, model, method):
                tops[s.split_id] = (s.project, set(order(store.g(s.split_id, model, method))[0, :k]))
        for scope in sorted({p for p, _ in tops.values()}) + ["ALL"]:
            sel = [t for p, t in tops.values() if scope in ("ALL", p)]
            for j, f in enumerate(feats):
                rows.append({"model": model, "method": method, "explanation": label(model, method), "scope": scope,
                             "feature": f, "n_splits": len(sel), "frequency": sum(j in t for t in sel) / len(sel)})  # fmt: skip
    return pd.DataFrame(rows)


def split_directions(store: ImportanceStore, splits, explanations) -> pd.DataFrame:
    """Step 4 direction statistic per split (Spearman of raw value vs signed attribution,
    averaged over seeds) and its sign."""
    rows = []
    feats = store.ex.features
    for model, method in explanations:
        for s in splits:
            seeds = store.seeds(s.split_id, model, method)
            if not seeds:
                continue
            rho = np.nanmean([_spearman_columns(store.raw(s.split_id, model, method, sd), store.attr(s.split_id, model, method, sd))
                              for sd in seeds], axis=0)  # fmt: skip
            for f, r in zip(feats, rho):
                rows.append({"project": s.project, "split_id": s.split_id, "model": model, "method": method,
                             "explanation": label(model, method), "feature": f, "rho": r,
                             "sign": np.sign(r) if np.isfinite(r) and r != 0 else np.nan})  # fmt: skip
    return pd.DataFrame(rows)


def direction_stability(directions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (project, explanation): share of features with the same sign in all splits;
    and the list of (project, explanation, feature) whose sign flips."""
    summary, flips = [], []
    for (project, expl), g in directions.groupby(["project", "explanation"], sort=False):
        same = 0
        n = 0
        for f, gf in g.groupby("feature", sort=False):
            signs = gf["sign"].dropna()
            if len(signs) < 2:
                continue
            n += 1
            if signs.nunique() == 1:
                same += 1
            else:
                flips.append({"project": project, "explanation": expl, "feature": f,
                              "signs": " ".join("+" if x > 0 else "−" for x in gf["sign"]),
                              "rho": ", ".join(f"{x:+.2f}" for x in gf["rho"])})  # fmt: skip
        summary.append({"project": project, "explanation": expl, "features": n, "same_sign_share": same / n if n else np.nan})
    return pd.DataFrame(summary), pd.DataFrame(flips)
