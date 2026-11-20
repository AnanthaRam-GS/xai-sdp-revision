"""Aggregation of long-format experiment results into summary and comparison tables.

Conventions
-----------
* Summaries (mean and percentile-bootstrap 95% CI) are computed over all rows of a
  group, i.e. over splits x seeds.
* Statistical comparisons (Scott-Knott ESD, Wilcoxon signed-rank with Holm correction,
  Cliff's delta) use the *per-split mean across seeds*, so that each split counts once.
  Pairs are matched on the split.
* For metrics where lower is better (NECM, Brier, ECE, IFA), the Scott-Knott ESD ranks
  are computed on the negated values so that rank 1 is always best; Cliff's delta is
  reported on the original values (positive = first model has larger values).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sdp.eval.stats import bootstrap_ci, cliffs_delta, cliffs_magnitude, holm, scott_knott_esd


@dataclass(frozen=True)
class MetricSpec:
    metric: str
    mode: str  # threshold_mode: none | fixed_0.5 | val_mcc
    label: str
    higher_is_better: bool = True


M = MetricSpec
MAIN_METRICS = [
    M("roc_auc", "none", "AUC"),
    M("pr_auc", "none", "PR-AUC"),
    M("mcc", "fixed_0.5", "MCC@0.5"),
    M("mcc", "val_mcc", "MCC@tuned"),
    M("f1", "fixed_0.5", "F1@0.5"),
    M("f1", "val_mcc", "F1@tuned"),
    M("recall", "fixed_0.5", "Recall@0.5"),
    M("recall", "val_mcc", "Recall@tuned"),
    M("balanced_accuracy", "fixed_0.5", "BalAcc@0.5"),
    M("balanced_accuracy", "val_mcc", "BalAcc@tuned"),
    M("brier", "none", "Brier", False),
    M("ece", "none", "ECE", False),
    M("popt", "none", "Popt"),
    M("recall_at_20pct_loc", "none", "Recall@20%LOC"),
    M("ifa", "none", "IFA", False),
    M("necm_15", "fixed_0.5", "NECM15@0.5", False),
    M("necm_15", "val_mcc", "NECM15@tuned", False),
]
RANK_METRICS = [
    M("roc_auc", "none", "AUC"),
    M("mcc", "val_mcc", "MCC@tuned"),
    M("popt", "none", "Popt"),
    M("recall_at_20pct_loc", "none", "Recall@20%LOC"),
    M("necm_15", "val_mcc", "NECM15@tuned", False),
]
PROJECT_METRICS = [M("roc_auc", "none", "AUC"), M("mcc", "val_mcc", "MCC@tuned"), M("popt", "none", "Popt")]
ABLATION_METRICS = [
    M("roc_auc", "none", "AUC"),
    M("mcc", "val_mcc", "MCC@tuned"),
    M("brier", "none", "Brier", False),
    M("popt", "none", "Popt"),
]


def select(df: pd.DataFrame, spec: MetricSpec) -> pd.DataFrame:
    return df[(df["metric"] == spec.metric) & (df["threshold_mode"] == spec.mode)]


def fmt_ci(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return "n/a"
    if pd.isna(lo):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# ----------------------------------------------------------------- summaries


def summary_long(
    df: pd.DataFrame, specs: list[MetricSpec], by: list[str], n_resamples=2000, confidence=0.95, seed=42
) -> pd.DataFrame:
    """Mean and bootstrap CI over all rows for each group in ``by`` and each metric."""
    rows = []
    for spec in specs:
        sub = select(df, spec)
        for key, g in sub.groupby(by, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            mean, lo, hi = bootstrap_ci(g["value"], n_resamples, confidence, seed)
            rows.append(dict(zip(by, key)) | {"metric": spec.label, "mean": mean, "ci_low": lo, "ci_high": hi,
                                              "n": int(g["value"].notna().sum())})  # fmt: skip
    return pd.DataFrame(rows)


def summary_wide(long: pd.DataFrame, column: str, order: list[str] | None = None, digits: int = 3) -> pd.DataFrame:
    """Metrics as rows, ``column`` values (e.g. models) as columns, cells 'mean [lo, hi]'."""
    cells = long.assign(cell=[fmt_ci(m, lo, hi, 1 if met == "IFA" else digits)
                              for m, lo, hi, met in zip(long["mean"], long["ci_low"], long["ci_high"], long["metric"])])  # fmt: skip
    wide = cells.pivot_table(index="metric", columns=column, values="cell", aggfunc="first", sort=False)
    if order:
        wide = wide[[c for c in order if c in wide.columns]]
    metric_order = list(dict.fromkeys(long["metric"]))
    return wide.reindex(metric_order).reset_index()


def project_table(df: pd.DataFrame, spec: MetricSpec, column: str = "model", order=None) -> pd.DataFrame:
    """Mean per project (rows) and model (columns), with an ALL row."""
    sub = select(df, spec)
    t = sub.pivot_table(index="project", columns=column, values="value", aggfunc="mean", sort=False)
    t.loc["ALL"] = sub.groupby(column, sort=False)["value"].mean()
    if order:
        t = t[[c for c in order if c in t.columns]]
    return t.map(lambda v: "n/a" if pd.isna(v) else f"{v:.3f}").reset_index()


# ---------------------------------------------------------------- comparisons


def per_split(df: pd.DataFrame, spec: MetricSpec, column: str = "model", block: str = "split_id") -> pd.DataFrame:
    """Per-split mean across seeds: blocks as rows, ``column`` values as columns."""
    return select(df, spec).pivot_table(index=block, columns=column, values="value", aggfunc="mean")


def skesd_ranks(df: pd.DataFrame, specs: list[MetricSpec], column: str = "model", order=None) -> pd.DataFrame:
    rows = []
    for spec in specs:
        wide = per_split(df, spec, column).dropna(how="all")
        data = wide if spec.higher_is_better else -wide
        ranks = scott_knott_esd(data).set_index("treatment")["rank"]
        rows.append({"metric": spec.label} | {c: int(ranks[c]) for c in (order or ranks.index) if c in ranks})
    return pd.DataFrame(rows)


def paired_comparison(a: pd.Series, b: pd.Series) -> dict:
    """Wilcoxon signed-rank (two-sided) and Cliff's delta for two aligned series."""
    from scipy.stats import wilcoxon

    pair = pd.concat([a, b], axis=1).dropna()
    x, y = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
    n = len(pair)
    if n == 0:
        p = np.nan
    elif np.allclose(x, y):
        p = 1.0
    else:
        p = float(wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)
    d = cliffs_delta(x, y) if n else np.nan
    return {"n": n, "mean_a": float(np.mean(x)) if n else np.nan, "mean_b": float(np.mean(y)) if n else np.nan,
            "p_value": p, "cliffs_delta": d, "magnitude": cliffs_magnitude(d)}  # fmt: skip


def reference_comparisons(
    df: pd.DataFrame, specs: list[MetricSpec], reference: str = "mlp", column: str = "model", order=None
) -> pd.DataFrame:
    """``reference`` vs every other value of ``column``; Holm correction within each metric."""
    rows = []
    for spec in specs:
        wide = per_split(df, spec, column)
        others = [c for c in (order or wide.columns) if c in wide.columns and c != reference]
        block = [{"metric": spec.label, "a": reference, "b": o} | paired_comparison(wide[reference], wide[o])
                 for o in others]  # fmt: skip
        adj = holm([r["p_value"] for r in block])
        for r, p in zip(block, adj):
            r["p_holm"] = p
        rows += block
    return _order_cols(pd.DataFrame(rows))


def all_pairs_comparisons(
    df: pd.DataFrame, specs: list[MetricSpec], column: str, within: str | None = None, block: str = "split_id"
) -> pd.DataFrame:
    """All pairs of ``column`` values (optionally within each value of ``within``);
    Holm correction within each (within, metric) family."""
    rows = []
    groups = df.groupby(within, sort=False) if within else [(None, df)]
    for key, g in groups:
        for spec in specs:
            wide = per_split(g, spec, column, block)
            cols = list(wide.columns)
            fam = []
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    fam.append(({within: key} if within else {}) | {"metric": spec.label, "a": cols[i], "b": cols[j]}
                               | paired_comparison(wide[cols[i]], wide[cols[j]]))  # fmt: skip
            adj = holm([r["p_value"] for r in fam])
            for r, p in zip(fam, adj):
                r["p_holm"] = p
            rows += fam
    return _order_cols(pd.DataFrame(rows))


def _order_cols(t: pd.DataFrame) -> pd.DataFrame:
    if t.empty:
        return t
    first = [c for c in t.columns if c not in ("n", "mean_a", "mean_b", "p_value", "p_holm", "cliffs_delta", "magnitude")]
    return t[first + ["n", "mean_a", "mean_b", "p_value", "p_holm", "cliffs_delta", "magnitude"]]


def format_comparisons(t: pd.DataFrame) -> pd.DataFrame:
    out = t.copy()
    for c in ("mean_a", "mean_b", "cliffs_delta"):
        out[c] = out[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "n/a")
    for c in ("p_value", "p_holm"):
        out[c] = out[c].map(lambda v: "n/a" if pd.isna(v) else ("<0.001" if v < 0.001 else f"{v:.3f}"))
    return out
