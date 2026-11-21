"""Analysis of the step 4 explanations: importance, direction, size share and agreement.

Conventions:

* Explanations are handled in *wide* form: one row per explained instance (split,
  model, method, seed, instance) and one column per metric. Columns are always
  matched to metrics **by name** (pivot on the stored ``feature`` column), never by
  position.
* Importance uses the absolute attribution, normalised per instance to unit L1 norm,
  so that methods with different output scales (probability vs log-odds) and
  magnitudes are comparable. Instances whose attributions are all zero are dropped.
* Aggregation: mean over instances within (split, seed); mean over seeds within a
  split; then the mean over the splits of a project, or over all splits (``ALL``), so
  that every split has equal weight.
* Saliency is stored as the signed gradient. Magnitude-based analyses use |gradient|
  (i.e. Saliency with ``abs=True``); sign-based ones use the sign of the gradient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sdp.eval.preprocess import FEATURES
from sdp.eval.stats import bootstrap_ci

logger = logging.getLogger(__name__)

SIZE_GROUP = ("loc", "wmc", "rfc", "npm")
MODEL_LABEL = {"mlp": "MLP", "lr": "LR", "rf": "RF", "xgb": "XGB"}
MODEL_ORDER = ["mlp", "lr", "rf", "xgb"]
MLP_METHODS = ["DeepLift", "DeepLiftShap", "GradientShap", "IntegratedGradients", "Saliency", "InputXGradient",
               "LIME", "KernelSHAP"]  # fmt: skip
METHOD_ORDER = MLP_METHODS + ["TreeSHAP", "LinearSHAP"]
CROSS_MODEL = [("mlp", "DeepLiftShap"), ("rf", "TreeSHAP"), ("xgb", "TreeSHAP"), ("lr", "LinearSHAP")]
LIME_ACROSS = [(m, "LIME") for m in MODEL_ORDER]
KEYS = ["split_id", "project", "model", "method", "seed", "instance_id"]
META = KEYS + ["class_name", "outcome", "y_true", "y_prob", "loc", "output_scale"]
GROUPS = ("all", "TP", "FP", "FN", "TN", "defective")
KS = (3, 5)


def label(model: str, method: str) -> str:
    return f"{MODEL_LABEL.get(model, model)}-{method}"


def explanation_order(pairs) -> list[tuple[str, str]]:
    """(model, method) pairs in a fixed reading order."""
    return sorted(set(pairs), key=lambda p: (MODEL_ORDER.index(p[0]), METHOD_ORDER.index(p[1])))


# ------------------------------------------------------------------------ loading


@dataclass
class Explanations:
    """Wide explanations: ``meta`` rows align with the rows of ``attr`` and ``raw``."""

    meta: pd.DataFrame
    attr: np.ndarray  # signed attributions (n × 20), columns = FEATURES
    raw: np.ndarray  # raw metric values (n × 20), columns = FEATURES

    def __len__(self) -> int:
        return len(self.meta)

    def subset(self, mask) -> "Explanations":
        mask = np.asarray(mask, dtype=bool)
        return Explanations(self.meta[mask].reset_index(drop=True), self.attr[mask], self.raw[mask])

    def select(self, model: str, method: str) -> "Explanations":
        return self.subset((self.meta["model"] == model).to_numpy() & (self.meta["method"] == method).to_numpy())

    def pairs(self) -> list[tuple[str, str]]:
        return explanation_order(map(tuple, self.meta[["model", "method"]].drop_duplicates().to_numpy()))

    @staticmethod
    def concat(parts: list["Explanations"]) -> "Explanations":
        return Explanations(pd.concat([p.meta for p in parts], ignore_index=True),
                            np.vstack([p.attr for p in parts]), np.vstack([p.raw for p in parts]))  # fmt: skip


def from_long(df: pd.DataFrame) -> Explanations:
    """Pivot long-format attributions to wide form, matching columns by feature name."""
    df = df.copy()
    for c in ("split_id", "project", "model", "method", "feature"):
        df[c] = df[c].astype(str)
    if set(df["feature"]) != set(FEATURES):
        raise ValueError("attributions must cover exactly the 20 metrics")
    attr = df.pivot(index=KEYS, columns="feature", values="attribution").reindex(columns=FEATURES)
    raw = df.pivot(index=KEYS, columns="feature", values="feature_value_raw").reindex(columns=FEATURES)
    meta = df.drop_duplicates(KEYS).set_index(KEYS)[[c for c in META if c not in KEYS]].reindex(attr.index)
    if attr.isna().any().any():
        raise ValueError("incomplete attributions: some instance lacks some feature")
    return Explanations(meta.reset_index(), attr.to_numpy(dtype="float64"), raw.to_numpy(dtype="float64"))


def load_explanations(root: Path, models=None, methods=None, seeds=None) -> Explanations:
    """Load (part of) the attribution store in wide form, one file at a time."""
    root = Path(root)
    files = sorted(
        f
        for f in root.glob("*/*/*.parquet")
        if not f.parent.parent.name.startswith("_")
        and (models is None or f.parent.parent.name in models)
        and (methods is None or f.parent.name in methods)
        and (seeds is None or int(f.stem.rsplit("__seed_", 1)[1]) in set(seeds))
    )
    if not files:
        raise FileNotFoundError(f"no attributions under {root}; run scripts.run_explanations first")
    out = Explanations.concat([from_long(pd.read_parquet(f)) for f in files])
    logger.info("Loaded %d explanations (%d files) from %s.", len(out), len(files), root)
    return out


# ------------------------------------------------------------------- aggregation


def normalised_abs(attr: np.ndarray) -> np.ndarray:
    """|attr| / sum|attr| per row (NaN rows when all attributions are zero)."""
    a = np.abs(np.asarray(attr, dtype="float64"))
    s = a.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, a / s, np.nan)


def aggregate(meta: pd.DataFrame, values: pd.DataFrame, by=("model", "method")) -> pd.DataFrame:
    """Instance values → split means (seeds averaged within split) → project / ALL means.

    Returns one row per (by..., scope, column) with the mean over splits (``value``)
    and the number of splits (``n_splits``); ``scope`` is a project name or ``ALL``.
    """
    by = list(by)
    df = pd.concat([meta[by + ["project", "split_id", "seed"]].reset_index(drop=True),
                    values.reset_index(drop=True)], axis=1)  # fmt: skip
    cols = list(values.columns)
    per_seed = df.groupby(by + ["project", "split_id", "seed"], observed=True)[cols].mean()
    per_split = per_seed.groupby(level=by + ["project", "split_id"], observed=True).mean()
    long = per_split.reset_index().melt(id_vars=by + ["project", "split_id"], var_name="column", value_name="value")
    long = long.dropna(subset=["value"])
    proj = long.groupby(by + ["project", "column"], observed=True)["value"].agg(["mean", "count"]).reset_index()
    proj = proj.rename(columns={"project": "scope"})
    allp = long.groupby(by + ["column"], observed=True)["value"].agg(["mean", "count"]).reset_index().assign(scope="ALL")
    out = pd.concat([allp, proj], ignore_index=True).rename(columns={"mean": "value", "count": "n_splits"})
    return out[by + ["scope", "column", "value", "n_splits"]]


def per_split_values(meta: pd.DataFrame, values: pd.DataFrame, by=("model", "method")) -> pd.DataFrame:
    by = list(by)
    df = pd.concat([meta[by + ["project", "split_id", "seed"]].reset_index(drop=True),
                    values.reset_index(drop=True)], axis=1)  # fmt: skip
    per_seed = df.groupby(by + ["project", "split_id", "seed"], observed=True)[list(values.columns)].mean()
    return per_seed.groupby(level=by + ["project", "split_id"], observed=True).mean().reset_index()


# ------------------------------------------------------------- a) global importance


def group_mask(meta: pd.DataFrame, group: str) -> np.ndarray:
    if group == "all":
        return np.ones(len(meta), dtype=bool)
    if group == "defective":
        return (meta["y_true"] == 1).to_numpy()
    return (meta["outcome"].astype(str) == group).to_numpy()


def global_importance(ex: Explanations, groups=GROUPS) -> pd.DataFrame:
    """Mean normalised |attribution| per feature, with ranks (1 = most important)."""
    norm = pd.DataFrame(normalised_abs(ex.attr), columns=FEATURES)
    parts = []
    for g in groups:
        mask = group_mask(ex.meta, g)
        if not mask.any():
            continue
        agg = aggregate(ex.meta[mask], norm[mask]).rename(columns={"column": "feature", "value": "importance"})
        parts.append(agg.assign(group=g))
    out = pd.concat(parts, ignore_index=True)
    out["rank"] = out.groupby(["model", "method", "group", "scope"])["importance"].rank(ascending=False, method="min")
    out["rank"] = out["rank"].astype(int)
    return out[["model", "method", "group", "scope", "feature", "importance", "rank", "n_splits"]]


def importance_matrix(imp: pd.DataFrame, group: str = "all", scope: str = "ALL") -> pd.DataFrame:
    """Features × model-method matrix of importances (labels from the data)."""
    sub = imp[(imp["group"] == group) & (imp["scope"] == scope)]
    sub = sub.assign(explanation=[label(m, k) for m, k in zip(sub["model"], sub["method"])])
    order = [label(m, k) for m, k in explanation_order(map(tuple, sub[["model", "method"]].to_numpy()))]
    mat = sub.pivot(index="feature", columns="explanation", values="importance")[order]
    return mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]


# ---------------------------------------------------------------------- b) direction


def _spearman_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation per column between x[:, j] and y[:, j] (NaN if constant)."""
    if len(x) < 3:
        return np.full(x.shape[1], np.nan)
    rx, ry = rankdata(x, axis=0), rankdata(y, axis=0)
    rx, ry = rx - rx.mean(axis=0), ry - ry.mean(axis=0)
    den = np.sqrt((rx**2).sum(axis=0) * (ry**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (rx * ry).sum(axis=0) / den, np.nan)


def direction(ex: Explanations, threshold: float = 0.1) -> pd.DataFrame:
    """Per (model, method, feature): Spearman(raw value, signed attribution) and the
    mean signed normalised attribution, aggregated like the importances (ALL and per
    project). ``direction`` is '+', '−' or '0' (|rho| < ``threshold``)."""
    keys = ["model", "method", "project", "split_id", "seed"]
    rows = []
    for key, idx in ex.meta.groupby(keys, observed=True, sort=False).indices.items():
        rho = _spearman_columns(ex.raw[idx], ex.attr[idx])
        rows.append(dict(zip(keys, key)) | dict(zip(FEATURES, rho)))
    rho = pd.DataFrame(rows)
    meta = rho[keys]
    agg_rho = aggregate(meta, rho[FEATURES]).rename(columns={"column": "feature", "value": "spearman"})
    pos = aggregate(meta, (rho[FEATURES] > 0).astype(float).where(rho[FEATURES].notna()))
    pos = pos.rename(columns={"column": "feature", "value": "share_splits_positive"}).drop(columns="n_splits")
    agg_rho = agg_rho.merge(pos, on=["model", "method", "scope", "feature"])
    s = np.abs(ex.attr).sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        signed = pd.DataFrame(np.where(s > 0, ex.attr / s, np.nan), columns=FEATURES)
    mean_signed = aggregate(ex.meta, signed).rename(columns={"column": "feature", "value": "mean_signed_attribution"})
    out = agg_rho.merge(mean_signed[["model", "method", "scope", "feature", "mean_signed_attribution"]],
                        on=["model", "method", "scope", "feature"])  # fmt: skip
    out["direction"] = np.select([out["spearman"] >= threshold, out["spearman"] <= -threshold], ["+", "−"], "0")
    return out


def direction_disagreements(dirs: pd.DataFrame, scope: str = "ALL") -> pd.DataFrame:
    """Features for which some explanations have '+' and others '−' (|rho| ≥ threshold),
    within each model and across all explanations."""
    d = dirs[dirs["scope"] == scope].assign(explanation=lambda t: [label(m, k) for m, k in zip(t["model"], t["method"])])
    rows = []
    for feature, g in d.groupby("feature"):
        for model, gm in list(g.groupby("model")) + [("all models", g)]:
            plus = sorted(gm.loc[gm["direction"] == "+", "explanation"])
            minus = sorted(gm.loc[gm["direction"] == "−", "explanation"])
            if plus and minus:
                rows.append({"feature": feature, "within": MODEL_LABEL.get(model, model), "positive": ", ".join(plus),
                             "negative": ", ".join(minus)})  # fmt: skip
    return pd.DataFrame(rows, columns=["feature", "within", "positive", "negative"])


# --------------------------------------------------------------------- c) size share


def size_share(ex: Explanations, size_group=SIZE_GROUP) -> pd.DataFrame:
    """Share of normalised |attribution| on loc and on the size group, and the
    percentage of instances whose top-1 feature is loc."""
    norm = normalised_abs(ex.attr)
    j_loc = FEATURES.index("loc")
    cols = [FEATURES.index(f) for f in size_group]
    top1 = np.argmax(np.abs(ex.attr), axis=1) == j_loc  # ties: first feature in canonical order
    vals = pd.DataFrame({"loc_share": norm[:, j_loc], "size_group_share": norm[:, cols].sum(axis=1),
                         "loc_top1_pct": 100.0 * top1})  # fmt: skip
    vals.loc[np.isnan(norm).all(axis=1), "loc_top1_pct"] = np.nan
    agg = aggregate(ex.meta, vals)
    return agg.pivot_table(index=["model", "method", "scope"], columns="column", values="value").reset_index()


# --------------------------------------------------------------------- d) agreement


def _positions(absA: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order (most important first; ties broken by canonical feature order) and the
    position of every feature in that order."""
    order = np.argsort(-absA, axis=1, kind="stable")
    pos = np.empty_like(order)
    rows = np.arange(len(absA))[:, None]
    pos[rows, order] = np.arange(absA.shape[1])[None, :]
    return order, pos


def topk_agreement(A: np.ndarray, B: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Krishna et al. (2022) feature, rank, sign and signed-rank agreement@k per row."""
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    _, pa = _positions(np.abs(A))
    _, pb = _positions(np.abs(B))
    in_a, in_b = pa < k, pb < k
    both = in_a & in_b
    same_rank = both & (pa == pb)
    same_sign = np.sign(A) == np.sign(B)
    return {
        f"feature_agreement@{k}": both.sum(axis=1) / k,
        f"rank_agreement@{k}": same_rank.sum(axis=1) / k,
        f"sign_agreement@{k}": (both & same_sign).sum(axis=1) / k,
        f"signed_rank_agreement@{k}": (same_rank & same_sign).sum(axis=1) / k,
    }


def rank_correlations(A: np.ndarray, B: np.ndarray) -> dict[str, np.ndarray]:
    """Spearman and Kendall tau-b between |A| and |B| per row, over all features
    (average ranks for ties; NaN when a row is constant)."""
    a, b = np.abs(np.atleast_2d(A)), np.abs(np.atleast_2d(B))
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra, rb = ra - ra.mean(axis=1, keepdims=True), rb - rb.mean(axis=1, keepdims=True)
    den = np.sqrt((ra**2).sum(axis=1) * (rb**2).sum(axis=1))
    i, j = np.triu_indices(a.shape[1], k=1)
    da, db = np.sign(a[:, i] - a[:, j]), np.sign(b[:, i] - b[:, j])
    n0 = len(i)
    kden = np.sqrt((n0 - (da == 0).sum(axis=1)) * (n0 - (db == 0).sum(axis=1)).astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        spearman = np.where(den > 0, (ra * rb).sum(axis=1) / den, np.nan)
        kendall = np.where(kden > 0, (da * db).sum(axis=1) / kden, np.nan)
    return {"spearman": spearman, "kendall": kendall}


AGREEMENT_METRICS = [f"{m}@{k}" for k in KS for m in ("feature_agreement", "rank_agreement", "sign_agreement",
                                                         "signed_rank_agreement")] + ["spearman", "kendall"]  # fmt: skip


def pair_agreement(ex: Explanations, a: tuple[str, str], b: tuple[str, str]) -> pd.DataFrame:
    """Per-instance agreement between explanations ``a`` and ``b`` (matched on split,
    seed and instance)."""
    on = ["split_id", "project", "seed", "instance_id"]
    ia = ex.meta.index[(ex.meta["model"] == a[0]) & (ex.meta["method"] == a[1])]
    ib = ex.meta.index[(ex.meta["model"] == b[0]) & (ex.meta["method"] == b[1])]
    left = ex.meta.loc[ia, on].assign(_a=ia)
    right = ex.meta.loc[ib, on].assign(_b=ib)
    m = left.merge(right, on=on, how="inner")
    A, B = ex.attr[m["_a"].to_numpy()], ex.attr[m["_b"].to_numpy()]
    vals = {}
    for k in KS:
        vals |= topk_agreement(A, B, k)
    vals |= rank_correlations(A, B)
    out = m[on].reset_index(drop=True).assign(a=label(*a), b=label(*b), **vals)
    return out


def agreement_pairs() -> list[tuple[str, tuple[str, str], tuple[str, str]]]:
    within = [("within MLP", ("mlp", x), ("mlp", y)) for x, y in combinations(MLP_METHODS, 2)]
    cross = [("cross-model SHAP-family", x, y) for x, y in combinations(CROSS_MODEL, 2)]
    lime = [("LIME across models", x, y) for x, y in combinations(LIME_ACROSS, 2)]
    return within + cross + lime


def summarise_agreement(per_instance: pd.DataFrame, n_resamples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Mean and 95% bootstrap CI over instances (seeds averaged per instance first),
    overall (``ALL``) and per project."""
    inst = per_instance.groupby(["family", "a", "b", "project", "split_id", "instance_id"], observed=True)[
        AGREEMENT_METRICS
    ].mean().reset_index()
    rows = []
    for (family, a, b), g in inst.groupby(["family", "a", "b"], sort=False):
        for scope, gs in [("ALL", g)] + list(g.groupby("project")):
            for metric in AGREEMENT_METRICS:
                mean, lo, hi = bootstrap_ci(gs[metric].to_numpy(), n_resamples, 0.95, seed)
                rows.append({"family": family, "a": a, "b": b, "scope": scope, "metric": metric, "mean": mean,
                             "ci_low": lo, "ci_high": hi, "n_instances": int(gs[metric].notna().sum())})  # fmt: skip
    return pd.DataFrame(rows)


def agreement(ex: Explanations, seeds=(0, 1, 2), n_resamples: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-instance agreement for every configured pair and its summary."""
    ex = ex.subset(ex.meta["seed"].isin(list(seeds)).to_numpy())
    available = set(ex.pairs())
    parts = []
    for family, a, b in agreement_pairs():
        if a in available and b in available:
            parts.append(pair_agreement(ex, a, b).assign(family=family))
        else:
            logger.warning("Agreement: skipping %s vs %s (not computed).", label(*a), label(*b))
    per_instance = pd.concat(parts, ignore_index=True)
    return per_instance, summarise_agreement(per_instance, n_resamples)


def agreement_matrix(summary: pd.DataFrame, metric: str, family: str = "within MLP", scope: str = "ALL",
                     order: list[str] | None = None) -> pd.DataFrame:  # fmt: skip
    """Symmetric explanation × explanation matrix of mean agreement (diagonal = 1)."""
    s = summary[(summary["family"] == family) & (summary["metric"] == metric) & (summary["scope"] == scope)]
    names = order or sorted(set(s["a"]) | set(s["b"]))
    mat = pd.DataFrame(np.nan, index=names, columns=names)
    for r in s.itertuples():
        mat.loc[r.a, r.b] = mat.loc[r.b, r.a] = r.mean
    for n in names:
        mat.loc[n, n] = 1.0
    return mat


# ------------------------------------------------------------------- e) sanity checks


def load_checks(root: Path) -> pd.DataFrame:
    files = sorted(Path(root, "_checks").glob("*/*.parquet"))
    parts = [pd.read_parquet(f) for f in files]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def check_summary(checks: pd.DataFrame, tolerance: float = 1e-3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median / 95th percentile / maximum error per (model, method), and the splits whose
    95th percentile exceeds ``tolerance`` (logged as warnings)."""
    q = lambda p: (lambda s: s.quantile(p))  # noqa: E731
    keys = ["model", "method", "check"]
    summ = checks.groupby(keys)["error"].agg(n="size", median="median", p95=q(0.95), max="max", mean="mean").reset_index()
    summ["passes"] = summ["p95"] <= tolerance
    per_split = checks.groupby(keys + ["split_id"])["error"].agg(p95=q(0.95), max="max").reset_index()
    flagged = per_split[per_split["p95"] > tolerance].sort_values(["model", "method", "p95"], ascending=[True, True, False])
    for r in summ[~summ["passes"]].itertuples():
        logger.warning("%s %s %s: 95th percentile %.2e exceeds %.0e (%d split(s) flagged).", r.model, r.method,
                       r.check, r.p95, tolerance, int(((flagged["model"] == r.model) & (flagged["method"] == r.method)).sum()))  # fmt: skip
    return summ, flagged.reset_index(drop=True)
