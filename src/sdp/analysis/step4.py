"""Step 4 reports: global importance, direction, size share, agreement, sanity checks, runtime."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.analysis import explanations as E
from sdp.analysis import figures as F
from sdp.analysis.step3 import machine_details
from sdp.config import Config
from sdp.data.audit import to_markdown

logger = logging.getLogger(__name__)

FIG_DIR = Path("figures") / "step4"
SCALE_NOTE = (
    "**Output scale.** Every explanation explains P(defective) except LR-LinearSHAP, which is exact "
    "on the log-odds scale (exact interventional SHAP on the probability scale is not linear). "
    "LR-LIME explains the probability. Because attributions are L1-normalised per instance before "
    "aggregation, and the comparisons are rank-based, the scale difference does not affect them."
)
AGG_NOTE = (
    "Absolute attributions are normalised per instance to unit L1 norm, averaged over the test "
    "instances of a split (and over the seeds of that split), then over splits; every split has the "
    "same weight. `ALL` = all 24 time-aware splits."
)


def _f(x, digits: int = 3) -> str:
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{digits}f}"


def _write(cfg: Config, name: str, md: list[str], csvs: dict[str, pd.DataFrame]) -> None:
    out = reports_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.md").write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    for suffix, df in csvs.items():
        df.to_csv(out / f"{name}{suffix}.csv", index=False)
    logger.info("Wrote %s", out / f"{name}.md")


def reports_dir(cfg: Config) -> Path:
    return Path(cfg.paths.reports_dir)


def _labels(pairs) -> list[str]:
    return [E.label(*p) for p in pairs]


# ----------------------------------------------------------------- a) importance


def top_table(imp: pd.DataFrame, pairs, k: int = 10, group: str = "all", scope: str = "ALL") -> pd.DataFrame:
    sub = imp[(imp["group"] == group) & (imp["scope"] == scope)]
    cols = {"rank": list(range(1, k + 1))}
    for model, method in pairs:
        s = sub[(sub["model"] == model) & (sub["method"] == method)].sort_values(["importance"], ascending=False)
        cells = [f"{r.feature} ({r.importance:.3f})" for r in s.head(k).itertuples()]
        cols[E.label(model, method)] = cells + [""] * (k - len(cells))
    return pd.DataFrame(cols)


def project_top(imp: pd.DataFrame, pairs, k: int = 5) -> pd.DataFrame:
    sub = imp[(imp["group"] == "all") & (imp["scope"] != "ALL")]
    rows = []
    for project in sorted(sub["scope"].unique()):
        row = {"project": project}
        for model, method in pairs:
            s = sub[(sub["scope"] == project) & (sub["model"] == model) & (sub["method"] == method)]
            row[E.label(model, method)] = ", ".join(s.sort_values("importance", ascending=False).head(k)["feature"])
        rows.append(row)
    return pd.DataFrame(rows)


def project_values(imp: pd.DataFrame, model: str, method: str, k: int = 5) -> pd.DataFrame:
    sub = imp[(imp["group"] == "all") & (imp["model"] == model) & (imp["method"] == method)]
    rows = []
    for scope in sorted(set(sub["scope"]) - {"ALL"}) + ["ALL"]:
        s = sub[sub["scope"] == scope].sort_values("importance", ascending=False).head(k)
        rows.append({"project": scope, "n_splits": int(s["n_splits"].iloc[0]),
                     **{f"#{i + 1}": f"{r.feature} ({r.importance:.3f})" for i, r in enumerate(s.itertuples())}})  # fmt: skip
    return pd.DataFrame(rows)


def outcome_table(imp: pd.DataFrame, model: str, method: str, k: int = 10) -> pd.DataFrame:
    sub = imp[(imp["scope"] == "ALL") & (imp["model"] == model) & (imp["method"] == method)]
    wide = sub.pivot(index="feature", columns="group", values="importance")
    groups = [g for g in E.GROUPS if g in wide.columns]
    wide = wide[groups].sort_values("all", ascending=False)
    ranks = wide.rank(ascending=False, method="min").astype(int)
    out = pd.DataFrame({"feature": wide.index})
    for g in groups:
        out[g] = [f"{v:.3f} ({r})" for v, r in zip(wide[g], ranks[g])]
    for g in ("TP", "FP", "FN", "TN"):
        if g in wide.columns:
            out[f"{g}−all"] = [f"{d:+.3f}" for d in wide[g] - wide["all"]]
    return out.head(k)


def importance_report(cfg: Config, ex: E.Explanations) -> pd.DataFrame:
    imp = E.global_importance(ex)
    pairs = ex.pairs()
    mlp = [p for p in pairs if p[0] == "mlp"]
    other = [p for p in pairs if p[0] != "mlp"]
    n_inst = ex.meta.drop_duplicates(["split_id", "instance_id"]).shape[0]
    md = [
        "# Step 4: global feature importance",
        "",
        f"Experiment `main_time_aware` (24 splits, {n_inst} test instances; all test instances explained). "
        "Seeds: 10 for the gradient-based methods, TreeSHAP and LinearSHAP; 3 (0–2) for LIME and KernelSHAP.",
        "",
        AGG_NOTE,
        "",
        SCALE_NOTE,
        "",
        "Cells: feature (mean normalised |attribution|). The full rankings (all groups, all projects, "
        "with ranks) are in `step4_global_importance.csv`.",
        "",
        "## Top-10 features, ALL splits, all test instances: MLP",
        "",
        to_markdown(top_table(imp, mlp)),
        "",
        "## Top-10 features, ALL splits, all test instances: LR, RF, XGBoost",
        "",
        to_markdown(top_table(imp, other)),
        "",
        "## Top-10 features, ALL splits, actual defective classes only",
        "",
        to_markdown(top_table(imp, [p for p in pairs if p in E.CROSS_MODEL or p[1] == "LIME"], group="defective")),
        "",
        "## MLP-DeepLiftShap: top-5 features per project",
        "",
        to_markdown(project_values(imp, "mlp", "DeepLiftShap")),
        "",
        "## Top-5 features per project: MLP-DeepLiftShap, RF-TreeSHAP and LIME (all models)",
        "",
        to_markdown(project_top(imp, [("mlp", "DeepLiftShap"), ("rf", "TreeSHAP")] + E.LIME_ACROSS)),
        "",
        "## Outcome groups: MLP-DeepLiftShap (ALL)",
        "",
        "Outcome groups at each model's validation-MCC threshold. Cells: importance (rank within "
        "the group); `X−all` = importance in group X minus importance over all instances. "
        "Top-10 features by overall importance.",
        "",
        to_markdown(outcome_table(imp, "mlp", "DeepLiftShap")),
        "",
    ]
    _write(cfg, "step4_global_importance", md, {"": imp})
    mat = E.importance_matrix(imp)
    F.heatmap(mat, "Global importance (mean normalised |attribution|, ALL splits)", "importance",
              reports_dir(cfg) / FIG_DIR / "global_importance_heatmap.png")  # fmt: skip
    return imp


# ------------------------------------------------------------------- b) direction


def direction_report(cfg: Config, ex: E.Explanations, imp: pd.DataFrame) -> pd.DataFrame:
    dirs = E.direction(ex)
    alls = dirs[dirs["scope"] == "ALL"].assign(explanation=lambda t: [E.label(m, k) for m, k in zip(t["model"], t["method"])])
    order = _labels(ex.pairs())
    feat_order = (imp[(imp["scope"] == "ALL") & (imp["group"] == "all")].groupby("feature")["importance"].mean()
                  .sort_values(ascending=False).index)  # fmt: skip
    rho = alls.pivot(index="feature", columns="explanation", values="spearman").reindex(index=feat_order)[order]
    sign = alls.pivot(index="feature", columns="explanation", values="direction").reindex(index=feat_order)[order]
    table = pd.DataFrame({"feature": rho.index})
    for c in order:
        table[c] = [f"{s} {v:+.2f}" if np.isfinite(v) else "" for v, s in zip(rho[c], sign[c])]
    msign = alls.pivot(index="feature", columns="explanation", values="mean_signed_attribution").reindex(index=feat_order)[order]
    ms = pd.DataFrame({"feature": msign.index})
    for c in order:
        ms[c] = ["+" if v > 0 else "−" if v < 0 else "0" for v in msign[c]]
    dis = E.direction_disagreements(dirs)
    sal = [c for c in order if c.endswith("Saliency")]
    md = [
        "# Step 4: direction of feature effects",
        "",
        "For each (model, method, feature): the Spearman correlation between the raw metric value and "
        "the signed attribution over the test instances of a split and seed, averaged over seeds and "
        "then over splits (ALL). `+` / `−` = mean correlation ≥ 0.1 / ≤ −0.1, `0` otherwise. "
        "Rows are ordered by mean importance over all explanations. Saliency uses the signed gradient. "
        "Full results (per project, share of splits with a positive correlation) are in "
        "`step4_direction.csv`.",
        "",
        SCALE_NOTE,
        "",
        "## Spearman(raw value, signed attribution), ALL",
        "",
        to_markdown(table),
        "",
        "## Features on which explanations disagree in direction (ALL)",
        "",
        "A feature is listed when, within a model (or across all explanations), at least one "
        "explanation has a positive direction and another a negative one.",
        "",
        to_markdown(dis),
        "",
        "## Sign of the mean signed (L1-normalised) attribution, ALL",
        "",
        "The sign says whether the feature on average pushes predictions up (+) or down (−) relative "
        "to the reference; it depends on the reference (background) and is not the same as the "
        "direction above.",
        "",
        to_markdown(ms),
        "",
    ]
    if sal:
        md += [f"Note: {', '.join(sal)} is a local gradient, not a difference from a reference; its "
               "sign is reported for completeness.", ""]  # fmt: skip
    _write(cfg, "step4_direction", md, {"": dirs, "_disagreements": dis})
    return dirs


# ------------------------------------------------------------------ c) size share


def size_share_report(cfg: Config, ex: E.Explanations) -> pd.DataFrame:
    ss = E.size_share(ex)
    ss["explanation"] = [E.label(m, k) for m, k in zip(ss["model"], ss["method"])]
    order = _labels(ex.pairs())
    alls = ss[ss["scope"] == "ALL"].set_index("explanation").loc[order]
    overall = pd.DataFrame({"explanation": order, "loc share": [_f(v) for v in alls["loc_share"]],
                            "size-group share": [_f(v) for v in alls["size_group_share"]],
                            "% instances with loc top-1": [_f(v, 1) for v in alls["loc_top1_pct"]]})  # fmt: skip

    def per_project(col: str, digits: int = 3) -> pd.DataFrame:
        p = ss[ss["scope"] != "ALL"].pivot(index="scope", columns="explanation", values=col)[order]
        t = pd.DataFrame({"project": p.index})
        for c in order:
            t[c] = [_f(v, digits) for v in p[c]]
        return t

    md = [
        "# Step 4: share of attribution on size",
        "",
        "Share of the total normalised |attribution| assigned to `loc` alone and to the size group "
        "{loc, wmc, rfc, npm}, and the percentage of test instances whose top-1 feature (largest "
        "|attribution|) is `loc`. Under equal attribution each feature would get 0.05 and the size "
        "group 0.20.",
        "",
        AGG_NOTE,
        "",
        SCALE_NOTE,
        "",
        "## Overall (ALL)",
        "",
        to_markdown(overall),
        "",
        "## loc share per project",
        "",
        to_markdown(per_project("loc_share")),
        "",
        "## Size-group share per project",
        "",
        to_markdown(per_project("size_group_share")),
        "",
        "## % instances with loc as top-1 feature, per project",
        "",
        to_markdown(per_project("loc_top1_pct", 1)),
        "",
    ]
    _write(cfg, "step4_size_share", md, {"": ss})
    bars = alls[["loc_share", "size_group_share"]].rename(columns={"loc_share": "loc",
                                                                   "size_group_share": "size group {loc, wmc, rfc, npm}"})  # fmt: skip
    F.size_share_bars(bars, "Share of normalised |attribution| on size (ALL splits)",
                      reports_dir(cfg) / FIG_DIR / "size_share_bars.png")  # fmt: skip
    return ss


# ------------------------------------------------------------------- d) agreement


def _ci(r) -> str:
    return f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]" if np.isfinite(r["mean"]) else ""


def agreement_report(cfg: Config, ex: E.Explanations) -> pd.DataFrame:
    _, summ = E.agreement(ex, seeds=(0, 1, 2), n_resamples=2000)
    mlp_order = [E.label("mlp", m) for m in E.MLP_METHODS]
    mats = []
    for metric in ("feature_agreement@5", "rank_agreement@5", "sign_agreement@5", "signed_rank_agreement@5",
                   "spearman", "kendall"):  # fmt: skip
        mat = E.agreement_matrix(summ, metric, order=mlp_order)
        short = mat.rename(index=lambda s: s.replace("MLP-", ""), columns=lambda s: s.replace("MLP-", ""))
        t = short.map(_f).reset_index().rename(columns={"index": "MLP"})
        mats.append((metric, t, mat))

    def pair_table(family: str) -> pd.DataFrame:
        s = summ[(summ["family"] == family) & (summ["scope"] == "ALL")]
        rows = []
        for (a, b), g in s.groupby(["a", "b"], sort=False):
            g = g.set_index("metric")
            rows.append({"pair": f"{a} vs {b}", "n": int(g["n_instances"].max()),
                         **{m: _ci(g.loc[m]) for m in E.AGREEMENT_METRICS if m.endswith("@5") or m in ("spearman", "kendall")}})  # fmt: skip
        return pd.DataFrame(rows)

    def pair_table_k3(family: str) -> pd.DataFrame:
        s = summ[(summ["family"] == family) & (summ["scope"] == "ALL")]
        rows = []
        for (a, b), g in s.groupby(["a", "b"], sort=False):
            g = g.set_index("metric")
            rows.append({"pair": f"{a} vs {b}", **{m: _ci(g.loc[m]) for m in E.AGREEMENT_METRICS if m.endswith("@3")}})
        return pd.DataFrame(rows)

    def project_table(family: str, metric: str) -> pd.DataFrame:
        s = summ[(summ["family"] == family) & (summ["metric"] == metric) & (summ["scope"] != "ALL")]
        s = s.assign(pair=s["a"] + " vs " + s["b"])
        p = s.pivot(index="scope", columns="pair", values="mean")
        p = p[list(dict.fromkeys(s["pair"]))]
        return p.map(_f).reset_index().rename(columns={"scope": "project"})

    md = [
        "# Step 4: explainer agreement",
        "",
        "Agreement metrics after Krishna et al. (2022), \"The Disagreement Problem in Explainable Machine "
        "Learning\", computed per test instance and pair of explanations of the same instance:",
        "",
        "* feature agreement@k: share of the top-k features (by |attribution|) common to both;",
        "* rank agreement@k: share of top-k positions holding the same feature in both;",
        "* sign agreement@k: share of common top-k features with the same attribution sign;",
        "* signed-rank agreement@k: same feature at the same position with the same sign;",
        "* Spearman and Kendall (tau-b) rank correlation of |attribution| over all 20 features.",
        "",
        "Ties in the top-k ordering are broken by the canonical feature order; rank correlations use "
        "average ranks. Seeds 0–2 (where all methods exist); each instance's agreement is averaged over "
        "the three seeds, and the mean and 95% percentile bootstrap CI (2000 resamples) are over "
        "instances. Saliency is compared as |gradient| for magnitudes and by the gradient's sign for the "
        "sign-based metrics.",
        "",
        "**Note.** DeepLift and DeepLiftShap share the same propagation rules (DeepLiftShap averages "
        "DeepLift over the background baselines instead of using their mean), so their high agreement is "
        "expected and is not independent evidence. The same holds, to a lesser degree, for "
        "IntegratedGradients vs DeepLift (same baseline) and for InputXGradient vs Saliency (same "
        "gradient).",
        "",
        SCALE_NOTE,
        "",
        "## Within MLP, k = 5 (mean over instances, ALL)",
        "",
    ]
    for metric, t, _ in mats:
        md += [f"### {metric}", "", to_markdown(t), ""]
    md += [
        "## Cross-model: MLP-DeepLiftShap, RF-TreeSHAP, XGB-TreeSHAP, LR-LinearSHAP (ALL)",
        "",
        "Different models explain the same instances; agreement here mixes explainer and model "
        "disagreement. Mean [95% CI].",
        "",
        to_markdown(pair_table("cross-model SHAP-family")),
        "",
        "## LIME across the four models (ALL)",
        "",
        to_markdown(pair_table("LIME across models")),
        "",
        "## Within MLP, all pairs (ALL), k = 5 with 95% CI",
        "",
        to_markdown(pair_table("within MLP")),
        "",
        "## k = 3 (ALL)",
        "",
        to_markdown(pd.concat([pair_table_k3(f) for f in ("within MLP", "cross-model SHAP-family",
                                                           "LIME across models")], ignore_index=True)),  # fmt: skip
        "",
        "## Per project: feature agreement@5",
        "",
        "### Cross-model",
        "",
        to_markdown(project_table("cross-model SHAP-family", "feature_agreement@5")),
        "",
        "### LIME across models",
        "",
        to_markdown(project_table("LIME across models", "feature_agreement@5")),
        "",
        "Per-project values for every pair and metric (with CIs) are in `step4_agreement.csv`.",
        "",
    ]
    _write(cfg, "step4_agreement", md, {"": summ})
    fa = mats[0][2].rename(index=lambda s: s.replace("MLP-", ""), columns=lambda s: s.replace("MLP-", ""))
    F.heatmap(fa, "MLP explainer agreement: feature agreement@5 (ALL)", "feature agreement@5",
              reports_dir(cfg) / FIG_DIR / "mlp_agreement_heatmap.png", vmin=0.0, vmax=1.0)  # fmt: skip
    return summ


# -------------------------------------------------------------- e) sanity checks


def sanity_report(cfg: Config, root: Path, tolerance: float, run_info: dict) -> pd.DataFrame:
    checks = E.load_checks(root)
    summ, flagged = E.check_summary(checks, tolerance)
    s = summ.assign(explanation=[E.label(m, k) for m, k in zip(summ["model"], summ["method"])])
    table = pd.DataFrame({"explanation": s["explanation"], "check": s["check"], "instances": s["n"],
                          "median": [f"{v:.2e}" for v in s["median"]], "95th pct": [f"{v:.2e}" for v in s["p95"]],
                          "max": [f"{v:.2e}" for v in s["max"]],
                          "status": ["pass" if p else f"WARNING: > {tolerance:.0e}" for p in s["passes"]]})  # fmt: skip
    per_split = (checks.groupby(["model", "method", "split_id", "seed"])["error"].quantile(0.95)
                 .rename("p95").reset_index())  # fmt: skip
    per_split = per_split[per_split["p95"] > tolerance]
    n_flag = per_split.groupby(["model", "method", "split_id"]).agg(seeds_flagged=("seed", "nunique"),
                                                                     worst_p95=("p95", "max")).reset_index()  # fmt: skip
    n_flag["explanation"] = [E.label(m, k) for m, k in zip(n_flag["model"], n_flag["method"])]
    n_flag = n_flag.sort_values("worst_p95", ascending=False)
    flag_table = pd.DataFrame({"explanation": n_flag["explanation"], "split": n_flag["split_id"],
                               "seeds flagged (of 10)": n_flag["seeds_flagged"],
                               "worst seed 95th pct": [f"{v:.2e}" for v in n_flag["worst_p95"]]})  # fmt: skip
    diff = run_info.get("max_prob_diff_vs_step3")
    md = [
        "# Step 4: completeness and additivity checks",
        "",
        "Per instance: |sum(attributions) − (f(x) − f(reference))|.",
        "",
        "* DeepLift and IntegratedGradients: reference = the background mean (single baseline);",
        "* DeepLiftShap: reference output = the mean output over the background set;",
        "* TreeSHAP: additivity, sum(SHAP) + expected value vs the predicted probability;",
        "* LinearSHAP (LR): additivity on the log-odds scale (decision function).",
        "",
        f"Tolerance: the 95th percentile must not exceed {tolerance:.0e}. Captum methods run on a float64 "
        "copy of the MLP.",
        "",
        "## Summary over all splits and seeds",
        "",
        to_markdown(table),
        "",
        "## Splits whose 95th percentile exceeds the tolerance",
        "",
        "Per split, pooled over seeds; the per-seed table below gives how many of the 10 seeds exceed it.",
        "",
        to_markdown(flagged.assign(explanation=[E.label(m, k) for m, k in zip(flagged["model"], flagged["method"])])
                    [["explanation", "split_id", "p95", "max"]].assign(p95=lambda t: t["p95"].map("{:.2e}".format),
                                                                         max=lambda t: t["max"].map("{:.2e}".format))
                    .rename(columns={"split_id": "split", "p95": "95th pct"})) if len(flagged) else "_(none)_",
        "",
        "### Per seed",
        "",
        to_markdown(flag_table) if len(flag_table) else "_(none)_",
        "",
        "## Notes",
        "",
        "* IntegratedGradients approximates a path integral with 64 Gauss-Legendre nodes; the MLP's "
        "ReLU kinks and the sigmoid make the integrand non-smooth, so a residual error is expected and "
        "shrinks with more steps. It is reported, not corrected.",
        "* TreeSHAP for the random forests: shap's interventional algorithm stores split thresholds as "
        "float32 while scikit-learn compares float32 inputs with float64 thresholds; where rounding a "
        "threshold crosses a data value shap took the wrong branch (additivity errors up to 0.02 in the "
        "smoke test). The forest given to shap is a copy with thresholds rounded down to float32, which "
        "partitions float32 inputs exactly as the original model; predictions are unchanged.",
        f"* Reproduction: every loaded model reproduces its stored step 3 test predictions; maximum "
        f"absolute difference {diff if diff is not None else 'n/a'} over "
        f"{run_info.get('n_checked_against_step3', 'n/a')} (split, model, seed) units.",
        "",
    ]
    _write(cfg, "step4_sanity_checks", md, {"": summ, "_flagged": flagged})
    return summ


# ------------------------------------------------------------------- runtime


def runtime_report(cfg: Config) -> pd.DataFrame:
    base = Path(cfg.paths.results_dir) / "explanations"
    full = pd.read_csv(base / "timings_full.csv")
    smoke = pd.read_csv(base / "timings_smoke.csv")
    info = json.loads((base / "run_info_full.json").read_text())
    est = json.loads((base / "runtime_estimate.json").read_text()) if (base / "runtime_estimate.json").exists() else {}
    per = (full.groupby(["model", "method"]).agg(units=("split_id", "size"), instances=("n_instances", "sum"),
                                                cpu_s=("time_s", "sum")).reset_index())  # fmt: skip
    per["ms_per_instance"] = (1000 * per["cpu_s"] / per["instances"]).round(2)
    per["cpu_h"] = (per["cpu_s"] / 3600).round(3)
    per["explanation"] = [E.label(m, k) for m, k in zip(per["model"], per["method"])]
    order = _labels(E.explanation_order(map(tuple, per[["model", "method"]].to_numpy())))
    per = per.set_index("explanation").loc[order].reset_index()
    sm = smoke.assign(explanation=[E.label(m, k) for m, k in zip(smoke["model"], smoke["method"])])
    sm_table = pd.DataFrame({"explanation": sm["explanation"], "instances": sm["n_instances"],
                             "time_s": sm["time_s"].round(2),
                             "ms_per_instance": (1000 * sm["time_s"] / sm["n_instances"]).round(2)})  # fmt: skip
    e = est.get("estimate", {})
    params = info.get("params", {})
    md = [
        "# Step 4 runtime",
        "",
        "## Machine",
        "",
        to_markdown(pd.DataFrame(list(machine_details().items()) + [("shap", _v("shap")), ("lime", _v("lime")),
                                                                     ("pyarrow", _v("pyarrow"))],
                                 columns=["item", "value"])),  # fmt: skip
        "",
        "## Smoke test and estimate",
        "",
        f"Smoke test: split `{est.get('smoke_split', 'n/a')}`, seed 0, all methods "
        f"({est.get('smoke', {}).get('wall_time_s', float('nan')):.1f} s wall).",
        "",
        to_markdown(sm_table),
        "",
        f"Estimated full run: {e.get('cpu_hours', float('nan')):.2f} CPU h, ~{e.get('wall_hours', float('nan')):.2f} h "
        f"wall with n_jobs={est.get('n_jobs', 'n/a')}. The estimate is below 8 h, so LIME (num_samples="
        f"{params.get('LIME', {}).get('num_samples')}) and KernelSHAP (nsamples="
        f"{params.get('KernelSHAP', {}).get('nsamples')}) were run at full size; no reduction was applied."
        if not info.get("reduced") else "The estimate exceeded 8 h; LIME and KernelSHAP were reduced.",
        "",
        "## Full run",
        "",
        f"{info.get('jobs')} jobs ((split, model, seed) units), n_jobs={info.get('n_jobs')}, "
        f"{info.get('jobs_from_cache')} reused from the smoke test. Wall time {info.get('wall_time_s', 0) / 60:.1f} min; "
        f"CPU time {full['time_s'].sum() / 3600:.2f} h in the explainers.",
        "",
        to_markdown(per[["explanation", "units", "instances", "cpu_h", "ms_per_instance"]]
                    .assign(cpu_h=lambda t: t["cpu_h"].map("{:.3f}".format))),  # fmt: skip
        "",
    ]
    _write(cfg, "step4_runtime", md, {"": per})
    return per


def _v(lib: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(lib)
    except metadata.PackageNotFoundError:
        return "n/a"
