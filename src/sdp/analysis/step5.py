"""Step 5 reports: LIME stability, background sensitivity, preprocessing, collinearity, runtime."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.analysis import explanations as E
from sdp.analysis import figures as F
from sdp.analysis.step3 import machine_details
from sdp.analysis.step4 import LIME_NOTE, _f, _write
from sdp.analysis.tables import _order_cols, format_comparisons, paired_comparison
from sdp.config import Config
from sdp.data.audit import to_markdown
from sdp.eval.preprocess import FEATURES
from sdp.eval.stats import holm
from sdp.robustness import background_study as B
from sdp.robustness import collinearity as C
from sdp.robustness import lime_study as L
from sdp.robustness import preprocessing_study as P
from sdp.robustness.common import output_root

logger = logging.getLogger(__name__)

FIG = Path("figures") / "step5"
MODEL_ORDER = ["mlp", "lr", "rf", "xgb"]


def _fig(cfg: Config, name: str) -> Path:
    return Path(cfg.paths.reports_dir) / FIG / name


def _fmt_frame(t: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = t.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: _f(v, digits))
    return out


# ------------------------------------------------------------------------ LIME


def lime_report(cfg: Config, rs: dict, lime_cv: pd.DataFrame, sample: pd.DataFrame, info: dict) -> None:
    runs = L.load_lime_runs(output_root(cfg, rs) / "lime")
    rr = L.run_to_run(runs)
    gv = L.grid_vs_main(runs)
    s_rr = L.summarise_run_to_run(rr)
    s_rr["model"] = s_rr["model"].map(E.MODEL_LABEL)
    s_rr = s_rr.set_index("model").loc[[E.MODEL_LABEL[m] for m in MODEL_ORDER]].reset_index()
    s_grid = L.summarise_grid(gv)
    names = [n for n, _ in L.grid_settings(rs["lime"])]
    wide = pd.DataFrame({"setting": names})
    for m in MODEL_ORDER:
        g = s_grid[s_grid["model"] == m].set_index("setting").reindex(names)
        wide[f"{E.MODEL_LABEL[m]} ρ"] = g["spearman"].map(_f).to_numpy()
        wide[f"{E.MODEL_LABEL[m]} FA@5"] = g["fa5"].map(_f).to_numpy()
        wide[f"{E.MODEL_LABEL[m]} R²"] = g["r2"].map(_f).to_numpy()
    by_outcome = rr.groupby(["model", "outcome"])["spearman"].mean().unstack().reindex(MODEL_ORDER)
    by_outcome.index = [E.MODEL_LABEL[m] for m in by_outcome.index]
    cv = lime_cv.pivot(index="model", columns="method", values="median_cv").reindex(MODEL_ORDER)
    cv_t = pd.DataFrame({"model": [E.MODEL_LABEL[m] for m in cv.index],
                         "v1 mean sampling (LIME_mean_sampling)": cv.get("LIME_mean_sampling", np.nan).map(_f),
                         "v2 instance sampling (LIME)": cv.get("LIME", np.nan).map(_f)})  # fmt: skip
    counts = sample.groupby("reference_outcome").size().reindex(["TP", "FP", "FN", "TN"]).fillna(0).astype(int)
    md = [
        "# Step 5: LIME stability and settings sensitivity",
        "",
        f"Seed-0 models of `main_time_aware`; per split up to {rs['lime']['instances_per_group']} test instances per "
        f"outcome group (groups of the seed-0 MLP; the same instances for all four models): {len(sample)} instances "
        f"({', '.join(f'{k} {v}' for k, v in counts.items())}). "
        + (f"Reduced to {info['reductions']['lime_instances_per_split']} instances per split. "
           if info.get("reductions", {}).get("lime_instances_per_split") else "")
        + "Main setting: `discretize_continuous=False`, `sample_around_instance=True`, 5000 samples, default "
        "kernel width (0.75 × √20), 20 features. The sample is in `results/robustness/lime_instance_sample.csv`; "
        "per-instance values are in `results/robustness/lime_*_instances.csv`.",
        "",
        LIME_NOTE,
        "",
        "## Run-to-run variability of the main setting (10 runs, random_state 0–9)",
        "",
        "Per instance: mean pairwise Spearman (|weights|, 20 features) and feature agreement@5 over the 45 run "
        "pairs; coefficient of variation std/|mean| of each feature's weight across runs (mean over features "
        "with a non-zero mean; median shown too because weights near 0 inflate it); whether the top-1 feature "
        "differs between any two runs; mean surrogate R². Averaged over instances.",
        "",
        to_markdown(_fmt_frame(s_rr.rename(columns={"spearman": "Spearman", "fa5": "FA@5", "cv_mean": "mean CV",
                                                    "cv_median": "median CV", "top1_change_pct": "% top-1 changes",
                                                    "r2": "R²"}))),  # fmt: skip
        "",
        "Run-to-run Spearman by outcome group:",
        "",
        to_markdown(_fmt_frame(by_outcome.reset_index().rename(columns={"index": "model"}))),
        "",
        "## Settings grid vs the main setting (random_state 0)",
        "",
        "One factor at a time; ρ = Spearman with the main setting's run 0, FA@5 = feature agreement@5, "
        "R² = local surrogate R² (`explanation.score`, a local-fidelity indicator). `main` shows the main "
        "setting's own R².",
        "",
        to_markdown(wide),
        "",
        "## Instance dependence (coefficient of variation across instances)",
        "",
        "Median over splits of the median over features of std/mean|w| across all test instances of a split "
        "(seed 0, full step 4 store). Higher = explanations differ more between instances.",
        "",
        to_markdown(cv_t),
        "",
    ]
    _write(cfg, "step5_lime_stability", md, {"": pd.concat([s_rr.assign(table="run_to_run"),
                                                            s_grid.assign(table="grid")], ignore_index=True),
                                             "_instance_cv": lime_cv})  # fmt: skip
    rr.to_csv(cfg.paths.results_dir / "robustness" / "lime_run_to_run_instances.csv", index=False)
    gv.to_csv(cfg.paths.results_dir / "robustness" / "lime_grid_instances.csv", index=False)
    data = {}
    for m in MODEL_ORDER:
        panel = {"run-to-run (main)": rr.loc[rr["model"] == m, "spearman"].to_numpy()}
        for n in names[1:]:
            panel[n] = gv.loc[(gv["model"] == m) & (gv["setting"] == n), "spearman"].to_numpy()
        data[E.MODEL_LABEL[m]] = panel
    F.box_panels(data, "LIME: Spearman between explanations, run-to-run and settings vs main", "Spearman ρ",
                 _fig(cfg, "lime_stability_spearman.png"))  # fmt: skip


# ---------------------------------------------------------------- background


def background_report(cfg: Config, rs: dict, info: dict) -> None:
    ex = B.load_background_runs(output_root(cfg, rs) / "background")
    ex.meta["method"] = ex.meta["method"].astype(str)
    agr = B.agreement_with_main(ex)
    summ = B.summarise(agr)
    noise = B.draw_noise(ex)
    nsum = B.summarise(noise, by=("model", "method"))
    summ["explanation"] = [E.label(m, k) for m, k in zip(summ["model"], summ["method"])]
    nsum["explanation"] = [E.label(m, k) for m, k in zip(nsum["model"], nsum["method"])]
    expl = [E.label(m, k) for m, ks in rs["background"]["models"].items() for k in ks]
    order = [v for v in B.variants(rs["background"]) if v != B.MAIN]

    def wide(metric: str) -> pd.DataFrame:
        t = summ.pivot(index="variant", columns="explanation", values=metric).reindex(order)[expl]
        return t.map(_f).reset_index()

    imp = E.global_importance(ex, groups=("all",), by=("model", "method", "variant"))
    rows = []
    for (m, k), g in imp.groupby(["model", "method"]):
        main = g[g["variant"] == B.MAIN]
        for v in order:
            gv = g[g["variant"] == v]
            changed_set, changed_order, scopes = 0, 0, 0
            for scope, gm in main.groupby("scope"):
                a = list(gm.sort_values("importance", ascending=False)["feature"].head(5))
                b = list(gv[gv["scope"] == scope].sort_values("importance", ascending=False)["feature"].head(5))
                if not b:
                    continue
                scopes += 1
                changed_set += set(a) != set(b)
                changed_order += a != b
            all_main = list(main[main["scope"] == "ALL"].sort_values("importance", ascending=False)["feature"].head(5))
            all_v = list(gv[gv["scope"] == "ALL"].sort_values("importance", ascending=False)["feature"].head(5))
            rows.append({"explanation": E.label(m, k), "variant": v, "scopes": scopes,
                         "top-5 set changed": changed_set, "top-5 order changed": changed_order,
                         "ALL top-5 (main)": ", ".join(all_main),
                         "ALL top-5 (variant)": ", ".join(all_v) if all_v != all_main else "same"})  # fmt: skip
    top_changes = pd.DataFrame(rows)
    ss = E.size_share(ex, by=("model", "method", "variant"))
    ss = ss[ss["scope"] == "ALL"].assign(explanation=lambda t: [E.label(m, k) for m, k in zip(t["model"], t["method"])])
    ss_t = ss.pivot(index="variant", columns="explanation", values="loc_share").reindex([B.MAIN] + order)[expl]
    ssg_t = ss.pivot(index="variant", columns="explanation", values="size_group_share").reindex([B.MAIN] + order)[expl]
    n_splits = ex.meta["split_id"].nunique()
    red = info.get("reductions", {}).get("background_projects")
    md = [
        "# Step 5: background-set sensitivity",
        "",
        f"Seed-0 models, all test instances of {n_splits} splits"
        + (f" (reduced to projects {', '.join(red)})" if red else " (all 24)") + ". Methods: MLP-DeepLiftShap, "
        "MLP-GradientShap, RF-TreeSHAP, XGB-TreeSHAP (interventional). Main background: 100 fitting rows, "
        "stratified by label, draw seed 0 (as in step 4). Each variant is compared with the main background "
        "per instance; values are means over instances within a split, then over splits.",
        "",
        "* `size=n`: stratified sample of n rows (`all` = every fitting row);",
        "* `scheme=random`: simple random sample of 100; `scheme=kmeans`: shap.kmeans with 100 weighted centres "
        "(each centre repeated by its cluster size); `scheme=defective` / `scheme=clean`: 100 rows of one class;",
        "* `draw=d`: stratified sample of 100 with draw seed d (draw 0 is the main background).",
        "",
        "GradientShap is itself stochastic (50 random baselines and path points per instance). With the same "
        "background and only a different random seed, its per-instance Spearman is about 0.83–0.88 (checked on "
        "the three ant splits), so its agreement of about 0.86 with the main background is its own Monte-Carlo "
        "noise, not an effect of the background.",
        "",
        "## Spearman with the main background",
        "",
        to_markdown(wide("spearman")),
        "",
        "## Feature agreement@5 with the main background",
        "",
        to_markdown(wide("fa5")),
        "",
        "## Sign agreement@5 with the main background",
        "",
        to_markdown(wide("sa5")),
        "",
        "## Mean absolute difference of L1-normalised attributions",
        "",
        to_markdown(wide("mad")),
        "",
        "## Noise floor: 5 independent stratified draws of size 100",
        "",
        "Mean pairwise agreement between the five draws (per instance, then over splits) and the mean per-feature "
        "variance of the L1-normalised attributions across draws. Differences from the main background smaller "
        "than this are within background-sampling noise.",
        "",
        to_markdown(_fmt_frame(nsum[["explanation", "spearman", "fa5", "sa5", "mad", "variance"]].rename(
            columns={"spearman": "Spearman", "fa5": "FA@5", "sa5": "SA@5", "mad": "MAD"}), 4)),
        "",
        "## Global top-5 changes",
        "",
        "Number of scopes (projects and ALL) whose global top-5 set or order differs from the main background, "
        "and the ALL top-5 when it differs.",
        "",
        to_markdown(top_changes),
        "",
        "## loc share (ALL)",
        "",
        to_markdown(ss_t.map(_f).reset_index()),
        "",
        "## Size-group share (ALL)",
        "",
        to_markdown(ssg_t.map(_f).reset_index()),
        "",
    ]
    _write(cfg, "step5_background_sensitivity", md, {"": summ, "_noise_floor": nsum, "_top5_changes": top_changes,
                                                      "_size_share": ss})  # fmt: skip
    agr.to_parquet(cfg.paths.results_dir / "robustness" / "background_agreement_instances.parquet", index=False)
    sizes = [f"size={s}" for s in rs["background"]["sizes"]]
    line = summ[summ["variant"].isin(sizes)].pivot(index="variant", columns="explanation", values="spearman")
    draws = summ[summ["variant"].str.startswith("draw=")].groupby("explanation")["spearman"].mean()
    line.loc[B.MAIN] = draws  # 100: independent stratified draws (the main background itself would be 1)
    line = line.reindex(sizes)[expl]
    line.index = [("100 (other draws)" if v == B.MAIN else v.split("=")[1]) for v in line.index]
    F.line_chart(line, "Agreement with the main background by background size", "background size",
                 "mean Spearman ρ with the main background", _fig(cfg, "background_size_agreement.png"))  # fmt: skip


# -------------------------------------------------------------- preprocessing


def preprocessing_report(cfg: Config, es, rs: dict) -> None:
    ex = P.load_preprocessing_explanations(cfg, es, rs)
    by = list(P.BY)
    imp = E.global_importance(ex, groups=("all",), by=by)
    ss = E.size_share(ex, by=by)
    per_split, tests = P.comparisons(ex)
    dirs = P.direction_table(ex)
    order = P.setting_order(map(tuple, ex.meta[by].drop_duplicates().to_numpy()))
    labels = [P.label(*o) for o in order]

    top_rows = []
    for o, lab in zip(order, labels):
        g = imp[(imp["model"] == o[0]) & (imp["method"] == o[1]) & (imp["preprocessing"] == o[2]) & (imp["scope"] == "ALL")]
        top_rows.append({"explanation": lab, "top-5 (ALL)": ", ".join(f"{r.feature} ({r.importance:.3f})" for r in
                                                                      g.sort_values("importance", ascending=False).head(5).itertuples())})  # fmt: skip
    proj_tables = []
    for o, lab in zip(order, labels):
        g = imp[(imp["model"] == o[0]) & (imp["method"] == o[1]) & (imp["preprocessing"] == o[2]) & (imp["scope"] != "ALL")]
        proj_tables.append((lab, {p: ", ".join(gp.sort_values("importance", ascending=False).head(5)["feature"])
                                  for p, gp in g.groupby("scope")}))  # fmt: skip
    projects = sorted(proj_tables[0][1])
    proj_df = pd.DataFrame({"project": projects} | {lab: [t.get(p, "") for p in projects] for lab, t in proj_tables})
    s_all = ss[ss["scope"] == "ALL"].copy()
    s_all["explanation"] = [P.label(m, k, p) for m, k, p in zip(s_all["model"], s_all["method"], s_all["preprocessing"])]
    s_all = s_all.set_index("explanation").loc[labels]
    ss_t = pd.DataFrame({"explanation": labels, "loc share": s_all["loc_share"].map(_f).to_numpy(),
                         "size-group share": s_all["size_group_share"].map(_f).to_numpy(),
                         "% loc top-1": s_all["loc_top1_pct"].map(lambda v: _f(v, 1)).to_numpy()})  # fmt: skip
    ss_proj = ss[ss["scope"] != "ALL"].copy()
    ss_proj["explanation"] = [P.label(m, k, p) for m, k, p in zip(ss_proj["model"], ss_proj["method"], ss_proj["preprocessing"])]
    loc_proj = ss_proj.pivot(index="scope", columns="explanation", values="loc_share")[labels].map(_f).reset_index().rename(columns={"scope": "project"})
    dir_t = dirs.pivot(index="explanation", columns="feature", values="spearman").loc[labels][P.DIRECTION_FEATURES]
    dir_s = dirs.pivot(index="explanation", columns="feature", values="direction").loc[labels][P.DIRECTION_FEATURES]
    dir_md = pd.DataFrame({"explanation": labels} | {f: [f"{s} {v:+.2f}" for v, s in zip(dir_t[f], dir_s[f])]
                                                     for f in P.DIRECTION_FEATURES})  # fmt: skip
    md = [
        "# Step 5: explanations across preprocessing settings",
        "",
        "Persisted `ablation_preprocessing` models (step 3): MLP and LR with preprocessing none / standard / "
        "log_standard, 10 seeds, 24 time-aware splits, all test instances. MLP: DeepLiftShap (step 4 background, "
        "in each model's own input space) and DeepLift (baseline = background mean); for preprocessing `none` "
        "also DeepLift with an all-zero baseline (the original paper's setting: unscaled inputs, zero "
        "reference). LR: LinearSHAP (log-odds). Aggregation as in step 4 (L1-normalised |attribution|, "
        "instances → split (seeds averaged) → ALL / project).",
        "",
        "## Global top-5 (ALL)",
        "",
        to_markdown(pd.DataFrame(top_rows)),
        "",
        "## Global top-5 per project",
        "",
        to_markdown(proj_df),
        "",
        "## Size share (ALL)",
        "",
        "Equal attribution would give loc 0.05 and the size group {loc, wmc, rfc, npm} 0.20.",
        "",
        to_markdown(ss_t),
        "",
        "## loc share per project",
        "",
        to_markdown(loc_proj),
        "",
        "## Direction (Spearman of raw value vs signed attribution, ALL)",
        "",
        "`+` / `−` = mean ρ ≥ 0.1 / ≤ −0.1, `0` otherwise.",
        "",
        to_markdown(dir_md),
        "",
        "## Statistical comparison of loc share and size-group share",
        "",
        "Per-split means (seeds averaged; 24 splits), Wilcoxon signed-rank (two-sided) between preprocessing "
        "settings within each explanation family, Holm-corrected within (family, metric), Cliff's delta "
        "(a vs b; positive = a larger). `none (zero baseline)` is DeepLift with the zero baseline.",
        "",
        to_markdown(format_comparisons(tests)),
        "",
    ]
    _write(cfg, "step5_preprocessing_explanations", md, {"": imp, "_size_share": ss, "_tests": tests,
                                                          "_per_split": per_split, "_direction": dirs})  # fmt: skip
    bars = s_all[["loc_share", "size_group_share"]].rename(columns={"loc_share": "loc",
                                                                    "size_group_share": "size group {loc, wmc, rfc, npm}"})  # fmt: skip
    F.size_share_bars(bars, "Share of normalised |attribution| on size by preprocessing (ALL splits)",
                      _fig(cfg, "preprocessing_size_share.png"))  # fmt: skip


# ---------------------------------------------------------------- collinearity


def _clusters_str(clusters) -> str:
    multi = [g for g in clusters if len(g) > 1]
    return "; ".join("{" + ", ".join(g) + "}" for g in multi) if multi else "(none)"


def collinearity_report(cfg: Config, es, rs: dict, splits, frames, ex_main: E.Explanations) -> None:
    cs = rs["collinearity"]
    diags = C.release_diagnostics(splits, frames, cs["threshold"], cs["vif_threshold"], cs.get("linkage", "average"))
    stable = C.stable_clusters([d["clusters"] for d in diags], cs["stable_share"])
    co = C.co_clustering([d["clusters"] for d in diags])
    pairs = [(a, b, co.loc[a, b]) for i, a in enumerate(FEATURES) for b in FEATURES[i + 1:] if co.loc[a, b] >= 0.5]
    pairs_t = pd.DataFrame(sorted(pairs, key=lambda t: -t[2]), columns=["metric a", "metric b", "share of releases"])
    rel_t = pd.DataFrame([{"split (training release)": d["split_id"], "n_fit": d["n_fit"],
                           "clusters (|ρ| ≥ 0.7)": _clusters_str(d["clusters"]),
                           "VIF ≥ 5": ", ".join(f"{k} ({v:.1f})" for k, v in d["vif"][d["vif"] >= 5].sort_values(ascending=False).items()) or "none",
                           "AutoSpearman selected": len(d["selected"])} for d in diags])  # fmt: skip
    sel = pd.DataFrame([{f: f in d["selected"] for f in FEATURES} for d in diags])
    freq = sel.mean().sort_values(ascending=False)
    freq_t = pd.DataFrame({"metric": freq.index, "kept in % of training releases": (100 * freq).map(lambda v: f"{v:.0f}")})
    sel_rows = pd.DataFrame([{"split": d["split_id"], "selected": ", ".join(d["selected"]),
                              "removed (step)": ", ".join(f"{r['removed']} ({r['step']})" for r in d["log"])} for d in diags])  # fmt: skip

    # grouped attributions (step 4 v2 store, all methods but LIME_mean_sampling)
    grouped = C.group_explanations(ex_main, stable)
    loc_cl = C.cluster_name(C.loc_cluster(stable))
    imp_f = E.global_importance(ex_main, groups=("all",))
    imp_c = E.global_importance(grouped, groups=("all",))
    top_rows = []
    for (m, k) in ex_main.pairs():
        f = imp_f[(imp_f["model"] == m) & (imp_f["method"] == k) & (imp_f["scope"] == "ALL")].sort_values("importance", ascending=False)
        c = imp_c[(imp_c["model"] == m) & (imp_c["method"] == k) & (imp_c["scope"] == "ALL")].sort_values("importance", ascending=False)
        top_rows.append({"explanation": E.label(m, k), "feature-level top-5": ", ".join(f.head(5)["feature"]),
                         "cluster-level top-5": ", ".join(f"{r.feature} ({r.importance:.3f})" for r in c.head(5).itertuples())})  # fmt: skip
    ss_f = E.size_share(ex_main)
    ss_c = E.size_share(grouped, size_group=[loc_cl], loc=loc_cl)
    ss_rows = []
    for (m, k) in ex_main.pairs():
        a = ss_f[(ss_f["model"] == m) & (ss_f["method"] == k) & (ss_f["scope"] == "ALL")].iloc[0]
        b = ss_c[(ss_c["model"] == m) & (ss_c["method"] == k) & (ss_c["scope"] == "ALL")].iloc[0]
        ss_rows.append({"explanation": E.label(m, k), "loc share": _f(a.loc_share), "size-group share": _f(a.size_group_share),
                        "loc-cluster share": _f(b.loc_share), "% loc top-1": _f(a.loc_top1_pct, 1),
                        "% loc-cluster top-1": _f(b.loc_top1_pct, 1)})  # fmt: skip
    sub_f = ex_main.subset(ex_main.meta["seed"].isin([0, 1, 2]).to_numpy())
    sub_c = grouped.subset(grouped.meta["seed"].isin([0, 1, 2]).to_numpy())
    agr_rows = []
    for family, a, b in E.agreement_pairs():
        if "within" in family:
            continue
        r = {"pair": f"{E.label(*a)} vs {E.label(*b)}"}
        for lab, ex_ in (("feature", sub_f), ("cluster", sub_c)):
            pa = E.pair_agreement(ex_, a, b)
            r[f"{lab} FA@5"] = _f(pa["feature_agreement@5"].mean())
            r[f"{lab} FA@3"] = _f(pa["feature_agreement@3"].mean())
            r[f"{lab} Spearman"] = _f(pa["spearman"].mean())
        agr_rows.append(r)

    retrain = retraining_comparison(cfg, es, rs, ex_main)
    md = [
        "# Step 5: multicollinearity",
        "",
        "Diagnostics on the fitting rows of each of the 24 training releases (raw metrics): Spearman correlation, "
        f"hierarchical clustering ({cs.get('linkage', 'average')} linkage) on 1 − |ρ| cut at |ρ| = {cs['threshold']}, "
        "and VIF on log1p-transformed metrics. Stable clusters link two metrics that are in the same cluster in "
        f"≥ {100 * cs['stable_share']:.0f}% of the training releases (connected components of these links).",
        "",
        "## Stable clusters",
        "",
        "; ".join("{" + ", ".join(g) + "}" for g in stable if len(g) > 1) + f". All other metrics are singletons "
        f"({sum(len(g) == 1 for g in stable)}). The size cluster (containing loc) is {{{', '.join(C.loc_cluster(stable))}}}.",
        "",
        "Pairs of metrics clustered together in at least half of the training releases:",
        "",
        to_markdown(pairs_t.assign(**{"share of releases": pairs_t["share of releases"].map(lambda v: f"{v:.2f}")})),
        "",
        "## Clusters and VIF per training release",
        "",
        to_markdown(rel_t),
        "",
        "## AutoSpearman (Jiarpakdee et al., 2018)",
        "",
        f"Correlation step: while a pair has |ρ| ≥ {cs['threshold']}, drop the metric of the most correlated pair "
        f"with the higher mean |ρ| to the remaining metrics; then drop the metric with the largest VIF while it is "
        f"≥ {cs['vif_threshold']} (VIF on log1p metrics). Selection frequency over the 24 training releases:",
        "",
        to_markdown(freq_t),
        "",
        "Selected metrics per training release:",
        "",
        to_markdown(sel_rows),
        "",
        "## Grouped attributions (stable clusters, step 4 v2 explanations)",
        "",
        "Signed attributions are summed within each stable cluster per instance; the step 4 analysis is then run "
        "on clusters. Cluster-level importances are not comparable one-to-one with feature-level ones (a cluster "
        "collects the attribution of several metrics).",
        "",
        to_markdown(pd.DataFrame(top_rows)),
        "",
        f"Size share: feature-level loc and size-group shares vs the share of the loc cluster ({loc_cl}).",
        "",
        to_markdown(pd.DataFrame(ss_rows)),
        "",
        "Cross-model agreement (seeds 0–2, mean over instances), feature level vs cluster level:",
        "",
        to_markdown(pd.DataFrame(agr_rows)),
        "",
    ] + retrain["md"]
    _write(cfg, "step5_collinearity", md, {"_release_diagnostics": rel_t, "_selection_frequency": freq_t,
                                           "_selected": sel_rows, "_co_clustering": co.reset_index(),
                                           "_grouped_importance": imp_c, "_retraining": retrain["tests"]})  # fmt: skip
    # figures: dendrogram of the release closest to the stable structure; selection frequency
    def closeness(d):
        same = C.co_clustering([d["clusters"]])
        return float((same.round(0) == (co >= cs["stable_share"]).astype(float)).to_numpy().mean())

    rep = max(diags, key=closeness)
    X = frames[rep["project"]]
    split = next(s for s in splits if s.split_id == rep["split_id"])
    from sdp.eval.preprocess import feature_frame

    Z, cols = C.correlation_linkage(feature_frame(X).iloc[split.fit_idx], cs.get("linkage", "average"))
    F.dendrogram_plot(Z, cols, f"Spearman clustering of the CK metrics: {rep['split_id']} (training rows)",
                      cut=1 - cs["threshold"], path=_fig(cfg, "collinearity_dendrogram.png"))  # fmt: skip
    F.single_bars(100 * freq, "AutoSpearman: how often each metric is kept (24 training releases)",
                  "% of training releases", _fig(cfg, "autospearman_selection_frequency.png"), xlim=(0, 100))  # fmt: skip


def retraining_comparison(cfg: Config, es, rs: dict, ex_main: E.Explanations) -> dict:
    """AutoSpearman models vs full-feature models (seeds 0–2): AUC, MCC@tuned, Popt; top-5 and size share."""
    exp = rs["collinearity"]["retrain_experiment"]
    path = cfg.paths.results_dir / exp / "metrics.csv"
    if not path.exists():
        return {"md": ["## AutoSpearman retraining", "", "_Not run._", ""], "tests": pd.DataFrame()}
    a = pd.read_csv(path)
    m = pd.read_csv(cfg.paths.results_dir / "main_time_aware" / "metrics.csv")
    m = m[m["model"].isin(["mlp", "rf"]) & m["seed"].isin([0, 1, 2])]
    specs = [("roc_auc", "none", "AUC"), ("mcc", "val_mcc", "MCC@tuned"), ("popt", "none", "Popt")]
    rows = []
    for model in ("mlp", "rf"):
        fam = []
        for metric, mode, lab in specs:
            def per_split(df):
                d = df[(df["model"] == model) & (df["metric"] == metric) & (df["threshold_mode"] == mode)]
                return d.groupby("split_id")["value"].mean()

            fam.append({"model": E.MODEL_LABEL[model], "metric": lab, "a": "AutoSpearman", "b": "all 20 metrics"}
                       | paired_comparison(per_split(a), per_split(m)))  # fmt: skip
        for r, p in zip(fam, holm([r["p_value"] for r in fam])):
            r["p_holm"] = p
        rows += fam
    tests = _order_cols(pd.DataFrame(rows))
    hp = pd.read_csv(cfg.paths.results_dir / exp / "hyperparameters.csv")
    a_es_root = Path(cfg.paths.artifacts_dir) / f"attributions_{exp}"
    lines = ["## AutoSpearman retraining", "",
             f"MLP (log_standard) and RF (standard) retrained on the AutoSpearman-selected metrics of each training "
             f"release with the step 3 engine (time-aware consecutive, class_weight, seeds 0–2, "
             f"{int(hp['n_trials'].iloc[0])} Optuna trials; persisted under `artifacts/models/{exp}/`). The full-feature "
             "models are the `main_time_aware` models with seeds 0–2 (tuned with 40 trials, so the comparison "
             "also differs in tuning budget). Per-split means over seeds; Wilcoxon signed-rank, Holm within "
             "model, Cliff's delta (AutoSpearman vs all metrics).", "",
             to_markdown(format_comparisons(tests)), ""]  # fmt: skip
    if a_es_root.exists():
        ex_a = E.load_explanations(a_es_root, fill_missing=True)
        main_sub = ex_main.subset((ex_main.meta["seed"].isin([0, 1, 2])
                                   & (((ex_main.meta["model"] == "mlp") & (ex_main.meta["method"] == "DeepLiftShap"))
                                      | ((ex_main.meta["model"] == "rf") & (ex_main.meta["method"] == "TreeSHAP")))).to_numpy())  # fmt: skip
        rows = []
        for lab, ex_ in (("all 20 metrics", main_sub), ("AutoSpearman", ex_a)):
            imp = E.global_importance(ex_, groups=("all",))
            ss = E.size_share(ex_)
            for (mm, k) in ex_.pairs():
                g = imp[(imp["model"] == mm) & (imp["method"] == k) & (imp["scope"] == "ALL")].sort_values("importance", ascending=False)
                s = ss[(ss["model"] == mm) & (ss["method"] == k) & (ss["scope"] == "ALL")].iloc[0]
                rows.append({"explanation": E.label(mm, k), "features": lab,
                             "top-5 (ALL)": ", ".join(f"{r.feature} ({r.importance:.3f})" for r in g.head(5).itertuples()),
                             "loc share": _f(s.loc_share), "size-group share": _f(s.size_group_share),
                             "% loc top-1": _f(s.loc_top1_pct, 1)})  # fmt: skip
        lines += ["Explanations (MLP-DeepLiftShap, RF-TreeSHAP; seeds 0–2). Metrics removed by AutoSpearman get "
                  "attribution 0; shares are relative to the metrics the model uses.", "",
                  to_markdown(pd.DataFrame(rows).sort_values(["explanation", "features"])), ""]  # fmt: skip
    return {"md": lines, "tests": tests}


# -------------------------------------------------------------------- runtime


def runtime_report(cfg: Config, rs: dict) -> None:
    base = cfg.paths.results_dir / "robustness"
    est = json.loads((base / "runtime_estimate.json").read_text()) if (base / "runtime_estimate.json").exists() else {}
    info = json.loads((base / "run_info.json").read_text()) if (base / "run_info.json").exists() else {}
    v2 = json.loads((cfg.paths.results_dir / "explanations" / "run_info_full.json").read_text())
    rows = []
    for study, v in est.get("studies", {}).items():
        rows.append({"study": study, "smoke wall (s)": _f(v.get("smoke_wall_s"), 1), "estimated CPU h": _f(v["cpu_s"] / 3600, 2),
                     "actual wall (min)": _f(info.get("wall_time_s", {}).get(study, info.get("wall_time_s", {}).get(f"{study}_training", np.nan)) / 60, 1)})  # fmt: skip
    timing_rows = []
    for name in ("lime", "background", "preprocessing"):
        p = base / f"timings_{name}.csv"
        if p.exists():
            t = pd.read_csv(p)
            timing_rows.append({"study": name, "units": len(t), "CPU h": _f(t["time_s"].sum() / 3600, 2)})
    a_t = cfg.paths.results_dir / rs["collinearity"]["retrain_experiment"] / "timings.csv"
    if a_t.exists():
        t = pd.read_csv(a_t)
        timing_rows.append({"study": "autospearman retraining", "units": len(t),
                            "CPU h": _f((t["tuning_time_s"].sum() + t["fit_time_s"].sum()) / 3600, 2)})
    red = info.get("reductions", est.get("reductions", {}))
    md = [
        "# Step 5 runtime",
        "",
        "## Machine",
        "",
        to_markdown(pd.DataFrame(list(machine_details().items()), columns=["item", "value"])),
        "",
        "## Step 4 recomputation (v2)",
        "",
        f"Integrated Gradients (512 steps) and LIME (instance sampling) recomputed with the step 4 runner: "
        f"{v2['wall_time_s'] / 60:.1f} min wall, {v2['jobs'] - v2['jobs_from_cache']} of {v2['jobs']} jobs recomputed.",
        "",
        "## Smoke test and estimate",
        "",
        f"Smoke split `{est.get('smoke_split', 'n/a')}` (seed 0). The AutoSpearman retraining is estimated from the "
        "step 3 timings (seeds 0–2, tuning time × 25/40).",
        "",
        to_markdown(pd.DataFrame(rows)) if rows else "_(no estimate)_",
        "",
        f"Estimated total: {est.get('cpu_hours', float('nan')):.2f} CPU h, ~{est.get('wall_hours', float('nan')):.2f} h wall "
        f"with n_jobs={est.get('n_jobs')} (assumed parallel efficiency {est.get('parallel_efficiency_assumed')}).",
        "",
        "Reductions: " + ("none (estimate below 8 h)." if not any((red or {}).values())
                          else f"background projects {red.get('background_projects')}, LIME instances per split "
                               f"{red.get('lime_instances_per_split')}."),
        "",
        "## Full run",
        "",
        to_markdown(pd.DataFrame(timing_rows)) if timing_rows else "",
        "",
        "Wall times (min): " + ", ".join(f"{k} {v / 60:.1f}" for k, v in info.get("wall_time_s", {}).items()) + ".",
        "",
    ]
    _write(cfg, "step5_runtime", md, {})
