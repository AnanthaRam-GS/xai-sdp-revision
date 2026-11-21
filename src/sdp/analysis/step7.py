"""Step 7 reports: cross-version, top-k frequency, unchanged classes, seeds, projects, drift, summary."""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.analysis import explanations as E
from sdp.analysis import figures as F
from sdp.analysis import tables as T
from sdp.analysis.step3 import machine_details
from sdp.analysis.step4 import _f, _write
from sdp.config import Config
from sdp.data.audit import to_markdown
from sdp.eval.stats import bootstrap_ci, holm
from sdp.stability.data import ALL, PRIMARY
from sdp.stability.drift import DRIFT_MEASURES, correlate

logger = logging.getLogger(__name__)

FIG = Path("figures") / "step7"
LAB = {(m, k): E.label(m, k) for m, k in ALL}
PRIMARY_LABELS = [E.label(m, k) for m, k in PRIMARY]
ALL_LABELS = [E.label(m, k) for m, k in ALL]
METRIC_LABEL = {"spearman": "Spearman ρ", "kendall": "Kendall τ-b", "jaccard5": "top-5 Jaccard",
                "jaccard3": "top-3 Jaccard", "rbo": "RBO (p = 0.9)"}  # fmt: skip
CROSS = [E.label(*x) for x in (("mlp", "DeepLiftShap"), ("rf", "TreeSHAP"), ("xgb", "TreeSHAP"), ("lr", "LinearSHAP"))]


def _fig(cfg: Config, name: str) -> Path:
    return Path(cfg.paths.reports_dir) / FIG / name


def _ci(values) -> str:
    m, lo, hi = bootstrap_ci(np.asarray(values, float))
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]" if np.isfinite(m) else "n/a"


def load(cfg: Config) -> dict:
    base = cfg.paths.results_dir / "stability"
    r = {"base": base, "null": pd.read_csv(base / "null.csv"), "cv": pd.read_parquet(base / "cross_version.parquet"),
         "topk": pd.read_parquet(base / "topk_frequency.parquet"), "dirs": pd.read_parquet(base / "directions.parquet"),
         "unch": pd.read_parquet(base / "unchanged.parquet"), "unch_counts": pd.read_csv(base / "unchanged_counts.csv", dtype={"release_a": str, "release_b": str}),
         "seeds": pd.read_parquet(base / "seed_pairs.parquet"), "mult": pd.read_csv(base / "multiplicity.csv"),
         "proj": pd.read_csv(base / "cross_project.csv"), "proj_mats": pd.read_csv(base / "cross_project_matrices.csv"),
         "drift": pd.read_csv(base / "drift.csv"), "info": json.loads((base / "run_info.json").read_text()),
         "clusters": json.loads((base / "clusters.json").read_text())}  # fmt: skip
    return r


def _null(r, level="feature") -> pd.DataFrame:
    return r["null"][r["null"]["level"] == level].set_index("metric")


# ------------------------------------------------------------------ cross-version


def _version_tables(r, level: str):
    cv = r["cv"][r["cv"]["level"] == level]
    null = _null(r, level)
    summ = []
    for lab in ALL_LABELS:
        g = cv[cv["explanation"] == lab]
        if g.empty:
            continue
        row = {"explanation": lab, "comparisons": g["comparison"].nunique()}
        for m in ("spearman", "kendall", "jaccard5", "jaccard3", "rbo"):
            gm = g[g["metric"] == m]
            row[METRIC_LABEL[m]] = _ci(gm["value"])
            row[f"{METRIC_LABEL[m]} data-only"] = _f(gm["data_only"].mean())
        row["seed SD (ρ)"] = _f(g.loc[g["metric"] == "spearman", "seed_sd"].mean())
        summ.append(row)
    summ = pd.DataFrame(summ)
    null_row = {"explanation": "random null (mean / 95th pct)", "comparisons": ""}
    for m in ("spearman", "kendall", "jaccard5", "jaccard3", "rbo"):
        null_row[METRIC_LABEL[m]] = f"{null.loc[m, 'null_mean']:.3f} / {null.loc[m, 'null_p95']:.3f}"
        null_row[f"{METRIC_LABEL[m]} data-only"] = ""
    summ = pd.concat([summ, pd.DataFrame([null_row])], ignore_index=True)
    return cv, summ


def _skesd_and_tests(cv: pd.DataFrame):
    long = cv.rename(columns={"comparison": "split_id"}).assign(threshold_mode="none", seed=0)
    specs = [T.MetricSpec("spearman", "none", "Spearman"), T.MetricSpec("rbo", "none", "RBO")]
    sk = T.skesd_ranks(long, specs, column="explanation", order=ALL_LABELS).set_index("metric").T.reset_index()
    sk = sk.rename(columns={"index": "explanation"})
    vs_data = []
    for m in ("spearman", "rbo"):
        fam = []
        for lab in ALL_LABELS:
            g = cv[(cv["explanation"] == lab) & (cv["metric"] == m)].set_index("comparison")
            if g["data_only"].notna().sum() == 0:
                continue
            fam.append({"metric": METRIC_LABEL[m], "a": lab, "b": "data-only reference"} | T.paired_comparison(g["value"], g["data_only"]))
        for row, p in zip(fam, holm([x["p_value"] for x in fam])):
            row["p_holm"] = p
        vs_data += fam
    cross = []
    for m in ("spearman", "rbo"):
        wide = cv[(cv["metric"] == m) & cv["explanation"].isin(CROSS)].pivot(index="comparison", columns="explanation", values="value")
        fam = [{"metric": METRIC_LABEL[m], "a": a, "b": b} | T.paired_comparison(wide[a], wide[b]) for a, b in combinations(CROSS, 2)]
        for row, p in zip(fam, holm([x["p_value"] for x in fam])):
            row["p_holm"] = p
        cross += fam
    return sk, T._order_cols(pd.DataFrame(vs_data)), T._order_cols(pd.DataFrame(cross))


def cross_version_report(cfg: Config, r: dict) -> pd.DataFrame:
    cv, summ = _version_tables(r, "feature")
    cvc, summ_c = _version_tables(r, "cluster")
    sk, vs_data, cross = _skesd_and_tests(cv)
    skc, vs_data_c, cross_c = _skesd_and_tests(cvc)
    per = cv[(cv["metric"] == "spearman") & cv["explanation"].isin(PRIMARY_LABELS)]
    per_t = per.assign(cell=[f"{v:.2f} [{lo:.2f}, {hi:.2f}] / {d:.2f}" for v, lo, hi, d in
                             zip(per["value"], per["ci_low"], per["ci_high"], per["data_only"])])  # fmt: skip
    per_t = per_t.pivot(index="comparison", columns="explanation", values="cell")[PRIMARY_LABELS].reset_index()
    per_r = cv[(cv["metric"] == "rbo") & cv["explanation"].isin(PRIMARY_LABELS)]
    per_r = per_r.assign(cell=[f"{v:.2f} [{lo:.2f}, {hi:.2f}] / {d:.2f}" for v, lo, hi, d in
                               zip(per_r["value"], per_r["ci_low"], per_r["ci_high"], per_r["data_only"])])  # fmt: skip
    per_r = per_r.pivot(index="comparison", columns="explanation", values="cell")[PRIMARY_LABELS].reset_index()
    ds, flips = _direction(r)
    clusters = "; ".join("{" + ", ".join(g) + "}" for g in r["clusters"] if len(g) > 1)
    md = [
        "# Step 7: cross-version stability of explanations",
        "",
        "Global importance g = mean L1-normalised |attribution| over all test instances of a split, averaged over "
        "the available seeds (10; LIME and KernelSHAP 3). Each of the 14 comparisons pairs consecutive "
        "time-aware splits of a project (e.g. ant 1.3→1.4 vs 1.4→1.5). For each metric: the value with a 95% CI "
        "from resampling the test instances of both splits (1,000 resamples); the **data-only reference** is the "
        "same metric between the permuted-label models (step 6 data randomisation, seed 0), which carry no "
        "defect signal; the **random null** is the metric between two independent random rankings "
        "(10,000 draws). Seed SD is the standard deviation over seeds of the same-seed comparison.",
        "",
        "## Summary per explanation (mean over the 14 comparisons, 95% CI over comparisons)",
        "",
        to_markdown(summ),
        "",
        "## Per comparison: Spearman ρ [instance-bootstrap CI] / data-only reference (primary explanations)",
        "",
        to_markdown(per_t),
        "",
        "## Per comparison: RBO [instance-bootstrap CI] / data-only reference (primary explanations)",
        "",
        to_markdown(per_r),
        "",
        "## Scott-Knott ESD ranks (per-comparison values; 1 = most stable)",
        "",
        to_markdown(sk),
        "",
        "## Each explanation vs its data-only reference (Wilcoxon signed-rank over comparisons, Holm, Cliff's delta)",
        "",
        to_markdown(T.format_comparisons(vs_data)),
        "",
        "## MLP-DeepLiftShap vs RF-TreeSHAP vs XGB-TreeSHAP vs LR-LinearSHAP",
        "",
        to_markdown(T.format_comparisons(cross)),
        "",
        "## Direction stability",
        "",
        "Sign of the step 4 direction statistic (Spearman of raw value vs signed attribution, seeds averaged) in "
        "every split of a project; share of the 20 features whose sign is the same in all splits (mean over "
        "projects). Flipping features are listed in `step7_cross_version_direction_flips.csv`.",
        "",
        to_markdown(ds),
        "",
        "Most frequent flips (primary explanations; number of projects in which the feature flips):",
        "",
        to_markdown(flips),
        "",
        f"## Cluster level (step 5 stable clusters: {clusters}; other metrics singletons)",
        "",
        "Attributions summed within each cluster per instance, then the same analysis; the random null is for "
        f"{len(r['clusters'])} items.",
        "",
        to_markdown(summ_c),
        "",
        "Scott-Knott ESD ranks at cluster level:",
        "",
        to_markdown(skc),
        "",
        "Cluster level vs data-only reference:",
        "",
        to_markdown(T.format_comparisons(vs_data_c)),
        "",
    ]
    _write(cfg, "step7_cross_version", md, {"": cv, "_cluster": cvc, "_tests_data_only": vs_data, "_tests_models": cross,
                                            "_direction_flips": r["dirs_flips"]})  # fmt: skip
    # figure: points with CIs (mean over comparisons) and reference bands
    null = _null(r)
    pts = cv[cv["metric"] == "spearman"].groupby("explanation").agg(mean=("value", "mean"), data=("data_only", "mean"))
    cis = {lab: bootstrap_ci(cv[(cv["metric"] == "spearman") & (cv["explanation"] == lab)]["value"].to_numpy())
           for lab in ALL_LABELS}  # fmt: skip
    F.point_ci(pd.DataFrame({"mean": [cis[l][0] for l in ALL_LABELS], "low": [cis[l][1] for l in ALL_LABELS],
                             "high": [cis[l][2] for l in ALL_LABELS], "reference": pts.reindex(ALL_LABELS)["data"].to_numpy()},
                            index=ALL_LABELS),  # fmt: skip
               "Cross-version stability of global importance (Spearman ρ, mean over 14 comparisons)", "Spearman ρ",
               band=(null.loc["spearman", "null_mean"], null.loc["spearman", "null_p95"]),
               band_label="random null (mean to 95th pct)", reference_label="data-only reference (permuted labels)",
               path=_fig(cfg, "cross_version_spearman.png"))  # fmt: skip
    return cv


def _direction(r):
    from sdp.stability.versions import direction_stability

    summ, flips = direction_stability(r["dirs"])
    r["dirs_flips"] = flips
    ds = summ.groupby("explanation")["same_sign_share"].agg(["mean", "min"]).reindex(ALL_LABELS).dropna(how="all")
    ds = ds.assign(mean=lambda t: t["mean"].map(_f), min=lambda t: t["min"].map(_f)).reset_index()
    ds = ds.rename(columns={"mean": "same sign in all splits (mean over projects)", "min": "worst project"})
    fp = flips[flips["explanation"].isin(PRIMARY_LABELS)]
    top = fp.groupby(["explanation", "feature"]).size().rename("projects").reset_index()
    top = top.sort_values(["explanation", "projects"], ascending=[True, False]).groupby("explanation").head(3)
    return ds, top


# ------------------------------------------------------------------ top-k


def topk_report(cfg: Config, r: dict) -> None:
    t = r["topk"]
    pooled = t[t["scope"] == "ALL"].pivot(index="feature", columns="explanation", values="frequency")
    pooled = pooled[[l for l in ALL_LABELS if l in pooled.columns]]
    pooled = pooled.loc[pooled[PRIMARY_LABELS].mean(axis=1).sort_values(ascending=False).index]
    pooled_md = pooled[PRIMARY_LABELS].map(lambda v: f"{v:.2f}").reset_index()

    def per_project(lab):
        g = t[(t["explanation"] == lab) & (t["scope"] != "ALL")]
        w = g.pivot(index="feature", columns="scope", values="frequency")
        n = g.groupby("scope")["n_splits"].first()
        w = w.loc[w.mean(axis=1).sort_values(ascending=False).index]
        w = w[w.max(axis=1) > 0]
        w.columns = [f"{c} ({n[c]})" for c in w.columns]
        return w.map(lambda v: f"{v:.2f}").reset_index()

    md = [
        "# Step 7: top-5 frequency (corrected Table 2)",
        "",
        "For each explanation, the share of splits in whose global top-5 (seed-averaged g) each metric appears. "
        "`pooled` = over all 24 time-aware splits; per project, the denominator is the project's number of splits "
        "(in brackets). This replaces the original paper's Table 2, which counted top features of a single "
        "explanation setting (unscaled inputs, zero baseline).",
        "",
        "## Pooled over all 24 splits (primary explanations)",
        "",
        to_markdown(pooled_md),
        "",
        "## Per project: MLP-DeepLiftShap",
        "",
        to_markdown(per_project("MLP-DeepLiftShap")),
        "",
        "## Per project: RF-TreeSHAP",
        "",
        to_markdown(per_project("RF-TreeSHAP")),
        "",
        "All 14 explanations, per project and pooled: `step7_topk_frequency.csv`.",
        "",
    ]
    _write(cfg, "step7_topk_frequency", md, {"": t})
    F.heatmap(pooled, "Share of splits with the metric in the global top-5 (pooled over 24 splits)", "frequency",
              _fig(cfg, "topk_frequency_heatmap.png"), vmin=0.0, vmax=1.0)  # fmt: skip


# ------------------------------------------------------------------ unchanged classes


def unchanged_report(cfg: Config, r: dict) -> None:
    u, c = r["unch"], r["unch_counts"]
    rows = []
    for (project, lab), g in u.groupby(["project", "explanation"]):
        if lab not in PRIMARY_LABELS:
            continue
        rows.append({"project": project, "explanation": lab, "classes": len(g), "Spearman": _ci(g["spearman"]),
                     "FA@5": _ci(g["fa5"]), "SA@5": _ci(g["sa5"]), "|Δp|": _f(g["dprob"].mean()),
                     "flip rate": _f(g["flip"].mean()), "seed floor ρ": _f(g["seed_spearman"].mean()) if "seed_spearman" in g else "n/a",
                     "seed floor FA@5": _f(g["seed_fa5"].mean()) if "seed_fa5" in g else "n/a"})  # fmt: skip
    overall = []
    for lab in ALL_LABELS:
        g = u[u["explanation"] == lab]
        if g.empty:
            continue
        overall.append({"explanation": lab, "classes": len(g), "Spearman (across versions)": _ci(g["spearman"]),
                        "seed floor ρ (same split)": _ci(g["seed_spearman"]) if g["seed_spearman"].notna().any() else "n/a",
                        "FA@5": _ci(g["fa5"]), "seed floor FA@5": _f(g["seed_fa5"].mean()), "SA@5": _ci(g["sa5"]),
                        "|Δp|": _f(g["dprob"].mean()), "flip rate": _f(g["flip"].mean())})  # fmt: skip
    ivy = c[c["project"] == "ivy"]
    ivy_txt = (f"ivy ({', '.join(ivy['comparison'])}): {int(ivy['matched_exact_name'].sum())} classes match by exact name; "
               f"mapping `fr.jayasoft.ivy.` → `org.apache.ivy.` adds {int(ivy['matched_via_mapping'].sum())} matches, of "
               f"which {int(ivy['unchanged_via_mapping'].sum())} are unchanged (identical metrics and label).") if len(ivy) else ""
    md = [
        "# Step 7: explanation drift of unchanged classes",
        "",
        "Classes present in both test releases n+1 and n+2 of consecutive splits with identical values on all 20 "
        "metrics and the same label. Their explanations from the model trained on n (explaining n+1) and on n+1 "
        "(explaining n+2) are compared seed by seed (common seeds) and averaged: Spearman of |attribution|, feature "
        "agreement@5 (FA@5), sign agreement@5 (SA@5); |Δp| is the change in the predicted probability and the flip "
        "rate the share of (class, seed) whose predicted label differs at the two models' tuned thresholds. The "
        "seed floor compares seeds 0 and 1 of the later split for the same classes. Means with 95% bootstrap CIs "
        "over classes. Per-class values: `results/stability/unchanged.parquet`.",
        "",
        "## Unchanged classes per pair",
        "",
        to_markdown(c[["project", "comparison", "release_a", "release_b", "matched", "matched_exact_name",
                       "matched_via_mapping", "unchanged", "unchanged_via_mapping"]]),  # fmt: skip
        "",
        ivy_txt,
        "",
        "## Overall",
        "",
        to_markdown(pd.DataFrame(overall)),
        "",
        "## Per project (primary explanations)",
        "",
        to_markdown(pd.DataFrame(rows)),
        "",
    ]
    _write(cfg, "step7_unchanged_classes", md, {"": pd.DataFrame(overall), "_per_project": pd.DataFrame(rows), "_counts": c})


# ------------------------------------------------------------------ seeds


def seed_report(cfg: Config, r: dict, cv: pd.DataFrame) -> None:
    s, mult = r["seeds"], r["mult"]
    per_split = s.groupby(["explanation", "split_id"])[["global_spearman", "global_rbo", "global_jaccard5",
                                                         "instance_spearman", "instance_fa5", "instance_sa5"]].mean().reset_index()  # fmt: skip
    rows = []
    for lab in ALL_LABELS:
        g = per_split[per_split["explanation"] == lab]
        if g.empty:
            continue
        n_seeds = s[s["explanation"] == lab].groupby("split_id")[["seed_a", "seed_b"]].apply(lambda t: len(set(t["seed_a"]) | set(t["seed_b"]))).max()
        rows.append({"explanation": lab, "seeds": int(n_seeds), "global ρ": _ci(g["global_spearman"]), "global RBO": _ci(g["global_rbo"]),
                     "global top-5 Jaccard": _ci(g["global_jaccard5"]), "instance ρ": _ci(g["instance_spearman"]),
                     "instance FA@5": _ci(g["instance_fa5"]), "instance SA@5": _ci(g["instance_sa5"])})  # fmt: skip
    mult_t = mult.groupby("model")[["disagreement_share", "mean_prob_sd", "mean_kappa"]].agg(["mean"]).droplevel(1, axis=1)
    mult_t = mult_t.reindex(E.MODEL_ORDER).map(_f).reset_index().rename(columns={
        "disagreement_share": "share of instances with ≥ 2 different labels", "mean_prob_sd": "mean SD of p",
        "mean_kappa": "mean pairwise Cohen's κ"})  # fmt: skip
    mult_t["model"] = mult_t["model"].map(E.MODEL_LABEL)
    # model comparison on seed stability (primary method per model)
    prim = {"mlp": "MLP-DeepLiftShap", "lr": "LR-LinearSHAP", "rf": "RF-TreeSHAP", "xgb": "XGB-TreeSHAP"}
    tests = []
    for metric, frame, col, lab in (("global ρ", per_split, "global_spearman", "explanation"),
                                    ("instance ρ", per_split, "instance_spearman", "explanation")):  # fmt: skip
        wide = frame[frame[lab].isin(prim.values())].pivot(index="split_id", columns=lab, values=col)
        fam = [{"measure": metric, "a": a, "b": b} | T.paired_comparison(wide[a], wide[b]) for a, b in combinations(prim.values(), 2)]
        for row, p in zip(fam, holm([x["p_value"] for x in fam])):
            row["p_holm"] = p
        tests += fam
    for col, name in (("disagreement_share", "label disagreement share"), ("mean_kappa", "mean Cohen's κ")):
        wide = mult.pivot(index="split_id", columns="model", values=col).rename(columns=E.MODEL_LABEL)
        fam = [{"measure": name, "a": a, "b": b} | T.paired_comparison(wide[a], wide[b]) for a, b in combinations(["MLP", "LR", "RF", "XGB"], 2)]
        for row, p in zip(fam, holm([x["p_value"] for x in fam])):
            row["p_holm"] = p
        tests += fam
    tests = T._order_cols(pd.DataFrame(tests))
    sources = _sources_of_variation(cfg, r, per_split, cv)
    md = [
        "# Step 7: seed stability and prediction multiplicity",
        "",
        "All pairs of training seeds of each split (45 pairs for 10 seeds; 3 for LIME and KernelSHAP). Global: the "
        "stability metrics between the two seeds' importance vectors; instance: per-instance Spearman of "
        "|attribution|, feature agreement@5 and sign agreement@5, averaged over the test instances. Per-split means "
        "over seed pairs, then mean and 95% bootstrap CI over the 24 splits.",
        "",
        "## Seed stability per explanation",
        "",
        to_markdown(pd.DataFrame(rows)),
        "",
        "## Prediction multiplicity across the 10 seeds (mean over splits)",
        "",
        to_markdown(mult_t),
        "",
        "## Model comparison (per-split values; Wilcoxon signed-rank, Holm within measure, Cliff's delta)",
        "",
        to_markdown(T.format_comparisons(tests)),
        "",
        "## Sources of variation",
        "",
        "One row per source; values are mean Spearman ρ (higher = more stable). *Instance* rows compare the "
        "explanations of the same instance; *global* rows compare importance vectors. The step 5 rows use their own "
        "samples: LIME runs = 10 runs with different random states on up to 40 instances per split (seed-0 models); "
        "background draws = 5 independent stratified backgrounds of 100 rows (seed-0 models, all instances).",
        "",
        to_markdown(sources),
        "",
    ]
    _write(cfg, "step7_seed_stability", md, {"": per_split, "_multiplicity": mult, "_tests": tests, "_sources": sources})
    panel = {lab: per_split.loc[per_split["explanation"] == lab, "global_spearman"].to_numpy() for lab in prim.values()}
    panel_i = {lab: per_split.loc[per_split["explanation"] == lab, "instance_spearman"].to_numpy() for lab in prim.values()}
    data = {"global importance": panel, "instance level": panel_i}
    F.box_panels(data, "Seed stability per model (per-split mean Spearman over seed pairs)", "Spearman ρ",
                 _fig(cfg, "seed_stability_boxplots.png"))  # fmt: skip


def _sources_of_variation(cfg: Config, r: dict, per_split: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    rep = Path(cfg.paths.reports_dir)
    cols = PRIMARY_LABELS
    out = []
    seed_g = per_split.groupby("explanation")["global_spearman"].mean()
    seed_i = per_split.groupby("explanation")["instance_spearman"].mean()
    out.append({"source": "training seed", "level": "global"} | {c: _f(seed_g.get(c, np.nan)) for c in cols})
    out.append({"source": "training seed", "level": "instance"} | {c: _f(seed_i.get(c, np.nan)) for c in cols})
    lime = pd.read_csv(rep / "step5_lime_stability.csv")
    lime = lime[lime["table"] == "run_to_run"].set_index("model")["spearman"]
    label_to_model = {v: k for k, v in E.MODEL_LABEL.items()}
    out.append({"source": "LIME random state (step 5)", "level": "instance"} | {
        c: (_f(lime.get(E.MODEL_LABEL[label_to_model[c.split("-")[0]]], np.nan)) if c.endswith("-LIME") else "") for c in cols})
    bg = pd.read_csv(rep / "step5_background_sensitivity_noise_floor.csv").set_index("explanation")["spearman"]
    out.append({"source": "background draw (step 5)", "level": "instance"} | {c: _f(bg[c]) if c in bg.index else "" for c in cols})
    v = cv[(cv["metric"] == "spearman") & (cv["level"] == "feature") if "level" in cv else (cv["metric"] == "spearman")]
    vm = v.groupby("explanation")["value"].mean()
    out.append({"source": "release (consecutive splits)", "level": "global"} | {c: _f(vm.get(c, np.nan)) for c in cols})
    p = r["proj"][r["proj"]["metric"] == "spearman"].assign(explanation=lambda t: [E.label(m, k) for m, k in zip(t["model"], t["method"])]).set_index("explanation")["mean"]
    out.append({"source": "project", "level": "global"} | {c: _f(p.get(c, np.nan)) for c in cols})
    null = _null(r).loc["spearman", "null_p95"]
    out.append({"source": "random null (95th pct)", "level": "global"} | {c: _f(null) for c in cols})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ cross-project


def cross_project_report(cfg: Config, r: dict) -> pd.DataFrame:
    p = r["proj"].assign(explanation=lambda t: [E.label(m, k) for m, k in zip(t["model"], t["method"])])
    null = _null(r)
    rows = []
    for lab in ALL_LABELS:
        g = p[p["explanation"] == lab].set_index("metric")
        if g.empty:
            continue
        rows.append({"explanation": lab,
                     "Spearman (mean off-diagonal)": f"{g.loc['spearman', 'mean']:.3f} [{g.loc['spearman', 'ci_low']:.3f}, {g.loc['spearman', 'ci_high']:.3f}]",
                     "Spearman data-only": _f(g.loc["spearman", "data_only"]),
                     "RBO (mean off-diagonal)": f"{g.loc['rbo', 'mean']:.3f} [{g.loc['rbo', 'ci_low']:.3f}, {g.loc['rbo', 'ci_high']:.3f}]",
                     "RBO data-only": _f(g.loc["rbo", "data_only"])})  # fmt: skip
    rows.append({"explanation": "random null (mean / 95th pct)",
                 "Spearman (mean off-diagonal)": f"{null.loc['spearman', 'null_mean']:.3f} / {null.loc['spearman', 'null_p95']:.3f}",
                 "Spearman data-only": "", "RBO (mean off-diagonal)": f"{null.loc['rbo', 'null_mean']:.3f} / {null.loc['rbo', 'null_p95']:.3f}",
                 "RBO data-only": ""})  # fmt: skip
    base = r["base"]
    common = []
    for m, k in ALL:
        f = base / f"common_top_{m}_{k}.csv"
        t = pd.read_csv(f) if f.exists() else pd.DataFrame()
        common.append({"explanation": E.label(m, k),
                       "in the top-5 of ≥ 7 of 10 projects": ", ".join(f"{a} ({b})" for a, b in zip(t.get("feature", []), t.get("projects_in_top5", []))) or "none"})  # fmt: skip
    md = [
        "# Step 7: cross-project stability",
        "",
        "Project-level importance = mean of the seed-averaged global importance over the project's splits. Mean of "
        "the 45 off-diagonal values of the 10 × 10 matrix with a 95% bootstrap CI over project pairs; data-only = "
        "the same for the permuted-label models (seed 0).",
        "",
        to_markdown(pd.DataFrame(rows)),
        "",
        "## Metrics in the top-5 of at least 7 of the 10 projects",
        "",
        to_markdown(pd.DataFrame(common)),
        "",
        "Full matrices: `step7_cross_project_matrices.csv`.",
        "",
    ]
    _write(cfg, "step7_cross_project", md, {"": p, "_matrices": r["proj_mats"]})
    mats = r["proj_mats"]
    for lab, (m, k) in (("MLP-DeepLiftShap", ("mlp", "DeepLiftShap")), ("RF-TreeSHAP", ("rf", "TreeSHAP"))):
        g = mats[(mats["model"] == m) & (mats["method"] == k) & (mats["metric"] == "spearman")].set_index("project")
        g = g[[c for c in g.index]]
        F.heatmap(g, f"Cross-project Spearman of project-level importance: {lab}", "Spearman ρ",
                  _fig(cfg, f"cross_project_{m}_{k}.png"), vmin=-1.0, vmax=1.0)  # fmt: skip
    return p


# ------------------------------------------------------------------ drift


def drift_report(cfg: Config, r: dict, cv: pd.DataFrame) -> None:
    d = r["drift"]
    tt = d[d["kind"] == "train to test"].copy()
    ct = d[d["kind"] == "consecutive test releases"].copy()
    m3 = pd.read_csv(cfg.paths.results_dir / "main_time_aware" / "metrics.csv")
    auc = m3[(m3["metric"] == "roc_auc") & m3["model"].isin(E.MODEL_ORDER)].groupby(["split_id", "model"])["value"].mean().unstack()
    auc.columns = [f"AUC {E.MODEL_LABEL[c]}" for c in auc.columns]
    tt = tt.merge(auc, left_on="split_id", right_index=True)
    rel_i = correlate(tt, DRIFT_MEASURES, list(auc.columns))
    stab = cv[(cv["metric"] == "spearman") & (cv["level"] == "feature") & cv["explanation"].isin(PRIMARY_LABELS)]
    stab = stab.pivot(index="comparison", columns="explanation", values="value")
    ct = ct.merge(stab, left_on="comparison", right_index=True)
    rel_ii = correlate(ct, DRIFT_MEASURES, PRIMARY_LABELS)

    def fmt(t):
        return t.assign(rho=t["rho"].map(lambda v: f"{v:+.2f}"), p_value=t["p_value"].map(lambda v: f"{v:.3f}"),
                        p_holm=t["p_holm"].map(lambda v: f"{v:.3f}"))  # fmt: skip

    cols = ["project", "comparison", "mean_ks", "max_ks", "max_ks_feature", "delta_ratio", "c2st_auc", "c2st_top_features"]
    md = [
        "# Step 7: data drift and its relation to stability",
        "",
        "Per release pair (raw metrics): mean and maximum two-sample Kolmogorov–Smirnov statistic over the 20 "
        "metrics, the change in the defective ratio, and a classifier two-sample test (random forest, 200 trees, "
        "5-fold stratified CV AUC, seed 0; 0.5 = indistinguishable) with its top-3 feature importances. "
        "Relationships: Spearman ρ with Holm correction across all tests of a family. **The number of pairs is "
        f"small (n = {len(tt)} train→test pairs, n = {len(ct)} consecutive pairs), so these correlations are "
        "descriptive and underpowered.**",
        "",
        "## Drift between consecutive test releases",
        "",
        to_markdown(ct[cols].assign(**{c: ct[c].map(_f) for c in ("mean_ks", "max_ks", "delta_ratio", "c2st_auc")})),
        "",
        "## Drift between training and test release (time-aware splits)",
        "",
        to_markdown(tt[["split_id", "mean_ks", "max_ks", "max_ks_feature", "delta_ratio", "c2st_auc", "c2st_top_features"]]
                    .assign(**{c: tt[c].map(_f) for c in ("mean_ks", "max_ks", "delta_ratio", "c2st_auc")})),  # fmt: skip
        "",
        "## (i) Train→test drift vs test AUC (step 3, mean over seeds)",
        "",
        to_markdown(fmt(rel_i)),
        "",
        "## (ii) Drift between consecutive test releases vs cross-version Spearman of the explanations",
        "",
        to_markdown(fmt(rel_ii)),
        "",
    ]
    _write(cfg, "step7_drift", md, {"": d, "_relationships_auc": rel_i, "_relationships_stability": rel_ii})
    F.scatter_panels(
        {f"{x} vs {y}": (ct[x].to_numpy(float), ct[y].to_numpy(float), x, y) for x, y in
         (("c2st_auc", "MLP-DeepLiftShap"), ("c2st_auc", "RF-TreeSHAP"), ("mean_ks", "MLP-DeepLiftShap"), ("mean_ks", "RF-TreeSHAP"))}
        | {f"{x} vs {y}": (tt[x].to_numpy(float), tt[y].to_numpy(float), x, y) for x, y in
           (("c2st_auc", "AUC MLP"), ("c2st_auc", "AUC RF"), ("mean_ks", "AUC MLP"), ("mean_ks", "AUC RF"))},
        "Drift vs cross-version explanation stability (top) and drift vs test AUC (bottom)", _fig(cfg, "drift_scatter.png"))  # fmt: skip
    r["rel_ii"], r["rel_i"] = rel_ii, rel_i


# ------------------------------------------------------------------ summary


def summary_report(cfg: Config, r: dict, cv: pd.DataFrame, proj: pd.DataFrame) -> None:
    null = _null(r)
    sp = cv[(cv["metric"] == "spearman") & (cv["level"] == "feature")]
    mlp = sp[sp["explanation"] == "MLP-DeepLiftShap"]["value"]
    rfv = sp[sp["explanation"] == "RF-TreeSHAP"]["value"]
    data_mlp = sp[sp["explanation"] == "MLP-DeepLiftShap"]["data_only"].mean()
    data_rf = sp[sp["explanation"] == "RF-TreeSHAP"]["data_only"].mean()
    t = r["topk"]
    store_top1 = r["base"] / "top1.csv"
    top1 = pd.read_csv(store_top1) if store_top1.exists() else None
    loc_top5 = t[(t["scope"] == "ALL") & (t["feature"] == "loc")].set_index("explanation")["frequency"]
    ag = pd.read_csv(Path(cfg.paths.reports_dir) / "step4_agreement.csv")
    dl = ag[(ag["a"] == "MLP-DeepLift") & (ag["b"] == "MLP-DeepLiftShap") & (ag["scope"] == "ALL")].set_index("metric")["mean"]
    pre = pd.read_csv(Path(cfg.paths.reports_dir) / "step5_preprocessing_explanations_size_share.csv")
    pre = pre[(pre["scope"] == "ALL")].set_index(["model", "method", "preprocessing"])
    p = proj[proj["metric"] == "spearman"].set_index("explanation")
    seeds = r["seeds"].groupby("explanation")["global_spearman"].mean()
    mult = r["mult"].groupby("model")["disagreement_share"].mean()
    claims = pd.DataFrame([
        {"original claim": "LOC is the top feature in all versions",
         "rebuilt pipeline": (f"loc is top-1 in {top1.loc[top1['explanation'] == 'MLP-DeepLiftShap', 'loc_top1_splits'].iloc[0]} of 24 splits for MLP-DeepLiftShap and "
                              f"{top1.loc[top1['explanation'] == 'RF-TreeSHAP', 'loc_top1_splits'].iloc[0]} for RF-TreeSHAP; in the top-5 of "
                              f"{loc_top5.get('MLP-DeepLiftShap', np.nan):.0%} / {loc_top5.get('RF-TreeSHAP', np.nan):.0%} of splits. "
                              f"With unscaled inputs and a zero baseline (the original setting, step 5) loc takes "
                              f"{pre.loc[('mlp', 'DeepLift_zero_baseline', 'none'), 'loc_share']:.2f} of the attribution and is top-1 for "
                              f"{pre.loc[('mlp', 'DeepLift_zero_baseline', 'none'), 'loc_top1_pct']:.0f}% of instances, against "
                              f"{pre.loc[('mlp', 'DeepLiftShap', 'log_standard'), 'loc_share']:.2f} with log-standardised inputs."),
         "verdict": "not supported: an artefact of unscaled inputs"},
        {"original claim": "Spearman 0.69–0.89 across versions",
         "rebuilt pipeline": (f"MLP-DeepLiftShap: mean {mlp.mean():.2f} (range {mlp.min():.2f}–{mlp.max():.2f}) over 14 comparisons; "
                              f"RF-TreeSHAP: mean {rfv.mean():.2f} ({rfv.min():.2f}–{rfv.max():.2f}). Data-only reference "
                              f"(permuted labels): {data_mlp:.2f} / {data_rf:.2f}; random null 95th pct {null.loc['spearman', 'null_p95']:.2f}."),
         "verdict": "not reproduced: far lower for the MLP, moderate for the tree models, and partly reached without any defect signal"},
        {"original claim": "DeepLIFT and DeepSHAP agree",
         "rebuilt pipeline": (f"MLP-DeepLift vs MLP-DeepLiftShap (step 4): feature agreement@5 {dl['feature_agreement@5']:.2f}, "
                              f"Spearman {dl['spearman']:.2f}; but both use the same propagation rules, so the agreement is "
                              "expected and not independent evidence; agreement with LIME and across models is much lower."),
         "verdict": "true but uninformative"},
    ])  # fmt: skip
    key = pd.DataFrame([
        {"quantity": "cross-version Spearman (MLP-DeepLiftShap / RF-TreeSHAP)", "value": f"{mlp.mean():.2f} / {rfv.mean():.2f}"},
        {"quantity": "data-only reference (permuted labels)", "value": f"{data_mlp:.2f} / {data_rf:.2f}"},
        {"quantity": "random null: mean / 95th percentile", "value": f"{null.loc['spearman', 'null_mean']:.2f} / {null.loc['spearman', 'null_p95']:.2f}"},
        {"quantity": "seed stability, global Spearman (MLP-DeepLiftShap / RF-TreeSHAP)",
         "value": f"{seeds.get('MLP-DeepLiftShap', np.nan):.2f} / {seeds.get('RF-TreeSHAP', np.nan):.2f}"},
        {"quantity": "cross-project Spearman (MLP-DeepLiftShap / RF-TreeSHAP)",
         "value": f"{p.loc['MLP-DeepLiftShap', 'mean']:.2f} / {p.loc['RF-TreeSHAP', 'mean']:.2f}"},
        {"quantity": "test instances whose label changes between seeds (MLP / LR / RF / XGB)",
         "value": " / ".join(f"{mult.get(m, np.nan):.1%}" for m in E.MODEL_ORDER)},
    ])  # fmt: skip
    md = [
        "# Step 7: stability summary",
        "",
        "Plain-language summary of the step 7 results (details in the other `step7_*` reports).",
        "",
        "## Key numbers",
        "",
        to_markdown(key),
        "",
    ] + r.get("summary_points", []) + [
        "",
        "## The original paper's claims vs the rebuilt pipeline",
        "",
        to_markdown(claims),
        "",
    ]
    _write(cfg, "step7_summary", md, {"": key, "_claims": claims})


# ------------------------------------------------------------------ runtime


def runtime_report(cfg: Config) -> None:
    base = cfg.paths.results_dir / "stability"
    full = json.loads((base / "run_info.json").read_text())
    smoke = json.loads((base / "_smoke" / "run_info.json").read_text()) if (base / "_smoke" / "run_info.json").exists() else {}
    perm = json.loads((base / "permuted_run_info.json").read_text()) if (base / "permuted_run_info.json").exists() else {}
    rows = [{"part": k, "smoke (s)": _f(smoke.get("timings_s", {}).get(k), 1), "full (s)": _f(v, 1)} for k, v in full["timings_s"].items()]
    est = smoke.get("estimate_full_s")
    md = [
        "# Step 7 runtime",
        "",
        "## Machine",
        "",
        to_markdown(pd.DataFrame(list(machine_details().items()), columns=["item", "value"])),
        "",
        "## Permuted-label refit (data-only reference)",
        "",
        f"{perm.get('models', 'n/a')} models refitted (seed 0, persisted hyperparameters, no re-tuning) and "
        f"{perm.get('explanations', 'n/a')} explanation sets stored in {perm.get('wall_time_s', float('nan')) / 60:.1f} min wall "
        "(10 workers). Reproduction of step 6: maximum absolute differences — AUC "
        f"{perm.get('max_diff_permuted_auc', float('nan')):.1e}, Spearman {perm.get('max_diff_spearman', float('nan')):.1e}, "
        f"FA@5 {perm.get('max_diff_fa5', float('nan')):.1e} over {perm.get('matched_to_step6', 'n/a')} (split, model, method) units. "
        "The refit was run directly, without its own smoke test: it repeats the step 6 data-randomisation refit, "
        "whose cost was known (step 6 runtime report), plus storing the explanations.",
        "",
        "## Smoke test (project ant) and full run of the analyses",
        "",
        f"Smoke test: {smoke.get('n_splits', 'n/a')} splits, {smoke.get('wall_time_s', float('nan')):.0f} s; estimated full run "
        f"{(est or float('nan')) / 60:.1f} min. Full run: {full['wall_time_s'] / 60:.1f} min (single process). No reductions.",
        "",
        to_markdown(pd.DataFrame(rows)),
        "",
    ]
    _write(cfg, "step7_runtime", md, {})


def summary_points(r: dict, cv: pd.DataFrame, proj: pd.DataFrame) -> list[str]:
    """Plain-language bullet points generated from the results."""
    sp = cv[(cv["metric"] == "spearman") & (cv["level"] == "feature")]
    m = sp.groupby("explanation").agg(value=("value", "mean"), data=("data_only", "mean"))
    above = [l for l in PRIMARY_LABELS if l in m.index and m.loc[l, "value"] > m.loc[l, "data"]]
    null95 = _null(r).loc["spearman", "null_p95"]
    seeds = r["seeds"].groupby("explanation")["global_spearman"].mean()
    u = r["unch"]
    uc = r["unch_counts"]
    p = proj[proj["metric"] == "spearman"].set_index("explanation")["mean"]
    rel = r.get("rel_ii")
    n_sig = int((rel["p_holm"] < 0.05).sum()) if rel is not None else 0
    n_sig_i = int((r["rel_i"]["p_holm"] < 0.05).sum()) if r.get("rel_i") is not None else 0
    ct = r["drift"][r["drift"]["kind"] == "consecutive test releases"]
    mult = r["mult"].groupby("model")["disagreement_share"].mean()
    _, vs_data, _ = _skesd_and_tests(cv[cv["level"] == "feature"])
    sig_data = list(vs_data[(vs_data["metric"] == METRIC_LABEL["spearman"]) & (vs_data["p_holm"] < 0.05)
                            & (vs_data["mean_a"] > vs_data["mean_b"])]["a"])  # fmt: skip
    prim = m.reindex(PRIMARY_LABELS)
    over_null = [l for l in PRIMARY_LABELS if prim.loc[l, "value"] > null95]
    under_null = [l for l in PRIMARY_LABELS if prim.loc[l, "value"] <= null95]
    mlp_like = [l for l in under_null if l.startswith("MLP")]
    sd = seeds.reindex(PRIMARY_LABELS)
    pts = [
        "## In plain language",
        "",
        f"* **Across releases**, the global rankings are only weakly to moderately stable: mean Spearman ρ over the 14 "
        f"consecutive-split comparisons ranges from {prim['value'].min():.2f} to {prim['value'].max():.2f}. Only "
        f"{', '.join(over_null) if over_null else 'none'} exceed the 95th percentile of the random null ({null95:.2f}); "
        f"the MLP explanations ({', '.join(l.replace('MLP-', '') for l in mlp_like)}) stay at ρ ≈ "
        f"{prim.loc[mlp_like, 'value'].min():.2f}–{prim.loc[mlp_like, 'value'].max():.2f}." if mlp_like else "",
        f"* **Much of the stability that exists does not need a defect signal**: the permuted-label models reach "
        f"ρ = {prim['data'].min():.2f}–{prim['data'].max():.2f} (highest for the tree models, whose importances follow the "
        f"metric distributions). Explanations whose mean ρ exceeds their data-only reference: "
        f"{', '.join(above) if above else 'none'}; after Holm correction the difference is significant (Spearman, "
        f"p < 0.05) only for {', '.join(sig_data) if sig_data else 'none of the explanations'} (n = 14 comparisons).",
        f"* **Across training seeds**, global rankings are very stable for the tree models (mean ρ "
        f"{sd.loc[['RF-TreeSHAP', 'XGB-TreeSHAP']].min():.2f}–{sd.loc[['RF-TreeSHAP', 'XGB-TreeSHAP']].max():.2f}) and only "
        f"moderately for the MLP ({sd.loc[[l for l in PRIMARY_LABELS if l.startswith('MLP')]].min():.2f}–"
        f"{sd.loc[[l for l in PRIMARY_LABELS if l.startswith('MLP')]].max():.2f}). Predictions themselves are not stable: "
        f"{mult.min():.0%}–{mult.max():.0%} of test instances change predicted label between at least two seeds "
        f"(MLP {mult.get('mlp', np.nan):.0%}).",
        f"* **Unchanged classes** ({int(uc['unchanged'].sum())} classes with identical metrics and label in consecutive "
        f"test releases) get explanations that agree with ρ ≈ {u[u['explanation'].isin(PRIMARY_LABELS)]['spearman'].mean():.2f} "
        f"across versions, against ≈ {u[u['explanation'].isin(PRIMARY_LABELS)]['seed_spearman'].mean():.2f} between two seeds of "
        "the same model: retraining on a new release changes the explanation of an unchanged class more than a new seed does.",
        f"* **Across projects**, project-level rankings agree weakly (mean ρ {p.reindex(PRIMARY_LABELS).min():.2f}–"
        f"{p.reindex(PRIMARY_LABELS).max():.2f}).",
        f"* **Data drift** between consecutive test releases is mostly small (mean KS {ct['mean_ks'].min():.2f}–"
        f"{ct['mean_ks'].max():.2f}; classifier two-sample AUC {ct['c2st_auc'].min():.2f}–{ct['c2st_auc'].max():.2f}). Its "
        f"correlation with explanation stability, and that of train→test drift with test AUC, is "
        f"{'not significant in any test' if n_sig + n_sig_i == 0 else f'significant in {n_sig + n_sig_i} test(s)'} after Holm "
        "correction (small n: 14 and 24 pairs).",
    ]
    return pts
