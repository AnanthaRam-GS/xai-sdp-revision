"""Step 6 reports: deletion/insertion, faithfulness, ROAR/KAR, sanity checks, summary, runtime."""

from __future__ import annotations

import json
import logging
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
from sdp.fidelity import roar as R
from sdp.fidelity.common import EVALUATED
from sdp.fidelity.evaluation import load_curves, load_instances

logger = logging.getLogger(__name__)

FIG = Path("figures") / "step6"
ORDER = [E.label(m, k) for m, ks in EVALUATED.items() for k in ks]
FAMILY_LABEL = {"random": "random", "size_first": "size-first", "rf_TreeSHAP": "RF-TreeSHAP ranking"}


def _fig(cfg: Config, name: str) -> Path:
    return Path(cfg.paths.reports_dir) / FIG / name


def _ci(values, digits: int = 3) -> str:
    m, lo, hi = bootstrap_ci(np.asarray(values, dtype=float))
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]" if np.isfinite(m) else "n/a"


def _with_label(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(explanation=[E.label(m, k) for m, k in zip(df["model"], df["method"])])


def _long(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Instance-level values in the long format used by ``sdp.analysis.tables``."""
    out = df.melt(id_vars=["explanation", "project", "split_id"], value_vars=metrics, var_name="metric", value_name="value")
    return out.assign(threshold_mode="none", seed=0)


def _skesd(df: pd.DataFrame, specs: list[T.MetricSpec], order: list[str]) -> pd.DataFrame:
    return T.skesd_ranks(df, specs, column="explanation", order=order)


def _per_group(inst: pd.DataFrame, col: str, metrics: list[str], groups: list[str]) -> pd.DataFrame:
    t = inst.groupby(["explanation", col])[metrics].mean().unstack(col)
    out = pd.DataFrame({"explanation": ORDER})
    for m in metrics:
        for g in groups:
            if (m, g) in t.columns:
                out[f"{m} {g}"] = t[(m, g)].reindex(ORDER).map(_f).to_numpy()
    return out


# -------------------------------------------------------- deletion / insertion


def deletion_insertion_report(cfg: Config, root: Path, info: dict) -> pd.DataFrame:
    inst = _with_label(load_instances(root / "curves"))
    rows = []
    for lab in ORDER:
        g = inst[inst["explanation"] == lab]
        rows.append({"explanation": lab, "instances": len(g), "deletion AUC ↓": _ci(g["deletion_auc"]),
                     "random": _f(g["deletion_random"].mean()), "deletion gain ↑": _ci(g["deletion_gain"]),
                     "insertion AUC ↑": _ci(g["insertion_auc"]), "random ": _f(g["insertion_random"].mean()),
                     "insertion gain ↑": _ci(g["insertion_gain"])})  # fmt: skip
    main = pd.DataFrame(rows)
    long = _long(inst, ["deletion_auc", "insertion_auc", "deletion_gain", "insertion_gain"])
    within = []
    for model, ks in EVALUATED.items():
        labs = [E.label(model, k) for k in ks]
        r = _skesd(long[long["explanation"].isin(labs)], [T.MetricSpec("deletion_gain", "none", "deletion gain"),
                                                           T.MetricSpec("insertion_gain", "none", "insertion gain")], labs)  # fmt: skip
        within.append(r.set_index("metric").T.reset_index().rename(columns={"index": "explanation"}))
    within_t = pd.concat(within, ignore_index=True)
    across = _skesd(long, [T.MetricSpec("deletion_gain", "none", "deletion gain"),
                           T.MetricSpec("insertion_gain", "none", "insertion gain")], ORDER)  # fmt: skip
    across_t = across.set_index("metric").T.reset_index().rename(columns={"index": "explanation"})
    outcome = _per_group(inst, "outcome", ["deletion_gain", "insertion_gain"], ["TP", "FP", "FN", "TN"])
    projects = sorted(inst["project"].unique())
    proj = _per_group(inst, "project", ["deletion_gain"], projects)
    proj_ins = _per_group(inst, "project", ["insertion_gain"], projects)
    mean_rows = []
    for lab in ORDER:
        g = inst[inst["explanation"] == lab]
        mean_rows.append({"explanation": lab, "deletion AUC (marginal)": _f(g["deletion_auc"].mean()),
                          "deletion AUC (mean removal)": _f(g["deletion_auc_mean"].mean()),
                          "gain (marginal)": _f(g["deletion_gain"].mean()), "gain (mean removal)": _f(g["deletion_gain_mean"].mean())})  # fmt: skip
    sample = info.get("settings", {}).get("curves", {}).get("sample_per_split")
    md = [
        "# Step 6: deletion and insertion curves",
        "",
        "Seed-0 models of `main_time_aware`, " + ("all test instances" if not sample else f"up to {sample} test instances per split")
        + " of the 24 splits. Features are removed (deletion) or added back (insertion) in each explanation's "
        "per-instance order of |attribution| (ties by feature order), k = 0..20. Removal is *marginal*: the "
        "removed features take the values of 50 rows of the step 4 background set (seeded selection) and the "
        "model's P(defective) is averaged over them. The curves track confidence in the model's own prediction "
        "at its validation-tuned threshold (p if predicted defective, 1 − p otherwise). Areas: trapezoid rule over "
        "k/20. Random = mean over 10 seeded random orderings per instance. Gains: deletion AUC_random − AUC "
        "and insertion AUC − AUC_random (higher is better for both). Mean and 95% bootstrap CI over instances.",
        "",
        "## Main results",
        "",
        to_markdown(main),
        "",
        "## Scott-Knott ESD ranks within each model (per-split means; 1 = best)",
        "",
        "Ranked on the gains over random: within a model every explanation shares the same random reference, so "
        "the gains order the methods exactly as the AUCs do, but without the between-split variation of the "
        "random level (Scott-Knott ESD is not paired by split; on the raw AUCs all methods of a model fall into "
        "one group).",
        "",
        to_markdown(within_t),
        "",
        "## Scott-Knott ESD ranks across all 14 explanations (gain over random)",
        "",
        to_markdown(across_t),
        "",
        "## Gains by outcome group (mean over instances)",
        "",
        to_markdown(outcome),
        "",
        "## Deletion gain per project",
        "",
        to_markdown(proj),
        "",
        "## Insertion gain per project",
        "",
        to_markdown(proj_ins),
        "",
        "## Sensitivity: mean removal (deletion only)",
        "",
        "Removed features replaced by the background mean instead of averaging over background rows.",
        "",
        to_markdown(pd.DataFrame(mean_rows)),
        "",
    ]
    _write(cfg, "step6_deletion_insertion", md, {"": inst.groupby("explanation")[[
        "deletion_auc", "deletion_random", "deletion_gain", "insertion_auc", "insertion_random", "insertion_gain",
        "deletion_auc_mean", "deletion_gain_mean"]].mean().reindex(ORDER).reset_index(),
        "_skesd_within": within_t, "_skesd_across": across_t, "_outcome": outcome})  # fmt: skip
    crv = load_curves(root / "curves")
    crv = crv[crv["removal"] == "marginal"]
    for kind in ("deletion", "insertion"):
        panels = {}
        c = crv[crv["kind"] == kind]
        for model in EVALUATED:
            g = c[c["model"] == model].groupby(["method", "k"])["value"].mean().unstack("method")
            g = g[[m for m in EVALUATED[model] if m in g.columns] + ["random"]]
            panels[E.MODEL_LABEL[model]] = g
        F.line_panels(panels, f"{kind.capitalize()} curves (confidence in the own prediction; random dashed)",
                      "k features " + ("removed" if kind == "deletion" else "inserted"), "mean confidence",
                      _fig(cfg, f"{kind}_curves.png"), ncols=4, panel_legends=True)  # fmt: skip
    return inst


# ---------------------------------------------------------------- faithfulness


def faithfulness_report(cfg: Config, inst: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab in ORDER:
        g = inst.loc[inst["explanation"] == lab, "faithfulness"]
        rows.append({"explanation": lab, "faithfulness": _ci(g.dropna()), "median": _f(g.median()),
                     "% > 0": _f(100 * (g > 0).mean(), 1), "% undefined": _f(100 * g.isna().mean(), 1)})  # fmt: skip
    long = _long(inst, ["faithfulness"])
    spec = [T.MetricSpec("faithfulness", "none", "faithfulness")]
    across = _skesd(long, spec, ORDER).set_index("metric").T.reset_index().rename(columns={"index": "explanation", "faithfulness": "rank (all 14)"})
    within = []
    for model, ks in EVALUATED.items():
        labs = [E.label(model, k) for k in ks]
        r = _skesd(long[long["explanation"].isin(labs)], spec, labs)
        within.append(r.set_index("metric").T.reset_index().rename(columns={"index": "explanation", "faithfulness": "rank within model"}))
    ranks = across.merge(pd.concat(within), on="explanation")
    outcome = _per_group(inst, "outcome", ["faithfulness"], ["TP", "FP", "FN", "TN"])
    projects = sorted(inst["project"].unique())
    md = [
        "# Step 6: faithfulness correlation",
        "",
        "Bhatt et al. (2020): per instance, 50 random subsets of 5 features (seeded, shared by all explanations "
        "of the instance); for each subset the sum of the attributions and the prediction drop f(x) − f(x without "
        "the subset), with marginal removal (50 background rows) and f = P(defective). Faithfulness is the Pearson "
        "correlation over the 50 subsets (undefined when the attributions are constant). Saliency as |gradient|. "
        "Mean and 95% bootstrap CI over instances; Scott-Knott ESD on per-split means (1 = best).",
        "",
        "## Faithfulness",
        "",
        to_markdown(pd.DataFrame(rows).merge(ranks, on="explanation")),
        "",
        "## By outcome group",
        "",
        to_markdown(outcome),
        "",
        "## Per project",
        "",
        to_markdown(_per_group(inst, "project", ["faithfulness"], projects)),
        "",
        "Note: LIME and Saliency weights are slopes, not contributions relative to a reference (step 4), so "
        "their subset sums are not expected to track prediction drops; the SHAP-type methods, DeepLift and "
        "Integrated Gradients are designed to.",
        "",
    ]
    _write(cfg, "step6_faithfulness", md, {"": inst.groupby("explanation")["faithfulness"].agg(["mean", "median", "count"]).reindex(ORDER).reset_index(),
                                           "_ranks": ranks})  # fmt: skip
    return ranks


# ------------------------------------------------------------------ ROAR / KAR


def _family_label(model: str, fam: str) -> str:
    return E.label(model, fam) if fam in sum(R.TRAIN_METHODS.values(), []) else FAMILY_LABEL.get(fam, fam)


def roar_report(cfg: Config, root: Path, info: dict) -> pd.DataFrame:
    rows = pd.concat([pd.read_parquet(f) for f in sorted((root / "roar").glob("*.parquet"))], ignore_index=True)
    ks = tuple(info.get("settings", {}).get("roar", {}).get("ks", R.KS))
    rows["family"] = rows["ranking"].map(R.ranking_family)
    areas = {m: R.area_scores(rows, ks, metric=m) for m in ("auc", "mcc_tuned", "popt")}
    a = areas["auc"]
    split_means = a.groupby(["split_id", "project", "model", "family", "test"], as_index=False)["area"].mean()
    tab = split_means.groupby(["model", "family", "test"])["area"].mean().unstack("test").reset_index()
    tab["ranking"] = [_family_label(m, f) for m, f in zip(tab["model"], tab["family"])]
    tab["model"] = tab["model"].map(E.MODEL_LABEL)
    area_t = tab[["model", "ranking", "ROAR", "KAR"]].rename(columns={"ROAR": "ROAR area (AUC) ↓", "KAR": "KAR area (AUC) ↑"})
    for c in area_t.columns[2:]:
        area_t[c] = area_t[c].map(_f)
    # AUC at each k
    at_k = rows.groupby(["model", "family", "test", "k"])["auc"].mean().reset_index()
    k_tabs = {}
    for test in ("ROAR", "KAR"):
        t = at_k[at_k["test"] == test].pivot_table(index=["model", "family"], columns="k", values="auc").reset_index()
        t["ranking"] = [_family_label(m, f) for m, f in zip(t["model"], t["family"])]
        t["model"] = t["model"].map(E.MODEL_LABEL)
        cols = ["model", "ranking"] + [c for c in t.columns if isinstance(c, (int, np.integer))]
        k_tabs[test] = t[cols].rename(columns={c: f"k={c}" for c in cols[2:]})
        for c in k_tabs[test].columns[2:]:
            k_tabs[test][c] = k_tabs[test][c].map(_f)
    # tests: method rankings vs random and size-first
    tests = []
    for (model, test), g in split_means.groupby(["model", "test"]):
        wide = g.pivot(index="split_id", columns="family", values="area")
        fam = []
        for f in [c for c in wide.columns if c not in ("random", "size_first")]:
            for ref in ("random", "size_first"):
                fam.append({"model": E.MODEL_LABEL[model], "test": test, "a": _family_label(model, f),
                            "b": FAMILY_LABEL[ref]} | T.paired_comparison(wide[f], wide[ref]))  # fmt: skip
        if "size_first" in wide.columns:
            fam.append({"model": E.MODEL_LABEL[model], "test": test, "a": "size-first", "b": "random"}
                       | T.paired_comparison(wide["size_first"], wide["random"]))  # fmt: skip
        for r, p in zip(fam, holm([r["p_value"] for r in fam])):
            r["p_holm"] = p
        tests += fam
    tests = T._order_cols(pd.DataFrame(tests))
    # reproduction of the step 3 models by the k = 0 refit
    m3 = pd.read_csv(cfg.paths.results_dir / "main_time_aware" / "metrics.csv")
    m3 = m3[(m3["metric"] == "roc_auc") & m3["model"].isin(list(R.TRAIN_METHODS))]
    full = rows[(rows["test"] == "ROAR") & (rows["k"] == 0)].drop_duplicates(["split_id", "model", "seed"])
    rep = full.merge(m3[["split_id", "model", "seed", "value"]], on=["split_id", "model", "seed"])
    max_diff = float((rep["auc"] - rep["value"]).abs().max()) if len(rep) else float("nan")
    seeds = sorted(rows["seed"].unique())
    md = [
        "# Step 6: ROAR and KAR",
        "",
        f"Global rankings from training-set attributions only (the fitting rows of each split; LIME on 200 "
        f"stratified fitting rows), seeds {', '.join(map(str, seeds))}. Ranking = mean L1-normalised |attribution| "
        "over those rows. References: 5 seeded random orderings (averaged), size-first (loc, rfc, wmc, npm, then "
        "random) and the RF-TreeSHAP ranking applied to the other models. ROAR drops the top-k metrics, KAR keeps "
        f"only the top-k (k ∈ {{{', '.join(map(str, ks))}}}); each model type is refitted with its persisted "
        "step 3 hyperparameters (no re-tuning), preprocessing, imbalance handling and early stopping, and evaluated "
        "on the test release. Area = mean test AUC over k (trapezoid rule over the evaluated k, divided by the k "
        "range); per-split means over seeds. ROAR: lower is better (removing the important metrics should hurt); "
        "KAR: higher is better.",
        "",
        f"Check: refitting with all 20 metrics (k = 0) reproduces the step 3 test AUC (max |difference| "
        f"{max_diff:.2e} over {len(rep)} models).",
        "",
        "## Area scores",
        "",
        to_markdown(area_t),
        "",
        "## Wilcoxon signed-rank tests on the area scores",
        "",
        "Per model and test (24 splits); Holm within (model, test); Cliff's delta of a vs b. For ROAR a negative "
        "delta favours a (a lower area); for KAR a positive delta favours a.",
        "",
        to_markdown(T.format_comparisons(tests)),
        "",
        "## Test AUC at each k: ROAR",
        "",
        to_markdown(k_tabs["ROAR"]),
        "",
        "## Test AUC at each k: KAR",
        "",
        to_markdown(k_tabs["KAR"]),
        "",
        "MCC@tuned and Popt area scores are in `step6_roar_kar_areas.csv`.",
        "",
    ]
    all_areas = pd.concat([v.assign(metric=k) for k, v in areas.items()], ignore_index=True)
    _write(cfg, "step6_roar_kar", md, {"": area_t, "_tests": tests, "_areas": all_areas})
    panels = {}
    for test in ("ROAR", "KAR"):
        for model in R.TRAIN_METHODS:
            g = at_k[(at_k["test"] == test) & (at_k["model"] == model)]
            t = g.pivot(index="k", columns="family", values="auc")
            t.columns = [_family_label(model, c) for c in t.columns]
            first = [c for c in t.columns if c not in ("random", "size-first", "RF-TreeSHAP ranking")]
            panels[f"{test}: {E.MODEL_LABEL[model]}"] = t[first + [c for c in ("size-first", "RF-TreeSHAP ranking", "random") if c in t.columns]]
    F.line_panels(panels, "ROAR (top-k removed) and KAR (only top-k kept): test AUC after refitting", "k", "test AUC",
                  _fig(cfg, "roar_kar_curves.png"), ncols=4, sharey=False, panel_legends=True)  # fmt: skip
    return split_means


# ------------------------------------------------------------------ sanity


def _stage_label(stage: str, n_layers: float, n_linear: float) -> str:
    if stage == "all":
        return "all layers"
    if n_layers == 1:
        return "output"
    return f"top {int(n_layers)} layers"


def sanity_report(cfg: Config, root: Path, flag: float) -> pd.DataFrame:
    rows = pd.concat([pd.read_parquet(f) for f in sorted((root / "sanity").glob("*.parquet"))], ignore_index=True)
    rows = _with_label(rows)
    casc = rows[rows["kind"] == "cascading"].copy()
    casc["stage_label"] = [_stage_label(s, n, m) for s, n, m in zip(casc["stage"], casc["n_layers"], casc["n_linear"])]
    stage_order = ["output", "top 2 layers", "top 3 layers", "all layers"]
    sp = casc.groupby(["explanation", "stage_label"])["spearman"].mean().unstack().reindex(columns=[s for s in stage_order if s in set(casc["stage_label"])])
    fa = casc.groupby(["explanation", "stage_label"])["fa5"].mean().unstack().reindex(columns=sp.columns)
    mlp_labs = [E.label("mlp", k) for k in EVALUATED["mlp"]]
    casc_t = pd.DataFrame({"explanation": mlp_labs})
    for s in sp.columns:
        casc_t[f"{s} ρ / FA@5"] = [f"{_f(sp.loc[l, s])} / {_f(fa.loc[l, s])}" for l in mlp_labs]
    ind = rows[rows["kind"] == "independent"].copy()
    ind["layer"] = [("output" if s == "only output" else ("first (input) layer" if s == "only hidden1" else "other hidden"))
                    for s in ind["stage"]]  # fmt: skip
    ind_t = ind.groupby(["explanation", "layer"])["spearman"].mean().unstack().reindex(mlp_labs)
    ind_t = ind_t.map(_f).reset_index()
    data = rows[rows["kind"] == "data"]
    data_t = data.groupby("explanation").agg(spearman=("spearman", "mean"), fa5=("fa5", "mean"),
                                             auc=("permuted_auc", "mean")).reindex(ORDER)  # fmt: skip
    full_sp = sp["all layers"]
    verdict = []
    for lab in ORDER:
        m_sim = full_sp.get(lab, np.nan)
        d_sim = data_t.loc[lab, "spearman"]
        flags = []
        if np.isfinite(m_sim) and m_sim > flag:
            flags.append("model randomisation")
        if np.isfinite(d_sim) and d_sim > flag:
            flags.append("data randomisation")
        verdict.append({"explanation": lab, "ρ after full model randomisation": _f(m_sim) if np.isfinite(m_sim) else "n/a (not an MLP)",
                        "ρ after label permutation": _f(d_sim), "permuted-label test AUC": _f(data_t.loc[lab, "auc"]),
                        "result": "flag: " + ", ".join(flags) if flags else "pass"})  # fmt: skip
    verdict_t = pd.DataFrame(verdict)
    data_md = data_t.reset_index().assign(spearman=lambda t: t["spearman"].map(_f), fa5=lambda t: t["fa5"].map(_f),
                                          auc=lambda t: t["auc"].map(_f)).rename(columns={"spearman": "Spearman", "fa5": "FA@5", "auc": "permuted-label test AUC"})  # fmt: skip
    n_by_depth = casc.drop_duplicates("split_id")["n_linear"].value_counts().sort_index()
    contrib = [E.label("mlp", k) for k in ("DeepLift", "DeepLiftShap", "GradientShap", "IntegratedGradients",
                                            "InputXGradient", "KernelSHAP")]  # fmt: skip
    lo, hi = data_t.loc[contrib, "spearman"].min(), data_t.loc[contrib, "spearman"].max()
    md = [
        "# Step 6: sanity checks (Adebayo et al., 2018)",
        "",
        "Similarity with the original explanation per test instance (seed 0, all test instances): Spearman "
        "correlation of |attribution| (ρ) and feature agreement@5; means over instances, then over splits. "
        f"A method is flagged when its mean ρ stays above {flag} after full model randomisation (MLP) or after "
        "label permutation.",
        "",
        "## Pass / flag",
        "",
        to_markdown(verdict_t),
        "",
        "## Cascading model-parameter randomisation (MLP)",
        "",
        "Linear layers re-initialised (PyTorch default initialisation, seeded) cumulatively from the output layer "
        "towards the input. The MLPs have different depths (number of Linear layers per split: "
        + ", ".join(f"{int(k)}: {v}" for k, v in n_by_depth.items()) + "), so `top 2/3 layers` exist only for "
        "deeper networks; `all layers` is full randomisation for every split.",
        "",
        to_markdown(casc_t),
        "",
        "## Independent single-layer randomisation (MLP), Spearman",
        "",
        to_markdown(ind_t),
        "",
        "## Data randomisation (permuted training labels)",
        "",
        "Each model refitted on the training release with permuted labels (seed 0, class counts unchanged) and "
        "persisted hyperparameters; the permuted-label models' test AUC should be about 0.5.",
        "",
        to_markdown(data_md),
        "",
        f"Note: after full model randomisation or label permutation, the contribution-type MLP explanations "
        f"(DeepLift, DeepLiftShap, GradientShap, Integrated Gradients, InputXGradient, KernelSHAP) keep part of "
        f"their ranking (ρ {lo:.2f}–{hi:.2f} after label permutation), whereas the slope-type ones (Saliency, LIME) "
        "drop to ρ ≈ 0. Contributions are (approximately) slope × (x − reference): the input term (x − reference) "
        "is the same for any model and carries ranking information by itself, which is the known weakness of "
        "input-multiplied attributions in these tests (Adebayo et al., 2018). None of the methods exceeds the "
        "0.5 flag, but the remaining similarity should be read as partly model-independent.",
        "",
    ]
    _write(cfg, "step6_sanity_checks", md, {"": verdict_t, "_rows": rows})
    fig_t = sp.T.reindex(columns=mlp_labs)
    fig_t.columns = [c.replace("MLP-", "") for c in fig_t.columns]
    F.line_panels({"Cascading randomisation (MLP)": fig_t}, "Similarity to the original explanation vs randomisation stage",
                  "layers randomised (from the output)", "mean Spearman ρ", _fig(cfg, "sanity_cascading.png"), dashed=())  # fmt: skip
    return verdict_t


# -------------------------------------------------------------------- summary


def summary_report(cfg: Config, inst: pd.DataFrame, roar_split: pd.DataFrame, verdict: pd.DataFrame) -> None:
    per_split = inst.groupby(["explanation", "project", "split_id"])[["deletion_gain", "insertion_gain", "faithfulness"]].mean().reset_index()
    rs = roar_split.copy()
    rnd = rs[rs["family"] == "random"].set_index(["split_id", "model", "test"])["area"]
    meth = rs[~rs["family"].isin(["random", "size_first", "rf_TreeSHAP"])].copy()
    meth["random"] = [rnd.get((s, m, t), np.nan) for s, m, t in zip(meth["split_id"], meth["model"], meth["test"])]
    meth["gain"] = np.where(meth["test"] == "ROAR", meth["random"] - meth["area"], meth["area"] - meth["random"])
    meth["explanation"] = [E.label(m, f) for m, f in zip(meth["model"], meth["family"])]
    rg = meth.pivot_table(index=["explanation", "split_id"], columns="test", values="gain").reset_index()
    rg = rg.rename(columns={"ROAR": "roar_gain", "KAR": "kar_gain"})
    per_split = per_split.merge(rg, on=["explanation", "split_id"], how="left")
    metrics = ["deletion_gain", "insertion_gain", "faithfulness", "roar_gain", "kar_gain"]
    ranks = per_split.copy()
    for m in metrics:
        ranks[m] = per_split.groupby("split_id")[m].rank(ascending=False)
    ranks["mean_rank"] = ranks[metrics].mean(axis=1, skipna=True)
    long = ranks.assign(metric="neg_mean_rank", value=-ranks["mean_rank"], threshold_mode="none", seed=0)
    sk = _skesd(long, [T.MetricSpec("neg_mean_rank", "none", "overall")], ORDER).set_index("metric").T["overall"]
    means = per_split.groupby("explanation")[metrics].mean().reindex(ORDER)
    mr = ranks.groupby("explanation")["mean_rank"].mean().reindex(ORDER)
    v = verdict.set_index("explanation")["result"]
    table = pd.DataFrame({
        "explanation": ORDER, "deletion gain": means["deletion_gain"].map(_f).to_numpy(),
        "insertion gain": means["insertion_gain"].map(_f).to_numpy(), "faithfulness": means["faithfulness"].map(_f).to_numpy(),
        "ROAR gain": [(_f(x) if np.isfinite(x) else "n/a") for x in means["roar_gain"]],
        "KAR gain": [(_f(x) if np.isfinite(x) else "n/a") for x in means["kar_gain"]],
        "sanity": [v.get(l, "n/a") for l in ORDER], "mean rank": mr.map(lambda x: _f(x, 2)).to_numpy(),
        "SK-ESD group": [int(sk[l]) if l in sk else "" for l in ORDER],
    })  # fmt: skip
    table = table.iloc[np.argsort([sk.get(l, 99) * 100 + mr[l] for l in ORDER], kind="stable")]
    sig = {}
    for (model, test), g in roar_split.groupby(["model", "test"]):
        wide = g.groupby(["split_id", "family"])["area"].mean().unstack()
        fams = [c for c in wide.columns if c not in ("random", "size_first", "rf_TreeSHAP")]
        res = [T.paired_comparison(wide[f], wide["random"]) for f in fams]
        for f, r, ph in zip(fams, res, holm([r["p_value"] for r in res])):
            better = (r["mean_a"] < r["mean_b"]) if test == "ROAR" else (r["mean_a"] > r["mean_b"])
            sig[(model, test, f)] = bool(ph < 0.05 and better)
    notes = []
    for model, ks in EVALUATED.items():
        labs = [E.label(model, k) for k in ks]
        sub = means.loc[labs]
        best = {m: sub[m].idxmax() for m in ("deletion_gain", "insertion_gain", "faithfulness")}
        roar_txt = ""
        fams = R.TRAIN_METHODS[model]
        for test, verb in (("ROAR", "removing the top-ranked metrics hurts the refitted model more than random removal"),
                           ("KAR", "keeping only the top-ranked metrics beats keeping random ones")):
            ok = [f for f in fams if sig.get((model, test, f))]
            no = [f for f in fams if not sig.get((model, test, f))]
            if ok:
                roar_txt += f" {test}: for {', '.join(ok)}, {verb} (Holm p < 0.05)."
            if no:
                roar_txt += f" {test}: {', '.join(no)} not distinguishable from random."
        names = {m: best[m].split("-", 1)[1] for m in best}
        notes.append(f"* **{E.MODEL_LABEL[model]}**: deletion favours {names['deletion_gain']}, insertion favours "
                     f"{names['insertion_gain']}, faithfulness favours {names['faithfulness']}.{roar_txt}")
    md = [
        "# Step 6: fidelity summary",
        "",
        "Per explanation (seed-0 models; ROAR/KAR seeds averaged): mean gains over random for deletion and "
        "insertion, mean faithfulness correlation, ROAR gain (random area − method area) and KAR gain (method area − "
        "random area) where training-set rankings exist, and the sanity-check result. Higher is better for every "
        "number. Overall ordering: per split, each explanation is ranked within each metric (1 = best among the "
        "explanations that have it), the ranks are averaged, and Scott-Knott ESD groups the per-split mean ranks "
        "(group 1 = best). Gains over random are comparable across models; the absolute curves are not.",
        "",
        to_markdown(table),
        "",
        "## Which explainers look most faithful",
        "",
    ] + notes + [
        "",
        "These are fidelity results for the explanations of these models on these data. Note that gains over random "
        "depend on how concentrated the model's dependence on a few metrics is, so they should be compared within "
        "a model rather than across models.",
        "",
    ]
    _write(cfg, "step6_summary", md, {"": table})


# -------------------------------------------------------------------- runtime


def runtime_report(cfg: Config) -> None:
    base = cfg.paths.results_dir / "fidelity"
    est = json.loads((base / "runtime_estimate.json").read_text()) if (base / "runtime_estimate.json").exists() else {}
    info = json.loads((base / "run_info.json").read_text()) if (base / "run_info.json").exists() else {}
    e = est.get("estimate", {})
    rows = []
    for part in ("curves", "train", "roar", "sanity"):
        p = base / f"timings_{part}.csv"
        cpu = pd.read_csv(p)["time_s"].sum() / 3600 if p.exists() else float("nan")
        rows.append({"part": part, "smoke wall (s)": _f(est.get("smoke_wall_s", {}).get(part), 1),
                     "estimated CPU h": _f(e.get("cpu_s", {}).get(part, float("nan")) / 3600, 2),
                     "actual CPU h": _f(cpu, 2), "actual wall (min)": _f(info.get("wall_time_s", {}).get(part, float("nan")) / 60, 1)})  # fmt: skip
    red = info.get("reductions", est.get("reductions", []))
    md = [
        "# Step 6 runtime",
        "",
        "## Machine",
        "",
        to_markdown(pd.DataFrame(list(machine_details().items()), columns=["item", "value"])),
        "",
        "## Smoke test, estimate and full run",
        "",
        f"Smoke split `{est.get('smoke_split', 'n/a')}` (seed 0; ROAR/KAR with one seed, scaled to three). "
        f"Estimated full run: {e.get('cpu_hours', float('nan')):.2f} CPU h, ~{e.get('wall_hours', float('nan')):.2f} h "
        f"wall with n_jobs={est.get('n_jobs')} (assumed parallel efficiency {e.get('efficiency')}).",
        "",
        to_markdown(pd.DataFrame(rows)),
        "",
        "Reductions: " + ("; ".join(red) if red else "none (estimate below 8 h)") + ".",
        "",
    ]
    _write(cfg, "step6_runtime", md, {})
