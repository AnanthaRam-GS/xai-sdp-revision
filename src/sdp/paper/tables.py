"""LaTeX tables for the paper (step 8), written to ``paper/tables/``.

Each file holds only the ``tabular`` part (booktabs, no vertical rules, ``\\small``),
preceded by a comment with the suggested caption and the source report, ready to
``\\input`` inside a ``table`` or ``table*`` environment. Numbers have three decimals
(two for percentages); confidence intervals are printed as [lo, hi] in a smaller font;
the best value per column is bold where that is meaningful. Inputs: ``reports/``,
``paper/data/`` and the configuration files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.config import CONFIG_DIR, REPO_ROOT

logger = logging.getLogger(__name__)

MODELS = ["mlp", "lr", "rf", "xgb", "manual_up", "manual_down", "all_defective", "majority"]
MODEL_NAME = {"mlp": "MLP", "lr": "LR", "rf": "RF", "xgb": "XGB", "manual_up": "ManualUp", "manual_down": "ManualDown",
              "all_defective": "All defective", "majority": "Majority"}  # fmt: skip
PRIMARY = ["MLP-DeepLiftShap", "MLP-KernelSHAP", "MLP-IntegratedGradients", "MLP-LIME", "MLP-Saliency",
           "LR-LinearSHAP", "RF-TreeSHAP", "XGB-TreeSHAP"]  # fmt: skip


def esc(text) -> str:
    s = str(text)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
                 ("{", r"\{"), ("}", r"\}"), ("→", r"$\rightarrow$"), ("ρ", r"$\rho$"), ("≥", r"$\geq$"),
                 ("≤", r"$\leq$"), ("×", r"$\times$"), ("−", "--"), ("–", "--"), ("√", r"$\surd$"), ("·", r"$\cdot$"),
                 ("Δ", r"$\Delta$"), ("κ", r"$\kappa$"), ("τ", r"$\tau$"), ("<", r"$<$"), (">", r"$>$")):  # fmt: skip
        s = s.replace(a, b)
    return s


def num(x, d: int = 3) -> str:
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def ci(m, lo, hi, d: int = 3) -> str:
    return f"{num(m, d)}\\,{{\\scriptsize[{num(lo, d)}, {num(hi, d)}]}}"


def bold(cell: str) -> str:
    head, sep, tail = cell.partition("\\,")
    return f"\\textbf{{{head}}}{sep}{tail}"


def tabular(df: pd.DataFrame, align: str, caption: str, source: str, header: list[str] | None = None) -> str:
    """The tabular part of a booktabs table (cells already formatted LaTeX)."""
    head = " & ".join(header or [esc(c) for c in df.columns]) + r" \\"
    body = [" & ".join(str(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    return "\n".join([f"% Suggested caption: {caption}", f"% Source: {source}", r"\small",
                      f"\\begin{{tabular}}{{{align}}}", r"\toprule", head, r"\midrule", *body, r"\bottomrule",
                      r"\end{tabular}", ""])  # fmt: skip


def _r(reports: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(reports) / f"{name}.csv")


def _summary_cells(t: pd.DataFrame, rows: list[str], row_col: str, metrics: list[tuple[str, bool]],
                   ranks: pd.DataFrame | None = None) -> pd.DataFrame:  # fmt: skip
    """Mean [CI] per row and metric; best (by mean) bold; optional SK-ESD rank superscripts."""
    out = pd.DataFrame({"": [esc(MODEL_NAME.get(r, r)) for r in rows]})
    for metric, higher in metrics:
        g = t[t["metric"] == metric].set_index(row_col)
        cells, means = [], []
        for r in rows:
            if r in g.index:
                x = g.loc[r]
                c = ci(x["mean"], x["ci_low"], x["ci_high"])
                if ranks is not None and metric in ranks.index and r in ranks.columns:
                    c = c.replace("\\,", f"$^{{{int(ranks.loc[metric, r])}}}$\\,", 1)
                cells.append(c)
                means.append(x["mean"])
            else:
                cells.append("--")
                means.append(np.nan)
        learned = [i for i, r in enumerate(rows) if r in ("mlp", "lr", "rf", "xgb") and np.isfinite(means[i])]
        if learned:  # bold: the best learned model (trivial baselines can win e.g. recall)
            vals = [means[i] for i in learned]
            best = learned[int(np.argmax(vals) if higher else np.argmin(vals))]
            cells[best] = bold(cells[best])
        out[esc(metric)] = cells
    return out


# ------------------------------------------------------------------ tables


def tab_dataset(reports: Path, data: Path) -> str:
    d = _r(reports, "study_dataset")
    pairs = _r(reports, "study_pairs").groupby("project").size()
    rows = []
    for p, g in d.groupby("project", sort=True):
        inc = g[g["included"]]
        exc = g[~g["included"]]
        rows.append({"Project": esc(p), "Releases": esc(", ".join(inc["version"].astype(str))) or "--",
                     "Classes": f"{int(inc['final_rows'].sum())}",
                     "Defective (\\%)": num(100 * inc["defective_count"].sum() / inc["final_rows"].sum(), 2) if len(inc) else "--",
                     "Pairs": str(int(pairs.get(p, 0))),
                     "Excluded (reason)": esc("; ".join(f"{v} ({r})" for v, r in zip(exc["version"].astype(str), exc["reason"])) or "--")})  # fmt: skip
    t = pd.DataFrame(rows)
    inc = d[d["included"]]
    t.loc[len(t)] = {"Project": r"\textit{Total}", "Releases": f"{len(inc)}", "Classes": f"{int(inc['final_rows'].sum())}",
                     "Defective (\\%)": num(100 * inc["defective_count"].sum() / inc["final_rows"].sum(), 2),
                     "Pairs": str(int(pairs.sum())), "Excluded (reason)": f"{int((~d['included']).sum())} releases"}  # fmt: skip
    return tabular(t, "llrrrp{5.2cm}", "Study dataset after cleaning and the inclusion rule (defective ratio between 5% and 75%, at least three releases per project); time-aware pairs train on release n and test on n+1.",
                   "reports/study_dataset.md", list(t.columns))  # fmt: skip


def tab_explainer_settings(reports: Path, data: Path) -> str:
    from sdp.explain.runner import load_explain_settings

    es = load_explain_settings(CONFIG_DIR / "explain.yaml")
    p = es.method_params
    bg = f"{es.background_size} training rows, stratified"
    rows = [
        ("DeepLift", "MLP", bg, "background mean", "--", "0--9"),
        ("DeepLiftShap", "MLP", bg, "each background row", "--", "0--9"),
        ("GradientShap", "MLP", bg, "background rows", f"{p('GradientShap')['n_samples']} samples, $\\sigma = {p('GradientShap')['stdevs']}$", "0--9"),
        ("IntegratedGradients", "MLP", bg, "background mean", f"{p('IntegratedGradients')['n_steps']} Gauss--Legendre steps", "0--9"),
        ("Saliency", "MLP", "--", "--", "signed gradient", "0--9"),
        ("InputXGradient", "MLP", "--", "--", "input $\\times$ gradient", "0--9"),
        ("KernelSHAP", "MLP", bg, "background rows", f"{p('KernelSHAP')['nsamples']} samples, no $\\ell_1$ selection", "0--2"),
        ("LIME", "all", "training data (statistics)", "--",
         f"{p('LIME')['num_samples']} samples, kernel width $0.75\\sqrt{{20}}$, around the instance, no discretisation", "0--2"),
        ("TreeSHAP", "RF, XGB", bg, "background rows (interventional)", "probability output", "0--9"),
        ("LinearSHAP", "LR", bg, "background rows (interventional)", "log-odds output", "0--9"),
    ]  # fmt: skip
    t = pd.DataFrame(rows, columns=["Explainer", "Models", "Background", "Baseline / reference", "Settings", "Seeds"])
    return tabular(t, "llllp{5cm}l", "Explainer configurations (step 4 v2). All explain P(defective) in the model's input space except LinearSHAP (log-odds).",
                   "configs/explain.yaml, reports/step4_global_importance.md", list(t.columns))  # fmt: skip


PERF = [("AUC", True), ("PR-AUC", True), ("MCC@tuned", True), ("F1@tuned", True), ("Recall@tuned", True),
        ("Brier", False), ("Popt", True), ("Recall@20%LOC", True), ("NECM15@tuned", False)]  # fmt: skip


def _perf(reports: Path, name: str, skesd: str) -> pd.DataFrame:
    t = _r(reports, name)
    ranks = _r(reports, skesd).set_index("metric")
    return _summary_cells(t, [m for m in MODELS if m in set(t["model"])], "model", PERF, ranks)


def tab_performance(reports: Path, data: Path) -> str:
    t = _perf(reports, "step3_main_time_aware", "step3_main_time_aware_skesd")
    t = t.set_index("").T.reset_index().rename(columns={"index": "Metric"})
    return tabular(t, "l" + "r" * (t.shape[1] - 1), "RQ1: time-aware performance (24 splits $\\times$ 10 seeds); mean [95% bootstrap CI]; superscripts: Scott-Knott ESD group (1 = best); bold: best learned model.",
                   "reports/step3_main_time_aware.md", ["Metric"] + [esc(c) for c in t.columns[1:]])  # fmt: skip


def tab_within_release(reports: Path, data: Path) -> str:
    t = _perf(reports, "step3_within_release", "step3_within_release_skesd")
    t = t.set_index("").T.reset_index().rename(columns={"index": "Metric"})
    return tabular(t, "l" + "r" * (t.shape[1] - 1), "Within-release performance (10 $\\times$ 5-fold CV); mean [95% bootstrap CI]; superscripts: Scott-Knott ESD group.",
                   "reports/step3_within_release.md", ["Metric"] + [esc(c) for c in t.columns[1:]])  # fmt: skip


def tab_ablation(reports: Path, data: Path) -> str:
    rows = []
    shares = _r(reports, "step5_preprocessing_explanations_size_share")
    shares = shares[shares["scope"] == "ALL"]
    for factor, name in (("preprocessing", "step3_ablation_preprocessing"), ("imbalance", "step3_ablation_imbalance")):
        t = _r(reports, name)
        for model in t["model"].unique():
            for setting in t[t["model"] == model][factor].unique():
                g = t[(t["model"] == model) & (t[factor] == setting)].set_index("metric")
                row = {"Ablation": factor, "Model": MODEL_NAME[model], "Setting": esc(setting)}
                for m in ("AUC", "MCC@tuned", "Popt"):
                    row[m] = ci(g.loc[m, "mean"], g.loc[m, "ci_low"], g.loc[m, "ci_high"])
                method = {"mlp": "DeepLiftShap", "lr": "LinearSHAP"}.get(model)
                sh = shares[(shares["model"] == model) & (shares["method"] == method) & (shares["preprocessing"] == setting)] \
                    if factor == "preprocessing" and method else pd.DataFrame()  # fmt: skip
                row["LOC share"] = num(sh["loc_share"].iloc[0]) if len(sh) else "--"
                row["Size share"] = num(sh["size_group_share"].iloc[0]) if len(sh) else "--"
                rows.append(row)
    t = pd.DataFrame(rows)
    return tabular(t, "lllrrrrr", "Preprocessing and imbalance ablations (time-aware): mean [95% CI] of AUC, MCC@tuned and Popt, and the share of normalised |attribution| on LOC and on the size group (MLP: DeepLiftShap, LR: LinearSHAP).",
                   "reports/step3_ablation_preprocessing.md, reports/step3_ablation_imbalance.md, reports/step5_preprocessing_explanations.md",
                   list(t.columns))  # fmt: skip


def tab_importance(reports: Path, data: Path) -> str:
    imp = _r(reports, "step4_global_importance")
    imp = imp[(imp["group"] == "all") & (imp["scope"] == "ALL")]
    imp["explanation"] = [f"{m.upper()}-{k}" for m, k in zip(imp["model"], imp["method"])]
    ss = _r(reports, "step4_size_share")
    ss = ss[ss["scope"] == "ALL"].set_index("explanation")
    rows = []
    for e in PRIMARY:
        g = imp[imp["explanation"] == e].sort_values("importance", ascending=False).head(5)
        rows.append({"Explanation": esc(e), "Top-5 metrics (mean normalised |attribution|)":
                     ", ".join(f"{f.upper().replace('_', chr(92) + '_')} ({v:.3f})" for f, v in zip(g["feature"], g["importance"])),
                     "LOC share": num(ss.loc[e, "loc_share"]), "Size share": num(ss.loc[e, "size_group_share"]),
                     "LOC top-1 (\\%)": num(ss.loc[e, "loc_top1_pct"], 2)})  # fmt: skip
    t = pd.DataFrame(rows)
    return tabular(t, "lp{7.2cm}rrr", "Global importance (all test instances, ALL splits) for the primary explanations: top-5 metrics, share of attribution on LOC and on the size group, and % of instances with LOC as the top metric.",
                   "reports/step4_global_importance.md, reports/step4_size_share.md", list(t.columns))  # fmt: skip


def tab_agreement(reports: Path, data: Path) -> str:
    ag = _r(reports, "step4_agreement")
    ag = ag[ag["scope"] == "ALL"]
    pairs = [("MLP-DeepLift", "MLP-DeepLiftShap"), ("MLP-DeepLiftShap", "MLP-KernelSHAP"), ("MLP-DeepLift", "MLP-IntegratedGradients"),
             ("MLP-DeepLiftShap", "MLP-LIME"), ("MLP-DeepLiftShap", "MLP-Saliency"), ("MLP-Saliency", "MLP-LIME"),
             ("MLP-DeepLiftShap", "RF-TreeSHAP"), ("MLP-DeepLiftShap", "LR-LinearSHAP"), ("RF-TreeSHAP", "XGB-TreeSHAP"),
             ("XGB-TreeSHAP", "LR-LinearSHAP"), ("RF-LIME", "XGB-LIME"), ("LR-LIME", "RF-LIME")]  # fmt: skip
    rows = []
    for a, b in pairs:
        g = ag[((ag["a"] == a) & (ag["b"] == b)) | ((ag["a"] == b) & (ag["b"] == a))].set_index("metric")
        rows.append({"Pair": esc(f"{a} vs {b}")} | {lab: ci(g.loc[m, "mean"], g.loc[m, "ci_low"], g.loc[m, "ci_high"])
                                                    for m, lab in (("feature_agreement@5", "FA@5"), ("sign_agreement@5", "SA@5"),
                                                                   ("spearman", "Spearman"))})  # fmt: skip
    t = pd.DataFrame(rows)
    return tabular(t, "lrrr", "Explainer agreement for key pairs (seeds 0--2, mean over instances [95% CI]); DeepLift and DeepLiftShap share propagation rules.",
                   "reports/step4_agreement.md", list(t.columns))  # fmt: skip


def tab_robustness(reports: Path, data: Path) -> str:
    lime = _r(reports, "step5_lime_stability")
    rr = lime[lime["table"] == "run_to_run"].set_index("model")
    grid = lime[lime["table"] == "grid"]
    rows = []
    for m in ("MLP", "LR", "RF", "XGB"):
        key = {"MLP": "mlp", "LR": "lr", "RF": "rf", "XGB": "xgb"}[m]
        g = grid[grid["model"] == key].set_index("setting")
        rows.append({"Part": "LIME", "Model / explanation": m, "Run-to-run $\\rho$": num(rr.loc[m, "spearman"]),
                     "Top-1 changes (\\%)": num(rr.loc[m, "top1_change_pct"], 2),
                     "Mean sampling $\\rho$": num(g.loc["sampling=mean", "spearman"]),
                     "Quartiles $\\rho$": num(g.loc["sampling=quartile", "spearman"]),
                     "1000 samples $\\rho$": num(g.loc["num_samples=1000", "spearman"])})  # fmt: skip
    bg = _r(reports, "step5_background_sensitivity").set_index(["explanation", "variant"])["spearman"]
    nf = _r(reports, "step5_background_sensitivity_noise_floor").set_index("explanation")["spearman"]
    for e in ("MLP-DeepLiftShap", "MLP-GradientShap", "RF-TreeSHAP", "XGB-TreeSHAP"):
        rows.append({"Part": "Background", "Model / explanation": esc(e), "Run-to-run $\\rho$": num(nf[e]),
                     "Top-1 changes (\\%)": "--", "Mean sampling $\\rho$": num(bg[(e, "size=10")]),
                     "Quartiles $\\rho$": num(bg[(e, "scheme=defective")]), "1000 samples $\\rho$": num(bg[(e, "scheme=kmeans")])})  # fmt: skip
    t = pd.DataFrame(rows)
    header = ["Part", "Model / explanation", "Repeat $\\rho$", "Top-1 changes (\\%)", "Variant A $\\rho$", "Variant B $\\rho$", "Variant C $\\rho$"]
    return tabular(t, "llrrrrr", "Robustness of the explanations (Spearman $\\rho$ with the main setting). LIME rows: repeat = 10 random states; A = mean sampling (v1), B = quartile discretisation, C = 1000 samples. Background rows: repeat = 5 independent draws of 100; A = 10 rows, B = defective rows only, C = k-means centres.",
                   "reports/step5_lime_stability.md, reports/step5_background_sensitivity.md", header)  # fmt: skip


def tab_fidelity(reports: Path, data: Path) -> str:
    t = _r(reports, "step6_summary")
    t = t[["explanation", "deletion gain", "insertion gain", "faithfulness", "ROAR gain", "KAR gain", "sanity", "SK-ESD group"]].copy()
    for c in ("deletion gain", "insertion gain", "faithfulness", "ROAR gain", "KAR gain"):
        vals = pd.to_numeric(t[c], errors="coerce")
        cells = [num(v) for v in vals]
        if vals.notna().any():
            b = int(np.nanargmax(vals.to_numpy()))
            cells[b] = f"\\textbf{{{cells[b]}}}"
        t[c] = cells
    t["explanation"] = t["explanation"].map(esc)
    t["sanity"] = t["sanity"].map(esc)
    return tabular(t, "lrrrrrlr", "Fidelity summary (seed-0 models): gains over random orderings, faithfulness correlation, ROAR/KAR gains over random rankings, sanity-check result and overall Scott-Knott ESD group of per-split mean ranks (1 = best).",
                   "reports/step6_summary.md", [esc(c) for c in t.columns])  # fmt: skip


def tab_stability(reports: Path, data: Path) -> str:
    src = _r(reports, "step7_seed_stability_sources").set_index(["source", "level"])
    cv = _r(reports, "step7_cross_version")
    cv = cv[cv["metric"] == "spearman"]
    mult = _r(reports, "step7_seed_stability_multiplicity").groupby("model")["disagreement_share"].mean()
    rows = []
    for e in PRIMARY:
        g = cv[cv["explanation"] == e]
        expl = [src.loc[("LIME random state (step 5)", "instance"), e], src.loc[("background draw (step 5)", "instance"), e]]
        expl = [float(x) for x in expl if pd.notna(x) and str(x) != ""]
        model = e.split("-")[0].lower()
        rows.append({"Explanation": esc(e), "Explainer randomness": num(max(expl)) if expl else "--",
                     "Seed (global)": num(src.loc[("training seed", "global"), e]),
                     "Release": num(g["value"].mean()), "Data-only": num(g["data_only"].mean()),
                     "Project": num(src.loc[("project", "global"), e]),
                     "Label changes across seeds (\\%)": num(100 * mult[model], 2)})  # fmt: skip
    t = pd.DataFrame(rows)
    return tabular(t, "lrrrrrr", "Sources of variation (mean Spearman $\\rho$; higher = more stable): explainer randomness (LIME runs or background draws, instance level), training seed, consecutive releases, data-only reference (permuted-label models), projects; and the share of test instances whose predicted label changes between seeds. Random null 95th percentile: 0.373.",
                   "reports/step7_seed_stability.md, reports/step7_cross_version.md", list(t.columns))  # fmt: skip


def tab_claims(reports: Path, data: Path) -> str:
    t = _r(reports, "step7_summary_claims")
    t = t.map(esc)
    return tabular(t, "p{3.2cm}p{9.5cm}p{3.2cm}", "Claims of the original paper and what the rebuilt pipeline finds.",
                   "reports/step7_summary.md", ["Original claim", "Rebuilt pipeline", "Verdict"])  # fmt: skip


def tab_per_project_auc(reports: Path, data: Path) -> str:
    t = _r(reports, "step3_main_time_aware_projects")
    t = t[t["metric"] == "AUC"].drop(columns="metric").sort_values("project")
    for m in MODELS:
        t[m] = t[m].map(lambda v: num(float(v)))
    t["project"] = t["project"].map(esc)
    return tabular(t, "l" + "r" * len(MODELS), "Time-aware test AUC per project (mean over splits and seeds).",
                   "reports/step3_main_time_aware.md", ["Project"] + [esc(MODEL_NAME[m]) for m in MODELS])  # fmt: skip


def tab_cost_ratio(reports: Path, data: Path) -> str:
    s = _r(reports, "step8_cost_ratio")
    be = _r(reports, "step8_cost_ratio_break_even").set_index(["model", "mode"])["break_even_c"]
    rows = []
    for m in MODELS:
        for mode, lab in (("val_mcc", "MCC-tuned"), ("cost_aware", "cost-aware")):
            g = s[(s["model"] == m) & (s["mode"] == mode)].set_index("c")
            if g.empty:
                continue
            row = {"Model": MODEL_NAME[m], "Threshold": lab}
            for c in (1, 5, 15):
                row[f"$c = {c}$"] = ci(g.loc[c, "mean"], g.loc[c, "ci_low"], g.loc[c, "ci_high"])
            b = be.get((m, mode), np.nan)
            row["Break-even $c$"] = "--" if not np.isfinite(b) or m not in ("mlp", "lr", "rf", "xgb") else num(b, 2)
            rows.append(row)
    t = pd.DataFrame(rows)
    return tabular(t, "llrrrr", "NECM at cost ratios 1, 5 and 15 (mean [95% CI] over splits) and the break-even cost ratio below which a learned model beats All defective and ManualUp.",
                   "reports/step8_cost_ratio.md", list(t.columns))  # fmt: skip


def tab_calibration(reports: Path, data: Path) -> str:
    from sdp.eval.stats import bootstrap_ci

    cal = _r(reports, "step8_calibration")
    ps = cal.groupby(["split_id", "model", "method"], as_index=False)[["ece", "brier", "nll", "auc", "mcc_tuned"]].mean()
    rows = []
    for m in ("mlp", "lr", "rf", "xgb"):
        for method in ("uncalibrated", "platt", "isotonic"):
            g = ps[(ps["model"] == m) & (ps["method"] == method)]
            row = {"Model": MODEL_NAME[m], "Method": method}
            for c, lab in (("ece", "ECE"), ("brier", "Brier"), ("nll", "NLL"), ("auc", "AUC"), ("mcc_tuned", "MCC@tuned")):
                row[lab] = ci(*bootstrap_ci(g[c].to_numpy()))
            rows.append(row)
    t = pd.DataFrame(rows)
    return tabular(t, "llrrrrr", "Calibration before and after Platt scaling and isotonic regression fitted on the validation split (mean [95% CI] over splits).",
                   "reports/step8_calibration.md", list(t.columns))  # fmt: skip


def tab_autospearman(reports: Path, data: Path) -> str:
    t = _r(reports, "step5_collinearity_retraining")
    t = t.assign(**{"AutoSpearman": t["mean_a"].map(num), "All 20 metrics": t["mean_b"].map(num),
                    "$p_{\\mathrm{Holm}}$": t["p_holm"].map(lambda p: "$<$0.001" if p < 0.001 else num(p)),
                    "Cliff's $\\delta$": t["cliffs_delta"].map(num)})  # fmt: skip
    t = t[["model", "metric", "AutoSpearman", "All 20 metrics", "$p_{\\mathrm{Holm}}$", "Cliff's $\\delta$"]]
    t["metric"] = t["metric"].map(esc)
    return tabular(t, "llrrrr", "Models retrained on the AutoSpearman-selected metrics vs all 20 metrics (seeds 0--2; per-split means; Wilcoxon signed-rank, Holm).",
                   "reports/step5_collinearity.md", ["Model", "Metric", "AutoSpearman", "All 20 metrics", "$p_{\\mathrm{Holm}}$", "Cliff's $\\delta$"])  # fmt: skip


TABLES = {f.__name__: f for f in (
    tab_dataset, tab_explainer_settings, tab_performance, tab_ablation, tab_importance, tab_agreement, tab_robustness,
    tab_fidelity, tab_stability, tab_claims, tab_within_release, tab_per_project_auc, tab_cost_ratio, tab_calibration,
    tab_autospearman)}  # fmt: skip


def make_all(reports: Path, data: Path, out: Path) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, f in TABLES.items():
        (out / f"{name}.tex").write_text(f(Path(reports), Path(data)), encoding="utf-8")
        written.append(out / f"{name}.tex")
    logger.info("Wrote %d tables to %s", len(written), out)
    return written
