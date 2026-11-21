"""Final paper figures (step 8), written to ``paper/figures/`` as PDF + 300-dpi PNG.

Inputs are the committed reports (``reports/``) and the collected paper inputs
(``paper/data/``, see :mod:`sdp.paper.collect`). File names are fixed (the paper refers to
them); every figure has a one-sentence factual caption in ``CAPTIONS.md``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from sdp.eval.preprocess import FEATURES
from sdp.eval.stats import bootstrap_ci
from sdp.paper import style as S

logger = logging.getLogger(__name__)

PRIMARY = ["MLP-DeepLiftShap", "MLP-KernelSHAP", "MLP-IntegratedGradients", "MLP-LIME", "MLP-Saliency",
           "LR-LinearSHAP", "RF-TreeSHAP", "XGB-TreeSHAP"]  # fmt: skip
ALL14 = ["MLP-DeepLift", "MLP-DeepLiftShap", "MLP-GradientShap", "MLP-IntegratedGradients", "MLP-Saliency",
         "MLP-InputXGradient", "MLP-LIME", "MLP-KernelSHAP", "LR-LinearSHAP", "LR-LIME", "RF-TreeSHAP", "RF-LIME",
         "XGB-TreeSHAP", "XGB-LIME"]  # fmt: skip
MLP8 = ["DeepLift", "DeepLiftShap", "GradientShap", "IntegratedGradients", "Saliency", "InputXGradient", "LIME", "KernelSHAP"]
MODEL_MARKER = {"MLP": "o", "LR": "s", "RF": "^", "XGB": "D"}

CAPTIONS = {
    "fig_performance": ("Per-split test AUC, MCC at the validation-tuned threshold and Popt of all models under time-aware "
                        "evaluation (boxes and points: 24 splits, seeds averaged); diamonds mark the within-release means.",
                        "reports/step3_main_time_aware.md, reports/step3_within_release.md"),
    "fig_cost_ratio": ("Normalised expected cost of misclassification (NECM) against the cost ratio c for every model at its "
                       "validation-MCC threshold, with 95% bootstrap confidence bands over splits; at this threshold the Majority "
                       "baseline flags every class and coincides with All defective.", "reports/step8_cost_ratio.md"),
    "fig_calibration": ("Reliability diagrams of the four learned models before calibration (test predictions pooled over "
                        "splits and seeds, ten equal-width bins).", "reports/step8_calibration.md"),
    "fig_global_importance": ("Mean L1-normalised absolute attribution of the 20 metrics for the 14 explanations (all test "
                              "instances, ALL splits), with metrics grouped by the stable correlation clusters.",
                              "reports/step4_global_importance.md, reports/step5_collinearity.md"),
    "fig_direction": ("Spearman correlation between each metric's value and its signed attribution (ALL splits) for the eight "
                      "primary explanations.", "reports/step4_direction.md"),
    "fig_agreement": ("Mean feature agreement at k = 5 between the MLP explanations (left) and between the SHAP-family "
                      "explanations of the four models (right), seeds 0–2.", "reports/step4_agreement.md"),
    "fig_scaling_effect": ("Share of the normalised absolute attribution on LOC and on the size group {LOC, WMC, RFC, NPM} by "
                           "preprocessing setting for the MLP and LR, including DeepLift with a zero baseline on unscaled "
                           "inputs.", "reports/step5_preprocessing_explanations.md"),
    "fig_fidelity": ("(a) Mean deletion curves (confidence in the model's own prediction against the number of removed "
                     "metrics) for the MLP and RF explanations, with the random ordering dashed; (b) ROAR and KAR area "
                     "scores of the training-set rankings against random and size-first rankings.",
                     "reports/step6_deletion_insertion.md, reports/step6_roar_kar.md"),
    "fig_sanity": ("Mean Spearman correlation between the original and recomputed MLP explanations as Linear layers are "
                   "re-initialised cumulatively from the output layer.", "reports/step6_sanity_checks.md"),
    "fig_stability": ("(a) Cross-version Spearman correlation of global importance per primary explanation (mean and 95% CI "
                      "over 14 comparisons) with the data-only reference and the random-null band; (b) mean Spearman "
                      "correlation per source of variation.", "reports/step7_cross_version.md, reports/step7_seed_stability.md"),
    "fig_background_sensitivity": ("Mean Spearman correlation with the main background explanation against the background "
                                   "size, with independent draws of size 100 as reference.",
                                   "reports/step5_background_sensitivity.md"),
    "fig_lime_settings": ("Mean Spearman correlation of LIME explanations between runs (run-to-run) and between each setting "
                          "and the main setting, per model.", "reports/step5_lime_stability.md"),
    "fig_collinearity_dendrogram": ("Hierarchical clustering (average linkage) of the 20 metrics on 1 − |Spearman ρ| for the "
                                    "jEdit 4.0 training release, with the cut at |ρ| = 0.7.", "reports/step5_collinearity.md"),
    "fig_topk_frequency": ("Share of the 24 time-aware splits in which each metric is among the five most important "
                           "metrics of each explanation.", "reports/step7_topk_frequency.md"),
}


def _r(reports: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(reports) / f"{name}.csv")


def _lab(label: str) -> str:
    return S.explanation_label(label)


# ------------------------------------------------------------------ RQ1


def fig_performance(reports: Path, data: Path, out: Path) -> list[Path]:
    per = pd.read_csv(Path(data) / "performance_per_split.csv")
    wr = pd.read_csv(Path(data) / "within_release_means.csv").set_index(["model", "label"])["value"]
    fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE, 58 * S.MM))
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes, ["AUC", "MCC@tuned", "Popt"]):
        for i, m in enumerate(S.MODEL_ORDER):
            v = per[(per["model"] == m) & (per["label"] == metric)]["value"].dropna().to_numpy()
            if not len(v):
                continue
            bp = ax.boxplot(v, positions=[i], widths=0.55, patch_artist=True, showfliers=False,
                            medianprops={"color": "black", "linewidth": 0.8}, whiskerprops={"linewidth": 0.6},
                            capprops={"linewidth": 0.6})  # fmt: skip
            bp["boxes"][0].set(facecolor=S.MODEL_COLOUR[m], alpha=0.45, linewidth=0.6)
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=4, color=S.MODEL_COLOUR[m], zorder=3, linewidths=0)
            if (m, metric) in wr.index:
                ax.scatter([i], [wr[(m, metric)]], marker="D", s=14, facecolor="white", edgecolor="black", linewidth=0.7, zorder=4)
        ax.set_xticks(range(len(S.MODEL_ORDER)), [S.MODEL_NAME[m] for m in S.MODEL_ORDER], rotation=45, ha="right")
        ax.set_title(metric.replace("@tuned", " (tuned threshold)"))
        ax.grid(axis="y", color=S.LIGHT, linewidth=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(handles=[Line2D([], [], marker="D", linestyle="", markerfacecolor="white", markeredgecolor="black",
                               markersize=4, label="within-release mean (diamond)")], loc="upper right", frameon=False)  # fmt: skip
    return S.save(fig, out, "fig_performance")


def fig_cost_ratio(reports: Path, data: Path, out: Path) -> list[Path]:
    t = _r(reports, "step8_cost_ratio")
    t = t[t["mode"] == "val_mcc"]
    fig, ax = plt.subplots(figsize=(S.SINGLE, 62 * S.MM))
    for m in S.MODEL_ORDER:
        g = t[t["model"] == m].sort_values("c")
        if g.empty or m == "majority":
            continue
        ax.plot(g["c"], g["mean"], color=S.MODEL_COLOUR[m], label=S.MODEL_NAME[m], linewidth=1.1,
                linestyle="--" if m in ("manual_up", "manual_down", "all_defective") else "-")  # fmt: skip
        ax.fill_between(g["c"], g["ci_low"], g["ci_high"], color=S.MODEL_COLOUR[m], alpha=0.12, linewidth=0)
    ax.set_xlabel("cost ratio c (cost of a missed defect / cost of a false alarm)")
    ax.set_ylabel("NECM$_c$ (lower is better)")
    ax.grid(color=S.LIGHT, linewidth=0.5)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.text(0.99, 0.02, "Majority = All defective", transform=ax.transAxes, ha="right", fontsize=6, color=S.GREY)
    fig.tight_layout()
    return S.save(fig, out, "fig_cost_ratio")


def fig_calibration(reports: Path, data: Path, out: Path) -> list[Path]:
    rel = _r(reports, "step8_calibration_reliability")
    rel = rel[rel["method"] == "uncalibrated"]
    fig, axes = plt.subplots(2, 2, figsize=(S.SINGLE, 82 * S.MM), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, ["mlp", "lr", "rf", "xgb"]):
        g = rel[(rel["model"] == m) & (rel["n"] > 0)]
        ax.plot([0, 1], [0, 1], linestyle="--", color=S.GREY, linewidth=0.7)
        ax.plot(g["mean_predicted"], g["observed"], marker="o", markersize=2.5, color=S.MODEL_COLOUR[m])
        ax.set_title(S.MODEL_NAME[m])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color=S.LIGHT, linewidth=0.5)
    for ax in axes[1]:
        ax.set_xlabel("mean predicted probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("observed defective rate")
    fig.tight_layout()
    return S.save(fig, out, "fig_calibration")


# ------------------------------------------------------------------ explanations


def _clusters(data: Path) -> list[list[str]]:
    p = Path(data) / "stable_clusters.json"
    return json.loads(p.read_text()) if p.exists() else [[f] for f in FEATURES]


def fig_global_importance(reports: Path, data: Path, out: Path) -> list[Path]:
    imp = _r(reports, "step4_global_importance")
    imp = imp[(imp["group"] == "all") & (imp["scope"] == "ALL")]
    imp = imp.assign(explanation=[f"{m.upper()}-{k}" for m, k in zip(imp["model"], imp["method"])])
    mat = imp.pivot(index="feature", columns="explanation", values="importance")[ALL14]
    clusters = _clusters(data)
    mean = mat.mean(axis=1)
    clusters = sorted(clusters, key=lambda g: -mean[g].max())
    rows = [f for g in clusters for f in sorted(g, key=lambda f: -mean[f])]
    mat = mat.loc[rows]
    fig, ax = plt.subplots(figsize=(S.DOUBLE, 105 * S.MM))
    im = ax.imshow(mat.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=float(np.nanmax(mat.to_numpy())))
    ax.set_xticks(range(mat.shape[1]), [_lab(c) for c in mat.columns], rotation=40, ha="right")
    ax.set_yticks(range(mat.shape[0]), [S.METRIC_LABEL[f] for f in mat.index])
    for i, f in enumerate(mat.index):
        for j, c in enumerate(mat.columns):
            v = mat.loc[f, c]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if v > 0.6 * np.nanmax(mat.to_numpy()) else "black")  # fmt: skip
    pos = 0
    for g in clusters:
        if len(g) > 1:
            y0, y1 = pos - 0.4, pos + len(g) - 0.6
            x = -2.6
            ax.plot([x, x], [y0, y1], color="black", linewidth=0.8, clip_on=False)
            ax.plot([x, x + 0.25], [y0, y0], color="black", linewidth=0.8, clip_on=False)
            ax.plot([x, x + 0.25], [y1, y1], color="black", linewidth=0.8, clip_on=False)
        pos += len(g)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("mean normalised |attribution|")
    cb.outline.set_visible(False)
    fig.tight_layout()
    return S.save(fig, out, "fig_global_importance")


def fig_direction(reports: Path, data: Path, out: Path) -> list[Path]:
    d = _r(reports, "step4_direction")
    d = d[d["scope"] == "ALL"]
    d = d.assign(explanation=[f"{m.upper()}-{k}" for m, k in zip(d["model"], d["method"])])
    d = d[d["explanation"].isin(PRIMARY)]
    order = d.groupby("feature")["spearman"].mean().sort_values().index
    y = {f: i for i, f in enumerate(order)}
    fig, ax = plt.subplots(figsize=(S.SINGLE, 100 * S.MM))
    ax.axvline(0, color=S.GREY, linewidth=0.6)
    for j, e in enumerate(PRIMARY):
        g = d[d["explanation"] == e]
        model, method = e.split("-", 1)
        ax.scatter(g["spearman"], [y[f] + (j - 3.5) * 0.08 for f in g["feature"]], s=9, marker=MODEL_MARKER[model],
                   color=S.METHOD_COLOUR[method], label=_lab(e), linewidths=0.3, edgecolors="black" if model != "MLP" else "none")  # fmt: skip
    ax.set_yticks(range(len(order)), [S.METRIC_LABEL[f] for f in order])
    ax.set_xlabel("Spearman ρ (metric value vs signed attribution)")
    ax.grid(axis="x", color=S.LIGHT, linewidth=0.5)
    ax.legend(frameon=False, fontsize=5.5, loc="lower right", ncol=1, handletextpad=0.2)
    fig.tight_layout()
    return S.save(fig, out, "fig_direction")


def _agreement_matrix(ag: pd.DataFrame, family: str, names: list[str]) -> pd.DataFrame:
    s = ag[(ag["family"] == family) & (ag["metric"] == "feature_agreement@5") & (ag["scope"] == "ALL")]
    m = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for r in s.itertuples():
        if r.a in names and r.b in names:
            m.loc[r.a, r.b] = m.loc[r.b, r.a] = r.mean
    return m


def _heat(ax, m: pd.DataFrame, labels: list[str]):
    im = ax.imshow(m.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = m.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="white" if v > 0.6 else "black")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def fig_agreement(reports: Path, data: Path, out: Path) -> list[Path]:
    ag = _r(reports, "step4_agreement")
    within = _agreement_matrix(ag, "within MLP", [f"MLP-{k}" for k in MLP8])
    cross_names = ["MLP-DeepLiftShap", "RF-TreeSHAP", "XGB-TreeSHAP", "LR-LinearSHAP"]
    cross = _agreement_matrix(ag, "cross-model SHAP-family", cross_names)
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 80 * S.MM), gridspec_kw={"width_ratios": [8, 4.6]})
    im = _heat(axes[0], within, MLP8)
    axes[0].set_title("(a) within the MLP, FA@5")
    _heat(axes[1], cross, [_lab(c) for c in cross_names])
    axes[1].set_title("(b) across models, FA@5")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("feature agreement@5")
    return S.save(fig, out, "fig_agreement")


def fig_scaling_effect(reports: Path, data: Path, out: Path) -> list[Path]:
    t = _r(reports, "step5_preprocessing_explanations_size_share")
    t = t[t["scope"] == "ALL"]
    rows = [("mlp", "DeepLiftShap", p) for p in ("none", "standard", "log_standard")] + [("mlp", "DeepLift_zero_baseline", "none")] \
        + [("lr", "LinearSHAP", p) for p in ("none", "standard", "log_standard")]  # fmt: skip
    prep_name = {"none": "unscaled", "standard": "standardised", "log_standard": "log + standardised"}
    labels, loc, size = [], [], []
    for m, k, p in rows:
        r = t[(t["model"] == m) & (t["method"] == k) & (t["preprocessing"] == p)].iloc[0]
        name = "DeepLift, zero baseline" if k == "DeepLift_zero_baseline" else k
        labels.append(f"{S.MODEL_NAME[m]} {name}, {prep_name[p]}")
        loc.append(r["loc_share"])
        size.append(r["size_group_share"])
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(S.SINGLE, 70 * S.MM))
    ax.barh(y + 0.19, loc, height=0.36, color=S.OKABE_ITO["blue"], label="LOC")
    ax.barh(y - 0.19, size, height=0.36, color=S.OKABE_ITO["orange"], label="size group (LOC, WMC, RFC, NPM)")
    ax.axvline(0.05, color=S.GREY, linestyle=":", linewidth=0.7)
    ax.axvline(0.20, color=S.GREY, linestyle=":", linewidth=0.7)
    ax.set_yticks(y, labels)
    ax.set_xlabel("share of normalised |attribution| (dotted: equal shares)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2)
    ax.grid(axis="x", color=S.LIGHT, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return S.save(fig, out, "fig_scaling_effect")


# ------------------------------------------------------------------ fidelity, sanity, stability


def fig_fidelity(reports: Path, data: Path, out: Path) -> list[Path]:
    crv = pd.read_csv(Path(data) / "deletion_curves.csv")
    crv = crv[crv["kind"] == "deletion"]
    areas = _r(reports, "step6_roar_kar_areas")
    areas = areas[areas["metric"] == "auc"].groupby(["model", "family", "test"], as_index=False)["area"].mean()
    fig, axes = plt.subplots(1, 4, figsize=(S.DOUBLE, 62 * S.MM))
    for ax, model in zip(axes[:2], ["mlp", "rf"]):
        g = crv[crv["model"] == model]
        for method, gm in g.groupby("method"):
            gm = gm.sort_values("k")
            if method == "random":
                ax.plot(gm["k"], gm["value"], color="black", linestyle="--", linewidth=0.9, label="random")
            else:
                ax.plot(gm["k"], gm["value"], color=S.METHOD_COLOUR.get(method, S.GREY), linewidth=1.0, label=method)
        ax.set_title(f"(a) deletion, {S.MODEL_NAME[model]}")
        ax.set_xlabel("metrics removed (k)")
        ax.grid(color=S.LIGHT, linewidth=0.5)
        ax.legend(frameon=False, fontsize=5, loc="upper right")
    axes[0].set_ylabel("confidence in own prediction")
    models = ["mlp", "lr", "rf", "xgb"]
    for ax, test in zip(axes[2:], ["ROAR", "KAR"]):
        g = areas[areas["test"] == test]
        for i, m in enumerate(models):
            gm = g[g["model"] == m]
            for r in gm.itertuples():
                if r.family == "random":
                    ax.scatter(r.area, i, marker="x", color="black", s=14, zorder=3)
                elif r.family == "size_first":
                    ax.scatter(r.area, i, marker="+", color="black", s=22, zorder=3)
                elif r.family == "rf_TreeSHAP":
                    continue
                else:
                    ax.scatter(r.area, i, marker="o", color=S.METHOD_COLOUR.get(r.family, S.GREY), s=12, zorder=3,
                               edgecolors="none")  # fmt: skip
        ax.set_yticks(range(len(models)), [S.MODEL_NAME[m] for m in models])
        ax.set_title(f"(b) {test} area ({'lower' if test == 'ROAR' else 'higher'} is better)")
        ax.set_xlabel("mean test AUC over k")
        ax.grid(axis="x", color=S.LIGHT, linewidth=0.5)
    handles = [Line2D([], [], marker="x", color="black", linestyle="", label="random"),
               Line2D([], [], marker="+", color="black", linestyle="", markersize=6, label="size-first")]
    handles += [Line2D([], [], marker="o", color=S.METHOD_COLOUR[k], linestyle="", markersize=4, label=k)
                for k in ("DeepLiftShap", "IntegratedGradients", "Saliency", "LIME", "LinearSHAP", "TreeSHAP")]  # fmt: skip
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=handles, frameon=False, fontsize=5.5, loc="lower right", ncol=4, bbox_to_anchor=(1.0, 0.0))
    return S.save(fig, out, "fig_fidelity")


def fig_sanity(reports: Path, data: Path, out: Path) -> list[Path]:
    rows = _r(reports, "step6_sanity_checks_rows")
    c = rows[rows["kind"] == "cascading"].copy()
    c["stage_label"] = np.where(c["stage"] == "all", "all layers",
                                np.where(c["n_layers"] == 1, "output", "top " + c["n_layers"].astype(int).astype(str) + " layers"))  # fmt: skip
    stages = [s for s in ["output", "top 2 layers", "top 3 layers", "all layers"] if s in set(c["stage_label"])]
    t = c.groupby(["method", "stage_label"])["spearman"].mean().unstack()[stages]
    fig, ax = plt.subplots(figsize=(S.SINGLE, 62 * S.MM))
    for k in MLP8:
        if k in t.index:
            ax.plot(range(len(stages)), t.loc[k], marker="o", markersize=2.5, color=S.METHOD_COLOUR[k], label=k)
    ax.axhline(0, color=S.GREY, linewidth=0.6)
    ax.set_xticks(range(len(stages)), stages)
    ax.set_xlabel("layers re-initialised, from the output layer")
    ax.set_ylabel("Spearman ρ with the original explanation")
    ax.grid(axis="y", color=S.LIGHT, linewidth=0.5)
    ax.legend(frameon=False, fontsize=5.5, ncol=2)
    fig.tight_layout()
    return S.save(fig, out, "fig_sanity")


def fig_stability(reports: Path, data: Path, out: Path) -> list[Path]:
    cv = _r(reports, "step7_cross_version")
    cv = cv[cv["metric"] == "spearman"]
    if "level" in cv.columns:
        cv = cv[cv["level"] == "feature"]
    null = pd.read_csv(Path(data) / "stability_null.csv")
    null = null[(null["level"] == "feature") & (null["metric"] == "spearman")].iloc[0]
    src = _r(reports, "step7_seed_stability_sources").set_index(["source", "level"])
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 70 * S.MM), sharey=True)
    y = np.arange(len(PRIMARY))[::-1]
    ax = axes[0]
    ax.axvspan(null["null_mean"], null["null_p95"], color=S.LIGHT, label="random null (mean to 95th pct)")
    for yi, e in zip(y, PRIMARY):
        g = cv[cv["explanation"] == e]
        m, lo, hi = bootstrap_ci(g["value"].to_numpy())
        ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt="o", color=S.OKABE_ITO["blue"], markersize=3, capsize=1.5, linewidth=0.8)
        ax.scatter(g["data_only"].mean(), yi, marker="D", color=S.OKABE_ITO["vermilion"], s=10, zorder=3)
    ax.set_yticks(y, [_lab(e) for e in PRIMARY])
    ax.set_xlabel("cross-version Spearman ρ")
    ax.set_title("(a) stability across releases")
    ax.legend(handles=[Line2D([], [], marker="o", color=S.OKABE_ITO["blue"], linestyle="", markersize=3, label="mean, 95% CI"),
                       Line2D([], [], marker="D", color=S.OKABE_ITO["vermilion"], linestyle="", markersize=3,
                              label="data-only reference"),
                       plt.Rectangle((0, 0), 1, 1, color=S.LIGHT, label="random null")],
              frameon=False, fontsize=5.5, loc="upper right")  # fmt: skip
    ax.grid(axis="x", color=S.LIGHT, linewidth=0.5)
    ax = axes[1]

    def val(source, level, e):
        try:
            v = src.loc[(source, level), e]
            return float(v) if pd.notna(v) and str(v) != "" else np.nan
        except KeyError:
            return np.nan

    def explainer_randomness(e):
        v = [x for x in (val("LIME random state (step 5)", "instance", e), val("background draw (step 5)", "instance", e))
             if np.isfinite(x)]
        return max(v) if v else np.nan

    sources = [("explainer randomness", explainer_randomness, S.OKABE_ITO["sky"]),
               ("training seed", lambda e: val("training seed", "global", e), S.OKABE_ITO["green"]),
               ("release", lambda e: val("release (consecutive splits)", "global", e), S.OKABE_ITO["orange"]),
               ("project", lambda e: val("project", "global", e), S.OKABE_ITO["purple"])]  # fmt: skip
    for k, (name, f, colour) in enumerate(sources):
        vals = [f(e) for e in PRIMARY]
        ax.scatter(vals, y + (1.5 - k) * 0.15, color=colour, s=12, label=name, zorder=3)
    ax.axvline(null["null_p95"], color=S.GREY, linestyle=":", linewidth=0.8, label="random null, 95th pct")
    ax.set_xlabel("mean Spearman ρ (higher = more stable)")
    ax.set_title("(b) sources of variation")
    ax.legend(frameon=False, fontsize=5.5, loc="lower left")
    ax.grid(axis="x", color=S.LIGHT, linewidth=0.5)
    fig.tight_layout()
    return S.save(fig, out, "fig_stability")


# ------------------------------------------------------------------ appendix


def fig_background_sensitivity(reports: Path, data: Path, out: Path) -> list[Path]:
    b = _r(reports, "step5_background_sensitivity")
    noise = _r(reports, "step5_background_sensitivity_noise_floor").set_index("explanation")["spearman"]
    sizes = ["size=10", "size=25", "size=50", "size=200", "size=all"]
    xlab = ["10", "25", "50", "100*", "200", "all"]
    fig, ax = plt.subplots(figsize=(S.SINGLE, 58 * S.MM))
    for e, g in b.groupby("explanation"):
        g = g.set_index("variant")["spearman"]
        vals = [g.get(s, np.nan) for s in sizes[:3]] + [noise.get(e, np.nan)] + [g.get(s, np.nan) for s in sizes[3:]]
        model, method = e.split("-", 1)
        ax.plot(range(6), vals, marker=MODEL_MARKER[model], markersize=3, color=S.METHOD_COLOUR.get(method, S.GREY),
                label=_lab(e))
    ax.set_xticks(range(6), xlab)
    ax.set_xlabel("background size (*: independent draws of 100)")
    ax.set_ylabel("Spearman ρ with the main background")
    ax.grid(axis="y", color=S.LIGHT, linewidth=0.5)
    ax.legend(frameon=False, fontsize=5.5)
    fig.tight_layout()
    return S.save(fig, out, "fig_background_sensitivity")


def fig_lime_settings(reports: Path, data: Path, out: Path) -> list[Path]:
    t = _r(reports, "step5_lime_stability")
    rr = t[t["table"] == "run_to_run"].assign(setting="run-to-run (main)")
    grid = t[(t["table"] == "grid") & (t["setting"] != "main")].copy()
    grid["model"] = grid["model"].map({"mlp": "MLP", "lr": "LR", "rf": "RF", "xgb": "XGB"}).fillna(grid["model"])
    allr = pd.concat([rr[["model", "setting", "spearman"]], grid[["model", "setting", "spearman"]]])
    settings = list(dict.fromkeys(allr["setting"]))
    y = {s: i for i, s in enumerate(settings[::-1])}
    fig, ax = plt.subplots(figsize=(S.SINGLE, 70 * S.MM))
    colour = {"MLP": S.MODEL_COLOUR["mlp"], "LR": S.MODEL_COLOUR["lr"], "RF": S.MODEL_COLOUR["rf"], "XGB": S.MODEL_COLOUR["xgb"]}
    for k, (m, g) in enumerate(allr.groupby("model", sort=False)):
        ax.scatter(g["spearman"], [y[s] + (k - 1.5) * 0.12 for s in g["setting"]], color=colour.get(m, S.GREY), s=10, label=m,
                   marker=MODEL_MARKER.get(m, "o"))  # fmt: skip
    ax.set_yticks(list(y.values()), [s.replace("×√20", " × √20") for s in y])
    ax.set_xlabel("mean Spearman ρ")
    ax.grid(axis="x", color=S.LIGHT, linewidth=0.5)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    return S.save(fig, out, "fig_lime_settings")


def fig_collinearity_dendrogram(reports: Path, data: Path, out: Path) -> list[Path]:
    from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
    from scipy.spatial.distance import squareform

    rho = pd.read_csv(Path(data) / "dendrogram_spearman.csv", index_col=0)
    rho = rho.dropna(how="all").dropna(axis=1, how="all")
    dist = np.clip(1 - rho.abs().to_numpy(), 0, None)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average")
    fig, ax = plt.subplots(figsize=(S.SINGLE, 60 * S.MM))
    set_link_color_palette([S.OKABE_ITO["blue"], S.OKABE_ITO["vermilion"], S.OKABE_ITO["green"], S.OKABE_ITO["purple"]])
    dendrogram(Z, labels=[S.METRIC_LABEL[f] for f in rho.index], ax=ax, color_threshold=0.3, above_threshold_color=S.GREY,
               leaf_rotation=90, leaf_font_size=6)  # fmt: skip
    set_link_color_palette(None)
    ax.axhline(0.3, color="black", linestyle="--", linewidth=0.7)
    ax.set_ylabel("1 − |Spearman ρ|")
    fig.tight_layout()
    return S.save(fig, out, "fig_collinearity_dendrogram")


def fig_topk_frequency(reports: Path, data: Path, out: Path) -> list[Path]:
    t = _r(reports, "step7_topk_frequency")
    t = t[t["scope"] == "ALL"]
    mat = t.pivot(index="feature", columns="explanation", values="frequency")[ALL14]
    mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(S.DOUBLE, 100 * S.MM))
    im = ax.imshow(mat.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), [_lab(c) for c in mat.columns], rotation=40, ha="right")
    ax.set_yticks(range(mat.shape[0]), [S.METRIC_LABEL[f] for f in mat.index])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="white" if v > 0.6 else "black")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("share of splits with the metric in the top-5")
    cb.outline.set_visible(False)
    fig.tight_layout()
    return S.save(fig, out, "fig_topk_frequency")


FIGURES = {f.__name__: f for f in (
    fig_performance, fig_cost_ratio, fig_calibration, fig_global_importance, fig_direction, fig_agreement,
    fig_scaling_effect, fig_fidelity, fig_sanity, fig_stability, fig_background_sensitivity, fig_lime_settings,
    fig_collinearity_dendrogram, fig_topk_frequency)}  # fmt: skip


def make_all(reports: Path, data: Path, out: Path) -> list[Path]:
    S.apply()
    written = []
    for name, f in FIGURES.items():
        written += f(Path(reports), Path(data), Path(out))
        logger.info("Figure %s written.", name)
    lines = ["# Figure captions (draft, factual)", "", "| file | caption | source |", "|---|---|---|"]
    for name in FIGURES:
        cap, src = CAPTIONS[name]
        lines.append(f"| `{name}.pdf` | {cap} | {src} |")
    (Path(out) / "CAPTIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written
