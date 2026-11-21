"""Step 8 reports: NECM cost-ratio sweep and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sdp.analysis import cost_calibration as CC
from sdp.analysis.step4 import _write
from sdp.analysis.tables import format_comparisons
from sdp.config import Config
from sdp.data.audit import to_markdown
from sdp.eval.stats import bootstrap_ci

ORDER = list(CC.LEARNED) + list(CC.BASELINES)


def cost_report(cfg: Config, ps: pd.DataFrame, summ: pd.DataFrame, be: pd.DataFrame, tests: pd.DataFrame, check: pd.DataFrame) -> None:
    def table(mode):
        t = summ[summ["mode"] == mode]
        w = t.assign(cell=[f"{m:.3f} [{lo:.3f}, {hi:.3f}]" for m, lo, hi in zip(t["mean"], t["ci_low"], t["ci_high"])])
        w = w.pivot(index="model", columns="c", values="cell")
        w = w.reindex([m for m in ORDER if m in w.index])
        w.index = [CC.MODEL_LABEL[m] for m in w.index]
        w.columns = [f"c = {c:g}" for c in w.columns]
        return w.reset_index().rename(columns={"index": "model"})

    be_t = be.assign(model=be["model"].map(CC.MODEL_LABEL),
                     break_even=[("never" if v == 0 else ("> 30" if np.isinf(v) else f"{v:.2f}")) for v in be["break_even_c"]])  # fmt: skip
    be_t = be_t.pivot(index="model", columns="mode", values="break_even").reset_index()
    be_t = be_t.rename(columns={"val_mcc": "MCC-tuned threshold", "cost_aware": "cost-aware threshold"})
    md = [
        "# Step 8: NECM cost-ratio sweep",
        "",
        "NECM_c = (FP + c·FN)/N on the test releases of the 24 time-aware splits (`main_time_aware` predictions), "
        "seeds averaged per split; mean and 95% bootstrap CI over splits. Thresholds: 0.5; the validation-MCC "
        "threshold stored in step 3 (all models); and for the learned models the cost-aware threshold minimising "
        "NECM_c on the validation split. Validation predictions come from the persisted step 3 models by inference "
        f"only; they reproduce the stored thresholds and test predictions exactly (maximum differences "
        f"{check['threshold_diff'].max():.1e} and {check['test_prob_max_diff'].max():.1e} over {len(check)} models).",
        "",
        "Note: the Majority baseline outputs a constant score per split (the training defective rate). Its "
        "validation-MCC threshold equals that score and predictions are p ≥ t, so at the tuned threshold it flags "
        "every class and coincides with all_defective; at 0.5 it flags none (step 3 behaviour, unchanged).",
        "",
        "## NECM at the validation-MCC threshold",
        "",
        to_markdown(table("val_mcc")),
        "",
        "## NECM at the cost-aware threshold (learned models)",
        "",
        to_markdown(table("cost_aware")),
        "",
        "## NECM at the 0.5 threshold",
        "",
        to_markdown(table("0.5")),
        "",
        "## Break-even cost ratio",
        "",
        "The largest cost ratio c below which the model's mean NECM is lower than both all_defective's and "
        "ManualUp's (at their validation-MCC thresholds) at every smaller c (grid step 0.05, up to 30). `never` = "
        "not even at c = 0.05.",
        "",
        to_markdown(be_t),
        "",
        "## Tests at c ∈ {1, 5, 15}",
        "",
        "Per-split NECM (seeds averaged), Wilcoxon signed-rank, Holm within each c, Cliff's delta (negative = the "
        "learned model has the lower cost).",
        "",
        to_markdown(format_comparisons(tests)),
        "",
    ]
    _write(cfg, "step8_cost_ratio", md, {"": summ, "_break_even": be, "_tests": tests, "_per_split": ps})


def calibration_report(cfg: Config, cal: pd.DataFrame, rel: pd.DataFrame, tests: pd.DataFrame) -> None:
    ps = cal.groupby(["split_id", "model", "method"], as_index=False)[["ece", "brier", "nll", "auc", "mcc_tuned"]].mean()
    rows = []
    for model in CC.LEARNED:
        for method in ("uncalibrated", "platt", "isotonic"):
            g = ps[(ps["model"] == model) & (ps["method"] == method)]
            row = {"model": CC.MODEL_LABEL[model], "method": method}
            for m in ("ece", "brier", "nll", "auc", "mcc_tuned"):
                mean, lo, hi = bootstrap_ci(g[m].to_numpy())
                row[m.upper() if m != "mcc_tuned" else "MCC@tuned"] = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
            rows.append(row)
    pl = cal[cal["method"] == "platt"]
    neg, n_platt = int((pl["platt_slope"] <= 0).sum()), len(pl)
    neg_by = pl[pl["platt_slope"] <= 0]["model"].value_counts().to_dict()
    key = ["split_id", "model", "seed"]
    merged = pl.merge(cal[cal["method"] == "uncalibrated"][key + ["auc"]], on=key, suffixes=("", "_raw"))
    pos = merged[merged["platt_slope"] > 0]
    auc_change_pos = float((pos["auc"] - pos["auc_raw"]).abs().mean())
    rel_t = rel[rel["method"] == "uncalibrated"].assign(model=lambda t: t["model"].map(CC.MODEL_LABEL))
    rel_w = rel_t.assign(cell=[("" if n == 0 else f"{mp:.2f} / {ob:.2f} ({n})") for mp, ob, n in
                               zip(rel_t["mean_predicted"], rel_t["observed"], rel_t["n"])])  # fmt: skip
    rel_w = rel_w.pivot(index="bin", columns="model", values="cell")
    rel_w.index = [f"[{b / 10:.1f}, {(b + 1) / 10:.1f})" for b in rel_w.index]
    md = [
        "# Step 8: calibration",
        "",
        "Learned models of `main_time_aware` (all 24 splits × 10 seeds). Platt scaling (logistic regression on the "
        "logit of the predicted probability) and isotonic regression are fitted on each model's validation split "
        "only and applied to its test predictions; the models are not refitted. MCC@tuned uses a threshold re-tuned "
        "on the calibrated validation probabilities. Per-split means over seeds; mean and 95% bootstrap CI over "
        "splits. ECE: 10 equal-width bins.",
        "",
        "## Summary",
        "",
        to_markdown(pd.DataFrame(rows)),
        "",
        "Notes:",
        "",
        f"* Platt scaling is monotone and leaves the ranking (AUC) unchanged when its fitted slope is positive. In "
        f"{neg} of {n_platt} (split, model, seed) units the slope on the validation split is negative ("
        + ", ".join(f"{CC.MODEL_LABEL[m]} {k}" for m, k in neg_by.items())
        + "): there the model ranks the validation classes worse than chance, Platt reverses the ranking, and the "
        "test AUC changes. Excluding these units, the mean change in test AUC is "
        f"{auc_change_pos:.1e}.",
        "* Isotonic regression produces probabilities of exactly 0 or 1 in sparse regions; one confident mistake "
        "then costs up to −log(1e−15) ≈ 34.5 in NLL, which explains the high isotonic NLL. Its ties also change "
        "the AUC slightly.",
        "",
        "## Tests (calibrated vs uncalibrated; Wilcoxon signed-rank, Holm within model, Cliff's delta)",
        "",
        to_markdown(format_comparisons(tests)),
        "",
        "## Reliability diagram data before calibration (pooled over splits and seeds)",
        "",
        "Cells: mean predicted probability / observed defective rate (instances).",
        "",
        to_markdown(rel_w.reset_index().rename(columns={"index": "bin"})),
        "",
    ]
    _write(cfg, "step8_calibration", md, {"": cal, "_reliability": rel, "_tests": tests})


