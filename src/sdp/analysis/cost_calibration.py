"""Closing analyses (step 8): NECM cost-ratio sweep and calibration.

**NECM sweep.** From the stored ``main_time_aware`` test predictions, NECM_c =
(FP + c·FN)/N for c ∈ {1, 2, 3, 5, 7.5, 10, 15, 20} at the 0.5 threshold, at each
model's validation-MCC threshold (stored in step 3, also for the baselines) and, for the
learned models, at the *cost-aware* threshold minimising NECM_c on the validation split
(validation predictions from :mod:`sdp.analysis.validation_predictions`). Per split, the
seeds are averaged; means and 95% bootstrap CIs are over the 24 splits. The break-even
cost ratio of a learned model is the largest c on a fine grid (0.05 steps) below which
its mean NECM is lower than both all_defective's and ManualUp's at every smaller c.

**Calibration.** For each learned model, split and seed: reliability-diagram data (10
equal-width bins), ECE, Brier and NLL on the test release; Platt scaling (logistic
regression on the logit of the validation probability) and isotonic regression are fitted
on the validation split only and applied to the test predictions; the MCC threshold is
re-tuned on the calibrated validation probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from sdp.analysis import tables as T
from sdp.eval.metrics import best_mcc_threshold, brier, confusion, ece, nll, roc_auc, threshold_metrics
from sdp.eval.stats import bootstrap_ci, holm

COST_RATIOS = (1, 2, 3, 5, 7.5, 10, 15, 20)
TEST_RATIOS = (1, 5, 15)
LEARNED = ("mlp", "lr", "rf", "xgb")
BASELINES = ("manual_up", "manual_down", "all_defective", "majority")
MODEL_LABEL = {"mlp": "MLP", "lr": "LR", "rf": "RF", "xgb": "XGB", "manual_up": "ManualUp", "manual_down": "ManualDown",
               "all_defective": "all_defective", "majority": "majority"}  # fmt: skip


# ------------------------------------------------------------------ NECM


def necm(y_true, y_pred, c: float) -> float:
    tp, fp, tn, fn = confusion(y_true, y_pred)
    n = tp + fp + tn + fn
    return (fp + c * fn) / n if n else np.nan


def best_cost_threshold(y_val, p_val, c: float) -> float:
    """Threshold minimising NECM_c on validation data (predictions are p ≥ t); the
    candidates are the distinct probabilities plus "predict none defective"; ties go
    to the smallest threshold."""
    y = np.asarray(y_val, dtype=int)
    p = np.asarray(p_val, dtype=float)
    candidates = np.r_[np.unique(p), np.inf]
    costs = [necm(y, p >= t, c) for t in candidates]
    return float(candidates[int(np.nanargmin(costs))])


def necm_sweep(pred: pd.DataFrame, thresholds: pd.DataFrame, val: pd.DataFrame, ratios=COST_RATIOS) -> pd.DataFrame:
    """Per (split, model, seed, threshold mode, c): NECM_c on the test release."""
    thr = thresholds.set_index(["split_id", "model", "seed"])["value"]
    val_groups = {k: g for k, g in val.groupby(["split_id", "model", "seed"], sort=False)}
    rows = []
    for (split, model, seed), g in pred.groupby(["split_id", "model", "seed"], sort=False):
        y, p = g["y_true"].to_numpy(int), g["y_prob"].to_numpy(float)
        base = {"split_id": split, "project": g["project"].iloc[0], "model": model, "seed": seed}
        t_mcc = float(thr.get((split, model, seed), np.nan))
        for c in ratios:
            rows.append(base | {"mode": "0.5", "c": c, "necm": necm(y, p >= 0.5, c)})
            rows.append(base | {"mode": "val_mcc", "c": c, "necm": necm(y, p >= t_mcc, c)})
            if model in LEARNED:
                v = val_groups[(split, model, seed)]
                t_c = best_cost_threshold(v["y_true"].to_numpy(int), v["y_prob"].to_numpy(float), c)
                rows.append(base | {"mode": "cost_aware", "c": c, "necm": necm(y, p >= t_c, c), "threshold": t_c})
    return pd.DataFrame(rows)


def per_split(sweep: pd.DataFrame) -> pd.DataFrame:
    return sweep.groupby(["split_id", "project", "model", "mode", "c"], as_index=False)["necm"].mean()


def sweep_summary(ps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, mode, c), g in ps.groupby(["model", "mode", "c"]):
        m, lo, hi = bootstrap_ci(g["necm"].to_numpy())
        rows.append({"model": model, "mode": mode, "c": c, "mean": m, "ci_low": lo, "ci_high": hi, "n_splits": len(g)})
    return pd.DataFrame(rows)


def break_even(ps: pd.DataFrame, modes=("val_mcc", "cost_aware"), grid=np.arange(0.05, 30.0001, 0.05)) -> pd.DataFrame:
    """Largest c below which the model's mean NECM beats all_defective and ManualUp at
    every smaller c. Means over splits; NECM is linear in c for fixed predictions, and the
    cost-aware curve is evaluated on the swept ratios and interpolated linearly."""
    out = []
    means = ps.groupby(["model", "mode", "c"])["necm"].mean()
    def curve(model, mode):
        s = means.loc[(model, mode)]
        return np.interp(grid, s.index.to_numpy(float), s.to_numpy(float))
    ref = np.minimum(curve("all_defective", "val_mcc"), curve("manual_up", "val_mcc"))
    for model in LEARNED:
        for mode in modes:
            cur = curve(model, mode)
            worse = np.flatnonzero(cur >= ref)
            if not len(worse):
                be = np.inf  # beats both baselines over the whole grid
            elif worse[0] == 0:
                be = 0.0  # never beats them, even at the smallest ratio
            else:
                be = float(grid[worse[0]])
            out.append({"model": model, "mode": mode, "break_even_c": be})
    return pd.DataFrame(out)


def sweep_tests(ps: pd.DataFrame, ratios=TEST_RATIOS) -> pd.DataFrame:
    rows = []
    for c in ratios:
        fam = []
        g = ps[ps["c"] == c]
        base = {b: g[(g["model"] == b) & (g["mode"] == "val_mcc")].set_index("split_id")["necm"] for b in ("all_defective", "manual_up")}
        for model in LEARNED:
            for mode in ("val_mcc", "cost_aware"):
                a = g[(g["model"] == model) & (g["mode"] == mode)].set_index("split_id")["necm"]
                for bname, b in base.items():
                    fam.append({"c": c, "a": f"{MODEL_LABEL[model]} ({mode})", "b": MODEL_LABEL[bname]} | T.paired_comparison(a, b))
        for r, p in zip(fam, holm([x["p_value"] for x in fam])):
            r["p_holm"] = p
        rows += fam
    return T._order_cols(pd.DataFrame(rows))


# ------------------------------------------------------------------ calibration


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt(p_val, y_val):
    """Platt scaling on the logit; the returned function carries the fitted slope in
    ``.slope`` (a negative slope reverses the ranking, i.e. changes the AUC)."""
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(_logit(p_val).reshape(-1, 1), np.asarray(y_val, dtype=int))

    def f(p):
        return lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]

    f.slope = float(lr.coef_[0, 0])
    return f


def isotonic(p_val, y_val):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(np.asarray(p_val, float), np.asarray(y_val, int))
    return lambda p: iso.predict(np.asarray(p, dtype=float))


def reliability(y, p, n_bins: int = 10) -> pd.DataFrame:
    p = np.asarray(p, dtype=float)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        rows.append({"bin": k, "lower": k / n_bins, "upper": (k + 1) / n_bins, "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()) if m.any() else np.nan,
                     "observed": float(np.asarray(y)[m].mean()) if m.any() else np.nan})  # fmt: skip
    return pd.DataFrame(rows)


def calibration(pred: pd.DataFrame, val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (split, model, seed, method): test ECE, Brier, NLL, AUC, MCC@tuned; and pooled
    reliability data per (model, method)."""
    rows, rel_inputs = [], {}
    val_groups = {k: g for k, g in val.groupby(["split_id", "model", "seed"], sort=False)}
    for (split, model, seed), g in pred[pred["model"].isin(LEARNED)].groupby(["split_id", "model", "seed"], sort=False):
        v = val_groups[(split, model, seed)]
        yv, pv = v["y_true"].to_numpy(int), v["y_prob"].to_numpy(float)
        yt, pt = g["y_true"].to_numpy(int), g["y_prob"].to_numpy(float)
        both = len(np.unique(yv)) == 2
        methods = {"uncalibrated": (lambda p: p)}
        if both:
            methods |= {"platt": platt(pv, yv), "isotonic": isotonic(pv, yv)}
        for name, f in methods.items():
            cal_v, cal_t = f(pv), f(pt)
            t = best_mcc_threshold(yv, cal_v)
            rows.append({"split_id": split, "project": g["project"].iloc[0], "model": model, "seed": seed, "method": name,
                         "ece": ece(yt, cal_t), "brier": brier(yt, cal_t), "nll": nll(yt, cal_t), "auc": roc_auc(yt, cal_t),
                         "mcc_tuned": threshold_metrics(yt, cal_t, t)["mcc"],
                         "platt_slope": getattr(f, "slope", np.nan)})  # fmt: skip
            rel_inputs.setdefault((model, name), []).append((yt, cal_t))
    rel = []
    for (model, name), parts in rel_inputs.items():
        y = np.concatenate([a for a, _ in parts])
        p = np.concatenate([b for _, b in parts])
        rel.append(reliability(y, p).assign(model=model, method=name))
    return pd.DataFrame(rows), pd.concat(rel, ignore_index=True)


def calibration_tests(cal: pd.DataFrame) -> pd.DataFrame:
    ps = cal.groupby(["split_id", "model", "method"], as_index=False)[["ece", "brier", "nll", "auc", "mcc_tuned"]].mean()
    rows = []
    for model in LEARNED:
        fam = []
        g = ps[ps["model"] == model]
        base = g[g["method"] == "uncalibrated"].set_index("split_id")
        for method in ("platt", "isotonic"):
            other = g[g["method"] == method].set_index("split_id")
            for metric in ("ece", "brier", "nll", "auc", "mcc_tuned"):
                fam.append({"model": MODEL_LABEL[model], "metric": metric, "a": method, "b": "uncalibrated"}
                           | T.paired_comparison(other[metric], base[metric]))  # fmt: skip
        for r, p in zip(fam, holm([x["p_value"] for x in fam])):
            r["p_holm"] = p
        rows += fam
    return T._order_cols(pd.DataFrame(rows))
