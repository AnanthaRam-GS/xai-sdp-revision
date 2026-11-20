"""Evaluation metrics.

All functions take a binary ``y_true`` and a predicted probability of the defective
class ``y_prob``; effort-aware metrics additionally take the original, unscaled LOC.

Degenerate cases (a single class in ``y_true``, no positive predictions, zero total
effort, ...) never raise: the affected metric is returned as NaN and a warning is
logged (once per distinct message, to keep logs readable).

Conventions (see also the README):

* Predictions at threshold ``t`` are ``y_prob >= t``.
* MCC is NaN when ``y_true`` has a single class. When both classes are present but the
  predictions are constant, MCC is 0 (the limit value; Chicco & Jurman, 2020; also
  scikit-learn's behaviour).
* Precision is NaN when there are no positive predictions; F1 is computed as
  ``2TP / (2TP + FP + FN)`` and is therefore 0 in that case if defects exist.
* Effort-aware ranking orders classes by predicted *risk density* ``y_prob / LOC``
  (descending). LOC is floored at 1 for the density only, so that classes with
  LOC = 0 do not divide by zero; their effort on the x-axis stays 0. Ties are broken
  by larger LOC first (pessimistic in effort), then by original order.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)

NAN = float("nan")
_warned: set[str] = set()

THRESHOLD_METRICS = (
    "precision", "recall", "f1", "mcc", "balanced_accuracy", "g_mean", "false_alarm_rate", "necm",
)  # fmt: skip


def _warn(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("%s Returning NaN (further occurrences are not logged).", message)


def _arrays(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype="int64").ravel()
    p = np.asarray(y_prob, dtype="float64").ravel()
    if y.shape != p.shape:
        raise ValueError(f"y_true and y_prob differ in length: {y.shape} vs {p.shape}")
    return y, p


def _both_classes(y: np.ndarray, metric: str) -> bool:
    if len(np.unique(y)) < 2:
        _warn(f"single-class:{metric}", f"{metric}: y_true contains a single class.")
        return False
    return True


# ------------------------------------------------------------ threshold-independent


def roc_auc(y_true, y_prob) -> float:
    y, p = _arrays(y_true, y_prob)
    return float(roc_auc_score(y, p)) if _both_classes(y, "roc_auc") else NAN


def pr_auc(y_true, y_prob) -> float:
    """Area under the precision-recall curve as average precision."""
    y, p = _arrays(y_true, y_prob)
    return float(average_precision_score(y, p)) if _both_classes(y, "pr_auc") else NAN


# --------------------------------------------------------------- threshold-dependent


def confusion(y_true, y_pred) -> tuple[int, int, int, int]:
    """Return (TP, FP, TN, FN)."""
    y = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    return int((y & yp).sum()), int((~y & yp).sum()), int((~y & ~yp).sum()), int((y & ~yp).sum())


def _mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    if (tp + fn) == 0 or (tn + fp) == 0:
        return NAN  # single class in y_true
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denom == 0 else (tp * tn - fp * fn) / denom


def threshold_metrics(y_true, y_prob, threshold: float = 0.5, cost_ratio: float = 15.0) -> dict[str, float]:
    """Confusion-matrix metrics at ``threshold``.

    ``necm`` is the normalised expected cost of misclassification,
    ``(FP + cost_ratio * FN) / N`` (cost ratio 15 following Herbold, 2018,
    arXiv:1801.04107).
    """
    y, p = _arrays(y_true, y_prob)
    tp, fp, tn, fn = confusion(y, p >= threshold)
    n_pos, n_neg = tp + fn, tn + fp

    if n_pos == 0:
        _warn("no-pos", "recall/MCC/balanced accuracy/G-mean: y_true has no defective instances.")
    if n_neg == 0:
        _warn("no-neg", "false alarm rate/MCC/balanced accuracy/G-mean: y_true has no clean instances.")
    if tp + fp == 0:
        _warn("no-pred-pos", "precision: no positive predictions.")

    recall = tp / n_pos if n_pos else NAN
    spec = tn / n_neg if n_neg else NAN
    f1_den = 2 * tp + fp + fn
    return {
        "precision": tp / (tp + fp) if (tp + fp) else NAN,
        "recall": recall,
        "f1": 2 * tp / f1_den if f1_den else NAN,
        "mcc": _mcc(tp, fp, tn, fn),
        "balanced_accuracy": (recall + spec) / 2,
        "g_mean": math.sqrt(recall * spec) if not (math.isnan(recall) or math.isnan(spec)) else NAN,
        "false_alarm_rate": fp / n_neg if n_neg else NAN,
        "necm": (fp + cost_ratio * fn) / len(y) if len(y) else NAN,
    }


def best_mcc_threshold(y_val, p_val, default: float = 0.5) -> float:
    """Threshold maximising MCC on validation data.

    Candidates are the distinct predicted probabilities (predictions are ``p >= t``).
    Ties are broken by the candidate closest to 0.5, then by the smaller value. If MCC
    is undefined (a single class in ``y_val``), ``default`` is returned with a warning.
    """
    y, p = _arrays(y_val, p_val)
    if len(np.unique(y)) < 2 or len(p) == 0:
        _warn("tune-single-class", f"threshold tuning: validation data have a single class; using {default}.")
        return default
    best_t, best_m = default, -np.inf
    for t in np.unique(p):
        m = _mcc(*confusion(y, p >= t))
        if math.isnan(m):
            continue
        if m > best_m + 1e-12 or (abs(m - best_m) <= 1e-12 and (abs(t - 0.5), t) < (abs(best_t - 0.5), best_t)):
            best_t, best_m = float(t), m
    return best_t


# ----------------------------------------------------------------------- calibration


def brier(y_true, y_prob) -> float:
    y, p = _arrays(y_true, y_prob)
    return float(np.mean((p - y) ** 2)) if len(y) else NAN


def nll(y_true, y_prob, eps: float = 1e-15) -> float:
    """Negative log-likelihood (log loss) with probabilities clipped to [eps, 1 - eps]."""
    y, p = _arrays(y_true, y_prob)
    if not len(y):
        return NAN
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected calibration error with ``n_bins`` equal-width bins on [0, 1].

    Bin b covers [b/n, (b+1)/n); the last bin also includes 1.0. ECE is the
    instance-weighted mean of |observed defect rate - mean predicted probability|.
    """
    y, p = _arrays(y_true, y_prob)
    if not len(y):
        return NAN
    bins = np.clip(np.floor(p * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in np.unique(bins):
        m = bins == b
        total += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(total / len(y))


# ----------------------------------------------------------------------- effort-aware


def effort_ranking(y_prob, loc) -> np.ndarray:
    """Inspection order: risk density desc, then larger LOC first, then original order."""
    p = np.asarray(y_prob, dtype="float64")
    loc = np.asarray(loc, dtype="float64")
    density = p / np.maximum(loc, 1.0)
    # np.lexsort sorts by the last key first; all keys ascending, so negate.
    return np.lexsort((np.arange(len(p)), -loc, -density))


def _curve_area(y: np.ndarray, loc: np.ndarray, order: np.ndarray) -> float:
    x = np.concatenate([[0.0], np.cumsum(loc[order]) / loc.sum()])
    d = np.concatenate([[0.0], np.cumsum(y[order]) / y.sum()])
    return float(np.trapezoid(d, x))


def _effort_ok(y: np.ndarray, loc: np.ndarray, metric: str) -> bool:
    if y.sum() == 0:
        _warn(f"effort-no-pos:{metric}", f"{metric}: no defective instances.")
        return False
    if loc.sum() <= 0:
        _warn(f"effort-no-loc:{metric}", f"{metric}: total LOC is zero.")
        return False
    return True


def popt(y_true, y_prob, loc) -> float:
    """Normalised effort-aware performance Popt (Mende & Koschke, 2010).

    Mende, T., & Koschke, R. (2010). Effort-aware defect prediction models. *CSMR 2010*,
    pp. 107-116. Cumulative % of defective classes found is plotted against cumulative
    % of LOC inspected, and

        Popt = 1 - (Area(optimal) - Area(model)) / (Area(optimal) - Area(worst)),

    so that the optimal model scores 1 and the worst 0. The optimal (worst) ranking
    orders classes by actual defect density descending (ascending), breaking ties by
    smaller (larger) LOC first. Defects are counted as defective classes (binary label).
    """
    y, p = _arrays(y_true, y_prob)
    loc = np.asarray(loc, dtype="float64")
    if not _effort_ok(y, loc, "popt"):
        return NAN
    actual = y / np.maximum(loc, 1.0)
    idx = np.arange(len(y))
    optimal = np.lexsort((idx, loc, -actual))
    worst = np.lexsort((idx, -loc, actual))
    a_opt, a_worst = _curve_area(y, loc, optimal), _curve_area(y, loc, worst)
    if a_opt - a_worst <= 1e-12:
        _warn("popt-degenerate", "popt: optimal and worst curves coincide.")
        return NAN
    a_model = _curve_area(y, loc, effort_ranking(p, loc))
    return float(1 - (a_opt - a_model) / (a_opt - a_worst))


def recall_at_effort(y_true, y_prob, loc, effort: float = 0.2) -> float:
    """Share of defective classes found when inspecting the top-ranked classes within
    ``effort`` (default 20%) of total LOC. Classes are inspected whole; the class that
    would exceed the budget is not inspected."""
    y, p = _arrays(y_true, y_prob)
    loc = np.asarray(loc, dtype="float64")
    if not _effort_ok(y, loc, "recall_at_effort"):
        return NAN
    order = effort_ranking(p, loc)
    inspected = np.cumsum(loc[order]) / loc.sum() <= effort + 1e-12
    return float(y[order][inspected].sum() / y.sum())


def ifa(y_true, y_prob, loc) -> float:
    """Initial false alarms: clean classes inspected before the first defective one,
    following the risk-density ranking."""
    y, p = _arrays(y_true, y_prob)
    if y.sum() == 0:
        _warn("ifa-no-pos", "ifa: no defective instances.")
        return NAN
    ranked = y[effort_ranking(p, loc)]
    return float(np.argmax(ranked == 1))


# ------------------------------------------------------------------------- bundle


def evaluate(
    y_true,
    y_prob,
    loc,
    tuned_threshold: float,
    cost_ratio: float = 15.0,
    effort: float = 0.2,
) -> list[tuple[str, str, float]]:
    """All metrics as ``(threshold_mode, metric, value)`` rows.

    ``threshold_mode`` is ``none`` (threshold-independent), ``fixed_0.5`` or ``val_mcc``
    (threshold maximising MCC on the validation split, also reported as ``threshold``).
    """
    necm_name = f"necm_{cost_ratio:g}"
    effort_name = f"recall_at_{100 * effort:g}pct_loc"
    rows = [
        ("none", "roc_auc", roc_auc(y_true, y_prob)),
        ("none", "pr_auc", pr_auc(y_true, y_prob)),
        ("none", "brier", brier(y_true, y_prob)),
        ("none", "nll", nll(y_true, y_prob)),
        ("none", "ece", ece(y_true, y_prob)),
        ("none", "popt", popt(y_true, y_prob, loc)),
        ("none", effort_name, recall_at_effort(y_true, y_prob, loc, effort)),
        ("none", "ifa", ifa(y_true, y_prob, loc)),
    ]
    for mode, t in (("fixed_0.5", 0.5), ("val_mcc", tuned_threshold)):
        for name, value in threshold_metrics(y_true, y_prob, t, cost_ratio).items():
            rows.append((mode, necm_name if name == "necm" else name, value))
    rows.append(("val_mcc", "threshold", float(tuned_threshold)))
    return rows


def metric_names(rows: Iterable[tuple[str, str, float]]) -> list[str]:
    return sorted({m for _, m, _ in rows})
