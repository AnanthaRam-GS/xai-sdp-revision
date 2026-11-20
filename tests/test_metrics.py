"""Metric tests against small, hand-computed examples."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sdp.eval import metrics as M

approx = pytest.approx


# ----------------------------------------------------------- threshold-independent


def test_roc_and_pr_auc_hand_computed():
    y, p = [1, 0, 1, 0], [0.9, 0.8, 0.4, 0.1]
    # Positive/negative pairs ranked correctly: (0.9>0.8), (0.9>0.1), (0.4>0.1) -> 3/4.
    assert M.roc_auc(y, p) == approx(0.75)
    # Ranking 1,0,1,0: precision 1/1 at recall 0.5, 2/3 at recall 1 -> 0.5*1 + 0.5*2/3.
    assert M.pr_auc(y, p) == approx(0.5 + 0.5 * 2 / 3)


# ------------------------------------------------------------- threshold-dependent


def test_threshold_metrics_balanced_case():
    # pred@0.5 = [1,1,0,0] -> TP=1, FP=1, TN=1, FN=1
    m = M.threshold_metrics([1, 0, 1, 0], [0.9, 0.8, 0.4, 0.1], 0.5, cost_ratio=15)
    assert m["precision"] == approx(0.5) and m["recall"] == approx(0.5) and m["f1"] == approx(0.5)
    assert m["mcc"] == approx(0.0)
    assert m["balanced_accuracy"] == approx(0.5) and m["g_mean"] == approx(0.5)
    assert m["false_alarm_rate"] == approx(0.5)
    assert m["necm"] == approx((1 + 15 * 1) / 4)


def test_threshold_metrics_unbalanced_case():
    # pred@0.5 = [1,1,1,0,0] vs y = [1,1,0,0,0] -> TP=2, FP=1, TN=2, FN=0
    m = M.threshold_metrics([1, 1, 0, 0, 0], [0.9, 0.6, 0.7, 0.2, 0.1], 0.5)
    assert m["precision"] == approx(2 / 3)
    assert m["recall"] == approx(1.0)
    assert m["f1"] == approx(0.8)
    assert m["mcc"] == approx(4 / 6)  # (2*2 - 1*0) / sqrt(3*2*3*2)
    assert m["false_alarm_rate"] == approx(1 / 3)
    assert m["balanced_accuracy"] == approx((1 + 2 / 3) / 2)
    assert m["g_mean"] == approx(math.sqrt(2 / 3))
    assert m["necm"] == approx(1 / 5)


def test_best_mcc_threshold_breaks_ties_towards_half():
    # t=0.8 and t=0.35 both give MCC = 2/sqrt(12); 0.35 is closer to 0.5.
    assert M.best_mcc_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == approx(0.35)


def test_constant_predictions_give_zero_mcc_and_nan_precision():
    m = M.threshold_metrics([1, 0, 0], [0.1, 0.1, 0.1], 0.5)
    assert m["mcc"] == 0.0
    assert math.isnan(m["precision"])
    assert m["f1"] == 0.0 and m["recall"] == 0.0
    assert m["necm"] == approx(15 / 3)


# ---------------------------------------------------------------------- calibration


def test_calibration_hand_computed():
    y, p = [1, 0], [0.8, 0.3]
    assert M.brier(y, p) == approx((0.2**2 + 0.3**2) / 2)
    assert M.nll(y, p) == approx(-(math.log(0.8) + math.log(0.7)) / 2)
    # Bins 8 and 3, one instance each: (|1-0.8| + |0-0.3|) / 2.
    assert M.ece(y, p) == approx(0.25)


def test_nll_clips_extreme_probabilities():
    assert math.isfinite(M.nll([1, 0], [0.0, 1.0]))


# ---------------------------------------------------------------------- effort-aware

Y, LOC = [1, 0, 1], [10, 10, 20]
# Optimal curve: (0,0) (.25,.5) (.75,1) (1,1) -> area 0.6875; worst: area 0.3125.


def test_popt_optimal_worst_and_intermediate():
    assert M.popt(Y, [0.9, 0.1, 0.5], LOC) == approx(1.0)  # same order as optimal
    assert M.popt(Y, [0.1, 0.9, 0.5], LOC) == approx(0.0)  # same order as worst
    # Order 2,1,0: (0,0) (.5,.5) (.75,.5) (1,1) -> area 0.4375 -> 1 - 0.25/0.375.
    assert M.popt(Y, [0.1, 0.2, 0.9], LOC) == approx(1 - 0.25 / 0.375)


def test_recall_at_effort_and_ifa():
    assert M.recall_at_effort(Y, [0.1, 0.2, 0.9], LOC, effort=0.2) == approx(0.0)  # 8 LOC budget
    assert M.recall_at_effort(Y, [0.1, 0.2, 0.9], LOC, effort=0.5) == approx(0.5)  # class 2 only
    assert M.ifa(Y, [0.1, 0.9, 0.5], LOC) == 1.0
    assert M.ifa(Y, [0.9, 0.1, 0.5], LOC) == 0.0


def test_zero_loc_does_not_crash_and_ties_are_deterministic():
    y, loc = [1, 0, 0, 1], [0, 5, 5, 10]
    p = [0.5, 0.5, 0.5, 0.5]
    order = M.effort_ranking(p, loc)
    assert list(order) == [0, 1, 2, 3]  # density 0.5, then 0.1 (tie -> original order), then 0.05
    assert math.isfinite(M.popt(y, p, loc))


# ---------------------------------------------------------------------- edge cases


def test_single_class_returns_nan_without_crashing():
    y, p, loc = [0, 0, 0], [0.2, 0.7, 0.4], [1, 2, 3]
    rows = M.evaluate(y, p, loc, tuned_threshold=M.best_mcc_threshold(y, p))
    values = {(mode, name): v for mode, name, v in rows}
    for name in ("roc_auc", "pr_auc", "popt", "ifa", "recall_at_20pct_loc"):
        assert math.isnan(values[("none", name)])
    assert math.isnan(values[("fixed_0.5", "mcc")])
    assert math.isfinite(values[("none", "brier")])
    assert values[("val_mcc", "threshold")] == 0.5


def test_all_ties_in_probabilities():
    y, p = [1, 0, 1, 0], [0.5] * 4
    assert M.roc_auc(y, p) == approx(0.5)
    assert M.best_mcc_threshold(y, p) == 0.5


def test_evaluate_names():
    rows = M.evaluate([1, 0, 1, 0], [0.9, 0.8, 0.4, 0.1], [1, 2, 3, 4], 0.5)
    values = {(mode, n): v for mode, n, v in rows}
    assert ("fixed_0.5", "necm_15") in values and ("val_mcc", "mcc") in values
    assert ("none", "recall_at_20pct_loc") in values
    assert np.isfinite(values[("none", "roc_auc")])
