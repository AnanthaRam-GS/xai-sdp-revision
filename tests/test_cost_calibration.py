"""Tests for the NECM cost-ratio sweep and the calibration analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdp.analysis import cost_calibration as CC
from sdp.eval.metrics import roc_auc


def test_necm_on_hand_made_confusion_counts():
    # TP = 2, FP = 1, TN = 5, FN = 2  (N = 10)
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    pred = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    assert CC.necm(y, pred, 1) == pytest.approx((1 + 2) / 10)
    assert CC.necm(y, pred, 15) == pytest.approx((1 + 30) / 10)
    assert CC.necm(y, np.ones(10), 5) == pytest.approx(6 / 10)  # all defective: FP = 6, FN = 0


def test_cost_aware_threshold_moves_with_the_cost_ratio():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.6, 0.7, 0.9])
    low, high = CC.best_cost_threshold(y, p, 0.5), CC.best_cost_threshold(y, p, 20)
    assert high <= low  # a higher FN cost flags more classes
    assert CC.best_cost_threshold(np.zeros(4, int), np.array([0.1, 0.2, 0.3, 0.4]), 1) == np.inf


def test_platt_scaling_leaves_auc_unchanged():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(0.3 * y + rng.uniform(0, 0.7, 400), 0.01, 0.99)
    f = CC.platt(p[:200], y[:200])
    assert f.slope > 0
    assert roc_auc(y[200:], f(p[200:])) == pytest.approx(roc_auc(y[200:], p[200:]))
    iso = CC.isotonic(p[:200], y[:200])(p[200:])
    assert iso.min() >= 0 and iso.max() <= 1


def test_reliability_bins():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.05, 0.15, 0.95, 1.0])
    r = CC.reliability(y, p).set_index("bin")
    assert r.loc[0, "n"] == 1 and r.loc[9, "n"] == 2 and r.loc[9, "observed"] == 0.5 and np.isnan(r.loc[5, "observed"])


def test_break_even_on_a_hand_made_sweep():
    rows = []
    for c in CC.COST_RATIOS:
        rows += [{"split_id": "s", "project": "p", "model": "all_defective", "mode": "val_mcc", "c": c, "necm": 0.6},
                 {"split_id": "s", "project": "p", "model": "manual_up", "mode": "val_mcc", "c": c, "necm": 1.0},
                 {"split_id": "s", "project": "p", "model": "lr", "mode": "val_mcc", "c": c, "necm": 0.1 * c},
                 {"split_id": "s", "project": "p", "model": "lr", "mode": "cost_aware", "c": c, "necm": 0.05 * c}]
    for m in ("mlp", "rf", "xgb"):
        for c in CC.COST_RATIOS:
            rows += [{"split_id": "s", "project": "p", "model": m, "mode": mode, "c": c, "necm": 9.0} for mode in ("val_mcc", "cost_aware")]
    be = CC.break_even(pd.DataFrame(rows)).set_index(["model", "mode"])["break_even_c"]
    assert be[("lr", "val_mcc")] == pytest.approx(6.0)  # 0.1 c = 0.6
    assert be[("lr", "cost_aware")] == pytest.approx(12.0)
    assert be[("mlp", "val_mcc")] == 0.0  # never better
