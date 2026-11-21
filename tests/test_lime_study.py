"""Tests for the LIME stability and settings study."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sdp.eval.preprocess import FEATURES
from sdp.robustness import lime_study as L
from sdp.robustness.common import load_robustness

LS = load_robustness()["lime"]


def test_grid_settings_vary_one_factor_from_main():
    grid = dict(L.grid_settings(LS))
    assert list(grid)[0] == "main" and len(grid) == 1 + 3 + 3 + 2
    assert grid["num_samples=1000"]["num_samples"] == 1000 and grid["num_samples=1000"]["sample_around_instance"]
    assert np.isclose(grid["kernel_width=2.0×√20"]["kernel_width"], 2.0 * np.sqrt(20))
    assert grid["sampling=mean"]["sample_around_instance"] is False
    assert grid["sampling=quartile"]["discretize_continuous"] is True
    assert "kernel_width" not in grid["main"]  # LIME's default width (0.75 x sqrt(20))


def test_instance_sample_is_stratified_seeded_and_capped():
    outcome = np.array(["TP"] * 3 + ["FP"] * 30 + ["FN"] * 12 + ["TN"] * 100)
    idx = L.sample_instances(outcome, 10, seed=7)
    counts = pd.Series(outcome[idx]).value_counts()
    assert counts["TP"] == 3 and counts["FP"] == 10 and counts["FN"] == 10 and counts["TN"] == 10
    np.testing.assert_array_equal(idx, L.sample_instances(outcome, 10, seed=7))
    small = L.sample_instances(outcome, 10, seed=7, total=20)
    assert len(small) == 20 and set(small) <= set(idx)
    assert pd.Series(outcome[small]).value_counts()["TP"] == 3


def fake_runs():
    rng = np.random.default_rng(0)
    base = rng.normal(size=20)
    rows = []
    for inst in range(3):
        for run in range(4):
            w = base + (0.0 if inst == 0 else 0.5) * rng.normal(size=20)
            rows.append({"model": "lr", "split_id": "s", "project": "p", "setting": "main", "run": run,
                         "instance_id": inst, "outcome": "TN", "y_true": 0, "score": 0.5, **dict(zip(FEATURES, w))})
        rows.append({"model": "lr", "split_id": "s", "project": "p", "setting": "num_samples=1000", "run": 0,
                     "instance_id": inst, "outcome": "TN", "y_true": 0, "score": 0.4, **dict(zip(FEATURES, -base))})
    return pd.DataFrame(rows)


def test_run_to_run_and_grid_metrics():
    rr = L.run_to_run(fake_runs()).set_index("instance_id")
    assert rr.loc[0, "spearman"] == 1.0 and rr.loc[0, "fa5"] == 1.0 and not rr.loc[0, "top1_changes"]
    assert rr.loc[0, "cv_mean"] == 0.0 and rr.loc[1, "spearman"] < 1.0
    gv = L.grid_vs_main(fake_runs())
    inst0 = gv[(gv["setting"] == "num_samples=1000") & (gv["instance_id"] == 0)].iloc[0]
    assert inst0["spearman"] == 1.0  # |-w| ranks like |w|
    assert set(gv["setting"]) == {"main", "num_samples=1000"}
