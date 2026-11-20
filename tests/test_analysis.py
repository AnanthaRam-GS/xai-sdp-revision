"""Tests for the result aggregation and comparison tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sdp.analysis import tables as T


def long_results(seed=0) -> pd.DataFrame:
    """Two models x 12 splits x 3 seeds; model a is clearly better on AUC."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(12):
        for k in range(3):
            for model, base in (("a", 0.8), ("b", 0.6)):
                rows.append({"project": f"p{s % 3}", "split_id": f"s{s}", "seed": k, "model": model,
                             "threshold_mode": "none", "metric": "roc_auc",
                             "value": base + 0.01 * s + rng.normal(0, 0.005)})  # fmt: skip
                rows.append({"project": f"p{s % 3}", "split_id": f"s{s}", "seed": k, "model": model,
                             "threshold_mode": "val_mcc", "metric": "necm_15",
                             "value": (2.0 if model == "a" else 3.0) + 0.01 * s})  # fmt: skip
    return pd.DataFrame(rows)


SPECS = [T.MetricSpec("roc_auc", "none", "AUC"), T.MetricSpec("necm_15", "val_mcc", "NECM15@tuned", False)]


def test_summary_counts_all_splits_and_seeds():
    long = T.summary_long(long_results(), SPECS, ["model"])
    auc = long[(long["metric"] == "AUC")].set_index("model")
    assert auc.loc["a", "n"] == 36
    assert auc.loc["a", "ci_low"] < auc.loc["a", "mean"] < auc.loc["a", "ci_high"]
    wide = T.summary_wide(long, "model", ["b", "a"])
    assert list(wide.columns) == ["metric", "b", "a"] and wide["metric"].tolist() == ["AUC", "NECM15@tuned"]


def test_per_split_means_collapse_seeds():
    wide = T.per_split(long_results(), SPECS[0])
    assert wide.shape == (12, 2)


def test_ranks_respect_direction():
    ranks = T.skesd_ranks(long_results(), SPECS, order=["a", "b"]).set_index("metric")
    assert ranks.loc["AUC", "a"] == 1 and ranks.loc["AUC", "b"] == 2
    assert ranks.loc["NECM15@tuned", "a"] == 1  # lower NECM is better


def test_reference_comparisons_and_holm():
    res = T.reference_comparisons(long_results(), SPECS, "a", order=["a", "b"])
    auc = res[res["metric"] == "AUC"].iloc[0]
    assert auc["n"] == 12 and auc["p_value"] < 0.01 and auc["cliffs_delta"] == 1.0
    assert auc["magnitude"] == "large" and auc["p_holm"] >= auc["p_value"]


def test_all_pairs_within_groups():
    df = long_results().rename(columns={"model": "setting"}).assign(model="m")
    res = T.all_pairs_comparisons(df, SPECS, "setting", within="model")
    assert set(res["model"]) == {"m"} and len(res) == 2
