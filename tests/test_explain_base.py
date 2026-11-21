"""Tests for the explainer conventions: background sets, results and the store."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdp.eval.preprocess import FEATURES
from sdp.explain import base
from tests.explain_helpers import make_context, synthetic


class ConstantByName(base.Explainer):
    """Returns attribution = feature position, but with columns in REVERSED order."""

    name = "Toy"

    def _attribute(self, ctx):
        values = np.tile(np.arange(len(FEATURES), dtype=float), (len(ctx.X_test), 1))
        return pd.DataFrame(values, columns=FEATURES).iloc[:, ::-1], None


def test_background_is_stratified_seeded_and_capped():
    X, y = synthetic(400)
    bg = base.sample_background(X, y, 100, seed=3)
    assert len(bg) == 100 and bg.index.is_unique
    assert abs(y[bg.index].mean() - y.mean()) < 0.02
    pd.testing.assert_frame_equal(bg, base.sample_background(X, y, 100, seed=3))
    assert not bg.index.equals(base.sample_background(X, y, 100, seed=4).index)
    small = base.sample_background(X.iloc[:60], y[:60], 100, seed=0)
    assert len(small) == 60


def test_background_round_trip(tmp_path):
    X, y = synthetic(200)
    bg = base.sample_background(X, y, 100, seed=0)
    base.save_background(bg, tmp_path / "bg.parquet")
    back = base.load_background(tmp_path / "bg.parquet")
    pd.testing.assert_frame_equal(bg, back, check_names=False)


def test_outcome_groups():
    out = base.outcome_groups([1, 0, 1, 0], [0.7, 0.6, 0.2, 0.1], 0.5)
    assert list(out) == ["TP", "FP", "FN", "TN"]
    assert list(base.outcome_groups([1], [0.5], 0.5)) == ["TP"]  # >= threshold, as in step 3


def test_results_are_matched_by_feature_name():
    ctx = make_context("lr", n_test=5)
    res = ConstantByName().explain(ctx)
    assert list(res.attributions.columns) == FEATURES
    np.testing.assert_array_equal(res.attributions.iloc[0].to_numpy(), np.arange(len(FEATURES)))
    with pytest.raises(ValueError):
        base.MethodResult("bad", pd.DataFrame(np.zeros((2, 3)), columns=["a", "b", "c"]))


def test_store_round_trip(tmp_path):
    ctx = make_context("lr", n_test=7)
    res = ConstantByName().explain(ctx)
    long = base.to_long(ctx, res)
    assert list(long.columns) == base.ATTRIBUTION_COLUMNS and len(long) == 7 * len(FEATURES)
    path = base.partition_path(tmp_path, "lr", "Toy", "ta-consecutive__toy__1_to_2", 0)
    base.write_long(long, path)
    back = base.read_attributions(tmp_path)
    assert len(back) == len(long)
    row = back[(back["instance_id"] == 3) & (back["feature"] == "loc")].iloc[0]
    assert row["attribution"] == FEATURES.index("loc")
    assert row["feature_value_raw"] == ctx.X_test_raw["loc"].iloc[3]
    assert row["class_name"] == ctx.class_name[3] and row["output_scale"] == "probability"
    assert row["outcome"] == ctx.outcome[3] and row["y_prob"] == pytest.approx(ctx.y_prob[3])
    assert len(base.read_attributions(tmp_path, models=["mlp"])) == 0
    assert len(base.read_attributions(tmp_path, methods=["Toy"], seeds=[0])) == len(long)
