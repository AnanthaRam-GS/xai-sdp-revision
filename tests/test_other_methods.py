"""Tests for LIME, KernelSHAP, TreeSHAP and linear SHAP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdp.eval.preprocess import FEATURES
from sdp.explain import other_methods as om
from sdp.explain.base import ExplainContext, sample_background
from tests.explain_helpers import make_context


class OnlyFeature3:
    """Synthetic model: P(defective) depends on column 3 (``cbo``) only."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        z = np.asarray(X, dtype=float)[:, 3]
        p = 1 / (1 + np.exp(-2.0 * z))
        return np.column_stack([1 - p, p])


def only_feature3_context(n_test=6, seed=0) -> ExplainContext:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(200 + n_test, len(FEATURES))), columns=FEATURES)
    y = (X["cbo"] > 0).astype(int).to_numpy()
    tr, te = X.iloc[:200], X.iloc[200:].reset_index(drop=True)
    return ExplainContext(
        split_id="s", project="toy", train_release="1", test_release="2", model_name="mlp", seed=seed,
        model=OnlyFeature3(), preprocessor=None, threshold=0.5, X_train=tr, y_train=y[:200], X_test=te,
        X_test_raw=te, background=sample_background(tr, y[:200], 50, seed), class_name=np.arange(n_test).astype(str),
        y_true=y[200:], loc=np.ones(n_test),
    )  # fmt: skip


def test_lime_maps_weights_by_index_not_list_order():
    assert FEATURES.index("cbo") == 3 and FEATURES[0] != "cbo"
    ctx = only_feature3_context()
    attr = om.LimeExplainer(num_samples=2000).explain(ctx).attributions
    top = attr.abs().idxmax(axis=1)
    assert (top == "cbo").all()
    assert (attr["cbo"] > 0).all()  # increasing in cbo


def test_lime_weights_from_as_map():
    class FakeExplanation:
        def as_map(self):
            return {1: [(3, 0.9), (0, -0.2), (17, 0.05)]}  # sorted by |weight|, not by index

    w = om.lime_weights(FakeExplanation(), 20)
    assert w[3] == 0.9 and w[0] == -0.2 and w[17] == 0.05 and np.count_nonzero(w) == 3


def test_lime_is_seeded():
    ctx = make_context("lr", n_test=3)
    a = om.LimeExplainer(num_samples=500).explain(ctx).attributions
    b = om.LimeExplainer(num_samples=500).explain(ctx).attributions
    pd.testing.assert_frame_equal(a, b)


def test_kernelshap_is_efficient_and_dense():
    ctx = make_context("mlp", n_test=4, background_size=20)
    res = om.KernelShapExplainer(nsamples=300).explain(ctx)
    total = res.attributions.sum(axis=1).to_numpy()
    np.testing.assert_allclose(total, ctx.y_prob - ctx.predict(ctx.background).mean(), atol=1e-6)
    assert (res.attributions.abs() > 0).sum(axis=1).min() > 10  # no lasso feature selection


@pytest.mark.parametrize("model", ["rf", "xgb"])
def test_treeshap_additivity_on_tiny_forest(model):
    ctx = make_context(model, n_test=15, background_size=30)
    res = om.TreeShapExplainer().explain(ctx)
    chk = res.checks
    assert set(chk["check"]) == {"additivity"} and len(chk) == 15
    np.testing.assert_allclose(chk["output"], ctx.y_prob)
    assert chk["error"].max() < 1e-6
    np.testing.assert_allclose(chk["reference"], ctx.predict(ctx.background).mean(), atol=1e-6)


def test_linear_shap_is_on_log_odds_scale():
    ctx = make_context("lr", n_test=10)
    res = om.LinearShapExplainer().explain(ctx)
    assert res.output_scale == "log_odds"
    assert res.checks["error"].max() < 1e-8
    coef = ctx.model.coef_[0]
    expected = coef * (ctx.X_test.to_numpy() - ctx.background.to_numpy().mean(axis=0))
    np.testing.assert_allclose(res.attributions.to_numpy(), expected, atol=1e-8)


def test_float32_safe_forest_keeps_predictions_and_rounds_down():
    ctx = make_context("rf", n_test=30)
    safe = om.float32_safe_forest(ctx.model)
    X32 = om._as_float32(ctx.X_test)
    np.testing.assert_array_equal(safe.predict_proba(X32), ctx.model.predict_proba(X32))
    for orig, new in zip(ctx.model.estimators_, safe.estimators_):
        split = orig.tree_.children_left >= 0
        t_old, t_new = orig.tree_.threshold[split], new.tree_.threshold[split]
        assert np.all(t_new <= t_old)
        np.testing.assert_array_equal(t_new, t_new.astype(np.float32).astype(np.float64))
    assert not np.shares_memory(safe.estimators_[0].tree_.threshold, ctx.model.estimators_[0].tree_.threshold)


class Quadratic:
    """Non-linear toy model: P(defective) rises with cbo**2 (U-shaped in cbo)."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        z = np.asarray(X, dtype=float)[:, 3]
        p = 1 / (1 + np.exp(-(z**2 - 1.0)))
        return np.column_stack([1 - p, p])


def test_lime_instance_sampling_gives_instance_dependent_weights():
    ctx = only_feature3_context(n_test=2)
    ctx.model = Quadratic()
    far = ctx.X_test.copy()
    far.loc[0, "cbo"], far.loc[1, "cbo"] = -2.5, 2.5
    import dataclasses

    ctx = dataclasses.replace(ctx, X_test=far, X_test_raw=far)
    local = om.LimeExplainer(num_samples=2000).explain(ctx).attributions["cbo"].to_numpy()
    assert local[0] < 0 < local[1]  # opposite slopes on the two arms of the U
    mean = om.LimeMeanSamplingExplainer(num_samples=2000).explain(ctx).attributions["cbo"].to_numpy()
    assert abs(mean[0] - mean[1]) < abs(local[1] - local[0])  # v1 setting: much less instance-dependent


def test_rename_method_keeps_old_results(tmp_path):
    from sdp.explain import base

    ctx = make_context("lr", n_test=3)
    res = om.LimeExplainer(num_samples=200).explain(ctx)
    base.write_long(base.to_long(ctx, res), base.partition_path(tmp_path, "lr", "LIME", "s", 0))
    moved = base.rename_method(tmp_path, "LIME", "LIME_mean_sampling")
    assert len(moved) == 1 and not (tmp_path / "lr" / "LIME").exists()
    back = base.read_attributions(tmp_path)
    assert set(back["method"]) == {"LIME_mean_sampling"} and len(back) == 3 * 20


def test_lime_records_surrogate_r2():
    ctx = make_context("lr", n_test=4)
    exp = om.LimeExplainer(num_samples=300)
    exp.explain(ctx)
    assert exp.scores_.shape == (4,) and np.all(exp.scores_ <= 1.0)
