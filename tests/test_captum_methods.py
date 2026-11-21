"""Tests for the Captum explainers on a tiny MLP."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sdp.eval.preprocess import FEATURES
from sdp.explain import captum_methods as cm
from tests.explain_helpers import make_context


@pytest.fixture(scope="module")
def ctx():
    return make_context("mlp", n_test=20, background_size=25)


@pytest.mark.parametrize("name", list(cm.CAPTUM_EXPLAINERS))
def test_shapes_and_feature_names(ctx, name):
    res = cm.CAPTUM_EXPLAINERS[name](n_samples=10, n_steps=16).explain(ctx)
    assert res.attributions.shape == (20, len(FEATURES)) and list(res.attributions.columns) == FEATURES
    assert np.isfinite(res.attributions.to_numpy()).all() and res.output_scale == "probability"


@pytest.mark.parametrize("name", ["DeepLift", "DeepLiftShap", "IntegratedGradients"])
def test_completeness_on_tiny_model(ctx, name):
    res = cm.CAPTUM_EXPLAINERS[name](n_steps=64).explain(ctx)
    chk = res.checks
    assert len(chk) == 20 and set(chk["check"]) == {"completeness"}
    np.testing.assert_allclose(chk["output"], ctx.y_prob, atol=1e-6)  # float64 copy matches the model
    tol = 1e-3 if name == "IntegratedGradients" else 1e-10
    assert chk["error"].quantile(0.95) < tol


def test_integrated_gradients_default_is_512_steps_and_converges(ctx):
    e512 = cm.IntegratedGradientsExplainer().explain(ctx).checks["error"]
    e16 = cm.IntegratedGradientsExplainer(n_steps=16).explain(ctx).checks["error"]
    assert e512.quantile(0.95) <= 1e-3 and e512.max() <= e16.max()


def test_deeplift_reference_is_background_mean(ctx):
    res = cm.DeepLiftExplainer().explain(ctx)
    module = cm.probability_module(ctx)
    ref = float(module(torch.as_tensor(ctx.background.mean().to_numpy()[None], dtype=torch.float64)))
    np.testing.assert_allclose(res.checks["reference"], ref)
    dls = cm.DeepLiftShapExplainer().explain(ctx)
    bg_mean_out = float(module(torch.as_tensor(ctx.background.to_numpy(), dtype=torch.float64)).mean())
    np.testing.assert_allclose(dls.checks["reference"], bg_mean_out)


def test_saliency_is_signed_gradient(ctx):
    sal = cm.SaliencyExplainer().explain(ctx).attributions.to_numpy()
    ixg = cm.InputXGradientExplainer().explain(ctx).attributions.to_numpy()
    assert (sal < 0).any()  # raw gradient, not |gradient|
    np.testing.assert_allclose(ixg, sal * ctx.X_test.to_numpy(), atol=1e-12)
