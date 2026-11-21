"""Gradient-based explainers for the MLP (Captum).

All methods explain ``ProbabilityModel`` (the trained network followed by a sigmoid),
i.e. P(defective), in the MLP's input space (after ``log_standard``). The network is
copied to float64 for explaining, so that completeness errors reflect the methods and
not float32 round-off; the float64 outputs match the stored float32 predictions to
about 1e-7.

* DeepLift: baseline = background mean.
* DeepLiftShap: baselines = the background set (attributions averaged over it).
* GradientShap: baselines = the background set, ``n_samples=50``, ``stdevs=0.0``.
* IntegratedGradients: baseline = background mean, ``n_steps=64``, Gauss-Legendre.
* Saliency: the raw (signed) gradient (``abs=False``); |gradient| is derived later.
* InputXGradient: input × gradient.

Completeness (DeepLift, DeepLiftShap, IntegratedGradients): per instance
``|sum(attr) − (f(x) − f(reference))|`` where the reference is the baseline, or, for
DeepLiftShap, the mean output over the background set.
"""

from __future__ import annotations

import copy
import logging
import warnings

import numpy as np
import torch
from captum.attr import (
    DeepLift,
    DeepLiftShap,
    GradientShap,
    InputXGradient,
    IntegratedGradients,
    Saliency,
)

from sdp.explain.base import ExplainContext, Explainer, check_frame

logger = logging.getLogger(__name__)
# Captum notes every time that it hooks the non-linear modules; that is expected here.
warnings.filterwarnings("ignore", message="Setting forward, backward hooks", category=UserWarning)

DTYPE = torch.float64


def probability_module(ctx: ExplainContext) -> torch.nn.Module:
    """A float64 copy of the MLP with a sigmoid output, in evaluation mode."""
    module = copy.deepcopy(ctx.model.get_probability_module()).to(DTYPE).eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def _tensor(frame) -> torch.Tensor:
    return torch.tensor(np.array(frame, dtype="float64"), dtype=DTYPE)


def _forward(module: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return module(x).reshape(-1).numpy()


class CaptumExplainer(Explainer):
    """Base class: runs a Captum method in chunks of instances."""

    models = ("mlp",)
    chunk_size = 256
    completeness = False  # whether to run the completeness check

    def _method(self, module):  # pragma: no cover - overridden
        raise NotImplementedError

    def _call(self, method, x: torch.Tensor, ctx: ExplainContext, bg: torch.Tensor) -> torch.Tensor:
        return method.attribute(x, target=0)

    def _reference(self, module, bg: torch.Tensor) -> float:
        return float(_forward(module, bg.mean(dim=0, keepdim=True))[0])

    def _attribute(self, ctx: ExplainContext):
        torch.manual_seed(ctx.seed)
        module = probability_module(ctx)
        method = self._method(module)
        x_all, bg = _tensor(ctx.X_test), _tensor(ctx.background)
        parts = []
        for i in range(0, len(x_all), self.chunk_size):
            x = x_all[i : i + self.chunk_size].clone().requires_grad_(True)
            parts.append(self._call(method, x, ctx, bg).detach().numpy())
        attr = np.concatenate(parts, axis=0) if parts else np.zeros((0, x_all.shape[1]))
        checks = None
        if self.completeness:
            out = _forward(module, x_all)
            checks = check_frame(ctx, self.name, "completeness", out, self._reference(module, bg), attr.sum(axis=1))
        return attr, checks


class DeepLiftExplainer(CaptumExplainer):
    name = "DeepLift"
    completeness = True

    def _method(self, module):
        return DeepLift(module)

    def _call(self, method, x, ctx, bg):
        return method.attribute(x, baselines=bg.mean(dim=0, keepdim=True), target=0)


class DeepLiftShapExplainer(CaptumExplainer):
    name = "DeepLiftShap"
    completeness = True
    chunk_size = 64  # each instance is paired with every background row

    def _method(self, module):
        return DeepLiftShap(module)

    def _call(self, method, x, ctx, bg):
        return method.attribute(x, baselines=bg, target=0)

    def _reference(self, module, bg):
        return float(_forward(module, bg).mean())


class GradientShapExplainer(CaptumExplainer):
    name = "GradientShap"
    chunk_size = 128

    def _method(self, module):
        return GradientShap(module)

    def _call(self, method, x, ctx, bg):
        return method.attribute(x, baselines=bg, target=0, n_samples=int(self.params.get("n_samples", 50)),
                                stdevs=float(self.params.get("stdevs", 0.0)))  # fmt: skip


class IntegratedGradientsExplainer(CaptumExplainer):
    name = "IntegratedGradients"
    completeness = True
    chunk_size = 64

    def _method(self, module):
        return IntegratedGradients(module)

    def _call(self, method, x, ctx, bg):
        return method.attribute(x, baselines=bg.mean(dim=0, keepdim=True), target=0,
                                n_steps=int(self.params.get("n_steps", 64)),
                                method=str(self.params.get("method", "gausslegendre")))  # fmt: skip


class SaliencyExplainer(CaptumExplainer):
    name = "Saliency"

    def _method(self, module):
        return Saliency(module)

    def _call(self, method, x, ctx, bg):
        return method.attribute(x, target=0, abs=False)


class InputXGradientExplainer(CaptumExplainer):
    name = "InputXGradient"

    def _method(self, module):
        return InputXGradient(module)


CAPTUM_EXPLAINERS = {
    cls.name: cls
    for cls in (
        DeepLiftExplainer,
        DeepLiftShapExplainer,
        GradientShapExplainer,
        IntegratedGradientsExplainer,
        SaliencyExplainer,
        InputXGradientExplainer,
    )
}
