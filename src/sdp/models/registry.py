"""Model registry: build any learned model or baseline by name."""

from __future__ import annotations

from sdp.models.baselines import BASELINES
from sdp.models.classic import XGBClassifierES, make_lr, make_rf
from sdp.models.imbalance import model_imbalance
from sdp.models.mlp import MLPClassifierTorch

LEARNED_MODELS = ("mlp", "lr", "rf", "xgb")
BASELINE_MODELS = tuple(BASELINES)
ALL_MODELS = LEARNED_MODELS + BASELINE_MODELS


def is_learned(name: str) -> bool:
    return name in LEARNED_MODELS


def make_model(name: str, params: dict | None = None, imbalance: str = "class_weight", seed: int = 0):
    """Unfitted model ``name`` with hyperparameters ``params``."""
    params = dict(params or {})
    imb = model_imbalance(imbalance)
    if name == "mlp":
        return MLPClassifierTorch(**params, imbalance=imb, random_state=seed)
    if name == "lr":
        return make_lr(**params, imbalance=imb, random_state=seed)
    if name == "rf":
        return make_rf(**params, imbalance=imb, random_state=seed)
    if name == "xgb":
        return XGBClassifierES(**params, imbalance=imb, random_state=seed)
    if name in BASELINES:
        return BASELINES[name]()
    raise ValueError(f"unknown model {name!r}; choose from {ALL_MODELS}")


def requires_raw_features(model) -> bool:
    return bool(getattr(model, "requires_raw_features", False))
