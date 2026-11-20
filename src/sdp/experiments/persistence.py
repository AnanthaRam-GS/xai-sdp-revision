"""Saving and loading fitted models with their fitted preprocessors.

Layout (step 4 loads exactly these)::

    artifacts/models/<experiment>/<split_dir>/<model>/<preprocessing>/seed_<k>/
        preprocessor.joblib
        mlp.pt + mlp.json        (MLP)   or   model.joblib   (other models)
        meta.json                hyperparameters, validation-tuned threshold,
                                 training/tuning time, feature order, ...

``split_dir`` is the split id made filesystem-safe (``:`` → ``__``, ``>`` → ``_to_``).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from sdp.models.mlp import MLPClassifierTorch

META = "meta.json"


def safe_split_id(split_id: str) -> str:
    return split_id.replace(":", "__").replace(">", "_to_").replace("/", "_")


def artifact_dir(root: Path, experiment: str, split_id: str, model: str, preprocessing: str, seed: int) -> Path:
    return Path(root) / experiment / safe_split_id(split_id) / model / preprocessing / f"seed_{seed}"


def save_artifact(path: Path, preprocessor, model, meta: dict) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path / "preprocessor.joblib")
    if isinstance(model, MLPClassifierTorch):
        model.save(path)
    else:
        joblib.dump(model, path / "model.joblib")
    (path / META).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def load_artifact(path: Path):
    """Return ``(preprocessor, model, meta)``."""
    path = Path(path)
    pre = joblib.load(path / "preprocessor.joblib")
    model = MLPClassifierTorch.load(path) if (path / "mlp.pt").exists() else joblib.load(path / "model.joblib")
    meta = json.loads((path / META).read_text(encoding="utf-8"))
    return pre, model, meta
