"""Small synthetic stores for the stability tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sdp.analysis.explanations import Explanations
from sdp.eval.preprocess import FEATURES
from sdp.stability.data import ImportanceStore


@dataclass(frozen=True)
class FakeSplit:
    split_id: str
    project: str
    train_releases: tuple
    test_release: str


def fake_store(spec: dict, n: int = 30, seeds=(0, 1), seed: int = 0, noise: float = 0.0) -> ImportanceStore:
    """``spec``: {(split_id, project, model, method): base importance vector (20,)}."""
    rng = np.random.default_rng(seed)
    metas, attrs, raws = [], [], []
    for (split, project, model, method), base in spec.items():
        for s in seeds:
            a = np.tile(base, (n, 1)) * (1 + noise * rng.normal(size=(n, 20)))
            raw = rng.normal(size=(n, 20))
            metas.append(pd.DataFrame({"split_id": split, "project": project, "model": model, "method": method,
                                       "seed": s, "instance_id": np.arange(n), "class_name": [f"C{i}" for i in range(n)],
                                       "outcome": "TN", "y_true": 0, "y_prob": 0.2}))
            attrs.append(a)
            raws.append(raw)
    return ImportanceStore(Explanations(pd.concat(metas, ignore_index=True), np.vstack(attrs), np.vstack(raws), list(FEATURES)))
