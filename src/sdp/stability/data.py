"""Access to the stored explanations for the stability analyses.

:class:`ImportanceStore` wraps a wide :class:`~sdp.analysis.explanations.Explanations`
(step 4 v2 store, or the permuted-label store) with the per-instance L1-normalised
|attribution| precomputed and an index by (split, model, method, seed), rows sorted by
instance id.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sdp.analysis.explanations import Explanations, load_explanations, normalised_abs
from sdp.fidelity.common import EVALUATED

PRIMARY = [("mlp", "DeepLiftShap"), ("mlp", "KernelSHAP"), ("mlp", "IntegratedGradients"), ("mlp", "LIME"),
           ("mlp", "Saliency"), ("lr", "LinearSHAP"), ("rf", "TreeSHAP"), ("xgb", "TreeSHAP")]  # fmt: skip
ALL = [(m, k) for m, ks in EVALUATED.items() for k in ks]


class ImportanceStore:
    def __init__(self, ex: Explanations):
        meta = ex.meta.copy()
        for c in ("split_id", "project", "model", "method", "class_name", "outcome"):
            if c in meta.columns:
                meta[c] = meta[c].astype(str)
        self.ex = Explanations(meta, ex.attr, ex.raw, ex.features)
        self.norm = normalised_abs(ex.attr)
        self.index = {}
        for key, idx in meta.groupby(["split_id", "model", "method", "seed"], sort=False).indices.items():
            idx = np.asarray(idx)
            self.index[(key[0], key[1], key[2], int(key[3]))] = idx[np.argsort(meta["instance_id"].to_numpy()[idx], kind="stable")]

    @classmethod
    def load(cls, root: Path, methods_exclude=("LIME_mean_sampling",)) -> "ImportanceStore":
        ex = load_explanations(root)
        keep = ~ex.meta["method"].astype(str).isin(list(methods_exclude)).to_numpy()
        return cls(ex.subset(keep))

    def seeds(self, split: str, model: str, method: str) -> list[int]:
        return sorted(s for (sp, m, k, s) in self.index if (sp, m, k) == (split, model, method))

    def rows(self, split: str, model: str, method: str, seed: int) -> np.ndarray:
        return self.index[(split, model, method, seed)]

    def has(self, split: str, model: str, method: str) -> bool:
        return bool(self.seeds(split, model, method))

    def instance_norm(self, split: str, model: str, method: str, seeds=None) -> np.ndarray:
        """Per-instance normalised |attribution| averaged over seeds (n × p)."""
        seeds = seeds if seeds is not None else self.seeds(split, model, method)
        return np.nanmean(np.stack([self.norm[self.rows(split, model, method, s)] for s in seeds]), axis=0)

    def g(self, split: str, model: str, method: str, seed: int | None = None) -> np.ndarray:
        """Global importance of one seed, or averaged over the available seeds."""
        if seed is None:
            return np.nanmean([self.g(split, model, method, s) for s in self.seeds(split, model, method)], axis=0)
        return np.nanmean(self.norm[self.rows(split, model, method, seed)], axis=0)

    def meta(self, split: str, model: str, method: str, seed: int):
        return self.ex.meta.iloc[self.rows(split, model, method, seed)].reset_index(drop=True)

    def attr(self, split: str, model: str, method: str, seed: int) -> np.ndarray:
        return self.ex.attr[self.rows(split, model, method, seed)]

    def raw(self, split: str, model: str, method: str, seed: int) -> np.ndarray:
        return self.ex.raw[self.rows(split, model, method, seed)]
