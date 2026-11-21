"""Seed stability and prediction multiplicity (step 7, section 4).

* For each split and explanation, every pair of training seeds (0–9; 0–2 for LIME and
  KernelSHAP): the global stability metrics between the two seeds' importance vectors,
  and the instance-level agreement (Spearman of |attribution|, feature agreement@5, sign
  agreement@5, averaged over the test instances).
* Prediction multiplicity per split and model across the 10 seeds: the share of test
  instances whose predicted label (validation-tuned threshold of each seed) differs
  between at least two seeds, the mean per-instance standard deviation of the predicted
  probability, and the mean pairwise Cohen's kappa of the predicted labels.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from sdp.analysis.explanations import label, rank_correlations, topk_agreement
from sdp.stability.data import ImportanceStore
from sdp.stability.reference import stability_metrics

# Any stored method of a model carries its predictions; these exist for all 10 seeds.
PREDICTION_SOURCE = {"mlp": "DeepLift", "lr": "LinearSHAP", "rf": "TreeSHAP", "xgb": "TreeSHAP"}


def seed_pairs(store: ImportanceStore, splits, explanations) -> pd.DataFrame:
    rows = []
    for model, method in explanations:
        for s in splits:
            seeds = store.seeds(s.split_id, model, method)
            for a, b in combinations(seeds, 2):
                glob = stability_metrics(store.g(s.split_id, model, method, a), store.g(s.split_id, model, method, b))
                A, B = store.attr(s.split_id, model, method, a), store.attr(s.split_id, model, method, b)
                top = topk_agreement(A, B, 5)
                rows.append({"project": s.project, "split_id": s.split_id, "model": model, "method": method,
                             "explanation": label(model, method), "seed_a": a, "seed_b": b,
                             **{f"global_{k}": float(v[0]) for k, v in glob.items()},
                             "instance_spearman": float(np.nanmean(rank_correlations(A, B)["spearman"])),
                             "instance_fa5": float(top["feature_agreement@5"].mean()),
                             "instance_sa5": float(top["sign_agreement@5"].mean())})  # fmt: skip
    return pd.DataFrame(rows)


def multiplicity_from_arrays(prob: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """``prob``, ``pred``: (n_seeds, n_instances)."""
    pred = np.asarray(pred, dtype=int)
    disagree = (pred.min(axis=0) != pred.max(axis=0)).mean()
    sd = np.asarray(prob, dtype=float).std(axis=0, ddof=1).mean()
    kappas = []
    for i, j in combinations(range(len(pred)), 2):
        if len(np.unique(pred[i])) == 1 and len(np.unique(pred[j])) == 1:
            kappas.append(1.0 if pred[i][0] == pred[j][0] else 0.0)  # degenerate: no variation in either seed
        else:
            kappas.append(cohen_kappa_score(pred[i], pred[j]))
    return {"disagreement_share": float(disagree), "mean_prob_sd": float(sd), "mean_kappa": float(np.mean(kappas))}


def multiplicity(store: ImportanceStore, splits) -> pd.DataFrame:
    rows = []
    for model, method in PREDICTION_SOURCE.items():
        for s in splits:
            seeds = store.seeds(s.split_id, model, method)
            metas = [store.meta(s.split_id, model, method, sd) for sd in seeds]
            prob = np.stack([m["y_prob"].to_numpy(float) for m in metas])
            pred = np.stack([np.isin(m["outcome"].astype(str).to_numpy(), ["TP", "FP"]) for m in metas])
            rows.append({"project": s.project, "split_id": s.split_id, "model": model, "n_seeds": len(seeds),
                         **multiplicity_from_arrays(prob, pred)})  # fmt: skip
    return pd.DataFrame(rows)
