"""Validation-split predictions of the persisted ``main_time_aware`` models.

Step 3 stored the test predictions and each model's validation-tuned threshold, but not
the validation predictions themselves. They are derived here by *inference only*: each
persisted preprocessor and model (``artifacts/models/main_time_aware``) predicts its
split's validation rows. Nothing is refitted. Two checks guard the derivation:

* the MCC-optimal threshold recomputed from these validation predictions must equal the
  stored ``threshold_val_mcc`` exactly;
* the same models must reproduce the stored test predictions (``predictions.csv``).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.config import Config
from sdp.data.clean import ID_COL, LABEL_COL
from sdp.eval.metrics import best_mcc_threshold
from sdp.eval.preprocess import feature_frame
from sdp.experiments.persistence import artifact_dir, load_artifact
from sdp.models.registry import positive_proba

LEARNED = {"mlp": "log_standard", "lr": "log_standard", "rf": "standard", "xgb": "standard"}


def split_validation_predictions(cfg: Config, split, df: pd.DataFrame, seeds) -> dict:
    t0 = time.perf_counter()
    root = Path(cfg.paths.artifacts_dir) / "models" / "main_time_aware"
    X = feature_frame(df)
    y = df[LABEL_COL].to_numpy(dtype=int)
    names = df[ID_COL].astype(str).to_numpy()
    val, test = split.val_idx, split.test_idx
    rows, checks = [], []
    for model, prep in LEARNED.items():
        for seed in seeds:
            pre, mdl, meta = load_artifact(artifact_dir(root, "", split.split_id, model, prep, seed))
            feats = list(meta["feature_order"])
            p_val = positive_proba(mdl, pre.transform(X.iloc[val][feats]))
            p_test = positive_proba(mdl, pre.transform(X.iloc[test][feats]))
            rows.append(pd.DataFrame({"split_id": split.split_id, "project": split.project, "model": model, "seed": seed,
                                      "class_name": names[val], "y_true": y[val], "y_prob": p_val}))  # fmt: skip
            checks.append({"split_id": split.split_id, "model": model, "seed": seed,
                           "threshold_stored": float(meta["threshold_val_mcc"]),
                           "threshold_recomputed": float(best_mcc_threshold(y[val], p_val)),
                           "p_test": p_test})  # fmt: skip
    return {"rows": pd.concat(rows, ignore_index=True), "checks": checks, "time_s": time.perf_counter() - t0}


def verify(checks: list[dict], stored_test: pd.DataFrame) -> pd.DataFrame:
    """Threshold and test-prediction differences against step 3."""
    out = []
    grouped = {k: g["y_prob"].to_numpy() for k, g in stored_test.groupby(["split_id", "model", "seed"], sort=False)}
    for c in checks:
        ref = grouped[(c["split_id"], c["model"], c["seed"])]
        out.append({"split_id": c["split_id"], "model": c["model"], "seed": c["seed"],
                    "threshold_diff": abs(c["threshold_stored"] - c["threshold_recomputed"]),
                    "test_prob_max_diff": float(np.max(np.abs(ref - c["p_test"])))})  # fmt: skip
    return pd.DataFrame(out)
