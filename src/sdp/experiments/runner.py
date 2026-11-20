"""Experiment runner.

For every split of a protocol and every model:

1. build ``Pipeline(preprocessor, model)`` from the model factory;
2. fit it on the training part of the split (``fit_idx``), so the preprocessor never
   sees validation or test data;
3. choose the MCC-maximising threshold on the validation hold-out (``val_idx``);
4. predict the test set and compute all metrics.

The model is *not* refitted on training + validation, because the tuned threshold
(and, later, early stopping) belong to the model fitted without the validation data.

Outputs, in ``results/<experiment>/``:

* ``metrics.csv`` in long format: experiment, project, train_releases, test_release,
  split_id, seed, model, preprocessing, threshold_mode, metric, value;
* ``predictions.csv`` per test instance: split_id, model, class_name, y_true, y_prob,
  loc (original LOC).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.config import Config
from sdp.data.clean import ID_COL, LABEL_COL
from sdp.data.study import StudyDataset
from sdp.eval.metrics import best_mcc_threshold, evaluate
from sdp.eval.preprocess import EFFORT_COL, feature_frame, make_pipeline
from sdp.eval.splits import Split, generate_splits, project_frame

logger = logging.getLogger(__name__)

METRICS_COLUMNS = [
    "experiment", "project", "train_releases", "test_release", "split_id", "seed",
    "model", "preprocessing", "threshold_mode", "metric", "value",
]  # fmt: skip
PREDICTION_COLUMNS = ["split_id", "model", "class_name", "y_true", "y_prob", "loc"]


@dataclass
class ExperimentSpec:
    name: str
    models: dict[str, Callable[[], object]]  # model name -> factory returning an unfitted estimator
    preprocessing: str = "standard"
    protocol: str = "time_aware"


def positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Probability of the defective class, also when the model saw only one class."""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(X))


def run_split(
    split: Split, df: pd.DataFrame, X: pd.DataFrame, spec: ExperimentSpec, cfg: Config
) -> tuple[list[dict], pd.DataFrame]:
    e = cfg.evaluation
    y = df[LABEL_COL].to_numpy(dtype="int64")
    loc = df[EFFORT_COL].to_numpy(dtype="float64")
    metric_rows, pred_frames = [], []
    for name, factory in spec.models.items():
        model = factory()
        # Rule-based baselines need raw LOC, so their input is never preprocessed.
        mode = "none" if getattr(model, "requires_raw_features", False) else spec.preprocessing
        pipe = make_pipeline(mode, model)
        pipe.fit(X.iloc[split.fit_idx], y[split.fit_idx])
        threshold = best_mcc_threshold(y[split.val_idx], positive_proba(pipe, X.iloc[split.val_idx]))
        p_test = positive_proba(pipe, X.iloc[split.test_idx])
        y_test, loc_test = y[split.test_idx], loc[split.test_idx]
        base = {
            "experiment": spec.name,
            "project": split.project,
            "train_releases": "+".join(split.train_releases),
            "test_release": split.test_release,
            "split_id": split.split_id,
            "seed": split.seed,
            "model": name,
            "preprocessing": mode,
        }
        for mode, metric, value in evaluate(y_test, p_test, loc_test, threshold, e.necm_cost_ratio, e.recall_at_loc):
            metric_rows.append(base | {"threshold_mode": mode, "metric": metric, "value": value})
        pred_frames.append(pd.DataFrame({
            "split_id": split.split_id, "model": name,
            "class_name": df[ID_COL].to_numpy()[split.test_idx],
            "y_true": y_test, "y_prob": p_test, "loc": loc_test,
        }))  # fmt: skip
    return metric_rows, pd.concat(pred_frames, ignore_index=True)


def run_experiment(
    cfg: Config,
    study: StudyDataset,
    spec: ExperimentSpec,
    frames: dict[str, pd.DataFrame] | None = None,
    splits: Iterable[Split] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run ``spec`` over all splits of its protocol and write the result files."""
    t0 = time.perf_counter()
    frames = frames or {p: project_frame(cfg, study, p) for p in study.projects}
    features = {p: feature_frame(df) for p, df in frames.items()}
    splits = splits if splits is not None else generate_splits(cfg, study, spec.protocol, frames)

    rows, preds, n = [], [], 0
    for split in splits:
        r, pr = run_split(split, frames[split.project], features[split.project], spec, cfg)
        rows.extend(r)
        preds.append(pr)
        n += 1
    metrics = pd.DataFrame(rows, columns=METRICS_COLUMNS)
    predictions = pd.concat(preds, ignore_index=True)[PREDICTION_COLUMNS] if preds else pd.DataFrame(
        columns=PREDICTION_COLUMNS)  # fmt: skip

    out: Path = cfg.paths.results_dir / spec.name
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    logger.info(
        "Experiment %s: %d splits x %d models in %.1f s -> %s",
        spec.name, n, len(spec.models), time.perf_counter() - t0, out,
    )  # fmt: skip
    return metrics, predictions
