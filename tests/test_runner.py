"""End-to-end test of the runner and baselines on a small synthetic study."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from sdp.config import Config, EvaluationSettings, Paths, ZenodoSettings
from sdp.data.clean import COUNT_COL, ID_COL, LABEL_COL, METRICS
from sdp.data.study import StudyDataset
from sdp.eval.splits import RELEASE_COL
from sdp.models.baselines import BASELINES
from sdp.experiments.runner import METRICS_COLUMNS, PREDICTION_COLUMNS, ExperimentSpec, run_experiment


def synthetic_project(rng) -> pd.DataFrame:
    parts = []
    for rel, n in (("1.0", 60), ("1.1", 70), ("1.2", 80)):
        df = pd.DataFrame(rng.gamma(2.0, 20.0, size=(n, len(METRICS))), columns=list(METRICS))
        df[LABEL_COL] = (rng.random(n) < 0.15 + df["loc"] / df["loc"].max() * 0.5).astype(int)
        df[COUNT_COL] = df[LABEL_COL] * 2
        df.insert(0, ID_COL, [f"{rel}.C{i}" for i in range(n)])
        df[RELEASE_COL] = rel
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def setup(tmp_path):
    p = tmp_path
    cfg = Config(
        seed=42,
        paths=Paths(p, p, p, p, results_dir=p / "results"),
        zenodo=ZenodoSettings(),
        projects={"x": ["1.0", "1.1", "1.2"]},
        evaluation=dataclasses.replace(EvaluationSettings(), n_repeats=2, n_resamples=200),
    )
    study = StudyDataset(table=pd.DataFrame(), releases={"x": ["1.0", "1.1", "1.2"]})
    return cfg, study, {"x": synthetic_project(np.random.default_rng(1))}


@pytest.mark.parametrize("protocol, n_splits", [("time_aware", 2), ("within_release", 3 * 5 * 2)])
def test_runner_writes_long_metrics_and_predictions(setup, protocol, n_splits):
    cfg, study, frames = setup
    spec = ExperimentSpec(name=f"t_{protocol}", models=BASELINES, preprocessing="standard", protocol=protocol)
    metrics, preds = run_experiment(cfg, study, spec, frames)

    out = cfg.paths.results_dir / spec.name
    assert list(pd.read_csv(out / "metrics.csv").columns) == METRICS_COLUMNS
    assert list(pd.read_csv(out / "predictions.csv").columns) == PREDICTION_COLUMNS
    assert metrics["split_id"].nunique() == n_splits
    assert set(metrics["model"]) == set(BASELINES)
    assert set(metrics["threshold_mode"]) == {"none", "fixed_0.5", "val_mcc"}

    def get(model, metric, mode="none"):
        m = metrics[(metrics["model"] == model) & (metrics["metric"] == metric) & (metrics["threshold_mode"] == mode)]
        return m["value"].to_numpy()

    # Constant predictors have AUC 0.5; all_defective has recall 1 at 0.5; majority MCC 0.
    np.testing.assert_allclose(get("all_defective", "roc_auc"), 0.5)
    np.testing.assert_allclose(get("majority", "roc_auc"), 0.5)
    np.testing.assert_allclose(get("all_defective", "recall", "fixed_0.5"), 1.0)
    np.testing.assert_allclose(get("majority", "mcc", "fixed_0.5"), 0.0)
    # Baselines are never preprocessed.
    assert set(metrics.loc[metrics["model"].isin(BASELINES), "preprocessing"]) == {"none"}
    # Every test instance appears once per split and model, with its original LOC.
    df = frames["x"]
    first = preds[(preds["model"] == "manual_down") & (preds["split_id"] == preds["split_id"].iloc[0])]
    lookup = df.set_index(ID_COL)["loc"]
    np.testing.assert_allclose(first["loc"].to_numpy(), lookup.loc[first["class_name"]].to_numpy())

