"""End-to-end tests of the step 3 experiment engine on a small synthetic study."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from sdp.config import Config, EvaluationSettings, Paths, TuningSettings, ZenodoSettings, load_config
from sdp.data.clean import COUNT_COL, ID_COL, LABEL_COL, METRICS
from sdp.data.study import StudyDataset
from sdp.eval.splits import RELEASE_COL, time_aware_splits
from sdp.experiments import engine
from sdp.experiments.config import ExperimentConfig, ModelSetting, load_experiment
from sdp.experiments.persistence import load_artifact

SPACES = load_config().tuning.spaces
TINY_SPACES = SPACES | {
    "mlp": {"units": {"type": "categorical", "choices": [8]}, "lr": {"type": "float", "low": 1e-3, "high": 1e-2}},
    "rf": {"n_estimators": {"type": "categorical", "choices": [20]}},
    "xgb": {"n_estimators": {"type": "int", "low": 20, "high": 40}},
}


def synthetic_frame(seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for rel, n in (("1.0", 70), ("1.1", 80), ("1.2", 90)):
        df = pd.DataFrame(rng.gamma(2.0, 20.0, size=(n, len(METRICS))), columns=list(METRICS))
        df[LABEL_COL] = (rng.random(n) < 0.15 + df["wmc"] / df["wmc"].max() * 0.5).astype(int)
        df[COUNT_COL] = df[LABEL_COL]
        df.insert(0, ID_COL, [f"{rel}.C{i}" for i in range(n)])
        df[RELEASE_COL] = rel
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def env(tmp_path, monkeypatch):
    p = tmp_path
    cfg = Config(
        seed=42,
        paths=Paths(p, p, p, p, results_dir=p / "results", artifacts_dir=p / "artifacts"),
        zenodo=ZenodoSettings(),
        projects={"x": ["1.0", "1.1", "1.2"]},
        evaluation=dataclasses.replace(EvaluationSettings(), n_repeats=1, n_resamples=100),
        tuning=TuningSettings(inner_folds=3, n_trials_time_aware=2, n_trials_within_release=2, spaces=TINY_SPACES),
    )
    frame = synthetic_frame()
    study = StudyDataset(table=pd.DataFrame(), releases={"x": ["1.0", "1.1", "1.2"]})
    monkeypatch.setattr(engine, "build_study", lambda cfg: study)
    monkeypatch.setattr(engine, "project_frame", lambda cfg, study, p: frame)
    return cfg, frame, study


def test_experiment_configs_load():
    for name in ("main_time_aware", "main_within_release", "ablation_preprocessing",
                 "ablation_imbalance", "main_time_aware_cumulative"):  # fmt: skip
        exp = load_experiment(f"configs/experiments/{name}.yaml")
        assert exp.name == name and exp.settings
    main = load_experiment("configs/experiments/main_time_aware.yaml")
    assert main.seeds == tuple(range(10)) and main.persist_models
    assert main.seeds_for("manual_up") == (0,)
    prep = {s.model: s.preprocessing for s in main.settings}
    assert prep == {"mlp": "log_standard", "lr": "log_standard", "rf": "standard", "xgb": "standard",
                    "manual_up": "none", "manual_down": "none", "all_defective": "none", "majority": "none"}


@pytest.mark.parametrize("protocol", ["time_aware", "within_release"])
def test_engine_end_to_end_with_persistence_and_cache(env, protocol):
    cfg, frame, _ = env
    exp = ExperimentConfig(
        name=f"t_{protocol}", protocol=protocol,
        settings=(ModelSetting("mlp", "log_standard", "class_weight"), ModelSetting("rf", "standard", "smote"),
                  ModelSetting("xgb", "standard", "class_weight"), ModelSetting("lr", "none", "none"),
                  ModelSetting("manual_up", "none", "none")),
        seeds=(0, 1), persist_models=(protocol == "time_aware"), n_jobs=2,
    )  # fmt: skip
    res = engine.run_experiment(cfg, exp)
    n_splits = 2 if protocol == "time_aware" else 3 * 5
    assert res.metrics["split_id"].nunique() == n_splits
    assert set(res.metrics["imbalance"]) == {"class_weight", "smote", "none"}
    assert set(res.metrics.loc[res.metrics["model"] == "manual_up", "seed"]) == {0}
    hp = res.hyperparameters
    if protocol == "within_release":
        # One tuning per (release, repeat), reused for all 5 folds.
        mlp = hp[(hp["model"] == "mlp") & (hp["seed"] == 0)]
        assert mlp.groupby("tuned_on_split")["split_id"].nunique().eq(5).all()
        assert mlp["tuned_on_split"].nunique() == 3
    for f in ("metrics.csv", "predictions.csv", "hyperparameters.csv", "timings.csv"):
        assert (cfg.paths.results_dir / exp.name / f).exists()

    if protocol == "time_aware":
        # Persisted models reproduce the test predictions exactly.
        split_id = "ta-consecutive:x:1.0>1.1"
        for model, prep in (("mlp", "log_standard"), ("rf", "standard"), ("xgb", "standard"), ("lr", "none")):
            path = engine.artifact_dir(cfg.paths.artifacts_dir / "models" / exp.name, "", split_id, model, prep, 1)
            pre, m, meta = load_artifact(path)
            assert meta["feature_order"] == list(METRICS) and "threshold_val_mcc" in meta
            test = frame[frame[RELEASE_COL] == "1.1"]
            p = m.predict_proba(pre.transform(test[list(METRICS)].astype(float)))[:, 1]
            pr = res.predictions
            expected = pr[(pr["split_id"] == split_id) & (pr["model"] == model) & (pr["seed"] == 1)]["y_prob"]
            np.testing.assert_allclose(p, expected.to_numpy(), rtol=0, atol=1e-12)

    # Re-running uses the cache and gives identical results.
    again = engine.run_experiment(cfg, exp)
    assert again.info["jobs_from_cache"] == again.info["jobs"]
    pd.testing.assert_frame_equal(
        res.metrics.sort_values(list(res.metrics.columns[:-1])).reset_index(drop=True),
        again.metrics.sort_values(list(again.metrics.columns[:-1])).reset_index(drop=True),
    )


def test_seeds_are_reproducible_across_processes(env):
    cfg, _, _ = env
    exp = ExperimentConfig(name="rep", protocol="time_aware", settings=(ModelSetting("mlp", "standard", "class_weight"),),
                           seeds=(3,), n_jobs=2)  # fmt: skip
    a = engine.run_experiment(cfg, exp).predictions
    for f in (cfg.paths.results_dir / "_cache").glob("*.pkl"):
        f.unlink()
    b = engine.run_experiment(cfg, exp).predictions
    key = ["split_id", "class_name"]
    np.testing.assert_array_equal(a.sort_values(key)["y_prob"].to_numpy(), b.sort_values(key)["y_prob"].to_numpy())


class Spy:
    """Records the row labels of every frame passed to fit / predict_proba."""

    log: list = []

    def __init__(self):
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, X_val=None, y_val=None):
        Spy.log.append(("fit", set(X.index)))
        if X_val is not None:
            Spy.log.append(("fit_val", set(X_val.index)))
        return self

    def predict_proba(self, X):
        Spy.log.append(("predict", set(X.index)))
        p = np.linspace(0.05, 0.95, len(X))
        return np.column_stack([1 - p, p])


def test_no_leakage_in_tuning_and_fitting(env, monkeypatch):
    """Tuning and fitting never see test rows; tuning never sees validation rows."""
    from sdp import tuning

    cfg, frame, study = env
    Spy.log = []
    monkeypatch.setattr(tuning, "make_model", lambda *a, **k: Spy())
    monkeypatch.setattr(engine, "make_model", lambda *a, **k: Spy())
    monkeypatch.setattr(engine, "is_learned", lambda name: True)
    split = next(time_aware_splits({"x": frame}, study, "consecutive", 0.2, 42))
    job = engine.Job("x", (split,), "spy", "standard", "class_weight", 0, n_trials=3)
    cfg = dataclasses.replace(cfg, tuning=dataclasses.replace(cfg.tuning, spaces={"spy": {"c": {"type": "float", "low": 0, "high": 1}}}))
    engine.execute_job(job, frame, cfg)

    fit, val, test = set(split.fit_idx), set(split.val_idx), set(split.test_idx)
    n_tuning_calls = 3 * 3 * 2  # trials x inner folds x (fit, predict)
    tuning_log, final_log = Spy.log[:n_tuning_calls + 3 * 3], Spy.log[n_tuning_calls + 3 * 3:]
    for kind, idx in tuning_log:
        assert idx <= fit, f"tuning {kind} touched rows outside the fitting part"
    for kind, idx in final_log:
        assert not idx & test or kind == "predict"
        if kind == "fit":
            assert idx == fit
        if kind == "fit_val":
            assert idx == val
    # The test release is only ever used for prediction, after fitting.
    assert any(kind == "predict" and idx == test for kind, idx in final_log)


def test_autospearman_feature_selection_jobs(env):
    cfg, frame, study = env
    frame["rfc"] = frame["wmc"] * 2 + 1  # perfectly rank-correlated with wmc: one of them must go
    exp = ExperimentConfig("fs", "time_aware", (ModelSetting("rf", "standard", "class_weight"),), (0,),
                           n_trials=1, persist_models=True,
                           feature_selection={"method": "autospearman", "threshold": 0.7, "vif_threshold": 5})  # fmt: skip
    jobs = engine.build_jobs(cfg, exp, study, {"x": frame})
    feats = jobs[0].features
    assert feats is not None and len({"wmc", "rfc"} & set(feats)) == 1
    plain = dataclasses.replace(jobs[0], features=None)
    assert plain.cache_key(cfg) != jobs[0].cache_key(cfg)
    res = engine.execute_job(jobs[0], frame, cfg)
    pre, model, meta = load_artifact(list(res["artifacts"].values())[0])
    assert meta["feature_order"] == list(feats) and model.n_features_in_ == len(feats)
    assert len(res["predictions"]) == len(jobs[0].splits[0].test_idx)
