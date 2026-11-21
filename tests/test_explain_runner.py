"""Tests for the explanation runner (settings, jobs, caching, reproduction check)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdp.eval.preprocess import FEATURES, make_preprocessor
from sdp.eval.splits import Split
from sdp.experiments.persistence import artifact_dir, save_artifact
from sdp.explain import runner
from sdp.explain.base import read_attributions
from sdp.models.imbalance import fit_model
from sdp.models.registry import make_model, positive_proba
from tests.explain_helpers import FAST_PARAMS, synthetic

SETTINGS = runner.load_explain_settings(Path(__file__).resolve().parents[1] / "configs" / "explain.yaml")


def test_settings_and_seed_policy():
    s = SETTINGS
    assert s.methods_for("mlp", 0)[-2:] == ("LIME", "KernelSHAP")
    assert "LIME" not in s.methods_for("rf", 5) and "TreeSHAP" in s.methods_for("rf", 5)
    assert s.methods_for("lr", 9) == ("LinearSHAP",)
    assert s.method_params("LIME")["num_samples"] == 5000
    red = s.with_reduction()
    assert red.method_params("LIME")["num_samples"] == 2000 and red.method_params("KernelSHAP")["nsamples"] == 500
    assert red.method_params("LIME")["num_features"] == 20


def test_settings_reject_inapplicable_methods(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(Path(runner.REPO_ROOT / "configs" / "explain.yaml").read_text().replace(
        "lr: [LinearSHAP, LIME]", "lr: [TreeSHAP, LIME]"))  # fmt: skip
    with pytest.raises(ValueError):
        runner.load_explain_settings(bad)


@pytest.fixture()
def toy(tmp_path):
    """A persisted toy model, its split and project frame."""
    X, y = synthetic(220, seed=1)
    df = X.assign(class_name=[f"C{i}" for i in range(len(X))], defective=y)
    fit, val, test = np.arange(0, 120), np.arange(120, 160), np.arange(160, 220)
    split = Split("ta-consecutive:toy:1>2", "time_aware", "consecutive", "toy", ("1",), "2", 42, None, None,
                  fit, val, test)  # fmt: skip
    models = {}
    for name, prep in (("lr", "log_standard"), ("xgb", "standard")):
        pre = make_preprocessor(prep).fit(X.iloc[fit])
        model = make_model(name, FAST_PARAMS[name], "class_weight", 0)
        fit_model(model, pre.transform(X.iloc[fit]), y[fit], pre.transform(X.iloc[val]), y[val], seed=0)
        path = artifact_dir(tmp_path / "models", "", split.split_id, name, prep, 0)
        meta = {"split_id": split.split_id, "seed": 0, "model": name, "threshold_val_mcc": 0.5,
                "feature_order": FEATURES}  # fmt: skip
        save_artifact(path, pre, model, meta)
        models[name] = (prep, path, positive_proba(model, pre.transform(X.iloc[test])))
    settings = dataclasses.replace(SETTINGS, params=SETTINGS.params | {"LIME": {"num_samples": 300, "num_features": 20}})
    return df, split, models, settings, tmp_path


def test_execute_job_writes_store_and_uses_cache(toy):
    df, split, models, settings, tmp = toy
    prep, path, stored = models["lr"]
    job = runner.ExplainJob(split, "lr", prep, 0, settings.methods_for("lr", 0), str(path))
    root = tmp / "attributions"
    res, cached = runner.run_cached(job, df, settings, str(root), stored)
    assert not cached and res["max_prob_diff"] < 1e-12 and {t["method"] for t in res["timings"]} == {"LinearSHAP", "LIME"}
    store = read_attributions(root)
    assert len(store) == 2 * 60 * len(FEATURES)
    assert set(store.loc[store["method"] == "LinearSHAP", "output_scale"]) == {"log_odds"}
    assert set(store.loc[store["method"] == "LIME", "output_scale"]) == {"probability"}
    assert set(store["outcome"]) <= {"TP", "FP", "FN", "TN"}
    assert all(p.exists() for p in job.outputs(root))
    bg = pd.read_parquet(job.background_path(root))
    assert len(bg) == 100 and set(bg["row"]) <= set(range(len(split.fit_idx)))
    _, cached = runner.run_cached(job, df, settings, str(root), stored)
    assert cached


def test_prediction_mismatch_is_an_error(toy):
    df, split, models, settings, tmp = toy
    prep, path, stored = models["xgb"]
    job = runner.ExplainJob(split, "xgb", prep, 0, ("TreeSHAP",), str(path))
    with pytest.raises(runner.PredictionMismatchError):
        runner.execute_job(job, df, settings, tmp / "a", stored + 0.01)
    out = runner.execute_job(job, df, settings, tmp / "a", stored)
    chk = pd.read_parquet(job.checks_path(tmp / "a"))
    assert out["max_prob_diff"] < 1e-12 and chk["error"].max() < 1e-6


def test_missing_artifacts_are_reported(toy):
    df, split, models, settings, tmp = toy
    job = runner.ExplainJob(split, "mlp", "log_standard", 3, ("DeepLift",), str(tmp / "nowhere"))
    assert runner.missing_artifacts([job]) == [str(tmp / "nowhere")]


def test_estimate_scales_with_instances():
    timings = pd.DataFrame({"model": ["lr"], "method": ["LIME"], "n_instances": [100], "n_fit": [50], "time_s": [10.0]})
    split = Split("s", "time_aware", "consecutive", "p", ("1",), "2", 0, None, None, np.arange(50), np.arange(0),
                  np.arange(300))  # fmt: skip
    jobs = [runner.ExplainJob(split, "lr", "log_standard", s, ("LIME",), "") for s in range(3)]
    est = runner.estimate_runtime(timings, jobs, n_jobs=3, efficiency=1.0)
    assert est["cpu_hours"] == pytest.approx(90 / 3600) and est["wall_hours"] == pytest.approx(30 / 3600)
    assert runner.estimate_runtime(timings, jobs, 3, 1.0, {"LIME": 0.4})["cpu_hours"] == pytest.approx(36 / 3600)


def test_a_job_without_checks_keeps_earlier_checks(toy):
    df, split, models, settings, tmp = toy
    prep, path, stored = models["lr"]
    root = tmp / "attributions"
    runner.execute_job(runner.ExplainJob(split, "lr", prep, 0, ("LinearSHAP",), str(path)), df, settings, root, stored)
    job = runner.ExplainJob(split, "lr", prep, 0, ("LIME",), str(path))
    before = pd.read_parquet(job.checks_path(root))
    assert len(before) > 0
    runner.execute_job(job, df, settings, root, stored)
    pd.testing.assert_frame_equal(pd.read_parquet(job.checks_path(root)), before)
