"""Tests for the ROAR/KAR benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdp.eval.preprocess import FEATURES
from sdp.explain.base import partition_path, to_long, write_long
from sdp.explain.other_methods import LinearShapExplainer
from sdp.fidelity import roar as R
from sdp.fidelity.common import refit
from sdp.fidelity.common import test_metrics as evaluate_on_test
from tests.explain_helpers import make_context, synthetic


def test_feature_sets_keep_names_aligned():
    rank = ["loc", "cbo", "wmc"] + [f for f in FEATURES if f not in ("loc", "cbo", "wmc")]
    s = R.feature_sets(rank, 2)
    assert s["KAR"] == ["cbo", "loc"] and "loc" not in s["ROAR"] and "cbo" not in s["ROAR"]
    assert s["ROAR"] == [f for f in FEATURES if f not in ("loc", "cbo")]  # canonical order kept
    assert len(s["ROAR"]) + len(s["KAR"]) == 20


def test_reference_rankings():
    r = R.reference_rankings("ta-consecutive:ant:1.3>1.4", 0)
    assert len([k for k in r if k.startswith("random_")]) == 5
    assert r["size_first"][:4] == ["loc", "rfc", "wmc", "npm"] and sorted(r["size_first"]) == sorted(FEATURES)
    assert r == R.reference_rankings("ta-consecutive:ant:1.3>1.4", 0)
    assert r["random_0"] != R.reference_rankings("ta-consecutive:ant:1.3>1.4", 1)["random_0"]


def test_rankings_use_training_rows_only(tmp_path):
    ctx = make_context("lr", n_test=10)
    res = LinearShapExplainer().explain(ctx)
    frame = to_long(ctx, res)
    path = partition_path(tmp_path, "lr", "LinearSHAP", "s", 0)
    write_long(frame.assign(subset="test"), path)
    with pytest.raises(ValueError):
        R.global_ranking(tmp_path, "s", "lr", "LinearSHAP", 0)
    write_long(frame.assign(subset="train"), path)
    rank = R.global_ranking(tmp_path, "s", "lr", "LinearSHAP", 0)
    assert sorted(rank) == sorted(FEATURES)
    imp = res.attributions.abs().div(res.attributions.abs().sum(axis=1), axis=0).mean()
    assert rank[0] == imp.idxmax()


def test_refit_on_a_feature_subset():
    X, y = synthetic(220, seed=4)
    fit, val, test = np.arange(140), np.arange(140, 180), np.arange(180, 220)
    feats = ["loc", "cbo"]
    r = refit("lr", {"C": 1.0}, "log_standard", "class_weight", 0, X.iloc[fit][feats], y[fit], X.iloc[val][feats], y[val])
    assert list(r.model.feature_names_in_) == feats
    m = evaluate_on_test(r, X.iloc[test][feats], y[test], X.iloc[test]["loc"].to_numpy())
    assert set(m) == {"auc", "mcc_tuned", "popt"} and 0.5 < m["auc"] <= 1.0


def test_area_scores():
    rows = []
    for ranking, base in (("TreeSHAP", 0.6), ("random_0", 0.7), ("random_1", 0.8)):
        for k in R.KS:
            rows.append({"split_id": "s", "project": "p", "model": "rf", "seed": 0, "ranking": ranking,
                         "test": "ROAR", "k": k, "auc": base})
    a = R.area_scores(pd.DataFrame(rows)).set_index("family")["area"]
    assert a["TreeSHAP"] == pytest.approx(0.6) and a["random"] == pytest.approx(0.75)
