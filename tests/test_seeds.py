"""Tests for seed stability and prediction multiplicity."""

from __future__ import annotations

import numpy as np
import pytest

from sdp.stability import seeds as Sd
from tests.stability_helpers import FakeSplit, fake_store


def test_multiplicity_on_hand_made_predictions():
    pred = np.array([[1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 0, 1]])
    prob = np.array([[0.9, 0.1, 0.4, 0.8], [0.9, 0.1, 0.6, 0.8], [0.9, 0.1, 0.4, 0.8]])
    m = Sd.multiplicity_from_arrays(prob, pred)
    assert m["disagreement_share"] == pytest.approx(0.25)  # only instance 2 changes label
    assert m["mean_prob_sd"] == pytest.approx(np.std([0.4, 0.6, 0.4], ddof=1) / 4)
    from sklearn.metrics import cohen_kappa_score

    k01 = cohen_kappa_score(pred[0], pred[1])
    assert m["mean_kappa"] == pytest.approx((k01 + k01 + 1.0) / 3)
    same = Sd.multiplicity_from_arrays(prob[[0, 0]], pred[[0, 0]])
    assert same["disagreement_share"] == 0 and same["mean_prob_sd"] == 0 and same["mean_kappa"] == 1.0


def test_seed_pairs_counts_all_pairs():
    s = FakeSplit("p:1>2", "p", ("1",), "2")
    store = fake_store({(s.split_id, "p", "rf", "TreeSHAP"): np.linspace(1, 2, 20)}, seeds=(0, 1, 2), noise=0.01)
    res = Sd.seed_pairs(store, [s], [("rf", "TreeSHAP")])
    assert len(res) == 3 and (res["global_spearman"] > 0.9).all() and (res["instance_fa5"] > 0.5).all()
