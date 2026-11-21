"""Tests for the unchanged-class matching and comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sdp.eval.preprocess import FEATURES
from sdp.stability import unchanged as U
from tests.stability_helpers import FakeSplit, fake_store


def release(names, values, labels):
    t = pd.DataFrame(np.asarray(values, float), columns=FEATURES)
    t.insert(0, "class_name", names)
    t["defective"] = labels
    return t.assign(instance_id=np.arange(len(t)))


def test_ivy_prefix_mapping_and_unchanged_flags():
    v = np.ones((4, 20))
    v2 = v.copy()
    v2[2, 5] = 9  # metric changed
    a = release(["fr.jayasoft.ivy.Ivy", "fr.jayasoft.ivy.util.Message", "fr.jayasoft.ivy.Main", "fr.jayasoft.Other"], v, [1, 0, 0, 1])
    b = release(["org.apache.ivy.Ivy", "org.apache.ivy.util.Message", "org.apache.ivy.Main", "org.apache.Other"], v2, [1, 1, 0, 1])
    m, counts = U.match_unchanged(a, b, "ivy")
    assert counts["matched"] == 3 and counts["matched_exact_name"] == 0 and counts["matched_via_mapping"] == 3
    assert counts["unchanged"] == 1 and counts["unchanged_via_mapping"] == 1  # Message changed label, Main changed metric
    assert m.iloc[0]["key"] == "org.apache.ivy.Ivy" and m.iloc[0]["instance_id_a"] == 0
    m2, c2 = U.match_unchanged(a, b, "ant")  # no mapping for other projects
    assert c2["matched"] == 0 and m2.empty
    assert U.canonical_name("fr.jayasoft.ivy.X", "ivy") == "org.apache.ivy.X"
    assert U.canonical_name("fr.jayasoft.ivy.X", "poi") == "fr.jayasoft.ivy.X"


def test_duplicated_names_are_left_out():
    v = np.ones((3, 20))
    a = release(["A", "A", "B"], v, [0, 0, 1])
    b = release(["A", "B", "C"], v, [0, 1, 1])
    m, counts = U.match_unchanged(a, b, "x")
    assert list(m["key"]) == ["B"] and counts["matched"] == 1


def test_identical_explanations_of_unchanged_classes():
    base = np.linspace(1, 2, 20)
    sa, sb = FakeSplit("p:1>2", "p", ("1",), "2"), FakeSplit("p:2>3", "p", ("2",), "3")
    store = fake_store({(sa.split_id, "p", "rf", "TreeSHAP"): base, (sb.split_id, "p", "rf", "TreeSHAP"): base}, n=5)
    matches = pd.DataFrame({"key": ["C1", "C3"], "instance_id_a": [1, 3], "instance_id_b": [0, 4], "mapped": False, "label": 0})
    res = U.compare_unchanged(store, sa, sb, matches, [("rf", "TreeSHAP")])
    assert len(res) == 2 and np.allclose(res["spearman"], 1.0) and np.allclose(res["fa5"], 1.0)
    assert np.allclose(res["flip"], 0.0) and np.allclose(res["seed_spearman"], 1.0)
