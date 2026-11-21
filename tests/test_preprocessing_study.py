"""Tests for the explanation study across preprocessing settings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sdp.analysis.explanations import Explanations
from sdp.eval.preprocess import FEATURES
from sdp.explain.runner import load_explain_settings
from sdp.robustness import preprocessing_study as P
from sdp.robustness.common import load_robustness

ROOT = Path(__file__).resolve().parents[1]
ES = load_explain_settings(ROOT / "configs" / "explain.yaml")
PS = load_robustness()["preprocessing"]


def test_settings_per_preprocessing():
    none = P.prep_settings(ES, PS, "none")
    assert none.experiment == "ablation_preprocessing" and none.models == {"mlp": "none", "lr": "none"}
    assert none.methods["mlp"] == ["DeepLiftShap", "DeepLift", "DeepLift_zero_baseline"]
    assert none.methods["lr"] == ["LinearSHAP"] and none.output_dir.endswith("/none")
    std = P.prep_settings(ES, PS, "standard")
    assert "DeepLift_zero_baseline" not in std.methods["mlp"] and std.methods_for("mlp", 7) == ("DeepLiftShap", "DeepLift")


def test_comparisons_pair_settings_within_family():
    rows, attrs = [], []
    rng = np.random.default_rng(0)
    for s in range(8):
        for prep, loc_w in (("none", 5.0), ("standard", 1.0), ("log_standard", 1.0)):
            for method in ("DeepLift",) + (("DeepLift_zero_baseline",) if prep == "none" else ()):
                for seed in (0, 1):
                    a = rng.uniform(0.5, 1.0, 20)
                    a[FEATURES.index("loc")] = loc_w * (3 if method == "DeepLift_zero_baseline" else 1)
                    rows.append({"split_id": f"s{s}", "project": f"p{s % 2}", "model": "mlp", "method": method,
                                 "preprocessing": prep, "seed": seed, "instance_id": 0, "outcome": "TN", "y_true": 0})
                    attrs.append(a)
    ex = Explanations(pd.DataFrame(rows), np.array(attrs), np.zeros((len(attrs), 20)))
    per_split, tests = P.comparisons(ex)
    assert set(per_split["setting"]) == {"none", "standard", "log_standard", "none (zero baseline)"}
    t = tests[(tests["metric"] == "loc share") & (tests["a"] == "none") & (tests["b"] == "standard")].iloc[0]
    assert t["n"] == 8 and t["cliffs_delta"] == 1.0 and t["p_holm"] >= t["p_value"]
    assert len(tests[tests["metric"] == "loc share"]) == 6  # 4 settings -> 6 pairs in the DeepLift family
