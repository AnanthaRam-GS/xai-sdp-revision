"""Tests for bootstrap CIs, Holm correction, Cliff's delta, Wilcoxon and Scott-Knott ESD."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sdp.eval import stats as S

approx = pytest.approx


def spread(mean: float, sd: float, n: int = 30) -> np.ndarray:
    """Deterministic sample with exactly the given mean and (population) sd."""
    z = np.linspace(-1, 1, n)
    return mean + sd * z / z.std()


# -------------------------------------------------------------------- bootstrap


def test_bootstrap_ci_constant_and_reproducible():
    assert S.bootstrap_ci([0.7] * 10) == approx((0.7, 0.7, 0.7))
    x = spread(0.5, 0.1, 40)
    a, b = S.bootstrap_ci(x, seed=1), S.bootstrap_ci(x, seed=1)
    assert a == b
    mean, lo, hi = a
    assert lo < mean < hi and mean == approx(0.5)
    # Roughly mean +/- 1.96 * sd / sqrt(n).
    assert hi - lo == approx(2 * 1.96 * 0.1 / math.sqrt(40), rel=0.15)


def test_bootstrap_ci_ignores_nan_and_handles_tiny_samples():
    assert S.bootstrap_ci([1.0, float("nan"), 3.0], n_resamples=500)[0] == approx(2.0)
    mean, lo, hi = S.bootstrap_ci([5.0])
    assert mean == 5.0 and math.isnan(lo) and math.isnan(hi)


# ----------------------------------------------------------------------- Holm


def test_holm_hand_computed():
    # Sorted: 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> monotone max 0.06.
    np.testing.assert_allclose(S.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(S.holm([0.5, 0.9]), [1.0, 1.0])


# ----------------------------------------------------------------- Cliff's delta


def test_cliffs_delta_and_magnitudes():
    assert S.cliffs_delta([1, 2, 3], [1, 1, 1]) == approx(6 / 9)
    assert S.cliffs_delta([1, 1], [1, 1]) == 0.0
    assert [S.cliffs_magnitude(d) for d in (0.1, -0.2, 0.4, 0.9)] == ["negligible", "small", "medium", "large"]
    assert S.cliffs_magnitude(0.147) == "small" and S.cliffs_magnitude(0.474) == "large"


# -------------------------------------------------------------------- Wilcoxon


def test_pairwise_wilcoxon_with_holm():
    rows = []
    for i in range(10):
        rows += [(f"s{i}", "A", 0.8 + 0.01 * i), (f"s{i}", "B", 0.6 + 0.012 * i), (f"s{i}", "C", 0.6 + 0.012 * i)]
    df = pd.DataFrame(rows, columns=["split_id", "model", "value"])
    res = S.pairwise_wilcoxon(df).set_index(["model_a", "model_b"])
    # A > B in all 10 pairs: exact two-sided p = 2 / 2^10.
    assert res.loc[("A", "B"), "p_value"] == approx(2 / 1024)
    assert res.loc[("A", "B"), "magnitude"] == "large"
    assert res.loc[("B", "C"), "p_value"] == 1.0  # identical
    assert (res["p_holm"] >= res["p_value"]).all()


# ------------------------------------------------------------- Scott-Knott ESD


def test_sk_lambda_hand_computed():
    # Means 4,3,1,0: best split {4,3}|{1,0}, B0 = 49/2 + 1/2 - 64/4 = 9.
    # sigma0^2 = (10 + 4*0.5) / (4 + 4) = 1.5; lambda = pi/(2(pi-2)) * 9 / 1.5.
    j, lam = S.sk_lambda(np.array([4.0, 3.0, 1.0, 0.0]), s2_mean=0.5, nu=4)
    assert j == 2
    assert lam == approx(math.pi / (2 * (math.pi - 2)) * 6)


def test_skesd_separates_clear_groups():
    data = {"A": spread(0.9, 0.01), "B": spread(0.9, 0.01), "C": spread(0.5, 0.01), "D": spread(0.2, 0.01)}
    ranks = S.scott_knott_esd(data).set_index("treatment")["rank"].to_dict()
    assert ranks == {"A": 1, "B": 1, "C": 2, "D": 3}


def test_skesd_identical_treatments_share_rank():
    x = spread(0.7, 0.05)
    ranks = S.scott_knott_esd({"A": x, "B": x.copy(), "C": x.copy()})["rank"]
    assert set(ranks) == {1}


def test_skesd_merges_significant_but_negligible_differences():
    # Means 0.800 vs 0.812, sd 0.1, n = 5000: significant, but Cohen's d ~ 0.1 (negligible).
    data = {"A": spread(0.812, 0.1, 5000), "B": spread(0.800, 0.1, 5000)}
    plain = S.scott_knott_esd(data, transform="none", negligible_d=0.0)
    esd = S.scott_knott_esd(data, transform="none")
    assert sorted(plain["rank"]) == [1, 2]
    assert sorted(esd["rank"]) == [1, 1]


def test_skesd_handles_negative_values_and_nan():
    data = pd.DataFrame({"A": spread(0.4, 0.05), "B": spread(-0.3, 0.05)})
    data.iloc[0, 0] = np.nan
    out = S.scott_knott_esd(data)
    assert out.iloc[0]["treatment"] == "A" and out["rank"].tolist() == [1, 2]
