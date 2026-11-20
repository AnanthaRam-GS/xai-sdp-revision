"""Tests for preprocessing (no leakage, feature set) and split protocols (no overlap)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdp.data.clean import COUNT_COL, ID_COL, LABEL_COL, METRICS
from sdp.data.study import StudyDataset
from sdp.eval.preprocess import FEATURES, NON_FEATURES, make_pipeline, make_preprocessor, split_xy
from sdp.eval.splits import RELEASE_COL, time_aware_splits, within_release_splits

RNG = np.random.default_rng(0)


def release_frame(n: int, release: str, pos_rate: float = 0.3, offset: float = 0.0) -> pd.DataFrame:
    df = pd.DataFrame(RNG.gamma(2.0, 10.0, size=(n, len(METRICS))) + offset, columns=list(METRICS))
    df.insert(0, ID_COL, [f"{release}.C{i}" for i in range(n)])
    df[COUNT_COL] = (RNG.random(n) < pos_rate).astype(int) * 2
    df[LABEL_COL] = (df[COUNT_COL] > 0).astype(int)
    df[RELEASE_COL] = release
    return df


@pytest.fixture
def project():
    frame = pd.concat([release_frame(60, "1.0"), release_frame(80, "1.1"), release_frame(70, "2.0")],
                      ignore_index=True)  # fmt: skip
    study = StudyDataset(table=pd.DataFrame(), releases={"p": ["1.0", "1.1", "2.0"]})
    return {"p": frame}, study


# ------------------------------------------------------------------ features


def test_features_are_the_20_metrics_only():
    df = release_frame(10, "1.0")
    xy = split_xy(df)
    assert list(xy.X.columns) == FEATURES == list(METRICS)
    for col in NON_FEATURES + (RELEASE_COL,):
        assert col not in xy.X.columns
    assert ID_COL in NON_FEATURES and COUNT_COL in NON_FEATURES and LABEL_COL in NON_FEATURES


@pytest.mark.parametrize("mode", ["none", "standard", "log_standard"])
def test_preprocessors_keep_feature_names_and_never_see_ids(mode):
    df = release_frame(30, "1.0")
    out = make_preprocessor(mode).fit(split_xy(df).X).transform(split_xy(df).X)
    assert list(out.columns) == FEATURES


def test_scaler_is_fitted_on_training_data_only():
    train, test = release_frame(50, "a"), release_frame(50, "b", offset=1000.0)
    pre = make_preprocessor("standard").fit(split_xy(train).X)
    np.testing.assert_allclose(pre.mean_, split_xy(train).X.mean().to_numpy())
    # Test data far from the training distribution stay far after scaling.
    assert pre.transform(split_xy(test).X).mean().min() > 10


def test_log_standard_logs_nonnegative_columns_and_clips_unseen_negatives():
    train = release_frame(40, "a")
    test = release_frame(5, "b")
    test.loc[0, "wmc"] = -3.0
    pre = make_preprocessor("log_standard").fit(split_xy(train).X)
    assert pre.named_steps["log1p"].log_columns_ == FEATURES
    assert np.isfinite(pre.transform(split_xy(test).X).to_numpy()).all()


def test_loc_is_kept_unscaled():
    df = release_frame(20, "1.0")
    xy = split_xy(df)
    make_pipeline("standard", None).steps[0][1].fit(xy.X)
    np.testing.assert_array_equal(xy.loc, df["loc"].to_numpy())


# ------------------------------------------------------------------- splits


def _check_disjoint(split, n_total):
    fit, val, test = set(split.fit_idx), set(split.val_idx), set(split.test_idx)
    assert not fit & val
    assert not (fit | val) & test, f"test overlaps training in {split.split_id}"
    assert max(fit | val | test) < n_total


@pytest.mark.parametrize("variant", ["consecutive", "cumulative"])
def test_time_aware_has_no_overlap_and_respects_time(project, variant):
    frames, study = project
    df = frames["p"]
    splits = list(time_aware_splits(frames, study, variant))
    assert [s.split_id for s in splits] == [
        f"ta-{variant}:p:1.0>1.1",
        f"ta-{variant}:p:{'1.0+1.1' if variant == 'cumulative' else '1.1'}>2.0",
    ]
    for s in splits:
        _check_disjoint(s, len(df))
        assert set(df.loc[s.test_idx, RELEASE_COL]) == {s.test_release}
        assert set(df.loc[s.train_idx, RELEASE_COL]) == set(s.train_releases)
        # Validation is a stratified ~20% of the training set.
        assert abs(len(s.val_idx) / len(s.train_idx) - 0.2) < 0.02
        assert df.loc[s.val_idx, LABEL_COL].sum() > 0


def test_within_release_folds_are_disjoint_complete_and_stable(project):
    frames, study = project
    df = frames["p"]
    splits = list(within_release_splits(frames, study, n_splits=5, n_repeats=10))
    assert len(splits) == 3 * 50
    assert len({s.split_id for s in splits}) == len(splits)
    for s in splits:
        _check_disjoint(s, len(df))
    # Each instance of a release is tested exactly once per repeat.
    first = [s for s in splits if s.test_release == "1.0" and s.repeat == 1]
    tested = np.concatenate([s.test_idx for s in first])
    assert sorted(tested) == sorted(np.flatnonzero(df[RELEASE_COL] == "1.0"))
    # Same seed -> identical splits.
    again = list(within_release_splits(frames, study, n_splits=5, n_repeats=10))
    for a, b in zip(splits, again):
        assert a.split_id == b.split_id
        np.testing.assert_array_equal(a.test_idx, b.test_idx)
        np.testing.assert_array_equal(a.val_idx, b.val_idx)
