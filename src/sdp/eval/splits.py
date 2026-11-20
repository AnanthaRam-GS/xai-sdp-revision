"""Split protocols.

* ``time_aware``: for each project, train on release n and test on release n+1
  (variant ``consecutive``), or train on all included releases <= n (variant
  ``cumulative``).
* ``within_release``: 10 x 5-fold :class:`~sklearn.model_selection.RepeatedStratifiedKFold`
  per release, with a fixed seed.

Inside every training set, a stratified, seeded hold-out (default 20%) is set aside as
the *validation* split for early stopping, threshold tuning and calibration checks.
Models are fitted on the remaining part (``fit_idx``). The test set is never used for
any fitting or tuning.

All indices are positions in the project frame returned by :func:`project_frame`.
Each split carries a stable, human-readable ``split_id``, for example
``ta-consecutive:ant:1.3>1.4`` or ``wr:ant:1.3:r01f2``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from sdp.config import Config
from sdp.data.clean import LABEL_COL
from sdp.data.study import StudyDataset, load_release

logger = logging.getLogger(__name__)

RELEASE_COL = "release"
PROTOCOLS = ("time_aware", "within_release")


@dataclass(frozen=True)
class Split:
    split_id: str
    protocol: str  # "time_aware" or "within_release"
    variant: str  # "consecutive", "cumulative" or "repeated_cv"
    project: str
    train_releases: tuple[str, ...]
    test_release: str
    seed: int
    repeat: int | None
    fold: int | None
    fit_idx: np.ndarray  # training part used to fit models
    val_idx: np.ndarray  # validation hold-out (inside the training set)
    test_idx: np.ndarray

    @property
    def train_idx(self) -> np.ndarray:
        return np.concatenate([self.fit_idx, self.val_idx])


def project_frame(cfg: Config, study: StudyDataset, project: str) -> pd.DataFrame:
    """All included releases of a project, concatenated in chronological order."""
    parts = [load_release(cfg, project, v).assign(**{RELEASE_COL: v}) for v in study.releases[project]]
    return pd.concat(parts, ignore_index=True)


def validation_holdout(
    train_idx: np.ndarray, y: np.ndarray, size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split training indices into (fit, validation), stratified by label and seeded.

    Falls back to an unstratified split (with a warning) if the minority class has
    fewer than two instances.
    """
    y_train = y[train_idx]
    stratify = y_train if np.bincount(y_train, minlength=2).min() >= 2 else None
    if stratify is None:
        logger.warning("Validation hold-out: too few minority instances to stratify; using a random split.")
    fit_idx, val_idx = train_test_split(train_idx, test_size=size, random_state=seed, stratify=stratify)
    return np.sort(fit_idx), np.sort(val_idx)


def time_aware_splits(
    frames: dict[str, pd.DataFrame],
    study: StudyDataset,
    variant: str = "consecutive",
    val_size: float = 0.2,
    seed: int = 42,
) -> Iterator[Split]:
    for pair in study.pairs(variant):
        df = frames[pair.project]
        rel = df[RELEASE_COL].to_numpy()
        y = df[LABEL_COL].to_numpy(dtype="int64")
        train_idx = np.flatnonzero(np.isin(rel, pair.train))
        test_idx = np.flatnonzero(rel == pair.test)
        fit_idx, val_idx = validation_holdout(train_idx, y, val_size, seed)
        yield Split(
            split_id=f"ta-{variant}:{pair.project}:{'+'.join(pair.train)}>{pair.test}",
            protocol="time_aware",
            variant=variant,
            project=pair.project,
            train_releases=pair.train,
            test_release=pair.test,
            seed=seed,
            repeat=None,
            fold=None,
            fit_idx=fit_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )


def within_release_splits(
    frames: dict[str, pd.DataFrame],
    study: StudyDataset,
    n_splits: int = 5,
    n_repeats: int = 10,
    val_size: float = 0.2,
    seed: int = 42,
) -> Iterator[Split]:
    for project, versions in study.releases.items():
        df = frames[project]
        rel = df[RELEASE_COL].to_numpy()
        y = df[LABEL_COL].to_numpy(dtype="int64")
        for version in versions:
            idx = np.flatnonzero(rel == version)
            rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
            for k, (tr, te) in enumerate(rskf.split(idx, y[idx])):
                repeat, fold = divmod(k, n_splits)
                fit_idx, val_idx = validation_holdout(idx[tr], y, val_size, seed)
                yield Split(
                    split_id=f"wr:{project}:{version}:r{repeat + 1:02d}f{fold + 1}",
                    protocol="within_release",
                    variant="repeated_cv",
                    project=project,
                    train_releases=(version,),
                    test_release=version,
                    seed=seed,
                    repeat=repeat + 1,
                    fold=fold + 1,
                    fit_idx=fit_idx,
                    val_idx=val_idx,
                    test_idx=np.sort(idx[te]),
                )


def generate_splits(
    cfg: Config, study: StudyDataset, protocol: str, frames: dict[str, pd.DataFrame]
) -> Iterator[Split]:
    """Splits for ``protocol`` using the settings in ``cfg.evaluation``."""
    e = cfg.evaluation
    if protocol == "time_aware":
        return time_aware_splits(frames, study, e.time_aware_variant, e.validation_size, cfg.seed)
    if protocol == "within_release":
        return within_release_splits(frames, study, e.n_splits, e.n_repeats, e.validation_size, cfg.seed)
    raise ValueError(f"unknown protocol {protocol!r}; choose from {PROTOCOLS}")
