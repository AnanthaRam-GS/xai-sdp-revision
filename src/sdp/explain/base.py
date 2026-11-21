"""Common conventions for every explainer: context, interface, background sets and store.

Conventions (identical for every model and method):

* **Explained output**: the predicted probability of the defective class. The only
  exception is linear SHAP for logistic regression, which is exact on the log-odds
  scale; its rows carry ``output_scale="log_odds"`` (all others ``"probability"``).
* **Input space**: attributions are computed in each model's input space, i.e. after
  its fitted preprocessor. The preprocessors are element-wise (``log1p`` and/or
  standardisation of each metric on its own), so attribution ``j`` belongs to exactly
  one original metric. Attributions are always stored with explicit feature names.
* **Background (reference) set**: one per split and seed, saved to disk: 100 rows of
  the training data, sampled with stratification by label using the model seed (all
  rows if there are fewer than 100). The training data are the rows the model and its
  preprocessor were fitted on (the fitting part, without the validation hold-out).
* **Single baseline** (DeepLift, Integrated Gradients): the mean of the background set
  in model input space.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sdp.eval.preprocess import FEATURES

logger = logging.getLogger(__name__)

PROBABILITY = "probability"
LOG_ODDS = "log_odds"
OUTPUT_SCALES = (PROBABILITY, LOG_ODDS)
OUTCOMES = ("TP", "FP", "FN", "TN")

ATTRIBUTION_COLUMNS = [
    "split_id", "project", "train_release", "test_release", "model", "method", "seed", "instance_id",
    "class_name", "outcome", "y_true", "y_prob", "loc", "feature", "feature_value_raw", "attribution",
    "output_scale",
]  # fmt: skip
CHECK_COLUMNS = ["split_id", "project", "model", "method", "seed", "instance_id", "check", "output", "reference",
                 "attribution_sum", "error"]  # fmt: skip
_CATEGORICAL = ["split_id", "project", "train_release", "test_release", "model", "method", "class_name", "outcome",
                "feature", "output_scale"]  # fmt: skip


# ---------------------------------------------------------------------- context


def outcome_groups(y_true, y_prob, threshold: float) -> np.ndarray:
    """TP/FP/FN/TN at ``threshold`` (prediction is ``y_prob >= threshold``, as in step 3)."""
    y = np.asarray(y_true, dtype=int).astype(bool)
    pred = np.asarray(y_prob, dtype=float) >= threshold
    return np.select([y & pred, ~y & pred, y & ~pred], ["TP", "FP", "FN"], default="TN").astype(object)


@dataclass
class ExplainContext:
    """Everything an explainer may use for one (split, model, seed)."""

    split_id: str
    project: str
    train_release: str
    test_release: str
    model_name: str
    seed: int
    model: object  # fitted model (MLPClassifierTorch, LogisticRegression, RF, XGBClassifierES)
    preprocessor: object  # fitted preprocessor (DataFrame in, DataFrame out)
    threshold: float  # validation-MCC threshold from step 3
    X_train: pd.DataFrame  # model input space, fitting part of the training data
    y_train: np.ndarray
    X_test: pd.DataFrame  # model input space
    X_test_raw: pd.DataFrame  # original metric values
    background: pd.DataFrame  # model input space
    class_name: np.ndarray
    y_true: np.ndarray
    loc: np.ndarray
    y_prob: np.ndarray = field(init=False)

    def __post_init__(self):
        cols = list(self.X_test.columns)
        if not cols or cols != [f for f in FEATURES if f in cols]:
            raise ValueError("explainer inputs must hold metrics in canonical order")
        for frame in (self.X_train, self.X_test_raw, self.background):
            if list(frame.columns) != cols:
                raise ValueError("all explainer inputs must have the same metrics")
        self.y_prob = self.predict(self.X_test.to_numpy())

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_test.columns)

    @property
    def outcome(self) -> np.ndarray:
        return outcome_groups(self.y_true, self.y_prob, self.threshold)

    def predict(self, X) -> np.ndarray:
        """P(defective) for rows in model input space (array or DataFrame)."""
        from sdp.models.registry import positive_proba

        X = pd.DataFrame(np.asarray(X, dtype="float64"), columns=self.feature_names)
        return positive_proba(self.model, X)

    def predict_proba(self, X) -> np.ndarray:
        """Both class probabilities, as LIME expects."""
        p = self.predict(X)
        return np.column_stack([1 - p, p])


# -------------------------------------------------------------------- interface


@dataclass
class MethodResult:
    """Attributions of one method for all test instances of one context."""

    method: str
    attributions: pd.DataFrame  # rows = test instances (in order), columns = feature names
    output_scale: str = PROBABILITY
    checks: pd.DataFrame | None = None  # per-instance completeness / additivity
    time_s: float = 0.0

    def __post_init__(self):
        if self.output_scale not in OUTPUT_SCALES:
            raise ValueError(f"unknown output scale {self.output_scale!r}")
        cols = set(self.attributions.columns)
        if not cols or not cols <= set(FEATURES) or len(cols) != self.attributions.shape[1]:
            raise ValueError(f"{self.method}: attributions must be labelled with metric names")
        self.attributions = self.attributions[[f for f in FEATURES if f in cols]]  # canonical order, by name


class Explainer(ABC):
    """An attribution method. Subclasses set ``name``, ``models`` and ``output_scale``."""

    name: str = ""
    models: tuple[str, ...] = ()
    output_scale: str = PROBABILITY

    def __init__(self, **params):
        self.params = params

    def explain(self, ctx: ExplainContext) -> MethodResult:
        t0 = time.perf_counter()
        attr, checks = self._attribute(ctx)
        frame = attr if isinstance(attr, pd.DataFrame) else pd.DataFrame(np.asarray(attr), columns=ctx.feature_names)
        frame.index = range(len(frame))
        if len(frame) != len(ctx.X_test):
            raise ValueError(f"{self.name}: {len(frame)} attribution rows for {len(ctx.X_test)} instances")
        return MethodResult(self.name, frame, self.output_scale, checks, time.perf_counter() - t0)

    @abstractmethod
    def _attribute(self, ctx: ExplainContext) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | None]:
        """Return attributions (n_instances × n_features, columns in ``ctx.feature_names``
        order if an array) and an optional per-instance check frame."""


def check_frame(ctx: ExplainContext, method: str, check: str, output, reference, attr_sum) -> pd.DataFrame:
    """Per-instance |sum(attr) − (output − reference)|."""
    output, attr_sum = np.asarray(output, float), np.asarray(attr_sum, float)
    reference = np.broadcast_to(np.asarray(reference, float), output.shape)
    return pd.DataFrame({
        "split_id": ctx.split_id, "project": ctx.project, "model": ctx.model_name, "method": method,
        "seed": ctx.seed, "instance_id": np.arange(len(output)), "check": check, "output": output,
        "reference": reference, "attribution_sum": attr_sum, "error": np.abs(attr_sum - (output - reference)),
    })[CHECK_COLUMNS]  # fmt: skip


def subset_context(ctx: ExplainContext, idx) -> ExplainContext:
    """The same context restricted to the test instances at positions ``idx``."""
    import dataclasses

    idx = np.asarray(idx, dtype=int)
    return dataclasses.replace(
        ctx, X_test=ctx.X_test.iloc[idx].reset_index(drop=True), X_test_raw=ctx.X_test_raw.iloc[idx].reset_index(drop=True),
        class_name=np.asarray(ctx.class_name)[idx], y_true=np.asarray(ctx.y_true)[idx], loc=np.asarray(ctx.loc)[idx],
    )  # fmt: skip


# ---------------------------------------------------------------- background set


def sample_background(X: pd.DataFrame, y, size: int, seed: int) -> pd.DataFrame:
    """``size`` rows of ``X`` sampled with stratification by ``y`` (all rows if fewer).

    The returned frame keeps the original row index so that the sample can be saved
    and re-applied to another model's input space.
    """
    y = np.asarray(y, dtype=int)
    if len(X) <= size:
        return X.copy()
    strat = y if np.bincount(y, minlength=2).min() >= 2 else None
    if strat is None:
        logger.warning("Background: a class has fewer than 2 rows; sampling without stratification.")
    idx, _ = train_test_split(np.arange(len(X)), train_size=size, random_state=seed, stratify=strat)
    return X.iloc[np.sort(idx)].copy()


def save_background(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.rename_axis("row").reset_index().to_parquet(path, index=False)


def load_background(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).set_index("row")


# --------------------------------------------------------------------- the store


def to_long(ctx: ExplainContext, result: MethodResult) -> pd.DataFrame:
    """Long-format rows (one per instance and feature), joined to features by name."""
    attr = result.attributions
    n, feats = len(attr), list(attr.columns)
    raw = ctx.X_test_raw[feats].to_numpy()
    rep = lambda a: np.repeat(np.asarray(a), len(feats))  # noqa: E731
    out = pd.DataFrame({
        "split_id": ctx.split_id, "project": ctx.project, "train_release": ctx.train_release,
        "test_release": ctx.test_release, "model": ctx.model_name, "method": result.method,
        "seed": np.int16(ctx.seed), "instance_id": rep(np.arange(n, dtype=np.int32)), "class_name": rep(ctx.class_name),
        "outcome": rep(ctx.outcome), "y_true": rep(ctx.y_true.astype(np.int8)), "y_prob": rep(ctx.y_prob),
        "loc": rep(ctx.loc), "feature": np.tile(feats, n), "feature_value_raw": raw.reshape(-1),
        "attribution": attr.to_numpy(dtype="float64").reshape(-1), "output_scale": result.output_scale,
    })  # fmt: skip
    return out[ATTRIBUTION_COLUMNS]


def partition_path(root: Path, model: str, method: str, split_dir: str, seed: int) -> Path:
    """``<root>/<model>/<method>/<split_dir>__seed_<k>.parquet``."""
    return Path(root) / model / method / f"{split_dir}__seed_{seed}.parquet"


def write_long(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def rename_method(root: Path, old: str, new: str) -> list[Path]:
    """Move every ``<model>/<old>/`` partition of the store to ``<model>/<new>/`` and
    rewrite its ``method`` column, so earlier results are kept under a new name."""
    moved = []
    for src_dir in sorted(Path(root).glob(f"*/{old}")):
        if src_dir.parent.name.startswith("_"):
            continue
        dst_dir = src_dir.parent / new
        if dst_dir.exists():
            raise FileExistsError(f"{dst_dir} exists already; refusing to overwrite")
        dst_dir.mkdir(parents=True)
        for f in sorted(src_dir.glob("*.parquet")):
            frame = pd.read_parquet(f)
            frame["method"] = new
            write_long(frame, dst_dir / f.name)
            f.unlink()
            moved.append(dst_dir / f.name)
        src_dir.rmdir()
    return moved


def read_attributions(root: Path, models=None, methods=None, columns=None, seeds=None) -> pd.DataFrame:
    """Read (a subset of) the attribution store; string columns become categoricals."""
    root = Path(root)
    files = sorted(
        f
        for f in root.glob("*/*/*.parquet")
        if not f.parent.parent.name.startswith("_")  # _checks, _backgrounds, _cache
        and (models is None or f.parent.parent.name in models)
        and (methods is None or f.parent.name in methods)
    )
    if seeds is not None:
        files = [f for f in files if int(f.stem.rsplit("__seed_", 1)[1]) in set(seeds)]
    if not files:
        return pd.DataFrame(columns=columns or ATTRIBUTION_COLUMNS)
    parts = [pd.read_parquet(f, columns=columns) for f in files]
    out = pd.concat(parts, ignore_index=True)
    for c in _CATEGORICAL:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out
