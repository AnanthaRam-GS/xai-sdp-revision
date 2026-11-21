"""Explanation runner over the persisted time-aware models (step 4).

A *job* is one (split, model, seed): it loads the persisted preprocessor and model
(nothing is retrained), rebuilds the split, checks that the loaded model reproduces
the stored step 3 test predictions, samples and saves the background set, runs every
configured method for that model and seed and writes:

* ``<artifacts>/attributions/<model>/<method>/<split_dir>__seed_<k>.parquet``
  (long format, see :mod:`sdp.explain.base`);
* ``<artifacts>/attributions/_checks/<model>/<split_dir>__seed_<k>.parquet``
  (per-instance completeness / additivity errors);
* ``<artifacts>/attributions/_backgrounds/<split_dir>/seed_<k>.parquet``
  (background rows, original metric values, with their row positions and labels).

Jobs run in parallel (joblib, loky) with one thread per worker. A job is skipped when
its cache marker ``_cache/<sha1>.json`` exists and all its outputs are present; the hash
covers the code version, the split indices, the method parameters and the bytes of the
persisted model files.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sdp.config import REPO_ROOT, Config
from sdp.data.clean import ID_COL, LABEL_COL
from sdp.data.study import build_study
from sdp.eval.preprocess import EFFORT_COL, FEATURES, feature_frame
from sdp.eval.splits import Split, project_frame, time_aware_splits
from sdp.experiments.persistence import artifact_dir, load_artifact, safe_split_id
from sdp.explain.base import (
    ExplainContext,
    Explainer,
    load_background,
    partition_path,
    sample_background,
    save_background,
    to_long,
    write_long,
)
from sdp.explain.captum_methods import CAPTUM_EXPLAINERS
from sdp.explain.other_methods import OTHER_EXPLAINERS

logger = logging.getLogger(__name__)

CODE_VERSION = "step4-v1"
EXPERIMENTS_DIR = REPO_ROOT / "configs" / "experiments"
EXPLAINERS: dict[str, type[Explainer]] = {**CAPTUM_EXPLAINERS, **OTHER_EXPLAINERS}
PROB_TOLERANCE = 1e-6  # loaded model vs stored step 3 predictions


class MissingArtifactError(RuntimeError):
    """Raised when a persisted model needed for explanation does not exist."""


class PredictionMismatchError(RuntimeError):
    """Raised when a loaded model does not reproduce its stored test predictions."""


@dataclass(frozen=True)
class ExplainSettings:
    experiment: str
    imbalance: str
    models: dict[str, str]
    methods: dict[str, list[str]]
    seeds: list[int]
    expensive_methods: list[str]
    expensive_seeds: list[int]
    background_size: int
    params: dict[str, dict]
    reduced_params: dict[str, dict] = field(default_factory=dict)
    max_hours: float = 8.0
    completeness_tolerance: float = 1e-3
    n_jobs: int = 10
    output_dir: str = "attributions"
    reduced: bool = False

    def methods_for(self, model: str, seed: int) -> tuple[str, ...]:
        return tuple(m for m in self.methods[model] if m not in self.expensive_methods or seed in self.expensive_seeds)

    def method_params(self, method: str) -> dict:
        p = dict(self.params.get(method, {}))
        if self.reduced:
            p |= self.reduced_params.get(method, {})
        return p

    def with_reduction(self) -> "ExplainSettings":
        return dataclasses.replace(self, reduced=True)

    def root(self, cfg: Config) -> Path:
        return Path(cfg.paths.artifacts_dir) / self.output_dir


def load_explain_settings(path: str | Path) -> ExplainSettings:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    unknown = {m for ms in raw["methods"].values() for m in ms} - set(EXPLAINERS)
    if unknown:
        raise ValueError(f"unknown explanation method(s): {sorted(unknown)}")
    for model, methods in raw["methods"].items():
        for m in methods:
            if model not in EXPLAINERS[m].models:
                raise ValueError(f"method {m} does not apply to model {model}")
    fields = {f.name for f in dataclasses.fields(ExplainSettings)}
    return ExplainSettings(**{k: v for k, v in raw.items() if k in fields})


# --------------------------------------------------------------------------- jobs


@dataclass(frozen=True)
class ExplainJob:
    split: Split
    model: str
    preprocessing: str
    seed: int
    methods: tuple[str, ...]
    artifact_path: str

    @property
    def label(self) -> str:
        return f"{self.split.split_id}|{self.model}|seed{self.seed}"

    @property
    def split_dir(self) -> str:
        return safe_split_id(self.split.split_id)

    def cost(self) -> float:
        weights = {"KernelSHAP": 40.0, "LIME": 10.0, "DeepLiftShap": 1.0, "GradientShap": 0.5, "TreeSHAP": 2.0}
        return len(self.split.test_idx) * sum(weights.get(m, 0.1) for m in self.methods)

    def cache_key(self, settings: ExplainSettings) -> str:
        h = hashlib.sha1()
        for arr in (self.split.fit_idx, self.split.val_idx, self.split.test_idx):
            h.update(np.ascontiguousarray(arr, dtype=np.int64).tobytes())
        for f in sorted(Path(self.artifact_path).iterdir()):
            h.update(f.name.encode())
            h.update(f.read_bytes())
        spec = {
            "code": CODE_VERSION, "split": self.split.split_id, "model": self.model, "seed": self.seed,
            "preprocessing": self.preprocessing, "background_size": settings.background_size,
            "methods": {m: settings.method_params(m) for m in self.methods}, "digest": h.hexdigest(),
        }  # fmt: skip
        return hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()

    def outputs(self, root: Path) -> list[Path]:
        files = [partition_path(root, self.model, m, self.split_dir, self.seed) for m in self.methods]
        return files + [self.checks_path(root), self.background_path(root)]

    def checks_path(self, root: Path) -> Path:
        return Path(root) / "_checks" / self.model / f"{self.split_dir}__seed_{self.seed}.parquet"

    def background_path(self, root: Path) -> Path:
        return Path(root) / "_backgrounds" / self.split_dir / f"seed_{self.seed}.parquet"


def build_jobs(cfg: Config, settings: ExplainSettings, frames: dict, splits: list[Split], seeds=None,
               models=None) -> list[ExplainJob]:  # fmt: skip
    root = Path(cfg.paths.artifacts_dir) / "models" / settings.experiment
    jobs = []
    for split in splits:
        for model, prep in settings.models.items():
            if models is not None and model not in models:
                continue
            for seed in seeds if seeds is not None else settings.seeds:
                path = artifact_dir(root, "", split.split_id, model, prep, seed)
                jobs.append(ExplainJob(split, model, prep, seed, settings.methods_for(model, seed), str(path)))
    return sorted(jobs, key=lambda j: -j.cost())


def missing_artifacts(jobs: list[ExplainJob]) -> list[str]:
    return [j.artifact_path for j in jobs if not (Path(j.artifact_path) / "meta.json").exists()]


def experiment_splits(cfg: Config, settings: ExplainSettings) -> tuple[list[Split], dict[str, pd.DataFrame]]:
    """The time-aware splits of the experiment, rebuilt exactly as in step 3."""
    from sdp.experiments.config import load_experiment

    exp = load_experiment(EXPERIMENTS_DIR / f"{settings.experiment}.yaml")
    if exp.protocol != "time_aware":
        raise ValueError("step 4 explains time-aware experiments only")
    study = build_study(cfg)
    frames = {p: project_frame(cfg, study, p) for p in study.projects}
    e = cfg.evaluation
    splits = list(time_aware_splits(frames, study, exp.time_aware_variant, e.validation_size, cfg.seed))
    return splits, frames


# ------------------------------------------------------------------------- worker


def build_context(job: ExplainJob, df: pd.DataFrame, settings: ExplainSettings, root: Path) -> ExplainContext:
    pre, model, meta = load_artifact(Path(job.artifact_path))
    if meta["split_id"] != job.split.split_id or int(meta["seed"]) != job.seed or meta["model"] != job.model:
        raise MissingArtifactError(f"artefact {job.artifact_path} does not belong to {job.label}")
    if list(meta["feature_order"]) != list(FEATURES):
        raise ValueError(f"{job.label}: unexpected feature order in the artefact")
    X = feature_frame(df)
    y = df[LABEL_COL].to_numpy(dtype="int64")
    fit, test = job.split.fit_idx, job.split.test_idx

    bg_path = job.background_path(root)
    if bg_path.exists():
        bg_raw = load_background(bg_path)[FEATURES]
    else:
        X_fit = X.iloc[fit].reset_index(drop=True)
        bg_raw = sample_background(X_fit, y[fit], settings.background_size, job.seed)
        save_background(bg_raw.assign(defective=y[fit][bg_raw.index]), bg_path)
    test_raw = X.iloc[test].reset_index(drop=True)
    return ExplainContext(
        split_id=job.split.split_id, project=job.split.project, train_release="+".join(job.split.train_releases),
        test_release=job.split.test_release, model_name=job.model, seed=job.seed, model=model, preprocessor=pre,
        threshold=float(meta["threshold_val_mcc"]), X_train=pre.transform(X.iloc[fit]).reset_index(drop=True),
        y_train=y[fit], X_test=pre.transform(test_raw), X_test_raw=test_raw,
        background=pre.transform(bg_raw).reset_index(drop=True),
        class_name=df[ID_COL].astype(str).to_numpy()[test], y_true=y[test],
        loc=df[EFFORT_COL].to_numpy(dtype="float64")[test],
    )  # fmt: skip


def execute_job(job: ExplainJob, df: pd.DataFrame, settings: ExplainSettings, root: Path,
                stored_prob: np.ndarray | None = None) -> dict:  # fmt: skip
    """Explain one (split, model, seed) in the current process (normally a joblib worker)."""
    from threadpoolctl import threadpool_limits

    from sdp.seed import set_seed

    set_seed(job.seed, torch_threads=1, quiet=True)
    t0 = time.perf_counter()
    timings, checks = [], []
    with threadpool_limits(limits=1):
        ctx = build_context(job, df, settings, root)
        diff = float(np.max(np.abs(ctx.y_prob - stored_prob))) if stored_prob is not None else float("nan")
        if stored_prob is not None and not diff <= PROB_TOLERANCE:
            raise PredictionMismatchError(f"{job.label}: loaded model differs from stored predictions by {diff:.2e}")
        load_s = time.perf_counter() - t0
        for name in job.methods:
            explainer = EXPLAINERS[name](**settings.method_params(name))
            set_seed(job.seed, torch_threads=1, quiet=True)
            res = explainer.explain(ctx)
            write_long(to_long(ctx, res), partition_path(root, job.model, name, job.split_dir, job.seed))
            if res.checks is not None:
                checks.append(res.checks)
                p95 = float(res.checks["error"].quantile(0.95))
                if p95 > settings.completeness_tolerance:
                    logger.warning("%s %s: 95th percentile of the %s error is %.2e > %.0e.", job.label, name,
                                   res.checks["check"].iloc[0], p95, settings.completeness_tolerance)  # fmt: skip
            timings.append({"split_id": job.split.split_id, "project": job.split.project, "model": job.model,
                            "method": name, "seed": job.seed, "n_instances": len(ctx.X_test),
                            "n_background": len(ctx.background), "n_fit": len(job.split.fit_idx),
                            "time_s": res.time_s})  # fmt: skip
    chk = pd.concat(checks, ignore_index=True) if checks else pd.DataFrame(columns=["split_id"])
    write_long(chk, job.checks_path(root))
    return {"label": job.label, "timings": timings, "load_s": load_s, "max_prob_diff": diff,
            "wall_s": time.perf_counter() - t0, "pid": os.getpid()}  # fmt: skip


def run_cached(job: ExplainJob, df: pd.DataFrame, settings: ExplainSettings, root: str,
               stored_prob: np.ndarray | None) -> tuple[dict, bool]:  # fmt: skip
    root = Path(root)
    marker = root / "_cache" / f"{job.cache_key(settings)}.json"
    if marker.exists() and all(p.exists() for p in job.outputs(root)):
        return json.loads(marker.read_text(encoding="utf-8")), True
    result = execute_job(job, df, settings, root, stored_prob)
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(result, default=float), encoding="utf-8")
    os.replace(tmp, marker)
    return result, False


# ------------------------------------------------------------------------- driver


def stored_predictions(cfg: Config, settings: ExplainSettings) -> dict[tuple, np.ndarray]:
    """Step 3 test predictions per (split_id, model, seed), in test-index order."""
    path = Path(cfg.paths.results_dir) / settings.experiment / "predictions.csv"
    if not path.exists():
        logger.warning("No stored predictions at %s; skipping the reproduction check.", path)
        return {}
    pred = pd.read_csv(path, usecols=["split_id", "model", "seed", "y_prob"])
    pred = pred[pred["model"].isin(list(settings.models))]
    return {k: g["y_prob"].to_numpy() for k, g in pred.groupby(["split_id", "model", "seed"], sort=False)}


def run_explanations(cfg: Config, settings: ExplainSettings, jobs: list[ExplainJob], frames: dict,
                     n_jobs: int | None = None, tag: str = "full") -> dict:  # fmt: skip
    """Run (or resume) the given jobs and write timings to ``results/explanations/``."""
    from joblib import Parallel, delayed

    missing = missing_artifacts(jobs)
    if missing:
        raise MissingArtifactError(f"{len(missing)} persisted model(s) are missing, e.g. {missing[:5]}")
    root = settings.root(cfg)
    stored = stored_predictions(cfg, settings)
    n_jobs = n_jobs or settings.n_jobs
    t0 = time.perf_counter()
    logger.info("Explaining %d jobs (%s), n_jobs=%d, reduced=%s -> %s", len(jobs), tag, n_jobs, settings.reduced, root)
    results, n_cached, last = [], 0, time.perf_counter()
    parallel = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered", batch_size=1)
    calls = (delayed(run_cached)(j, frames[j.split.project], settings, str(root),
                                 stored.get((j.split.split_id, j.model, j.seed))) for j in jobs)  # fmt: skip
    for i, (res, cached) in enumerate(parallel(calls), start=1):
        results.append(res | {"from_cache": cached})
        n_cached += cached
        if time.perf_counter() - last > 60 or i == len(jobs):
            logger.info("%s: %d/%d jobs done (%d from cache), %.0f s elapsed.", tag, i, len(jobs), n_cached,
                        time.perf_counter() - t0)  # fmt: skip
            last = time.perf_counter()
    wall = time.perf_counter() - t0
    timings = pd.DataFrame([t for r in results for t in r["timings"]])
    out = Path(cfg.paths.results_dir) / "explanations"
    out.mkdir(parents=True, exist_ok=True)
    timings.to_csv(out / f"timings_{tag}.csv", index=False)
    diffs = [r["max_prob_diff"] for r in results if np.isfinite(r["max_prob_diff"])]
    info = {"tag": tag, "jobs": len(jobs), "jobs_from_cache": n_cached, "n_jobs": n_jobs, "wall_time_s": wall,
            "cpu_time_s": float(sum(r["wall_s"] for r in results if not r["from_cache"])),
            "reduced": settings.reduced, "params": {m: settings.method_params(m) for m in EXPLAINERS},
            "max_prob_diff_vs_step3": max(diffs) if diffs else None, "n_checked_against_step3": len(diffs)}  # fmt: skip
    (out / f"run_info_{tag}.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    logger.info("Explanations (%s) finished in %.0f s (%d/%d from cache).", tag, wall, n_cached, len(jobs))
    return info | {"timings": timings}


def estimate_runtime(smoke_timings: pd.DataFrame, jobs: list[ExplainJob], n_jobs: int, efficiency: float,
                     scale: dict[str, float] | None = None) -> dict:  # fmt: skip
    """CPU and wall time of ``jobs`` from per-instance smoke-test rates.

    TreeSHAP and LIME rates are additionally scaled by the size of the training data
    (tree size and the cost of the background statistics grow with it). ``scale``
    multiplies the rate of a method (e.g. 0.4 for 2000 instead of 5000 LIME samples).
    """
    t = smoke_timings.assign(rate=smoke_timings["time_s"] / smoke_timings["n_instances"])
    per_fit = {"TreeSHAP"}
    rate = t.groupby(["model", "method"])["rate"].mean().to_dict()
    fit_ref = t.groupby(["model", "method"])["n_fit"].mean().to_dict()
    scale = scale or {}
    cpu_by_method: dict[str, float] = {}
    for job in jobs:
        for m in job.methods:
            r = rate.get((job.model, m), 0.0) * scale.get(m, 1.0) * len(job.split.test_idx)
            if m in per_fit:
                r *= len(job.split.fit_idx) / max(fit_ref.get((job.model, m), 1.0), 1.0)
            cpu_by_method[f"{job.model}:{m}"] = cpu_by_method.get(f"{job.model}:{m}", 0.0) + r
    cpu = sum(cpu_by_method.values())
    eff = max(min(efficiency, 1.0), 0.3)
    return {"cpu_hours": cpu / 3600, "wall_hours": cpu / (n_jobs * eff) / 3600, "parallel_efficiency": eff,
            "cpu_hours_by_method": {k: v / 3600 for k, v in sorted(cpu_by_method.items())}}  # fmt: skip
