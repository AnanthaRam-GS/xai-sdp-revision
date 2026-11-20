"""Experiment engine for learned models and baselines (step 3).

A *job* is one (model, preprocessing, imbalance, seed) setting on:

* one split for the time-aware protocol, or
* the 5 folds of one repeat of one release for the within-release protocol; the model
  is tuned once on the first fold's fitting part and the parameters are reused for the
  other folds of that repeat.

For every split of a job:

1. tune (learned models only, first split of the job) on the 80% fitting part with
   inner CV (:mod:`sdp.tuning`);
2. fit the preprocessor on the fitting part, then the model on the fitting part
   (SMOTE-resampled if requested) with early stopping on the validation part;
3. choose the MCC-maximising threshold on the validation part;
4. evaluate on the test release / fold and, if requested, persist the fitted model and
   preprocessor (:mod:`sdp.experiments.persistence`).

Each job runs in its own worker process with its own seeds and one thread
(PyTorch, BLAS and OpenMP). Job results are cached in ``results/_cache/`` under a hash
of everything that determines them, so interrupted runs resume and identical units
shared by several experiments are computed once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.config import Config
from sdp.data.clean import ID_COL, LABEL_COL
from sdp.data.study import StudyDataset, build_study
from sdp.eval.metrics import best_mcc_threshold, evaluate
from sdp.eval.preprocess import EFFORT_COL, FEATURES, feature_frame, make_preprocessor
from sdp.eval.splits import Split, project_frame, time_aware_splits, within_release_splits
from sdp.experiments.config import ExperimentConfig
from sdp.experiments.persistence import META, artifact_dir, save_artifact
from sdp.models.imbalance import fit_model
from sdp.models.registry import is_learned, make_model, positive_proba, requires_raw_features

logger = logging.getLogger(__name__)

CODE_VERSION = "step3-v1"
METRIC_COLUMNS = [
    "experiment", "project", "train_releases", "test_release", "split_id", "seed", "model",
    "preprocessing", "imbalance", "threshold_mode", "metric", "value",
]  # fmt: skip
PREDICTION_COLUMNS = ["split_id", "model", "preprocessing", "imbalance", "seed", "class_name", "y_true", "y_prob", "loc"]
MODEL_COST = {"mlp": 10.0, "xgb": 3.0, "rf": 3.0, "lr": 1.0}  # rough relative cost, for scheduling


@dataclass(frozen=True)
class Job:
    project: str
    splits: tuple[Split, ...]
    model: str
    preprocessing: str
    imbalance: str
    seed: int
    n_trials: int
    artifact_root: str | None = None  # <artifacts>/<experiment>, or None for no persistence
    experiment: str = ""

    @property
    def label(self) -> str:
        return f"{self.splits[0].split_id}|{self.model}|{self.preprocessing}|{self.imbalance}|seed{self.seed}"

    def cost(self) -> float:
        n = sum(len(s.fit_idx) for s in self.splits[:1])
        return MODEL_COST.get(self.model, 0.01) * max(self.n_trials, 1) * n + len(self.splits)

    def cache_key(self, cfg: Config) -> str:
        h = hashlib.sha1()
        for s in self.splits:
            for arr in (s.fit_idx, s.val_idx, s.test_idx):
                h.update(np.ascontiguousarray(arr, dtype=np.int64).tobytes())
        spec = {
            "code": CODE_VERSION,
            "splits": [s.split_id for s in self.splits],
            "split_hash": h.hexdigest(),
            "model": self.model,
            "preprocessing": self.preprocessing,
            "imbalance": self.imbalance,
            "seed": self.seed,
            "n_trials": self.n_trials,
            "space": cfg.tuning.spaces.get(self.model) if is_learned(self.model) else None,
            "inner_folds": cfg.tuning.inner_folds,
            "necm": cfg.evaluation.necm_cost_ratio,
            "effort": cfg.evaluation.recall_at_loc,
        }
        return hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()


# ------------------------------------------------------------------------- worker


def execute_job(job: Job, df: pd.DataFrame, cfg: Config) -> dict:
    """Run one job in the current process (normally a joblib worker)."""
    from threadpoolctl import threadpool_limits

    from sdp.seed import set_seed
    from sdp.tuning import tune

    set_seed(job.seed, torch_threads=1, quiet=True)
    X = feature_frame(df)
    y = df[LABEL_COL].to_numpy(dtype="int64")
    loc = df[EFFORT_COL].to_numpy(dtype="float64")
    names = df[ID_COL].astype(str).to_numpy()
    e = cfg.evaluation

    out = {"metrics": [], "predictions": [], "hyperparameters": [], "timings": [], "artifacts": {}}
    params, tuning_time, inner_auc, tuned_on = {}, 0.0, float("nan"), ""
    with threadpool_limits(limits=1):
        for i, split in enumerate(job.splits):
            fit, val, test = split.fit_idx, split.val_idx, split.test_idx
            this_tuning = 0.0
            if is_learned(job.model) and i == 0 and job.n_trials > 0:
                res = tune(job.model, X.iloc[fit], y[fit], cfg.tuning.spaces.get(job.model, {}), job.n_trials,
                           job.preprocessing, job.imbalance, job.seed, cfg.tuning.inner_folds)  # fmt: skip
                params, tuning_time, inner_auc, tuned_on = res.params, res.time_s, res.best_value, split.split_id
                this_tuning = res.time_s

            t0 = time.perf_counter()
            model = make_model(job.model, params, job.imbalance, job.seed)
            prep = "none" if requires_raw_features(model) else job.preprocessing
            pre = make_preprocessor(prep).fit(X.iloc[fit])
            X_fit, X_val, X_test = pre.transform(X.iloc[fit]), pre.transform(X.iloc[val]), pre.transform(X.iloc[test])
            fit_model(model, X_fit, y[fit], X_val, y[val], imbalance=job.imbalance, seed=job.seed)
            fit_time = time.perf_counter() - t0
            threshold = best_mcc_threshold(y[val], positive_proba(model, X_val))
            p_test = positive_proba(model, X_test)

            base = {"project": split.project, "train_releases": "+".join(split.train_releases),
                    "test_release": split.test_release, "split_id": split.split_id, "seed": job.seed,
                    "model": job.model, "preprocessing": prep, "imbalance": job.imbalance}  # fmt: skip
            for mode, metric, value in evaluate(y[test], p_test, loc[test], threshold, e.necm_cost_ratio, e.recall_at_loc):
                out["metrics"].append(base | {"threshold_mode": mode, "metric": metric, "value": value})
            out["predictions"].append(pd.DataFrame({
                "split_id": split.split_id, "model": job.model, "preprocessing": prep, "imbalance": job.imbalance,
                "seed": job.seed, "class_name": names[test], "y_true": y[test], "y_prob": p_test, "loc": loc[test],
            }))  # fmt: skip
            n_iter = getattr(model, "best_epoch_", getattr(model, "best_iteration_", None))
            if is_learned(job.model):
                out["hyperparameters"].append(base | {
                    "tuned_on_split": tuned_on, "n_trials": job.n_trials, "inner_cv_auc": inner_auc,
                    "params": json.dumps(params, sort_keys=True, default=str), "best_iteration": n_iter,
                    "threshold_val_mcc": threshold,
                })  # fmt: skip
            out["timings"].append(base | {"n_fit": len(fit), "tuning_time_s": this_tuning,
                                          "fit_time_s": fit_time, "best_iteration": n_iter})  # fmt: skip

            if job.artifact_root and is_learned(job.model):
                path = artifact_dir(Path(job.artifact_root), "", split.split_id, job.model, prep, job.seed)
                meta = base | {
                    "experiment": job.experiment, "hyperparameters": params, "threshold_val_mcc": threshold,
                    "training_time_s": fit_time, "tuning_time_s": tuning_time, "tuned_on_split": tuned_on,
                    "inner_cv_auc": inner_auc, "feature_order": list(FEATURES), "n_fit": len(fit),
                    "n_val": len(val), "best_iteration": n_iter, "code_version": CODE_VERSION,
                }  # fmt: skip
                save_artifact(path, pre, model, meta)
                out["artifacts"][split.split_id] = str(path)
    out["predictions"] = pd.concat(out["predictions"], ignore_index=True)
    return out


def _relocate_artifacts(result: dict, job: Job) -> None:
    """Copy artifacts computed by another experiment into this experiment's tree."""
    for split in job.splits:
        src = Path(result["artifacts"][split.split_id])
        dst = artifact_dir(Path(job.artifact_root), "", split.split_id, job.model, src.parent.name, job.seed)
        if src.resolve() != dst.resolve():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            meta = json.loads((dst / META).read_text(encoding="utf-8"))
            meta["experiment"], meta["copied_from"] = job.experiment, str(src)
            (dst / META).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        result["artifacts"][split.split_id] = str(dst)


def run_cached(job: Job, df: pd.DataFrame, cfg: Config, cache_dir: str) -> tuple[dict, bool]:
    """Return ``(result, from_cache)``; computes and caches the job if needed."""
    path = Path(cache_dir) / f"{job.cache_key(cfg)}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            result = pickle.load(fh)
        if not (job.artifact_root and is_learned(job.model)):
            return result, True
        if len(result.get("artifacts", {})) == len(job.splits) and all(
            Path(p, META).exists() for p in result["artifacts"].values()
        ):
            _relocate_artifacts(result, job)
            return result, True
    result = execute_job(job, df, cfg)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        pickle.dump(result, fh)
    os.replace(tmp, path)
    return result, False


# ------------------------------------------------------------------------- driver


def experiment_splits(cfg: Config, exp: ExperimentConfig, study: StudyDataset, frames) -> list[tuple[Split, ...]]:
    """Groups of splits that form one job each."""
    e = cfg.evaluation
    if exp.protocol == "time_aware":
        return [(s,) for s in time_aware_splits(frames, study, exp.time_aware_variant, e.validation_size, cfg.seed)]
    groups: dict[tuple, list[Split]] = {}
    for s in within_release_splits(frames, study, e.n_splits, e.n_repeats, e.validation_size, cfg.seed):
        groups.setdefault((s.project, s.test_release, s.repeat), []).append(s)
    return [tuple(g) for g in groups.values()]


def build_jobs(cfg: Config, exp: ExperimentConfig, study: StudyDataset, frames) -> list[Job]:
    n_trials = exp.n_trials if exp.n_trials is not None else cfg.tuning.n_trials(exp.protocol)
    root = str(cfg.paths.artifacts_dir / "models" / exp.name) if exp.persist_models else None
    jobs = []
    for group in experiment_splits(cfg, exp, study, frames):
        for st in exp.settings:
            for seed in exp.seeds_for(st.model):
                jobs.append(Job(group[0].project, group, st.model, st.preprocessing, st.imbalance, seed,
                                n_trials if is_learned(st.model) else 0, root, exp.name))  # fmt: skip
    return sorted(jobs, key=lambda j: -j.cost())


@dataclass
class ExperimentResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    hyperparameters: pd.DataFrame
    timings: pd.DataFrame
    info: dict


def run_experiment(cfg: Config, exp: ExperimentConfig, n_jobs: int | None = None) -> ExperimentResult:
    """Run (or resume) an experiment and write its outputs to ``results/<name>/``."""
    from joblib import Parallel, delayed

    t0 = time.perf_counter()
    study = build_study(cfg)
    if exp.projects:
        study = StudyDataset(study.table, {p: v for p, v in study.releases.items() if p in exp.projects})
    frames = {p: project_frame(cfg, study, p) for p in study.projects}
    jobs = build_jobs(cfg, exp, study, frames)
    cache_dir = cfg.paths.results_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = n_jobs or exp.n_jobs
    logger.info("Experiment %s: %d jobs (%d splits), n_jobs=%d.", exp.name, len(jobs),
                len({s.split_id for j in jobs for s in j.splits}), n_jobs)  # fmt: skip

    results, n_cached, last = [], 0, time.perf_counter()
    parallel = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered", batch_size=1)
    for i, (res, cached) in enumerate(
        parallel(delayed(run_cached)(j, frames[j.project], cfg, str(cache_dir)) for j in jobs), start=1
    ):
        results.append(res)
        n_cached += cached
        if time.perf_counter() - last > 60 or i == len(jobs):
            el = time.perf_counter() - t0
            computed = max(i - n_cached, 1)
            remaining = (len(jobs) - i) * el / computed if i > n_cached else float("nan")
            logger.info("%s: %d/%d jobs done (%d from cache), %.0f s elapsed, ~%.0f s remaining.",
                        exp.name, i, len(jobs), n_cached, el, remaining)  # fmt: skip
            last = time.perf_counter()

    def frame(key: str, columns=None) -> pd.DataFrame:
        parts = [pd.DataFrame(r[key]) if isinstance(r[key], list) else r[key] for r in results]
        parts = [p for p in parts if len(p)]
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
        if columns and "experiment" in columns:
            out.insert(0, "experiment", exp.name)
        return out

    res = ExperimentResult(
        metrics=frame("metrics", METRIC_COLUMNS)[METRIC_COLUMNS],
        predictions=frame("predictions", PREDICTION_COLUMNS)[PREDICTION_COLUMNS],
        hyperparameters=frame("hyperparameters"),
        timings=frame("timings"),
        info={"experiment": exp.name, "jobs": len(jobs), "jobs_from_cache": n_cached, "n_jobs": n_jobs,
              "wall_time_s": time.perf_counter() - t0, "config": exp.__dict__ | {"settings": [s.label for s in exp.settings]}},
    )  # fmt: skip
    out = cfg.paths.results_dir / exp.name
    out.mkdir(parents=True, exist_ok=True)
    res.metrics.to_csv(out / "metrics.csv", index=False)
    res.predictions.to_csv(out / "predictions.csv", index=False)
    res.hyperparameters.to_csv(out / "hyperparameters.csv", index=False)
    res.timings.to_csv(out / "timings.csv", index=False)
    (out / "run_info.json").write_text(json.dumps(res.info, indent=2, default=str), encoding="utf-8")
    logger.info("Experiment %s finished in %.0f s (%d/%d jobs from cache) -> %s",
                exp.name, res.info["wall_time_s"], n_cached, len(jobs), out)  # fmt: skip
    return res
