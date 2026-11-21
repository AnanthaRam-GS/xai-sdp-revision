"""Shared set-up for the step 5 studies: settings, seed-0 contexts and parallel jobs.

Every study reuses the step 4 machinery: the explanation settings
(``configs/explain.yaml``), the rebuilt time-aware splits, the persisted models and
the saved background sets (:mod:`sdp.explain.runner`).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from sdp.config import CONFIG_DIR, Config, REPO_ROOT
from sdp.eval.splits import Split
from sdp.experiments.persistence import safe_split_id
from sdp.explain.base import ExplainContext
from sdp.explain.runner import ExplainSettings, build_context, build_jobs

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS = CONFIG_DIR / "robustness.yaml"


def load_robustness(path: str | Path = DEFAULT_SETTINGS) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def output_root(cfg: Config, rs: dict) -> Path:
    return Path(cfg.paths.artifacts_dir) / rs.get("output_dir", "robustness")


def seed_context(cfg: Config, es: ExplainSettings, split: Split, frames: dict, model: str, seed: int = 0) -> ExplainContext:
    """The step 4 context (persisted model, saved background) of one model and seed."""
    job = build_jobs(cfg, es, frames, [split], seeds=[seed], models=[model])[0]
    return build_context(job, frames[split.project], es, es.root(cfg))


def split_dir(split: Split) -> str:
    return safe_split_id(split.split_id)


def run_parallel(func, tasks: list[tuple], n_jobs: int, label: str) -> list:
    """``func(*task)`` for every task in loky workers (one thread each), logging progress."""
    from joblib import Parallel, delayed

    t0, last, out = time.perf_counter(), time.perf_counter(), []
    parallel = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered", batch_size=1)
    for i, res in enumerate(parallel(delayed(_call)(func, t) for t in tasks), start=1):
        out.append(res)
        if time.perf_counter() - last > 60 or i == len(tasks):
            logger.info("%s: %d/%d tasks, %.0f s elapsed.", label, i, len(tasks), time.perf_counter() - t0)
            last = time.perf_counter()
    return out


def _call(func, task):
    from threadpoolctl import threadpool_limits

    from sdp.log import setup_logging
    from sdp.seed import set_seed

    setup_logging()
    set_seed(0, torch_threads=1, quiet=True)
    with threadpool_limits(limits=1):
        return func(*task)


def timing_frame(results: list[dict], study: str) -> pd.DataFrame:
    rows = [r["timing"] | {"study": study} for r in results if "timing" in r]
    return pd.DataFrame(rows)
