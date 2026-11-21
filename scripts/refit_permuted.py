"""Refit and persist the seed-0 permuted-label models and their explanations (step 7).

Usage::

    python -m scripts.refit_permuted [--n-jobs 10]

Repeats the step 6 data-randomisation refit (same permutation, hyperparameters,
preprocessing and imbalance handling; no re-tuning), saves the 96 models under
``artifacts/models/permuted_labels/`` and their test-set explanations under
``artifacts/attributions/_permuted/`` (``subset="permuted_labels"``), and checks that
the refit reproduces the step 6 test AUCs and similarities
(``results/stability/permuted_reproduction.csv``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import pandas as pd

from sdp.config import CONFIG_DIR, DEFAULT_CONFIG, REPO_ROOT, load_config
from sdp.explain.runner import experiment_splits, load_explain_settings
from sdp.log import setup_logging
from sdp.robustness.common import run_parallel
from sdp.seed import set_seed
from sdp.stability.permuted import permuted_job, reproduction_check, step6_rows

logger = logging.getLogger("refit_permuted")
MODELS = ("mlp", "lr", "rf", "xgb")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--projects", nargs="*", default=None, help="restrict to these projects (smoke test)")
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    es = load_explain_settings(CONFIG_DIR / "explain.yaml")
    splits, frames = experiment_splits(cfg, es)
    if args.projects:
        splits = [s for s in splits if s.project in args.projects]
    t0 = time.perf_counter()
    tasks = [(cfg, es, s, frames[s.project], m) for s in splits for m in MODELS]
    tasks.sort(key=lambda t: -len(t[2].test_idx) * (8 if t[4] == "mlp" else 1))
    res = run_parallel(permuted_job, tasks, args.n_jobs, "Permuted-label refit")
    wall = time.perf_counter() - t0
    rows = pd.DataFrame([r for x in res for r in x["rows"]])
    check = reproduction_check(rows, step6_rows(cfg.paths.artifacts_dir / "fidelity"))
    out = cfg.paths.results_dir / "stability"
    out.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if args.projects else ""
    check.to_csv(out / f"permuted_reproduction{tag}.csv", index=False)
    pd.DataFrame([x["timing"] for x in res]).to_csv(out / f"timings_permuted{tag}.csv", index=False)
    summary = {"models": len(res), "explanations": len(rows), "matched_to_step6": len(check), "wall_time_s": wall,
               **{f"max_{c}": float(check[c].max()) for c in check.columns if c.startswith("diff_")}}  # fmt: skip
    (out / f"permuted_run_info{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Permuted-label refit: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
