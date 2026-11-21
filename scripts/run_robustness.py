"""Run the step 5 robustness and sensitivity studies.

Usage::

    python -m scripts.run_robustness [--studies lime background preprocessing autospearman]
        [--smoke] [--estimate-only] [--n-jobs 10]

``--smoke`` first runs the LIME, background and preprocessing studies on one split
(seed 0) and estimates the full runtime (the AutoSpearman retraining is estimated from
the step 3 timings). If the estimate exceeds ``max_hours`` (8 h), the background study
is restricted to the splits of five projects and then, if still needed, the LIME
settings study to 20 instances per split; reductions are logged and recorded in
``results/robustness/runtime_estimate.json``.

``autospearman`` retrains MLP and RF on the AutoSpearman-selected metrics (step 3
engine, ``configs/experiments/autospearman_time_aware.yaml``) and explains them with
DeepLiftShap / TreeSHAP (step 4 runner).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from sdp.config import CONFIG_DIR, DEFAULT_CONFIG, REPO_ROOT, load_config
from sdp.experiments.config import load_experiment
from sdp.experiments.engine import run_experiment
from sdp.explain.runner import build_jobs, experiment_splits, load_explain_settings, run_explanations
from sdp.log import setup_logging
from sdp.robustness import background_study as B
from sdp.robustness import lime_study as L
from sdp.robustness import preprocessing_study as P
from sdp.robustness.common import load_robustness, output_root, timing_frame
from sdp.seed import set_seed

logger = logging.getLogger("run_robustness")
STUDIES = ("lime", "background", "preprocessing", "autospearman")


def autospearman_settings(es, rs):
    exp = rs["collinearity"]["retrain_experiment"]
    seeds = list(load_experiment(CONFIG_DIR / "experiments" / f"{exp}.yaml").seeds)
    return dataclasses.replace(
        es, experiment=exp, models={"mlp": "log_standard", "rf": "standard"},
        methods={"mlp": ["DeepLiftShap"], "rf": ["TreeSHAP"]}, seeds=seeds, expensive_methods=[],
        output_dir=f"attributions_{exp}",
    )  # fmt: skip


def estimate_retraining_cpu_s(cfg, n_trials: int = 25) -> float:
    """CPU seconds of the AutoSpearman retraining from the step 3 timings (same models,
    seeds 0–2), scaling the tuning time by the number of trials (25 instead of 40)."""
    t = pd.read_csv(cfg.paths.results_dir / "main_time_aware" / "timings.csv")
    t = t[t["model"].isin(["mlp", "rf"]) & t["seed"].isin([0, 1, 2])]
    return float(t["tuning_time_s"].sum() * n_trials / cfg.tuning.n_trials_time_aware + t["fit_time_s"].sum())


def smoke_and_estimate(cfg, es, rs, splits, frames, n_jobs: int, studies) -> dict:
    by_size = sorted(splits, key=lambda s: len(s.test_idx))
    smoke = by_size[len(by_size) // 2]
    root = output_root(cfg, rs) / "_smoke"
    out = {"smoke_split": smoke.split_id, "studies": {}}
    n_test = {s.split_id: len(s.test_idx) for s in splits}
    n_fit = {s.split_id: len(s.fit_idx) for s in splits}

    if "lime" in studies:
        sample = L.instance_samples(cfg, es, [smoke], rs["lime"])
        t0 = time.perf_counter()
        res = L.run_lime_study(cfg, es, rs, [smoke], frames, sample, n_jobs, root / "lime")
        tf = timing_frame(res, "lime")
        per_inst = (tf["time_s"] / tf["n_instances"]).mean()
        full_inst = len(L.instance_samples(cfg, es, splits, rs["lime"]))
        red_inst = len(L.instance_samples(cfg, es, splits, rs["lime"], reduced=True))
        out["studies"]["lime"] = {"smoke_wall_s": time.perf_counter() - t0, "cpu_s_per_instance_model": per_inst,
                                  "cpu_s": per_inst * full_inst * len(rs["lime"]["models"]),
                                  "cpu_s_reduced": per_inst * red_inst * len(rs["lime"]["models"])}  # fmt: skip
    if "background" in studies:
        t0 = time.perf_counter()
        res = B.run_background_study(cfg, es, rs, [smoke], frames, n_jobs, root / "background")
        tf = timing_frame(res, "background")
        scaled = {"size=all", "scheme=kmeans"}

        def cpu(split_ids):
            total = 0.0
            for r in tf.itertuples():
                rate = r.time_s / r.n_instances
                for sid in split_ids:
                    f = n_fit[sid] / n_fit[smoke.split_id] if r.variant in scaled else 1.0
                    total += rate * n_test[sid] * f
            return total

        reduced = [s.split_id for s in splits if s.project in rs["background"]["reduced_projects"]]
        out["studies"]["background"] = {"smoke_wall_s": time.perf_counter() - t0, "cpu_s": cpu(list(n_test)),
                                        "cpu_s_reduced": cpu(reduced)}  # fmt: skip
    if "preprocessing" in studies:
        t0 = time.perf_counter()
        infos = P.run_preprocessing_study(cfg, es, rs, n_jobs, splits=[smoke], seeds=[0])
        cpu_smoke = sum(i["timings"]["time_s"].sum() for i in infos)
        total_test = sum(n_test.values())
        out["studies"]["preprocessing"] = {"smoke_wall_s": time.perf_counter() - t0,
                                           "cpu_s": cpu_smoke / n_test[smoke.split_id] * total_test * 10}  # fmt: skip
    if "autospearman" in studies:
        out["studies"]["autospearman"] = {"cpu_s": estimate_retraining_cpu_s(cfg) * 1.05}  # + explanations (small)
    eff = 0.7
    cpu = sum(v["cpu_s"] for v in out["studies"].values())
    out["cpu_hours"] = cpu / 3600
    out["wall_hours"] = cpu / (n_jobs * eff) / 3600
    out["parallel_efficiency_assumed"] = eff
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--studies", nargs="+", default=list(STUDIES), choices=STUDIES)
    parser.add_argument("--settings", default=str(CONFIG_DIR / "robustness.yaml"))
    parser.add_argument("--explain-settings", default=str(CONFIG_DIR / "explain.yaml"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    es = load_explain_settings(args.explain_settings)
    rs = load_robustness(args.settings)
    n_jobs = args.n_jobs or rs.get("n_jobs", 10)
    splits, frames = experiment_splits(cfg, es)
    res_dir = cfg.paths.results_dir / "robustness"
    res_dir.mkdir(parents=True, exist_ok=True)
    reductions = {"background_projects": None, "lime_instances_per_split": None}

    if args.smoke:
        est = smoke_and_estimate(cfg, es, rs, splits, frames, n_jobs, args.studies)
        logger.info("Estimated full step 5 run: %.2f CPU h, ~%.2f h wall (n_jobs=%d).", est["cpu_hours"],
                    est["wall_hours"], n_jobs)  # fmt: skip
        wall = est["wall_hours"]
        to_h = lambda s: s / (n_jobs * est["parallel_efficiency_assumed"]) / 3600  # noqa: E731
        if wall > rs["max_hours"] and "background" in est["studies"]:
            b = est["studies"]["background"]
            wall -= to_h(b["cpu_s"] - b["cpu_s_reduced"])
            reductions["background_projects"] = rs["background"]["reduced_projects"]
            logger.warning("Estimate > %s h: background study restricted to %s (new estimate %.2f h).",
                           rs["max_hours"], reductions["background_projects"], wall)  # fmt: skip
        if wall > rs["max_hours"] and "lime" in est["studies"]:
            lm = est["studies"]["lime"]
            wall -= to_h(lm["cpu_s"] - lm["cpu_s_reduced"])
            reductions["lime_instances_per_split"] = rs["lime"]["reduced_instances_per_split"]
            logger.warning("Estimate still > %s h: LIME study reduced to %d instances per split (new estimate %.2f h).",
                           rs["max_hours"], reductions["lime_instances_per_split"], wall)  # fmt: skip
        est |= {"reductions": reductions, "wall_hours_after_reductions": wall, "n_jobs": n_jobs}
        (res_dir / "runtime_estimate.json").write_text(json.dumps(est, indent=2, default=str), encoding="utf-8")
        if args.estimate_only:
            return 0

    timings = {}
    if "lime" in args.studies:
        t0 = time.perf_counter()
        sample = L.instance_samples(cfg, es, splits, rs["lime"], reduced=bool(reductions["lime_instances_per_split"]))
        sample.to_csv(res_dir / "lime_instance_sample.csv", index=False)
        res = L.run_lime_study(cfg, es, rs, splits, frames, sample, n_jobs)
        timing_frame(res, "lime").to_csv(res_dir / "timings_lime.csv", index=False)
        timings["lime"] = time.perf_counter() - t0
    if "background" in args.studies:
        t0 = time.perf_counter()
        use = [s for s in splits if reductions["background_projects"] is None or s.project in reductions["background_projects"]]
        res = B.run_background_study(cfg, es, rs, use, frames, n_jobs)
        timing_frame(res, "background").to_csv(res_dir / "timings_background.csv", index=False)
        timings["background"] = time.perf_counter() - t0
    if "preprocessing" in args.studies:
        t0 = time.perf_counter()
        infos = P.run_preprocessing_study(cfg, es, rs, n_jobs)
        pd.concat([i["timings"].assign(preprocessing=i["preprocessing"]) for i in infos]).to_csv(
            res_dir / "timings_preprocessing.csv", index=False)
        timings["preprocessing"] = time.perf_counter() - t0
    if "autospearman" in args.studies:
        t0 = time.perf_counter()
        exp = load_experiment(CONFIG_DIR / "experiments" / f"{rs['collinearity']['retrain_experiment']}.yaml")
        run_experiment(cfg, exp, n_jobs=n_jobs)
        timings["autospearman_training"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        a_es = autospearman_settings(es, rs)
        a_splits, a_frames = experiment_splits(cfg, a_es)
        info = run_explanations(cfg, a_es, build_jobs(cfg, a_es, a_frames, a_splits), a_frames, n_jobs=n_jobs,
                                tag="autospearman")  # fmt: skip
        timings["autospearman_explanations"] = time.perf_counter() - t1
        logger.info("AutoSpearman models reproduce their predictions (max diff %s).", info["max_prob_diff_vs_step3"])
    info_path = res_dir / "run_info.json"
    prev = json.loads(info_path.read_text()) if info_path.exists() else {}
    prev.setdefault("wall_time_s", {}).update(timings)
    prev["reductions"] = reductions if args.smoke else prev.get("reductions", reductions)
    prev["n_jobs"] = n_jobs
    info_path.write_text(json.dumps(prev, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
