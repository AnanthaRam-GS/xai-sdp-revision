"""Step 3 reports: main results, within-release, ablations, cumulative check, runtime."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pandas as pd

from sdp.analysis import tables as T
from sdp.config import Config
from sdp.data.audit import to_markdown
from sdp.eval.stats import holm

logger = logging.getLogger(__name__)

MODEL_ORDER = ["mlp", "lr", "rf", "xgb", "manual_up", "manual_down", "all_defective", "majority"]
LEARNED = ["mlp", "lr", "rf", "xgb"]


def load(cfg: Config, experiment: str, what: str = "metrics") -> pd.DataFrame:
    path = cfg.paths.results_dir / experiment / f"{what}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the experiment first")
    return pd.read_csv(path, dtype={"train_releases": str, "test_release": str})


def run_info(cfg: Config, experiment: str) -> dict:
    p = cfg.paths.results_dir / experiment / "run_info.json"
    info = json.loads(p.read_text()) if p.exists() else {}
    hp = cfg.paths.results_dir / experiment / "hyperparameters.csv"
    if hp.exists():
        trials = sorted(pd.read_csv(hp, usecols=["n_trials"])["n_trials"].unique())
        info["n_trials_used"] = ", ".join(str(int(t)) for t in trials)
    return info


def _write(cfg: Config, name: str, md: list[str], csvs: dict[str, pd.DataFrame]) -> None:
    rep = cfg.paths.reports_dir
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for suffix, df in csvs.items():
        df.to_csv(rep / f"{name}{suffix}.csv", index=False)
    logger.info("Wrote %s", rep / f"{name}.md")


def _setup_lines(df: pd.DataFrame, info: dict) -> list[str]:
    hp = info.get("config", {})
    settings = df.groupby("model", sort=False)[["preprocessing", "imbalance"]].first()
    settings = settings.reindex([m for m in MODEL_ORDER if m in settings.index])
    desc = ", ".join(f"`{m}` ({r.preprocessing}, {r.imbalance})" for m, r in settings.iterrows())
    seeds = df.groupby("model")["seed"].nunique()
    return [
        f"- Splits: {df['split_id'].nunique()}; seeds per learned model: "
        f"{int(seeds.reindex(LEARNED).max()) if seeds.reindex(LEARNED).notna().any() else 1}; "
        f"rule-based baselines: 1 (deterministic).",
        f"- Models (preprocessing, imbalance): {desc}.",
        f"- Optuna trials per tuning run: {info.get('n_trials_used') or hp.get('n_trials') or 'protocol default'} "
        "(chosen hyperparameters in `results/<experiment>/hyperparameters.csv`).",
    ]


STATS_NOTE = [
    "Summaries: mean and percentile-bootstrap 95% CI (2,000 resamples) over all splits x seeds. "
    "Statistical comparisons use the per-split mean across seeds (each split counts once). "
    "`@0.5` = 0.5 threshold; `@tuned` = MCC-maximising threshold chosen on the validation split. "
    "Lower is better for Brier, ECE, IFA and NECM15; higher for all other metrics.",
]


def model_report(cfg: Config, experiment: str, name: str, title: str) -> pd.DataFrame:
    """Summary, per-project tables, Scott-Knott ESD ranks and MLP-vs-others tests."""
    df = load(cfg, experiment)
    e = cfg.evaluation
    long = T.summary_long(df, T.MAIN_METRICS, ["model"], e.n_resamples, e.confidence, cfg.seed)
    wide = T.summary_wide(long, "model", MODEL_ORDER)
    ranks = T.skesd_ranks(df, T.RANK_METRICS, order=MODEL_ORDER)
    tests = T.reference_comparisons(df, T.RANK_METRICS, "mlp", order=MODEL_ORDER)
    md = [f"# {title}", "", f"Experiment `{experiment}`.", "", *_setup_lines(df, run_info(cfg, experiment)), "",
          *STATS_NOTE, "", "## Summary over all splits", "", to_markdown(wide), ""]  # fmt: skip
    projects = []
    for spec in T.PROJECT_METRICS:
        t = T.project_table(df, spec, order=MODEL_ORDER)
        projects.append(t.assign(metric=spec.label))
        md += [f"## Per project: {spec.label} (mean over splits x seeds)", "", to_markdown(t), ""]
    md += [
        "## Scott-Knott ESD ranks (1 = best)",
        "",
        "On per-split means across seeds; lower-is-better metrics are negated before ranking. "
        "Models sharing a rank are not distinguishable (non-negligible effect size required).",
        "",
        to_markdown(ranks),
        "",
        "## MLP vs each other model: Wilcoxon signed-rank (Holm) and Cliff's delta",
        "",
        "Paired by split (per-split means across seeds); Holm correction across the comparisons "
        "of each metric. `cliffs_delta` > 0 means MLP has larger values (for NECM15 larger is worse). "
        "Magnitudes follow Romano et al. (negligible < 0.147 <= small < 0.33 <= medium < 0.474 <= large).",
        "",
        to_markdown(T.format_comparisons(tests)),
        "",
    ]
    _write(cfg, name, md, {"": long, "_projects": pd.concat(projects), "_skesd": ranks, "_wilcoxon": tests})
    return long


def ablation_report(cfg: Config, experiment: str, name: str, title: str, factor: str) -> pd.DataFrame:
    """Settings of ``factor`` (preprocessing or imbalance) compared within each model."""
    df = load(cfg, experiment)
    e = cfg.evaluation
    long = T.summary_long(df, T.ABLATION_METRICS, ["model", factor], e.n_resamples, e.confidence, cfg.seed)
    tests = T.all_pairs_comparisons(df, T.ABLATION_METRICS, factor, within="model")
    md = [f"# {title}", "", f"Experiment `{experiment}`.", "", *_setup_lines(df, run_info(cfg, experiment)), "",
          *STATS_NOTE, ""]  # fmt: skip
    for model in [m for m in MODEL_ORDER if m in set(df["model"])]:
        sub = long[long["model"] == model]
        md += [f"## {model}", "", to_markdown(T.summary_wide(sub, factor)), ""]
    md += [
        f"## Pairwise comparisons between {factor} settings",
        "",
        f"Wilcoxon signed-rank on per-split means across seeds, Holm correction within each "
        f"(model, metric) family; `cliffs_delta` > 0 means setting `a` has larger values.",
        "",
        to_markdown(T.format_comparisons(tests)),
        "",
    ]
    _write(cfg, name, md, {"": long, "_wilcoxon": tests})
    return long


def cumulative_report(cfg: Config, name: str = "step3_cumulative") -> pd.DataFrame:
    cons = load(cfg, "main_time_aware")
    cum = load(cfg, "main_time_aware_cumulative")
    seeds = sorted(int(s) for s in cum["seed"].unique())
    cons = cons[cons["model"].isin(LEARNED) & cons["seed"].isin(seeds)].assign(variant="consecutive")
    cum = cum.assign(variant="cumulative")
    df = pd.concat([cons, cum], ignore_index=True)
    df["pair"] = df["project"] + ":" + df["test_release"].astype(str)
    e = cfg.evaluation
    long = T.summary_long(df, T.ABLATION_METRICS, ["model", "variant"], e.n_resamples, e.confidence, cfg.seed)

    rows = []
    for spec in T.ABLATION_METRICS:
        fam = []
        for model in LEARNED:
            wide = T.per_split(df[df["model"] == model], spec, "variant", block="pair")
            fam.append({"metric": spec.label, "model": model, "a": "consecutive", "b": "cumulative"}
                       | T.paired_comparison(wide["consecutive"], wide["cumulative"]))  # fmt: skip
        for r, p in zip(fam, holm([r["p_value"] for r in fam])):
            r["p_holm"] = p
        rows += fam
    tests = T._order_cols(pd.DataFrame(rows))
    same = (cons.groupby("split_id")["train_releases"].first().str.count(r"\+") == 0).sum()
    md = [
        "# Consecutive vs cumulative time-aware training",
        "",
        f"Robustness check for the four learned models with seeds {seeds}. Consecutive results are the "
        "`main_time_aware` runs with the same seeds; cumulative training uses all included releases "
        "up to n. Both are tested on release n+1; pairs are matched on (project, test release). The "
        f"first pair of each project has identical training data in both variants ({len(set(cum['project']))} of "
        f"{cum['split_id'].nunique()} pairs).",
        "",
        *STATS_NOTE,
        "",
    ]
    for model in LEARNED:
        md += [f"## {model}", "", to_markdown(T.summary_wide(long[long["model"] == model], "variant")), ""]
    md += ["## Wilcoxon (Holm across the four models per metric) and Cliff's delta", "",
           "`cliffs_delta` > 0 means consecutive has larger values.", "",
           to_markdown(T.format_comparisons(tests)), ""]  # fmt: skip
    del same
    _write(cfg, name, md, {"": long, "_wilcoxon": tests})
    return long


# --------------------------------------------------------------------- runtime


def machine_details() -> dict:
    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    cpu = sysctl("machdep.cpu.brand_string") or platform.processor()
    mem = sysctl("hw.memsize")
    libs = ["numpy", "pandas", "scikit-learn", "scipy", "torch", "xgboost", "optuna", "imbalanced-learn",
            "captum", "joblib"]  # fmt: skip
    return {
        "CPU": cpu,
        "Logical cores": os.cpu_count(),
        "Performance / efficiency cores": f"{sysctl('hw.perflevel0.physicalcpu')} / {sysctl('hw.perflevel1.physicalcpu')}"
        if sysctl("hw.perflevel0.physicalcpu") else "n/a",
        "RAM (GB)": round(int(mem) / 2**30, 1) if mem else "n/a",
        "OS": f"{platform.system()} {platform.mac_ver()[0] or platform.release()} ({platform.machine()})",
        "Python": sys.version.split()[0],
        **{lib: _version(lib) for lib in libs},
    }


def _version(lib: str) -> str:
    try:
        return metadata.version(lib)
    except metadata.PackageNotFoundError:
        return "n/a"


def runtime_report(cfg: Config, experiments: list[str], name: str = "step3_runtime") -> pd.DataFrame:
    key = ["split_id", "model", "preprocessing", "imbalance", "seed"]
    per_exp, timings = [], []
    for exp in experiments:
        t = load(cfg, exp, "timings")
        info = run_info(cfg, exp)
        timings.append(t.assign(experiment=exp))
        per_exp.append({
            "experiment": exp, "jobs": info.get("jobs"), "jobs_reused_from_cache": info.get("jobs_from_cache"),
            "n_jobs": info.get("n_jobs"), "wall_time_h": round(info.get("wall_time_s", float("nan")) / 3600, 2),
            "tuning_cpu_h": round(t["tuning_time_s"].sum() / 3600, 2),
            "fit_cpu_h": round(t["fit_time_s"].sum() / 3600, 2),
        })  # fmt: skip
    t = pd.concat(timings, ignore_index=True)
    unique = t.drop_duplicates(key)  # units shared between experiments are computed once
    per_model = (
        unique.groupby("model")
        .agg(units=("split_id", "size"), tuning_cpu_h=("tuning_time_s", "sum"), fit_cpu_h=("fit_time_s", "sum"),
             mean_tuning_s=("tuning_time_s", lambda s: s[s > 0].mean()), mean_fit_s=("fit_time_s", "mean"))  # fmt: skip
        .reindex([m for m in MODEL_ORDER if m in set(unique["model"])])
        .reset_index()
    )
    for c in ("tuning_cpu_h", "fit_cpu_h"):
        per_model[c] = (per_model[c] / 3600).round(3)
    for c in ("mean_tuning_s", "mean_fit_s"):
        per_model[c] = per_model[c].round(2)
    est_path = cfg.paths.results_dir / "runtime_estimate.json"
    est = json.loads(est_path.read_text()) if est_path.exists() else {}
    per_exp_df = pd.DataFrame(per_exp)
    total_wall = per_exp_df["wall_time_h"].sum()
    md = [
        "# Step 3 runtime",
        "",
        "## Machine",
        "",
        to_markdown(pd.DataFrame(list(machine_details().items()), columns=["item", "value"])),
        "",
        "## Experiments",
        "",
        "CPU hours are summed over jobs (single-threaded workers). Jobs reused from the cache were "
        "computed by an earlier experiment with identical settings (e.g. the `log_standard` MLP/LR "
        "units of the preprocessing ablation), so their CPU time is listed under both experiments "
        "but spent once.",
        "",
        to_markdown(per_exp_df),
        "",
        f"Total wall time: {total_wall:.2f} h. Total CPU time of unique units: "
        f"{(unique['tuning_time_s'].sum() + unique['fit_time_s'].sum()) / 3600:.2f} h.",
        "",
        "## Per model (unique units over all experiments)",
        "",
        "`units` = fitted (split, setting, seed) units; `mean_tuning_s` = mean time of one tuning run "
        "(Optuna study); `mean_fit_s` = mean time of the final fit on the fitting part.",
        "",
        to_markdown(per_model),
        "",
    ]
    if est:
        md += [
            "## Smoke test and estimate",
            "",
            f"Smoke test (`main_time_aware` on ant, 2 seeds, 5 trials): {est.get('smoke_wall_s', float('nan')):.1f} s. "
            f"Estimated before the full run: {est.get('total_cpu_hours', float('nan')):.1f} CPU h, "
            f"~{est.get('estimated_wall_hours', float('nan')):.1f} h wall with n_jobs={est.get('n_jobs')}. "
            f"The estimate was below 8 h, so the time-aware tuning budget was kept at "
            f"{cfg.tuning.n_trials_time_aware} trials.",
            "",
        ]
    _write(cfg, name, md, {"": per_model, "_experiments": per_exp_df})
    return per_model
