"""One-command reproduction of steps 1–8 (full or quick) and the paper assets.

* ``full``: the complete pipeline from the raw data (downloads only if a raw file listed
  in ``MANIFEST.csv`` is missing), with the expected runtime of each step printed
  from the step runtime reports (``paper/data/runtime_totals.csv``).
* ``quick``: the same pipeline on two projects (ant, synapse), two seeds, five Optuna
  trials, LIME/KernelSHAP on 20 instances per split and smaller study settings. Its
  configuration is derived from ``configs/`` (see :func:`write_quick_configs`) and points
  every output to ``results_quick/``, ``reports_quick/`` and ``artifacts_quick/``, so it
  never overwrites the real results.
* ``paper-assets``: regenerate ``paper/figures``, ``paper/tables`` and ``paper/numbers.tex``
  from the existing results and check the numbers.

Each step runs the documented entry points as subprocesses (``python -m scripts.<name>``)
with ``SDP_CONFIG_DIR`` selecting the configuration set.
"""

from __future__ import annotations

import copy
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from sdp.config import REPO_ROOT

logger = logging.getLogger(__name__)

EXPERIMENTS = ["main_time_aware", "main_within_release", "ablation_preprocessing", "ablation_imbalance",
               "main_time_aware_cumulative"]  # fmt: skip
QUICK_PROJECTS = ["ant", "synapse"]
QUICK_DIRS = {"results_dir": "results_quick", "reports_dir": "reports_quick", "artifacts_dir": "artifacts_quick",
              "processed_dir": "results_quick/data/processed"}  # fmt: skip
PROTECTED = ("results", "reports", "artifacts", "data/processed")  # quick mode must never write here


@dataclass
class Step:
    number: int
    title: str
    commands: list[list[str]]
    expected: str


def _expected(step: str, totals: pd.DataFrame | None) -> str:
    if totals is None or step not in set(totals["step"]):
        return "a few minutes"
    h = float(totals.set_index("step").loc[step, "wall_hours"])
    return f"about {h * 60:.0f} min" if h < 1.5 else f"about {h:.1f} h"


def plan(mode: str, config_dir: Path, offline: bool) -> list[Step]:
    """The commands of every step; ``config_dir`` holds the configuration set."""
    py = [sys.executable, "-m"]
    totals_path = REPO_ROOT / "paper" / "data" / "runtime_totals.csv"
    totals = pd.read_csv(totals_path) if totals_path.exists() and mode == "full" else None
    exp = [str(Path(config_dir) / "experiments" / f"{e}.yaml") for e in EXPERIMENTS]
    q = mode == "quick"
    steps = [
        Step(1, "data: download (if missing), clean, audit, study dataset",
             [py + ["scripts.prepare_data"] + (["--offline"] if offline else [])], "about 1 min"),
        Step(2, "evaluation framework: trivial baselines", [py + ["scripts.run_baselines"]], "about 1 min"),
        Step(3, "models, tuning and experiments", [py + ["scripts.run_experiment", *exp], py + ["scripts.make_step3_reports"]],
             _expected("step3_experiments", totals) if not q else "a few minutes"),
        Step(4, "explanations (v2) and the kept v1 LIME setting",
             [py + ["scripts.run_explanations", "--smoke"], py + ["scripts.run_explanations", "--methods", "LIME_mean_sampling"],
              py + ["scripts.make_step4_reports"]],
             _expected("step4_explanations_v2", totals) if not q else "a few minutes"),
        Step(5, "robustness and sensitivity", [py + ["scripts.run_robustness", "--smoke"], py + ["scripts.make_step5_reports"]],
             _expected("step5_robustness", totals) if not q else "a few minutes"),
        Step(6, "fidelity", [py + ["scripts.run_fidelity", "--smoke"], py + ["scripts.make_step6_reports"]],
             _expected("step6_fidelity", totals) if not q else "a few minutes"),
        Step(7, "stability and drift (incl. the permuted-label refit)",
             [py + ["scripts.refit_permuted"], py + ["scripts.run_stability"], py + ["scripts.make_step7_reports"]],
             _expected("step7_stability", totals) if not q else "about 1 min"),
        Step(8, "closing analyses, Scott-Knott ESD cross-check (if R is available) and paper assets",
             [py + ["scripts.run_closing_analyses"], py + ["scripts.run_skesd_crosscheck"],
              py + ["scripts.make_paper_assets", "--collect"] + (["--out", "results_quick/paper"] if q else [])],
             "a few minutes"),
    ]  # fmt: skip
    return steps


# ------------------------------------------------------------------ quick configuration


def _load(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _dump(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def quick_overrides(src: Path = REPO_ROOT / "configs") -> dict[str, dict]:
    """The quick-mode configuration files (relative path → content), derived from ``src``."""
    out = {}
    d = copy.deepcopy(_load(src / "default.yaml"))
    d["paths"] |= QUICK_DIRS
    d["projects"] = {p: v for p, v in d["projects"].items() if p in QUICK_PROJECTS}
    d.setdefault("evaluation", {})
    d["evaluation"].setdefault("within_release", {})["n_repeats"] = 1
    d.setdefault("tuning", {}).setdefault("n_trials", {}).update({"time_aware": 5, "within_release": 5})
    out["default.yaml"] = d
    for f in sorted((src / "experiments").glob("*.yaml")):
        e = copy.deepcopy(_load(f))
        e["seeds"] = [0] if e.get("protocol") == "within_release" else [0, 1]
        e["n_trials"] = 5
        out[f"experiments/{f.name}"] = e
    x = copy.deepcopy(_load(src / "explain.yaml"))
    x |= {"seeds": [0, 1], "expensive_seeds": [0, 1], "expensive_instances": 20}
    out["explain.yaml"] = x
    r = copy.deepcopy(_load(src / "robustness.yaml"))
    r["lime"] |= {"instances_per_group": 2, "runs": 3}
    out["robustness.yaml"] = r
    f6 = copy.deepcopy(_load(src / "fidelity.yaml"))
    f6["curves"] |= {"sample_per_split": 50, "n_random": 3, "faith_subsets": 20}
    f6["roar"] |= {"ks": [1, 3, 5], "seeds": [0], "n_random": 2}
    out["fidelity.yaml"] = f6
    return out


def write_quick_configs(target: Path, src: Path = REPO_ROOT / "configs") -> Path:
    for rel, content in quick_overrides(src).items():
        _dump(content, Path(target) / rel)
    return Path(target)


def quick_paths_are_separate(config_dir: Path) -> bool:
    """True when no output path of the configuration lies in a protected folder."""
    paths = _load(Path(config_dir) / "default.yaml")["paths"]
    outputs = [paths[k] for k in ("results_dir", "reports_dir", "artifacts_dir", "processed_dir")]
    return all(not any(Path(o) == Path(p) or Path(p) in Path(o).parents for p in PROTECTED) for o in outputs)


# ------------------------------------------------------------------ running


def raw_data_complete(raw_dir: Path = REPO_ROOT / "data" / "raw") -> bool:
    manifest = Path(raw_dir) / "MANIFEST.csv"
    if not manifest.exists():
        return False
    m = pd.read_csv(manifest)
    return all((Path(raw_dir) / f).exists() for f in m["file_name"])


def run(mode: str, dry_run: bool = False, steps=None) -> int:
    if mode == "quick":
        config_dir = write_quick_configs(REPO_ROOT / "results_quick" / "configs")
        if not quick_paths_are_separate(config_dir):
            raise RuntimeError("quick configuration would write to the real results")
    else:
        config_dir = REPO_ROOT / "configs"
    env = dict(os.environ, SDP_CONFIG_DIR=str(config_dir))
    offline = raw_data_complete()
    todo = [s for s in plan(mode, config_dir, offline) if steps is None or s.number in steps]
    t_all = time.perf_counter()
    for s in todo:
        print(f"[step {s.number}] {s.title} — expected {s.expected}", flush=True)
        for cmd in s.commands:
            print("   $ " + " ".join(c if " " not in c else repr(c) for c in cmd[1:]), flush=True)
            if dry_run:
                continue
            t0 = time.perf_counter()
            res = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
            if res.returncode != 0:
                print(f"[step {s.number}] failed after {time.perf_counter() - t0:.0f} s (exit {res.returncode})", flush=True)
                return res.returncode
        print(f"[step {s.number}] done ({time.perf_counter() - t_all:.0f} s elapsed)", flush=True)
    return 0
