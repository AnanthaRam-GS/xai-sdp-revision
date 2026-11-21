"""Collect the small derived inputs of the paper assets into ``paper/data/``.

The paper figures, tables and numbers are generated from ``reports/`` (committed) and
from the files written here, so that ``make paper-assets`` also works in a fresh clone or
from the replication archive, where the git-ignored ``results/`` and ``artifacts/`` are
absent. Everything is derived from existing results without recomputing any experiment:

* ``performance_per_split.csv``: time-aware test metrics per split and model (seeds
  averaged), from ``results/main_time_aware/metrics.csv``;
* ``within_release_means.csv``: within-release mean per model and metric;
* ``ablation_per_split.csv``: the preprocessing and imbalance ablations per split;
* ``deletion_curves.csv``: mean deletion/insertion curves per explanation (step 6);
* ``stability_null.csv``: the step 7 random-null summary;
* ``dendrogram_<release>.csv``: the Spearman matrix of the step 5 representative release.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sdp.config import CONFIG_DIR, Config

logger = logging.getLogger(__name__)

PERFORMANCE = [("roc_auc", "none", "AUC"), ("pr_auc", "none", "PR-AUC"), ("mcc", "val_mcc", "MCC@tuned"),
               ("f1", "val_mcc", "F1@tuned"), ("recall", "val_mcc", "Recall@tuned"), ("brier", "none", "Brier"),
               ("popt", "none", "Popt"), ("recall_at_20pct_loc", "none", "Recall@20%LOC"),
               ("necm_15", "val_mcc", "NECM15@tuned")]  # fmt: skip
DENDROGRAM_SPLIT = "ta-consecutive:jedit:4.0>4.1"  # the step 5 representative release


def _metric_frame(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path, dtype={"train_releases": str, "test_release": str})
    keep = pd.Series(False, index=m.index)
    for metric, mode, _ in PERFORMANCE:
        keep |= (m["metric"] == metric) & (m["threshold_mode"] == mode)
    m = m[keep].copy()
    lab = {(a, b): c for a, b, c in PERFORMANCE}
    m["label"] = [lab[(a, b)] for a, b in zip(m["metric"], m["threshold_mode"])]
    return m


def runtime_totals(res: Path) -> pd.DataFrame:
    """Wall-clock hours per step from the run information files (10 workers)."""
    import json

    def load(p):
        p = res / p
        return json.loads(p.read_text()) if p.exists() else {}

    rows = []
    step3 = sum(load(f"{e}/run_info.json").get("wall_time_s", 0.0) for e in (
        "main_time_aware", "main_within_release", "ablation_preprocessing", "ablation_imbalance", "main_time_aware_cumulative"))
    rows.append({"step": "step3_experiments", "wall_hours": step3 / 3600})
    rows.append({"step": "step4_explanations_v1", "wall_hours": load("explanations/v1/run_info_full.json").get("wall_time_s", 0.0) / 3600})
    rows.append({"step": "step4_explanations_v2", "wall_hours": load("explanations/run_info_full.json").get("wall_time_s", 0.0) / 3600})
    rows.append({"step": "step5_robustness", "wall_hours": sum(load("robustness/run_info.json").get("wall_time_s", {}).values()) / 3600})
    rows.append({"step": "step6_fidelity", "wall_hours": sum(load("fidelity/run_info.json").get("wall_time_s", {}).values()) / 3600})
    rows.append({"step": "step7_stability", "wall_hours": (load("stability/run_info.json").get("wall_time_s", 0.0)
                                                           + load("stability/permuted_run_info.json").get("wall_time_s", 0.0)) / 3600})
    t = pd.DataFrame(rows)
    return pd.concat([t, pd.DataFrame([{"step": "total", "wall_hours": t["wall_hours"].sum()}])], ignore_index=True)


def collect(cfg: Config, out: Path, fidelity_root: Path | None = None) -> list[Path]:
    """Refresh the paper inputs whose sources exist; an input whose sources are missing
    (e.g. the git-ignored artefacts in the replication archive) keeps its committed file."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res = Path(cfg.paths.results_dir)
    written, skipped = [], []

    def item(name: str, make) -> None:
        try:
            frame = make()
        except (FileNotFoundError, KeyError, ValueError, StopIteration) as exc:
            skipped.append(f"{name} ({exc.__class__.__name__}: {exc})")
            return
        if frame is None:
            skipped.append(f"{name} (no source)")
            return
        path = out / name
        if name.endswith(".json"):
            path.write_text(frame, encoding="utf-8")
        else:
            frame.to_csv(path, index=name.startswith("dendrogram"))
        written.append(path)

    def performance():
        m = _metric_frame(res / "main_time_aware" / "metrics.csv")
        return m.groupby(["project", "split_id", "model", "label"], as_index=False)["value"].mean()

    def within():
        return _metric_frame(res / "main_within_release" / "metrics.csv").groupby(["model", "label"], as_index=False)["value"].mean()

    def ablation():
        parts = []
        for exp, factor in (("ablation_preprocessing", "preprocessing"), ("ablation_imbalance", "imbalance")):
            a = _metric_frame(res / exp / "metrics.csv")
            a = a.groupby(["split_id", "model", factor, "label"], as_index=False)["value"].mean()
            parts.append(a.rename(columns={factor: "setting"}).assign(ablation=factor))
        return pd.concat(parts)

    def curves():
        if fidelity_root is None or not (Path(fidelity_root) / "curves").exists():
            return None
        from sdp.fidelity.evaluation import load_curves

        c = load_curves(Path(fidelity_root) / "curves")
        c = c[c["removal"] == "marginal"]
        # the stored curves are means over a split's instances; average splits with equal weight
        return c.groupby(["model", "method", "kind", "k"], as_index=False)["value"].mean()

    def runtime():
        rt = runtime_totals(res)
        return rt if rt["wall_hours"].sum() > 0 else None

    def clusters():
        p = res / "stability" / "clusters.json"
        return p.read_text(encoding="utf-8") if p.exists() else None

    def null():
        p = res / "stability" / "null.csv"
        return pd.read_csv(p) if p.exists() else None

    def dendrogram():
        from sdp.eval.preprocess import feature_frame
        from sdp.explain.runner import experiment_splits, load_explain_settings

        splits, frames = experiment_splits(cfg, load_explain_settings(CONFIG_DIR / "explain.yaml"))
        split = next((s for s in splits if s.split_id == DENDROGRAM_SPLIT), None)
        if split is None:  # e.g. quick mode without jedit: the first consecutive split instead
            split = next((s for s in splits if s.split_id.startswith("ta-consecutive:")), None)
            if split is None:
                return None
            logger.warning("Representative split %s absent; the dendrogram uses %s", DENDROGRAM_SPLIT, split.split_id)
        return feature_frame(frames[split.project]).iloc[split.fit_idx].astype(float).corr(method="spearman")

    for name, make in (("performance_per_split.csv", performance), ("within_release_means.csv", within),
                       ("ablation_per_split.csv", ablation), ("deletion_curves.csv", curves),
                       ("runtime_totals.csv", runtime), ("stable_clusters.json", clusters), ("stability_null.csv", null),
                       ("dendrogram_spearman.csv", dendrogram)):  # fmt: skip
        item(name, make)
    for s_ in skipped:
        logger.warning("Paper input not refreshed (the committed file is kept): %s", s_)
    logger.info("Collected %d paper input file(s) into %s", len(written), out)
    return written
