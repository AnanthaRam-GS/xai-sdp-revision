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

from sdp.config import Config

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


def collect(cfg: Config, out: Path, fidelity_root: Path | None = None) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res = Path(cfg.paths.results_dir)
    written = []

    m = _metric_frame(res / "main_time_aware" / "metrics.csv")
    per = m.groupby(["project", "split_id", "model", "label"], as_index=False)["value"].mean()
    per.to_csv(out / "performance_per_split.csv", index=False)
    written.append(out / "performance_per_split.csv")

    w = _metric_frame(res / "main_within_release" / "metrics.csv")
    w.groupby(["model", "label"], as_index=False)["value"].mean().to_csv(out / "within_release_means.csv", index=False)
    written.append(out / "within_release_means.csv")

    parts = []
    for exp, factor in (("ablation_preprocessing", "preprocessing"), ("ablation_imbalance", "imbalance")):
        a = _metric_frame(res / exp / "metrics.csv")
        a = a.groupby(["split_id", "model", factor, "label"], as_index=False)["value"].mean()
        parts.append(a.rename(columns={factor: "setting"}).assign(ablation=factor))
    pd.concat(parts).to_csv(out / "ablation_per_split.csv", index=False)
    written.append(out / "ablation_per_split.csv")

    if fidelity_root is not None and (Path(fidelity_root) / "curves").exists():
        from sdp.fidelity.evaluation import load_curves

        c = load_curves(Path(fidelity_root) / "curves")
        c = c[c["removal"] == "marginal"]
        # mean over instances within a split is stored; average splits with equal weight
        c.groupby(["model", "method", "kind", "k"], as_index=False)["value"].mean().to_csv(out / "deletion_curves.csv", index=False)
        written.append(out / "deletion_curves.csv")

    clusters = res / "stability" / "clusters.json"
    if clusters.exists():
        (out / "stable_clusters.json").write_text(clusters.read_text(encoding="utf-8"), encoding="utf-8")
        written.append(out / "stable_clusters.json")
    null = res / "stability" / "null.csv"
    if null.exists():
        pd.read_csv(null).to_csv(out / "stability_null.csv", index=False)
        written.append(out / "stability_null.csv")

    try:
        from sdp.eval.preprocess import feature_frame
        from sdp.explain.runner import experiment_splits, load_explain_settings
        from sdp.config import REPO_ROOT

        es = load_explain_settings(REPO_ROOT / "configs" / "explain.yaml")
        splits, frames = experiment_splits(cfg, es)
        split = next((s for s in splits if s.split_id == DENDROGRAM_SPLIT), None)
        if split is not None:
            X = feature_frame(frames[split.project]).iloc[split.fit_idx]
            rho = X.astype(float).corr(method="spearman")
            rho.to_csv(out / "dendrogram_spearman.csv")
            written.append(out / "dendrogram_spearman.csv")
    except FileNotFoundError as exc:  # processed data absent
        logger.warning("Dendrogram input not collected: %s", exc)
    logger.info("Collected %d paper input file(s) into %s", len(written), out)
    return written

