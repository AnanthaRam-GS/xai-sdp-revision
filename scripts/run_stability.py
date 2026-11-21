"""Compute the step 7 stability and drift analyses (no retraining).

Usage::

    python -m scripts.run_stability [--projects ant] [--n-boot 1000]

Reads the step 4 (v2) store, the permuted-label explanations (``python -m
scripts.refit_permuted``) and the step 5 stable clusters, and writes all results to
``results/stability/`` (``--projects`` restricts to a smoke test and writes to
``results/stability/_smoke/``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from sdp.data.clean import ID_COL, LABEL_COL
from sdp.eval.preprocess import feature_frame
from sdp.explain.runner import experiment_splits, load_explain_settings
from sdp.log import setup_logging
from sdp.robustness import collinearity as C
from sdp.robustness.common import load_robustness
from sdp.seed import set_seed
from sdp.stability import drift as D
from sdp.stability import projects as Pj
from sdp.stability import seeds as Sd
from sdp.stability import unchanged as U
from sdp.stability import versions as V
from sdp.stability.data import ALL, PRIMARY, ImportanceStore
from sdp.stability.permuted import permuted_root
from sdp.stability.reference import null_summary, random_null

logger = logging.getLogger("run_stability")


def names(df: pd.DataFrame, project: str) -> list[str]:
    return [U.canonical_name(n, project) for n in df[ID_COL].astype(str)]


def grouped(store: ImportanceStore, clusters) -> ImportanceStore:
    return ImportanceStore(C.group_explanations(store.ex, clusters))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--projects", nargs="*", default=None)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    es = load_explain_settings(REPO_ROOT / "configs" / "explain.yaml")
    rs = load_robustness()
    all_splits, frames = experiment_splits(cfg, es)
    splits = [s for s in all_splits if not args.projects or s.project in args.projects]
    out = cfg.paths.results_dir / "stability" / ("_smoke" if args.projects else "")
    out.mkdir(parents=True, exist_ok=True)
    timings, t_all = {}, time.perf_counter()

    def tick(name, t0):
        timings[name] = time.perf_counter() - t0
        logger.info("%s done in %.1f s.", name, timings[name])

    t0 = time.perf_counter()
    store = ImportanceStore.load(es.root(cfg))
    perm_path = permuted_root(es.root(cfg))
    perm = ImportanceStore.load(perm_path) if perm_path.exists() else None
    if perm is None:
        logger.warning("No permuted-label explanations at %s: the data-only reference is skipped.", perm_path)
    cs = rs["collinearity"]
    diags = C.release_diagnostics(all_splits, frames, cs["threshold"], cs["vif_threshold"], cs.get("linkage", "average"))
    clusters = C.stable_clusters([d["clusters"] for d in diags], cs["stable_share"])
    gstore = grouped(store, clusters)
    gperm = grouped(perm, clusters) if perm is not None else None
    tick("load", t0)

    t0 = time.perf_counter()
    nulls = pd.concat([null_summary(random_null(20, 10_000, 0)).assign(level="feature"),
                       null_summary(random_null(len(clusters), 10_000, 1)).assign(level="cluster")])  # fmt: skip
    nulls.to_csv(out / "null.csv", index=False)
    pairs = V.consecutive_pairs(splits)
    cv = V.cross_version(store, perm, pairs, ALL, args.n_boot, seed=0).assign(level="feature")
    cvc = V.cross_version(gstore, gperm, pairs, ALL, args.n_boot, seed=50_000).assign(level="cluster")
    pd.concat([cv, cvc]).to_parquet(out / "cross_version.parquet", index=False)
    V.topk_frequency(store, splits, ALL).to_parquet(out / "topk_frequency.parquet", index=False)
    loc = store.ex.features.index("loc")
    pd.DataFrame([{"explanation": f"{V.label(m, k)}", "loc_top1_splits": sum(int(np.argmax(store.g(s.split_id, m, k)) == loc)
                                                                          for s in splits if store.has(s.split_id, m, k))}
                  for m, k in ALL]).to_csv(out / "top1.csv", index=False)  # fmt: skip
    dirs = V.split_directions(store, splits, ALL)
    dirs.to_parquet(out / "directions.parquet", index=False)
    (out / "clusters.json").write_text(json.dumps(clusters), encoding="utf-8")
    tick("cross_version", t0)

    t0 = time.perf_counter()
    counts, comps = [], []
    for a, b in pairs:
        df = frames[a.project]
        m, c = U.match_unchanged(U.split_test_rows(df, a), U.split_test_rows(df, b), a.project)
        counts.append({"project": a.project, "comparison": V.pair_id(a, b), "release_a": a.test_release,
                       "release_b": b.test_release, **c})  # fmt: skip
        comps.append(U.compare_unchanged(store, a, b, m, ALL).assign(comparison=V.pair_id(a, b)))
    pd.DataFrame(counts).to_csv(out / "unchanged_counts.csv", index=False)
    pd.concat([c for c in comps if len(c)], ignore_index=True).to_parquet(out / "unchanged.parquet", index=False)
    tick("unchanged", t0)

    t0 = time.perf_counter()
    Sd.seed_pairs(store, splits, ALL).to_parquet(out / "seed_pairs.parquet", index=False)
    Sd.multiplicity(store, splits).to_csv(out / "multiplicity.csv", index=False)
    tick("seeds", t0)

    t0 = time.perf_counter()
    rows, mats = [], {}
    feats = store.ex.features
    for model, method in ALL:
        vec = Pj.project_vectors(store, splits, model, method)
        pvec = Pj.project_vectors(perm, splits, model, method, seeds=[0]) if perm is not None and perm.has(splits[0].split_id, model, method) else None
        for metric in ("spearman", "rbo"):
            m = Pj.project_matrix(vec, metric)
            mats[(model, method, metric)] = m
            mean, lo, hi = Pj.off_diagonal(m)
            dmean = Pj.off_diagonal(Pj.project_matrix(pvec, metric))[0] if pvec else np.nan
            rows.append({"model": model, "method": method, "metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi,
                         "data_only": dmean, "n_projects": len(vec)})  # fmt: skip
        top = Pj.common_top_features(vec, feats).assign(model=model, method=method)
        top.to_csv(out / f"common_top_{model}_{method}.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "cross_project.csv", index=False)
    pd.concat([m.assign(model=k[0], method=k[1], metric=k[2]).rename_axis("project").reset_index()
               for k, m in mats.items()]).to_csv(out / "cross_project_matrices.csv", index=False)  # fmt: skip
    tick("projects", t0)

    t0 = time.perf_counter()
    drift_rows, imps = [], []
    for a, b in pairs:
        df = frames[a.project]
        A, B = df.iloc[a.test_idx], df.iloc[b.test_idx]
        d = D.drift_pair(feature_frame(A).assign(defective=A[LABEL_COL].to_numpy()),
                         feature_frame(B).assign(defective=B[LABEL_COL].to_numpy()), seed=0,
                         names_a=names(A, a.project), names_b=names(B, a.project))  # fmt: skip
        drift_rows.append({"kind": "consecutive test releases", "project": a.project, "comparison": V.pair_id(a, b),
                           "split_id": "", **{k: v for k, v in d.items() if k not in ("ks", "importance")}})  # fmt: skip
        imps.append(d["importance"].rename(V.pair_id(a, b)))
    for s in splits:
        df = frames[s.project]
        tr, te = df.iloc[np.r_[s.fit_idx, s.val_idx]], df.iloc[s.test_idx]
        d = D.drift_pair(feature_frame(tr).assign(defective=tr[LABEL_COL].to_numpy()),
                         feature_frame(te).assign(defective=te[LABEL_COL].to_numpy()), seed=0,
                         names_a=names(tr, s.project), names_b=names(te, s.project))  # fmt: skip
        drift_rows.append({"kind": "train to test", "project": s.project, "comparison": "", "split_id": s.split_id,
                           **{k: v for k, v in d.items() if k not in ("ks", "importance")}})  # fmt: skip
        imps.append(d["importance"].rename(s.split_id))
    pd.DataFrame(drift_rows).to_csv(out / "drift.csv", index=False)
    pd.concat(imps, axis=1).T.rename_axis("pair").reset_index().to_csv(out / "drift_importance.csv", index=False)
    tick("drift", t0)

    wall = time.perf_counter() - t_all
    info = {"timings_s": timings, "wall_time_s": wall,
            "estimate_full_s": wall * len(all_splits) / len(splits) if args.projects else None, "projects": args.projects or "all",
            "n_splits": len(splits), "n_comparisons": len(pairs), "n_boot": args.n_boot,
            "permuted_reference": perm is not None, "n_clusters": len(clusters),
            "primary": [f"{m}:{k}" for m, k in PRIMARY]}  # fmt: skip
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    logger.info("Step 7 analyses finished in %.0f s -> %s", info["wall_time_s"], out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
