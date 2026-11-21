"""Explanation drift of unchanged classes (step 7, section 3).

For consecutive splits (n→n+1) and (n+1→n+2) of a project, an *unchanged class* appears
in both test releases n+1 and n+2 with identical values on all 20 metrics and the same
label. Classes are matched by name; for ivy the package was renamed between 1.4 and 2.0,
so ``fr.jayasoft.ivy.`` is mapped to ``org.apache.ivy.`` before matching. Class names that
occur more than once in a release are left out.

For each unchanged class and explanation, the explanation from the model trained on n
(explaining n+1) is compared with the one from the model trained on n+1 (explaining n+2)
— seed by seed, then averaged over the common seeds — with the step 4 instance metrics
(Spearman of |attribution|, feature agreement@5, sign agreement@5). The predicted
probabilities are compared and a *flip* is a different predicted label at the two models'
validation-tuned thresholds. The seed-noise floor compares the same class's explanations
from seeds 0 and 1 of the later split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sdp.analysis.explanations import label, rank_correlations, topk_agreement
from sdp.data.clean import ID_COL, LABEL_COL
from sdp.eval.preprocess import FEATURES
from sdp.stability.data import ImportanceStore

IVY_PREFIX = ("fr.jayasoft.ivy.", "org.apache.ivy.")


def canonical_name(name: str, project: str) -> str:
    old, new = IVY_PREFIX
    return new + name[len(old):] if project == "ivy" and name.startswith(old) else name


def split_test_rows(df: pd.DataFrame, split) -> pd.DataFrame:
    """The split's test rows in test-index order, with their instance id."""
    t = df.iloc[split.test_idx].reset_index(drop=True)
    return t.assign(instance_id=np.arange(len(t)))


def match_unchanged(a: pd.DataFrame, b: pd.DataFrame, project: str) -> tuple[pd.DataFrame, dict]:
    """Match two test releases by (canonical) class name; flag unchanged classes."""
    def prep(t):
        t = t.assign(raw_name=t[ID_COL].astype(str))
        t["key"] = [canonical_name(n, project) for n in t["raw_name"]]
        return t[~t["key"].duplicated(keep=False)]

    pa, pb = prep(a), prep(b)
    m = pa.merge(pb, on="key", suffixes=("_a", "_b"))
    exact = int((m["raw_name_a"] == m["raw_name_b"]).sum())
    same_metrics = np.all(np.isclose(m[[f"{f}_a" for f in FEATURES]].to_numpy(float),
                                     m[[f"{f}_b" for f in FEATURES]].to_numpy(float), rtol=0, atol=0), axis=1)  # fmt: skip
    m["unchanged"] = same_metrics & (m[f"{LABEL_COL}_a"] == m[f"{LABEL_COL}_b"]).to_numpy()
    m["mapped"] = (m["raw_name_a"] != m["raw_name_b"]).to_numpy()
    counts = {"matched": len(m), "matched_exact_name": exact, "matched_via_mapping": len(m) - exact,
              "unchanged": int(m["unchanged"].sum()), "unchanged_via_mapping": int((m["unchanged"] & m["mapped"]).sum())}
    out = m.loc[m["unchanged"], ["key", "instance_id_a", "instance_id_b", "mapped", f"{LABEL_COL}_a"]]
    return out.rename(columns={f"{LABEL_COL}_a": "label"}).reset_index(drop=True), counts


def _predicted(outcome) -> np.ndarray:
    return np.isin(np.asarray(outcome, dtype=str), ["TP", "FP"])


def compare_unchanged(store: ImportanceStore, a, b, matches: pd.DataFrame, explanations) -> pd.DataFrame:
    rows = []
    if matches.empty:
        return pd.DataFrame()
    ia, ib = matches["instance_id_a"].to_numpy(), matches["instance_id_b"].to_numpy()
    for model, method in explanations:
        seeds = sorted(set(store.seeds(a.split_id, model, method)) & set(store.seeds(b.split_id, model, method)))
        if not seeds:
            continue
        acc = {k: [] for k in ("spearman", "fa5", "sa5", "dprob", "flip")}
        for s in seeds:
            A, B = store.attr(a.split_id, model, method, s)[ia], store.attr(b.split_id, model, method, s)[ib]
            ma, mb = store.meta(a.split_id, model, method, s), store.meta(b.split_id, model, method, s)
            top = topk_agreement(A, B, 5)
            acc["spearman"].append(rank_correlations(A, B)["spearman"])
            acc["fa5"].append(top["feature_agreement@5"])
            acc["sa5"].append(top["sign_agreement@5"])
            acc["dprob"].append(np.abs(mb["y_prob"].to_numpy()[ib] - ma["y_prob"].to_numpy()[ia]))
            acc["flip"].append((_predicted(ma["outcome"].to_numpy()[ia]) != _predicted(mb["outcome"].to_numpy()[ib])).astype(float))
        floor = {}
        sb = store.seeds(b.split_id, model, method)
        if len(sb) >= 2:
            X, Y = store.attr(b.split_id, model, method, sb[0])[ib], store.attr(b.split_id, model, method, sb[1])[ib]
            ft = topk_agreement(X, Y, 5)
            floor = {"seed_spearman": rank_correlations(X, Y)["spearman"], "seed_fa5": ft["feature_agreement@5"],
                     "seed_sa5": ft["sign_agreement@5"]}  # fmt: skip
        mean = {k: np.nanmean(np.stack(v), axis=0) for k, v in acc.items()}
        frame = pd.DataFrame({"project": a.project, "split_a": a.split_id, "split_b": b.split_id, "model": model,
                              "method": method, "explanation": label(model, method), "class_key": matches["key"],
                              "mapped": matches["mapped"], "n_seeds": len(seeds), **mean, **floor})  # fmt: skip
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)

