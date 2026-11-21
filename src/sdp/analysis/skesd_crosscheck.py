"""Cross-check of the in-house Scott-Knott ESD against the R package ScottKnottESD.

Three datasets are ranked by both implementations: the step 3 per-split test AUC of all
models (``main_time_aware``), the step 6 per-split mean ranks behind the fidelity summary
ordering, and the step 7 per-comparison cross-version Spearman ρ of the 14 explanations.
Our implementation log-transforms non-negative data (``log1p``) before ranking; the R
function does not transform, so R receives exactly the data our implementation ranks
(transformed where ours transforms). Groups are compared as partitions and by order.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.eval.stats import scott_knott_esd


def model_input(wide: pd.DataFrame) -> pd.DataFrame:
    """The data our implementation actually ranks (``transform="auto"``)."""
    wide = wide.astype(float)
    return np.log1p(wide) if (wide.to_numpy()[~np.isnan(wide.to_numpy())] >= 0).all() else wide


def ours(wide: pd.DataFrame) -> pd.Series:
    r = scott_knott_esd(wide)
    return r.set_index("treatment")["rank"].astype(int)


def duplicate_columns(wide: pd.DataFrame) -> dict[str, str]:
    """{duplicate: kept twin} for columns with identical values (R's sk_esd fails on
    exact ties between treatments, so only one copy is sent)."""
    seen, dup = {}, {}
    for c in wide.columns:
        key = tuple(np.round(wide[c].to_numpy(float), 12))
        if key in seen:
            dup[c] = seen[key]
        else:
            seen[key] = c
    return dup


def run_r(inputs: dict[str, pd.DataFrame], work: Path, lib: str, script: Path) -> tuple[dict[str, pd.Series], str, dict]:
    in_dir, out_dir = Path(work) / "in", Path(work) / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    dups, safe = {}, {}
    for name, wide in inputs.items():
        data = model_input(wide).dropna()
        dups[name] = duplicate_columns(data)
        data = data.drop(columns=list(dups[name]))
        safe[name] = {f"t{i}": c for i, c in enumerate(data.columns)}  # syntactic names for R's data.frame()
        data.set_axis(list(safe[name]), axis=1).to_csv(in_dir / f"{name}.csv", index=False)
    proc = subprocess.run(["Rscript", str(script), str(in_dir), str(out_dir), lib], capture_output=True, text=True, check=True)
    out = {}
    for name in inputs:
        g = pd.read_csv(out_dir / f"{name}.csv").set_index("treatment")["group"].astype(int)
        g.index = [safe[name][t] for t in g.index]
        for d, twin in dups[name].items():
            g[d] = g[twin]
        out[name] = g
    return out, proc.stdout.strip(), dups


def compare(a: pd.Series, b: pd.Series) -> dict:
    """Identical partitions and identical group numbers, plus the treatments that differ."""
    a, b = a.sort_index(), b.reindex(a.sort_index().index)
    part_a = sorted(tuple(sorted(a[a == g].index)) for g in a.unique())
    part_b = sorted(tuple(sorted(b[b == g].index)) for g in b.unique())
    diff = [t for t in a.index if a[t] != b[t]]
    return {"treatments": len(a), "groups_ours": a.nunique(), "groups_r": b.nunique(),
            "same_partition": part_a == part_b, "same_group_numbers": not diff,
            "differences": "; ".join(f"{t}: ours {a[t]}, R {b[t]}" for t in diff) or "none"}  # fmt: skip
