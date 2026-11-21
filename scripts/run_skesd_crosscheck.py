"""Cross-check our Scott-Knott ESD against the R package ScottKnottESD (step 8).

Usage::

    python -m scripts.run_skesd_crosscheck [--r-lib ~/Library/R/sdp-user-lib]

Skips (and says so) when ``Rscript`` or the package is not available; R is never
installed by this script. Writes ``reports/step8_skesd_crosscheck.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

from sdp.analysis import skesd_crosscheck as X
from sdp.analysis import step6
from sdp.analysis.step4 import _write
from sdp.config import CONFIG_DIR, DEFAULT_CONFIG, REPO_ROOT, load_config
from sdp.data.audit import to_markdown
from sdp.log import setup_logging


def inputs(cfg) -> dict[str, pd.DataFrame]:
    m = pd.read_csv(cfg.paths.results_dir / "main_time_aware" / "metrics.csv")
    auc = m[(m["metric"] == "roc_auc") & (m["threshold_mode"] == "none")]
    auc = auc.groupby(["split_id", "model"])["value"].mean().unstack()
    fs = yaml.safe_load((CONFIG_DIR / "fidelity.yaml").read_text(encoding="utf-8"))
    root = Path(cfg.paths.artifacts_dir) / fs.get("output_dir", "fidelity")
    info = json.loads((cfg.paths.results_dir / "fidelity" / "run_info.json").read_text())
    from sdp.fidelity.evaluation import load_instances

    inst = step6._with_label(load_instances(root / "curves"))
    roar_rows = pd.concat([pd.read_parquet(f) for f in sorted((root / "roar").glob("*.parquet"))], ignore_index=True)
    from sdp.fidelity import roar as R

    roar_rows["family"] = roar_rows["ranking"].map(R.ranking_family)
    areas = R.area_scores(roar_rows, tuple(info["settings"]["roar"]["ks"]))
    roar_split = areas.groupby(["split_id", "project", "model", "family", "test"], as_index=False)["area"].mean()
    _, ranks = step6.summary_ranks(inst, roar_split)
    fid = (-ranks.pivot(index="split_id", columns="explanation", values="mean_rank"))
    cv = pd.read_parquet(cfg.paths.results_dir / "stability" / "cross_version.parquet")
    cv = cv[(cv["metric"] == "spearman") & (cv["level"] == "feature")].pivot(index="comparison", columns="explanation", values="value")
    return {"step3_auc": auc, "step6_fidelity_rank": fid, "step7_cross_version_spearman": cv}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--r-lib", default=os.path.expanduser("~/Library/R/sdp-user-lib"))
    parser.add_argument("--r-lib-v1", default=os.path.expanduser("~/Library/R/sdp-user-lib-skv1"))
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    if shutil.which("Rscript") is None:
        _write(cfg, "step8_skesd_crosscheck", ["# Step 8: Scott-Knott ESD cross-check", "", "Skipped: `Rscript` is not on PATH."], {})
        return 0
    data = inputs(cfg)
    versions = {"current CRAN release": args.r_lib, "original (2017 procedure)": f"{args.r_lib_v1}:{args.r_lib}"}
    rows, used = [], []
    for label, libs in versions.items():
        if not Path(libs.split(":")[0]).exists():
            continue
        work = cfg.paths.results_dir / "skesd_crosscheck" / label.split()[0]
        r_groups, version, dups = X.run_r(data, work, libs, REPO_ROOT / "scripts" / "skesd_crosscheck.R")
        used.append(f"{version} ({label})")
        for name, wide in data.items():
            rows.append({"R package": version, "dataset": name, "blocks": len(wide.dropna())}
                        | X.compare(X.ours(wide.dropna()), r_groups[name])
                        | {"identical treatments sent once": ", ".join(f"{d} = {t}" for d, t in dups[name].items()) or "none"})  # fmt: skip
    t = pd.DataFrame(rows)
    md = [
        "# Step 8: Scott-Knott ESD cross-check against R",
        "",
        f"R packages: {'; '.join(used)}, each `sk_esd(x, alpha = 0.05)`, installed into user libraries; no system-wide "
        "changes. Our implementation (`sdp.eval.stats.scott_knott_esd`) log-transforms non-negative data before "
        "ranking; R receives exactly the data our implementation ranks (R does not transform). Datasets: step 3 "
        "per-split test AUC (8 models, 24 splits); step 6 per-split negative mean rank behind the fidelity summary "
        "(14 explanations); step 7 per-comparison cross-version Spearman ρ (14 explanations, 14 comparisons). "
        "Treatments with identical values are sent once (R's partition fails on exact ties) and given their twin's "
        "group.",
        "",
        "The two R versions implement different procedures. Version 1.x (Tantithamthavorn et al., 2017) runs the "
        "Scott-Knott test (`ScottKnott::SK`, likelihood-ratio statistic, alpha) and then merges adjacent treatments "
        "whose Cohen's d is negligible; our implementation follows this procedure but merges adjacent *groups* "
        "(pooled observations). Version 2.x partitions until the extreme treatments of a group differ by a "
        "negligible Cohen's d and does not use the significance test.",
        "",
        to_markdown(t),
        "",
    ]
    _write(cfg, "step8_skesd_crosscheck", md, {"": t})
    return 0


if __name__ == "__main__":
    sys.exit(main())
