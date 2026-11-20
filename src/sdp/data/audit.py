"""Data audit: per-release summary, cross-release overlap and quality flags.

Outputs
-------
``reports/data_audit.csv``      one row per release
``reports/release_overlap.csv`` one row per pair of consecutive releases in a project
``reports/data_audit.md``       both tables, data provenance and flags

Nothing is excluded here; releases and projects are only *flagged*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sdp.config import Config, version_sort_key
from sdp.data.clean import ID_COL, LABEL_COL, METRICS, CleaningStats

logger = logging.getLogger(__name__)

AUDIT_COLUMNS = [
    "project",
    "version",
    "raw_rows",
    "invalid_rows",
    "inconsistent_rows_removed",
    "duplicate_rows_removed",
    "final_rows",
    "defective_count",
    "defective_ratio_pct",
    "mean_loc",
    "median_loc",
]
OVERLAP_COLUMNS = [
    "project",
    "release_n",
    "release_n1",
    "classes_n",
    "classes_n1",
    "shared_classes",
    "shared_pct_of_n",
    "shared_pct_of_n1",
    "rows_n1",
    "unchanged_rows",
    "unchanged_pct_of_n1",
]


@dataclass
class AuditInputs:
    """Everything the audit needs besides the processed CSVs themselves."""

    manifest: pd.DataFrame
    stats: dict[tuple[str, str], CleaningStats]
    missing_releases: dict[str, list[str]] = field(default_factory=dict)
    unexpected_releases: dict[str, list[str]] = field(default_factory=dict)
    legacy: pd.DataFrame | None = None


# ------------------------------------------------------------------------ tables


def release_summary(
    releases: dict[tuple[str, str], pd.DataFrame], stats: dict[tuple[str, str], CleaningStats]
) -> pd.DataFrame:
    rows = []
    for (project, version), df in releases.items():
        s = stats[(project, version)]
        n_def = int(df[LABEL_COL].sum())
        rows.append(
            {
                "project": project,
                "version": version,
                "raw_rows": s.raw_rows,
                "invalid_rows": s.invalid_rows,
                "inconsistent_rows_removed": s.inconsistent_rows,
                "duplicate_rows_removed": s.duplicate_rows,
                "final_rows": len(df),
                "defective_count": n_def,
                "defective_ratio_pct": round(100 * n_def / len(df), 2) if len(df) else float("nan"),
                "mean_loc": round(float(df["loc"].mean()), 1) if len(df) else float("nan"),
                "median_loc": round(float(df["loc"].median()), 1) if len(df) else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def pair_overlap(prev: pd.DataFrame, nxt: pd.DataFrame) -> dict:
    """Class-name overlap and unchanged carry-over between two consecutive releases.

    A row of release n+1 is an *unchanged carry-over* if its 20 metric values and its
    label are identical to those of any row in release n (class name is not required
    to match).
    """
    classes_n, classes_n1 = set(prev[ID_COL]), set(nxt[ID_COL])
    shared = len(classes_n & classes_n1)

    key = [*METRICS, LABEL_COL]
    seen = set(prev[key].itertuples(index=False, name=None))
    unchanged = sum(t in seen for t in nxt[key].itertuples(index=False, name=None))

    def pct(a: int, b: int) -> float:
        return round(100 * a / b, 2) if b else float("nan")

    return {
        "classes_n": len(classes_n),
        "classes_n1": len(classes_n1),
        "shared_classes": shared,
        "shared_pct_of_n": pct(shared, len(classes_n)),
        "shared_pct_of_n1": pct(shared, len(classes_n1)),
        "rows_n1": len(nxt),
        "unchanged_rows": unchanged,
        "unchanged_pct_of_n1": pct(unchanged, len(nxt)),
    }


def release_overlap(releases: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    by_project: dict[str, list[str]] = {}
    for project, version in releases:
        by_project.setdefault(project, []).append(version)

    rows = []
    for project, versions in by_project.items():
        versions = sorted(versions, key=version_sort_key)
        for a, b in zip(versions, versions[1:]):
            rows.append(
                {"project": project, "release_n": a, "release_n1": b}
                | pair_overlap(releases[(project, a)], releases[(project, b)])
            )
    return pd.DataFrame(rows, columns=OVERLAP_COLUMNS)


def flag_releases(summary: pd.DataFrame, lo_pct: float, hi_pct: float) -> pd.DataFrame:
    mask = (summary["defective_ratio_pct"] < lo_pct) | (summary["defective_ratio_pct"] > hi_pct)
    return summary.loc[mask, ["project", "version", "final_rows", "defective_ratio_pct"]]


def flag_projects(summary: pd.DataFrame, projects: list[str], min_releases: int) -> dict[str, int]:
    usable = summary[summary["final_rows"] > 0].groupby("project").size()
    counts = {p: int(usable.get(p, 0)) for p in projects}
    return {p: n for p, n in counts.items() if n < min_releases}


# ---------------------------------------------------------------------- markdown


def to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra dependency)."""
    if df.empty:
        return "_(none)_"

    def fmt(v) -> str:
        if v is None or v is pd.NA:
            return ""
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:.2f}"
        return str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def render_markdown(
    cfg: Config,
    inputs: AuditInputs,
    summary: pd.DataFrame,
    overlap: pd.DataFrame,
) -> str:
    a = cfg.audit
    lo, hi = 100 * a.min_defective_ratio, 100 * a.max_defective_ratio
    flagged = flag_releases(summary, lo, hi)
    thin = flag_projects(summary, list(cfg.projects), a.min_releases)

    m = inputs.manifest
    sources = (
        m.groupby("project", sort=False)
        .agg(
            zenodo_record_id=("zenodo_record_id", "first"),
            doi=("doi", "first"),
            releases=("version", lambda s: ", ".join(sorted(map(str, s), key=version_sort_key))),
        )
        .reset_index()
    )
    sources["doi"] = sources["doi"].map(lambda d: f"[{d}](https://doi.org/{d})")

    totals = summary[["raw_rows", "invalid_rows", "inconsistent_rows_removed", "duplicate_rows_removed",
                      "final_rows", "defective_count"]].sum()  # fmt: skip

    out = [
        "# Data audit",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `python -m scripts.prepare_data`.",
        "",
        "Source: Jureczko CK-metric datasets (SEACRAFT / tera-PROMISE) on Zenodo. "
        "Cleaning was applied to each release separately: invalid rows dropped, then "
        "inconsistent groups removed, then duplicates removed (Sun et al., 2018).",
        "",
        "## Data sources",
        "",
        to_markdown(sources),
        "",
        "## Differences from the expected release list",
        "",
    ]
    if not inputs.missing_releases and not inputs.unexpected_releases:
        out.append("All expected releases were found on Zenodo, and no others.")
    for p, vs in inputs.missing_releases.items():
        out.append(f"- **{p}**: expected but **not available on Zenodo**: {', '.join(vs)}")
    for p, vs in inputs.unexpected_releases.items():
        out.append(f"- **{p}**: on Zenodo but not in the expected list: {', '.join(vs)}")
    out.append("")

    if inputs.legacy is not None and not inputs.legacy.empty:
        cols = [c for c in ["project", "version", "status", "legacy_rows", "zenodo_rows",
                            "byte_identical", "identical_ignoring_line_endings", "differing_rows"]
                if c in inputs.legacy.columns]  # fmt: skip
        out += ["## Legacy files vs Zenodo", "", to_markdown(inputs.legacy[cols]), ""]

    out += [
        "## Per-release summary",
        "",
        to_markdown(summary),
        "",
        f"Totals: {int(totals['raw_rows'])} raw rows; {int(totals['invalid_rows'])} invalid, "
        f"{int(totals['inconsistent_rows_removed'])} inconsistent and "
        f"{int(totals['duplicate_rows_removed'])} duplicate rows removed; "
        f"{int(totals['final_rows'])} rows remain, of which {int(totals['defective_count'])} are defective.",
        "",
        "## Overlap between consecutive releases",
        "",
        "`shared_classes`: classes (by `class_name`) present in both releases. "
        "`unchanged_rows`: rows of release n+1 whose 20 metric values **and** label are "
        "identical to some row of release n (unchanged carry-over).",
        "",
        to_markdown(overlap),
        "",
        "## Flags (nothing excluded yet)",
        "",
        f"### Releases with a defective ratio < {lo:g}% or > {hi:g}%",
        "",
        to_markdown(flagged),
        "",
        f"### Projects with fewer than {a.min_releases} usable releases",
        "",
    ]
    if thin:
        out += [f"- **{p}**: {n} usable release(s)" for p, n in thin.items()]
    else:
        out.append("_(none)_")

    disjoint = overlap.loc[overlap["shared_classes"] == 0, ["project", "release_n", "release_n1"]]
    out += [
        "",
        "### Consecutive releases with no class names in common",
        "",
        "Class tracking by `class_name` is impossible across these pairs (for example "
        "after a package rename).",
        "",
        to_markdown(disjoint),
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------- run


def load_processed(cfg: Config, manifest: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    releases = {}
    for r in manifest.itertuples():
        path = cfg.paths.processed_dir / f"{r.project}-{r.version}.csv"
        releases[(r.project, str(r.version))] = pd.read_csv(path)
    return releases


def run_audit(cfg: Config, inputs: AuditInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute and write the audit tables and the Markdown report."""
    releases = load_processed(cfg, inputs.manifest)
    summary = release_summary(releases, inputs.stats)
    overlap = release_overlap(releases)

    reports: Path = cfg.paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    summary.to_csv(reports / "data_audit.csv", index=False)
    overlap.to_csv(reports / "release_overlap.csv", index=False)
    if inputs.legacy is not None and not inputs.legacy.empty:
        inputs.legacy.to_csv(reports / "legacy_comparison.csv", index=False)
    (reports / "data_audit.md").write_text(render_markdown(cfg, inputs, summary, overlap), encoding="utf-8")
    logger.info("Wrote data audit to %s.", reports)
    return summary, overlap
