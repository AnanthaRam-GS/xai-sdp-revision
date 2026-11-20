"""Study inclusion rule and the final study dataset.

Rule (thresholds in ``configs/default.yaml``, section ``study``):

1. Exclude every release whose defective ratio after cleaning is below
   ``min_defective_ratio`` or above ``max_defective_ratio``.
2. Then keep a project only if at least ``min_releases`` releases remain.

Releases are kept in the chronological order given explicitly in the configuration
(``projects``); versions are never string-sorted. Time-aware release pairs are formed
between consecutive *included* releases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sdp.config import Config
from sdp.data.audit import to_markdown
from sdp.data.clean import LABEL_COL

logger = logging.getLogger(__name__)

STUDY_COLUMNS = [
    "project",
    "version",
    "order",
    "final_rows",
    "defective_count",
    "defective_ratio_pct",
    "included",
    "reason",
]


@dataclass(frozen=True)
class ReleasePair:
    """A time-aware train/test pair. ``train`` has one release (consecutive) or several (cumulative)."""

    project: str
    train: tuple[str, ...]
    test: str


@dataclass
class StudyDataset:
    table: pd.DataFrame  # one row per configured release, see STUDY_COLUMNS
    releases: dict[str, list[str]]  # included releases per project, chronological

    @property
    def projects(self) -> list[str]:
        return list(self.releases)

    def pairs(self, variant: str = "consecutive") -> list[ReleasePair]:
        """Time-aware pairs: train on release n (or all releases <= n), test on n+1."""
        if variant not in ("consecutive", "cumulative"):
            raise ValueError(f"unknown time-aware variant: {variant!r}")
        out = []
        for project, versions in self.releases.items():
            for i in range(1, len(versions)):
                train = tuple(versions[:i]) if variant == "cumulative" else (versions[i - 1],)
                out.append(ReleasePair(project, train, versions[i]))
        return out


def apply_inclusion_rule(
    stats: pd.DataFrame, cfg: Config
) -> StudyDataset:
    """Apply the inclusion rule to per-release statistics.

    ``stats`` needs the columns project, version, final_rows, defective_count.
    """
    s = cfg.study
    lo, hi = 100 * s.min_defective_ratio, 100 * s.max_defective_ratio
    rows = []
    for project in cfg.projects:
        present = stats[stats["project"] == project].set_index("version")
        for i, version in enumerate(cfg.release_order(project)):
            if version not in present.index:
                rows.append({"project": project, "version": version, "order": i, "final_rows": 0,
                             "defective_count": 0, "defective_ratio_pct": float("nan"),
                             "included": False, "reason": "release not available"})  # fmt: skip
                continue
            r = present.loc[version]
            n, d = int(r["final_rows"]), int(r["defective_count"])
            ratio = round(100 * d / n, 2) if n else float("nan")
            if not n:
                reason = "no rows after cleaning"
            elif ratio < lo:
                reason = f"defective ratio {ratio:.2f}% < {lo:g}%"
            elif ratio > hi:
                reason = f"defective ratio {ratio:.2f}% > {hi:g}%"
            else:
                reason = ""
            rows.append({"project": project, "version": version, "order": i, "final_rows": n,
                         "defective_count": d, "defective_ratio_pct": ratio,
                         "included": reason == "", "reason": reason})  # fmt: skip
    table = pd.DataFrame(rows, columns=STUDY_COLUMNS)

    releases: dict[str, list[str]] = {}
    for project in cfg.projects:
        mask = (table["project"] == project) & table["included"]
        kept = table.loc[mask, "version"].tolist()
        if len(kept) < s.min_releases:
            table.loc[mask, "included"] = False
            table.loc[mask, "reason"] = (
                f"project has only {len(kept)} release(s) left after the ratio rule (< {s.min_releases})"
            )
            logger.info("Study: project %s excluded (%d release(s) remain).", project, len(kept))
        else:
            releases[project] = kept
    for r in table[~table["included"]].itertuples():
        logger.info("Study: excluded %s %s: %s.", r.project, r.version, r.reason)
    logger.info(
        "Study dataset: %d projects, %d releases.", len(releases), sum(map(len, releases.values()))
    )
    return StudyDataset(table=table, releases=releases)


def release_stats(cfg: Config) -> pd.DataFrame:
    """Final rows and defective counts for every processed release on disk."""
    rows = []
    for project in cfg.projects:
        for version in cfg.release_order(project):
            path = cfg.paths.processed_dir / f"{project}-{version}.csv"
            if path.exists():
                df = pd.read_csv(path, usecols=[LABEL_COL])
                rows.append({"project": project, "version": version, "final_rows": len(df),
                             "defective_count": int(df[LABEL_COL].sum())})  # fmt: skip
    return pd.DataFrame(rows, columns=["project", "version", "final_rows", "defective_count"])


def build_study(cfg: Config) -> StudyDataset:
    """Build the study dataset from the processed releases."""
    return apply_inclusion_rule(release_stats(cfg), cfg)


def load_release(cfg: Config, project: str, version: str) -> pd.DataFrame:
    """Load one processed release."""
    return pd.read_csv(cfg.paths.processed_dir / f"{project}-{version}.csv", dtype={"class_name": str})


# ------------------------------------------------------------------------ report


def pairs_table(study: StudyDataset) -> pd.DataFrame:
    t = study.table.set_index(["project", "version"])
    rows = []
    cumulative = {(p.project, p.test): p.train for p in study.pairs("cumulative")}
    for p in study.pairs("consecutive"):
        cum = cumulative[(p.project, p.test)]
        rows.append({
            "project": p.project,
            "train (consecutive)": p.train[0],
            "train (cumulative)": ", ".join(cum),
            "test": p.test,
            "train_rows": int(t.loc[(p.project, p.train[0]), "final_rows"]),
            "train_rows_cumulative": int(sum(t.loc[(p.project, v), "final_rows"] for v in cum)),
            "test_rows": int(t.loc[(p.project, p.test), "final_rows"]),
            "test_defective_ratio_pct": float(t.loc[(p.project, p.test), "defective_ratio_pct"]),
        })  # fmt: skip
    return pd.DataFrame(rows)


def write_study_report(cfg: Config, study: StudyDataset) -> None:
    reports: Path = cfg.paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    study.table.to_csv(reports / "study_dataset.csv", index=False)
    pairs = pairs_table(study)
    pairs.to_csv(reports / "study_pairs.csv", index=False)

    s = cfg.study
    inc = study.table[study.table["included"]]
    exc = study.table[~study.table["included"]]
    lines = [
        "# Study dataset",
        "",
        "Inclusion rule (configurable in `configs/default.yaml`, section `study`):",
        "",
        f"1. Exclude releases whose defective ratio after cleaning is < {100 * s.min_defective_ratio:g}% "
        f"or > {100 * s.max_defective_ratio:g}%.",
        f"2. Then keep a project only if it has >= {s.min_releases} remaining releases.",
        "",
        "Releases are ordered chronologically as listed explicitly in the configuration.",
        "",
        f"**Included:** {len(study.releases)} projects, {len(inc)} releases, "
        f"{int(inc['final_rows'].sum())} classes ({int(inc['defective_count'].sum())} defective, "
        f"{100 * inc['defective_count'].sum() / inc['final_rows'].sum():.2f}%).",
        "",
        "## Included releases",
        "",
        to_markdown(inc.drop(columns=["included", "reason"])),
        "",
        "## Excluded releases",
        "",
        to_markdown(exc.drop(columns=["included"])),
        "",
        "## Time-aware release pairs",
        "",
        "Pairs are formed between consecutive *included* releases (train = release n, test = "
        "release n+1). The cumulative variant trains on all included releases up to n.",
        "",
        to_markdown(pairs),
        "",
        f"{len(pairs)} pairs in total.",
        "",
    ]
    (reports / "study_dataset.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote study dataset report to %s.", reports)
