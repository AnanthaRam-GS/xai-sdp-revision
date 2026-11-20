"""Loading and validation of the YAML configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    processed_dir: Path
    legacy_dir: Path
    reports_dir: Path
    results_dir: Path = REPO_ROOT / "results"


@dataclass(frozen=True)
class ZenodoSettings:
    api_url: str = "https://zenodo.org/api"
    creator_names: tuple[str, ...] = ("Jureczko", "Jureckzo")
    timeout_seconds: float = 60.0
    max_retries: int = 3


@dataclass(frozen=True)
class MirrorSettings:
    repository: str
    commit: str
    directory: str = "bug-data"
    api_url: str = "https://api.github.com"
    raw_url: str = "https://raw.githubusercontent.com"
    releases: dict[str, list[str]] = field(default_factory=dict)
    verify_against_zenodo: dict[str, list[str]] = field(default_factory=dict)

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"


@dataclass(frozen=True)
class StudySettings:
    min_defective_ratio: float = 0.05
    max_defective_ratio: float = 0.75
    min_releases: int = 3


@dataclass(frozen=True)
class EvaluationSettings:
    time_aware_variant: str = "consecutive"
    n_splits: int = 5
    n_repeats: int = 10
    validation_size: float = 0.2
    preprocessing: str = "standard"
    recall_at_loc: float = 0.2
    necm_cost_ratio: float = 15.0
    n_resamples: int = 2000
    confidence: float = 0.95

    @classmethod
    def from_dict(cls, d: dict) -> "EvaluationSettings":
        wr, eff, bs = d.get("within_release", {}), d.get("effort", {}), d.get("bootstrap", {})
        flat = {k: v for k, v in d.items() if k not in ("within_release", "effort", "bootstrap")}
        return cls(**flat, **wr, **eff, **bs)


@dataclass(frozen=True)
class TuningSettings:
    inner_folds: int = 3
    n_trials_time_aware: int = 40
    n_trials_within_release: int = 15
    spaces: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "TuningSettings":
        n = d.get("n_trials", {})
        return cls(
            inner_folds=int(d.get("inner_folds", 3)),
            n_trials_time_aware=int(n.get("time_aware", 40)),
            n_trials_within_release=int(n.get("within_release", 15)),
            spaces=dict(d.get("spaces", {})),
        )

    def n_trials(self, protocol: str) -> int:
        return self.n_trials_time_aware if protocol == "time_aware" else self.n_trials_within_release


@dataclass(frozen=True)
class Config:
    seed: int
    paths: Paths
    zenodo: ZenodoSettings
    projects: dict[str, list[str]]
    study: StudySettings = field(default_factory=StudySettings)
    github_mirror: MirrorSettings | None = None
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    tuning: TuningSettings = field(default_factory=TuningSettings)

    def release_order(self, project: str, versions: list[str] | None = None) -> list[str]:
        """Return ``versions`` (default: all configured) in the configured chronological order.

        Releases not listed in the configuration are appended in natural version order
        (with a warning from the caller where relevant).
        """
        configured = self.projects.get(project, [])
        if versions is None:
            return list(configured)
        known = [v for v in configured if v in versions]
        extra = sorted((v for v in versions if v not in configured), key=version_sort_key)
        return known + extra


def version_sort_key(version: str) -> tuple:
    """Order release labels naturally, with ``init`` before any numbered release."""
    if version == "init":
        return (-1,)
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"[.\-]", version))


def _resolve(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def load_config(path: str | Path = DEFAULT_CONFIG, root: Path = REPO_ROOT) -> Config:
    """Read a YAML configuration file and resolve relative paths against ``root``."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    paths = Paths(**{k: _resolve(v, root) for k, v in raw["paths"].items()})
    zen = raw.get("zenodo", {})
    if "creator_names" in zen:
        zen["creator_names"] = tuple(zen["creator_names"])
    projects = {str(p): [str(v) for v in vs] for p, vs in raw["projects"].items()}
    if not projects:
        raise ValueError("The configuration lists no projects.")

    mirror = None
    if raw.get("github_mirror"):
        m = dict(raw["github_mirror"])
        for key in ("releases", "verify_against_zenodo"):
            m[key] = {str(p): [str(v) for v in vs] for p, vs in (m.get(key) or {}).items()}
        mirror = MirrorSettings(**m)

    return Config(
        seed=int(raw["seed"]),
        paths=paths,
        zenodo=ZenodoSettings(**zen),
        projects=projects,
        study=StudySettings(**raw.get("study", {})),
        github_mirror=mirror,
        evaluation=EvaluationSettings.from_dict(raw.get("evaluation", {})),
        tuning=TuningSettings.from_dict(raw.get("tuning", {})),
    )
