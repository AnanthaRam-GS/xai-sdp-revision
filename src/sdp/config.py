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
class AuditSettings:
    min_defective_ratio: float = 0.05
    max_defective_ratio: float = 0.75
    min_releases: int = 3


@dataclass(frozen=True)
class Config:
    seed: int
    paths: Paths
    zenodo: ZenodoSettings
    projects: dict[str, list[str]]
    audit: AuditSettings = field(default_factory=AuditSettings)
    github_mirror: MirrorSettings | None = None


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
        audit=AuditSettings(**raw.get("audit", {})),
        github_mirror=mirror,
    )
