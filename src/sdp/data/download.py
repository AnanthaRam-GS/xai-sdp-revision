"""Acquisition of the Jureczko CK-metric datasets from Zenodo.

Each project is a separate Zenodo record (SEACRAFT / tera-PROMISE mirror). Records are
located through the Zenodo REST search API, every CSV is verified against the md5
checksum published by Zenodo, and provenance is written to ``data/raw/MANIFEST.csv``.

Raw files are stored byte-for-byte as published, so their md5 matches Zenodo's. If a
single CSV holds several releases, it is split by its ``version`` column instead.

Nothing is ever fabricated: if Zenodo cannot be reached or a record cannot be found,
the affected projects are reported and the pipeline stops.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sdp.config import Config, version_sort_key

logger = logging.getLogger(__name__)

MANIFEST_NAME = "MANIFEST.csv"
MANIFEST_COLUMNS = [
    "project",
    "version",
    "file_name",
    "zenodo_record_id",
    "doi",
    "download_url",
    "md5",
    "retrieval_date_utc",
    "row_count",
    "source_file",
    "version_label",
]
USER_AGENT = "xai-sdp-revision-data-pipeline/0.1"


class DownloadError(RuntimeError):
    """Raised when one or more projects could not be acquired from Zenodo."""

    def __init__(self, failures: dict[str, str]):
        self.failures = failures
        lines = "\n".join(f"  - {p}: {reason}" for p, reason in failures.items())
        super().__init__(
            f"{len(failures)} project(s) could not be acquired from Zenodo:\n{lines}\n"
            "No data were invented. Please download these manually."
        )


@dataclass(frozen=True)
class ZenodoFile:
    key: str
    md5: str
    url: str
    size: int


@dataclass(frozen=True)
class ZenodoRecord:
    record_id: int
    doi: str
    title: str
    files: tuple[ZenodoFile, ...]


@dataclass
class DownloadResult:
    manifest: pd.DataFrame
    missing_releases: dict[str, list[str]] = field(default_factory=dict)
    unexpected_releases: dict[str, list[str]] = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers


def md5_of(path_or_bytes: Path | bytes) -> str:
    """Return the hexadecimal md5 digest of a file or a byte string."""
    h = hashlib.md5()
    if isinstance(path_or_bytes, bytes):
        h.update(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
    return h.hexdigest()


def csv_pattern(project: str) -> re.Pattern[str]:
    """File-name pattern for a project's release CSVs, e.g. ``camel-1.4.csv``."""
    return re.compile(rf"^{re.escape(project)}-(?P<version>[^/]+)\.csv$", re.IGNORECASE)


def make_session(max_retries: int) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = USER_AGENT
    return session


def _parse_record(hit: dict) -> ZenodoRecord:
    files = tuple(
        ZenodoFile(
            key=f["key"],
            md5=f["checksum"].removeprefix("md5:"),
            url=f["links"]["self"],
            size=int(f.get("size", 0)),
        )
        for f in hit.get("files", [])
    )
    return ZenodoRecord(
        record_id=int(hit["id"]),
        doi=hit.get("doi") or hit["metadata"].get("doi", ""),
        title=hit["metadata"].get("title", ""),
        files=files,
    )


def find_record(session: requests.Session, cfg: Config, project: str) -> ZenodoRecord:
    """Locate the Jureczko Zenodo record for ``project`` via the search API.

    Among the hits authored by Jureczko, the record with the most ``<project>-*.csv``
    files is chosen (e.g. the full ``camel`` record over the single-release
    ``camel 1.0`` record). Other candidate records are logged.
    """
    creators = " OR ".join(cfg.zenodo.creator_names)
    query = f'title:"{project}" AND creators.name:({creators})'
    resp = session.get(
        f"{cfg.zenodo.api_url}/records",
        params={"q": query, "size": 25},
        timeout=cfg.zenodo.timeout_seconds,
    )
    resp.raise_for_status()
    hits = resp.json()["hits"]["hits"]

    pattern = csv_pattern(project)
    candidates = []
    for hit in hits:
        rec = _parse_record(hit)
        n = sum(bool(pattern.match(f.key)) for f in rec.files)
        if n:
            candidates.append((n, -rec.record_id, rec))
    if not candidates:
        raise LookupError(f"no Jureczko record with '{project}-*.csv' files (query: {query})")

    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    chosen = candidates[0][2]
    for _, _, other in candidates[1:]:
        chosen_md5 = {f.md5 for f in chosen.files}
        redundant = all(f.md5 in chosen_md5 for f in other.files if pattern.match(f.key))
        logger.info(
            "%s: also found record %d (%r); %s.",
            project,
            other.record_id,
            other.title,
            "all its CSVs are identical to the chosen record's"
            if redundant
            else "it contains CSVs NOT in the chosen record (ignored)",
        )
    logger.info(
        "%s: using Zenodo record %d (DOI %s, title %r).",
        project,
        chosen.record_id,
        chosen.doi,
        chosen.title,
    )
    return chosen


def read_raw_csv(source: Path | bytes) -> pd.DataFrame:
    """Read a raw Jureczko CSV. Duplicate ``name`` headers become ``name`` / ``name.1``."""
    buf = io.BytesIO(source) if isinstance(source, bytes) else source
    return pd.read_csv(buf, dtype={"version": str})


def split_by_version(content: bytes, file_version: str) -> dict[str, tuple[bytes, str]]:
    """Split a CSV by its ``version`` column when it holds several releases.

    Returns ``{version: (bytes, version_label)}``. A single-release file is returned
    unchanged (so its md5 still matches Zenodo) under ``file_version``.
    """
    df = read_raw_csv(content)
    labels = sorted(df["version"].astype(str).unique(), key=version_sort_key)
    if len(labels) <= 1:
        return {file_version: (content, labels[0] if labels else "")}
    logger.info("File holds %d releases (%s); splitting by 'version'.", len(labels), labels)
    out = {}
    for label in labels:
        part = df[df["version"].astype(str) == label]
        out[label] = (part.to_csv(index=False).encode("utf-8"), label)
    return out


def _count_rows(content: bytes) -> int:
    return len(read_raw_csv(content))


def _download(session: requests.Session, f: ZenodoFile, timeout: float) -> bytes:
    resp = session.get(f.url, timeout=timeout)
    resp.raise_for_status()
    content = resp.content
    digest = md5_of(content)
    if digest != f.md5:
        raise ValueError(f"md5 mismatch for {f.key}: expected {f.md5}, got {digest}")
    return content


def load_manifest(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / MANIFEST_NAME
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return pd.read_csv(path, dtype=str)


# --------------------------------------------------------------------------- main API


def acquire_project(
    session: requests.Session,
    cfg: Config,
    project: str,
    previous: pd.DataFrame,
) -> list[dict]:
    """Download (or re-verify) every release CSV of one project; return manifest rows."""
    record = find_record(session, cfg, project)
    pattern = csv_pattern(project)
    raw_dir = cfg.paths.raw_dir
    rows: list[dict] = []

    for f in sorted(record.files, key=lambda f: f.key):
        m = pattern.match(f.key)
        if not m:
            continue
        file_version = m.group("version")
        target = raw_dir / f"{project}-{file_version}.csv"
        prev = previous[(previous["project"] == project) & (previous["source_file"] == f.key)]

        # Idempotency: a present file whose md5 matches Zenodo's is not downloaded again.
        if target.exists() and md5_of(target) == f.md5:
            content = target.read_bytes()
            retrieved = (
                prev["retrieval_date_utc"].iloc[0]
                if len(prev)
                else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            logger.info("%s: %s already present and verified (md5 %s); skipping.", project, f.key, f.md5)
        else:
            logger.info("%s: downloading %s (%d bytes).", project, f.key, f.size)
            content = _download(session, f, cfg.zenodo.timeout_seconds)
            retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            logger.info("%s: %s verified against Zenodo md5 %s.", project, f.key, f.md5)

        for version, (part, label) in split_by_version(content, file_version).items():
            out = raw_dir / f"{project}-{version}.csv"
            if not (out.exists() and out.read_bytes() == part):
                out.write_bytes(part)
            rows.append(
                {
                    "project": project,
                    "version": version,
                    "file_name": out.name,
                    "zenodo_record_id": record.record_id,
                    "doi": record.doi,
                    "download_url": f.url,
                    "md5": f.md5,
                    "retrieval_date_utc": retrieved,
                    "row_count": _count_rows(part),
                    "source_file": f.key,
                    "version_label": label,
                }
            )
    return rows


def compare_with_expected(
    manifest: pd.DataFrame, expected: dict[str, list[str]]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return releases expected but absent, and releases present but not expected."""
    missing, unexpected = {}, {}
    for project, versions in expected.items():
        found = set(manifest.loc[manifest["project"] == project, "version"].astype(str))
        miss = [v for v in versions if v not in found]
        extra = sorted(found - set(versions), key=version_sort_key)
        if miss:
            missing[project] = miss
        if extra:
            unexpected[project] = extra
    return missing, unexpected


def download_all(cfg: Config, offline: bool = False) -> DownloadResult:
    """Acquire all configured projects from Zenodo and write ``MANIFEST.csv``.

    With ``offline=True`` no network access is attempted; the existing manifest and
    the md5 of every listed raw file are verified instead.
    """
    raw_dir = cfg.paths.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    previous = load_manifest(raw_dir)

    if offline:
        manifest = verify_offline(cfg, previous)
    else:
        session = make_session(cfg.zenodo.max_retries)
        rows: list[dict] = []
        failures: dict[str, str] = {}
        for project in cfg.projects:
            try:
                rows.extend(acquire_project(session, cfg, project, previous))
            except (requests.RequestException, LookupError, ValueError) as exc:
                logger.error("%s: acquisition failed: %s", project, exc)
                failures[project] = str(exc)
        if failures:
            raise DownloadError(failures)

        manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
        manifest["_order"] = manifest["version"].map(version_sort_key)
        manifest["_p"] = manifest["project"].map(list(cfg.projects).index)
        manifest = manifest.sort_values(["_p", "_order"]).drop(columns=["_p", "_order"])
        manifest.to_csv(raw_dir / MANIFEST_NAME, index=False)
        logger.info("Wrote %s (%d releases).", raw_dir / MANIFEST_NAME, len(manifest))

    missing, unexpected = compare_with_expected(manifest, cfg.projects)
    for project, versions in missing.items():
        logger.warning("%s: expected release(s) NOT on Zenodo: %s", project, ", ".join(versions))
    for project, versions in unexpected.items():
        logger.warning("%s: unexpected release(s) on Zenodo: %s", project, ", ".join(versions))
    return DownloadResult(manifest, missing, unexpected)


def verify_offline(cfg: Config, manifest: pd.DataFrame) -> pd.DataFrame:
    """Check that every file in an existing manifest is present and unchanged."""
    if manifest.empty:
        raise DownloadError({"*": "offline mode requested but no MANIFEST.csv exists"})
    failures = {}
    for row in manifest.itertuples():
        path = cfg.paths.raw_dir / row.file_name
        if not path.exists():
            failures[f"{row.project}-{row.version}"] = "file missing"
        elif row.file_name == row.source_file and md5_of(path) != row.md5:
            failures[f"{row.project}-{row.version}"] = "md5 differs from MANIFEST"
    if failures:
        raise DownloadError(failures)
    logger.info("Offline mode: %d raw files verified against MANIFEST.csv.", len(manifest))
    return manifest
