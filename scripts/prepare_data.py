"""Step 1 entry point: download → clean → audit.

Usage::

    python -m scripts.prepare_data [--config configs/default.yaml] [--offline]

Idempotent: raw files already present whose md5 matches Zenodo are not downloaded again.
"""

from __future__ import annotations

import argparse
import logging
import sys

from sdp.config import DEFAULT_CONFIG, load_config
from sdp.data.audit import AuditInputs, run_audit
from sdp.data.clean import clean_file
from sdp.data.download import DownloadError, download_all
from sdp.data.legacy import compare_legacy
from sdp.log import setup_logging
from sdp.seed import set_seed

logger = logging.getLogger("prepare_data")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML configuration file")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not contact Zenodo; verify existing raw files against MANIFEST.csv instead",
    )
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    # 1. Download
    try:
        result = download_all(cfg, offline=args.offline)
    except DownloadError as exc:
        logger.error("%s", exc)
        return 1

    # 1b. Legacy comparison (ANT only; skipped if the folder is absent)
    legacy = compare_legacy(cfg.paths.legacy_dir, cfg.paths.raw_dir, "ant")

    # 2. Clean each release separately
    stats = {}
    for r in result.manifest.itertuples():
        key = (r.project, str(r.version))
        stats[key] = clean_file(
            cfg.paths.raw_dir / r.file_name,
            cfg.paths.processed_dir / f"{r.project}-{r.version}.csv",
            label=f"{r.project}-{r.version}",
        )

    # 3. Audit
    run_audit(
        cfg,
        AuditInputs(
            manifest=result.manifest,
            stats=stats,
            missing_releases=result.missing_releases,
            unexpected_releases=result.unexpected_releases,
            legacy=legacy,
        ),
    )
    logger.info("Data preparation finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
