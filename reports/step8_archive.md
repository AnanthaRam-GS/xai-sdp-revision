# Step 8: replication archive

Built with `python -m scripts.build_archive` (`make archive`) from commit `515856afb8f739c7ded0850a51df4caf6b2d9de3` with a clean working tree.

| item | value |
|---|---|
| file | `dist/replication_package_v1.0.0.zip` (git-ignored; not uploaded) |
| files | 516 (all committed files, the result tables under `results/`, and `ARCHIVE_INFO.txt`) |
| size | 26,563,035 bytes (25.3 MiB) |
| SHA-256 | `41126404a1888dbf3dc17fdc8162bcd9f4fcc8a35ad2a437a140b805333f7ded` |

Excluded: the attribution store `artifacts/attributions/` (about 300 MB), the persisted models and the other files under `artifacts/`, quick-mode outputs and the virtual environment. Without `artifacts/`, `make paper-assets` regenerates the figures, tables and `paper/numbers.tex` from `reports/` and `paper/data/`. Rerunning the experiments recreates the artefacts. Entries carry a fixed timestamp (2026-01-01), so rebuilding from the same commit and results gives the same contents. This record was committed after the archive was built, so it is not inside the archive.
