# xai-sdp-revision

Experimental pipeline for the revision of a research paper on explainable software defect prediction (Journal of Systems and Software).

This repository is built in steps. **Step 1** (this stage) covers project set-up and the data pipeline: acquisition, cleaning and auditing. No models or explainers are included yet.

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv            # or: uv venv --python 3.11 .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                    # makes the `sdp` package importable
```

## Running step 1

```bash
python -m scripts.prepare_data              # download → clean → audit
python -m scripts.prepare_data --offline    # no network: verify raw files against MANIFEST.csv
pytest                                      # unit tests
```

The command is idempotent: raw files that are already present and match Zenodo's md5 checksum are not downloaded again.

| Output | Description |
|---|---|
| `data/raw/<project>-<version>.csv` | Raw CSVs exactly as published on Zenodo (committed) |
| `data/raw/MANIFEST.csv` | Provenance: record ID, DOI, URL, md5, retrieval date (UTC), row count |
| `data/processed/<project>-<version>.csv` | Cleaned releases (not committed; regenerated) |
| `reports/data_audit.{md,csv}` | Per-release summary and flags |
| `reports/release_overlap.csv` | Class overlap and unchanged carry-over between consecutive releases |
| `reports/legacy_comparison.csv` | Comparison of the locally held ANT files with Zenodo (when `data/legacy/ant/` exists) |

Configuration (paths, projects, expected releases, seed = 42, audit thresholds) lives in `configs/default.yaml`.

## Data source

The Jureczko CK-metric defect datasets (Jureczko & Madeyski, 2010), from the SEACRAFT / tera-PROMISE mirror on Zenodo. Each project is a separate record. Note that the creator is listed on Zenodo as "Marian Jureckzo" (sic); the downloader searches both spellings.

| Project | Zenodo record | DOI | Releases on Zenodo |
|---|---|---|---|
| ant | 268440 | [10.5281/zenodo.268440](https://doi.org/10.5281/zenodo.268440) | 1.7 |
| camel | 268442 | [10.5281/zenodo.268442](https://doi.org/10.5281/zenodo.268442) | 1.0, 1.2, 1.4, 1.6 |
| ivy | 322436 | [10.5281/zenodo.322436](https://doi.org/10.5281/zenodo.322436) | 1.1, 1.4, 2.0 |
| jedit | 268458 | [10.5281/zenodo.268458](https://doi.org/10.5281/zenodo.268458) | 3.2, 4.0, 4.1, 4.2, 4.3 |
| log4j | 268451 | [10.5281/zenodo.268451](https://doi.org/10.5281/zenodo.268451) | 1.0, 1.1, 1.2 |
| lucene | 322458 | [10.5281/zenodo.322458](https://doi.org/10.5281/zenodo.322458) | 2.0, 2.2, 2.4 |
| poi | 322443 | [10.5281/zenodo.322443](https://doi.org/10.5281/zenodo.322443) | 1.5, 2.0, 2.5, 3.0 |
| synapse | 322449 | [10.5281/zenodo.322449](https://doi.org/10.5281/zenodo.322449) | 1.0, 1.1, 1.2 |
| velocity | 322455 | [10.5281/zenodo.322455](https://doi.org/10.5281/zenodo.322455) | 1.4, 1.5, 1.6 |
| xalan | 268436 | [10.5281/zenodo.268436](https://doi.org/10.5281/zenodo.268436) | 2.4, 2.5, 2.6, 2.7 |
| xerces | 268474 | [10.5281/zenodo.268474](https://doi.org/10.5281/zenodo.268474) | init, 1.2, 1.3, 1.4 |

**Known gap:** the ANT record contains only release 1.7. ANT 1.3–1.6 are not on Zenodo.

Notes:
- A second, single-release record for camel 1.0 exists (268429); its file is byte-identical to the one in 268442.
- File names are used as canonical release labels. The raw `version` column sometimes differs (e.g. `poi-2.0.csv` holds `2.0RC1`, `jedit-3.2.csv` holds `3.2.1`); the original label is kept in `MANIFEST.csv` (`version_label`).
- Zenodo serves the CSVs with CRLF line endings. `.gitattributes` stores them byte-for-byte so that their md5 checksums still match.

## Cleaning rules (`src/sdp/data/clean.py`)

Applied to each release separately:

1. Column names are lower-cased. The class identifier (`name.1` in the raw files) is renamed to `class_name`. It is kept only to track classes across releases and is **never** used as a feature. `bug` is renamed to `bug_count`.
2. The 20 CK metrics must all be present, otherwise cleaning fails: `wmc, dit, noc, cbo, rfc, lcom, ca, ce, npm, lcom3, loc, dam, moa, mfa, cam, ic, cbm, amc, max_cc, avg_cc`.
3. Missing or non-numeric values are reported and the affected rows dropped (none occur in the current data).
4. `defective = 1 if bug_count > 0 else 0`.
5. Following Sun et al. (2018, *An empirical study of public data quality problems in cross project defect prediction*, arXiv:1805.10787):
   - **Inconsistent cases** (identical on all 20 metrics but with different `defective` labels): all rows of such groups are removed.
   - **Duplicate cases** (identical on all 20 metrics with the same label): one row per group is kept.

   Inconsistent cases are removed before duplicates.
6. Output columns: `class_name`, the 20 metrics, `bug_count`, `defective`.

No scaling, normalisation or feature selection is done here. These happen inside the training pipeline (step 2) to avoid leakage.

## Audit (`src/sdp/data/audit.py`)

`reports/data_audit.md` reports, for each release, the raw rows, invalid rows, inconsistent and duplicate rows removed, final rows, defective count and ratio, and mean/median LOC. For each pair of consecutive releases it reports the classes shared by name and the rows of release n+1 that are unchanged carry-overs (identical metrics **and** label to some row of release n). It flags releases with a defective ratio below 5% or above 75%, projects with fewer than 3 usable releases, and consecutive releases with no class names in common. Nothing is excluded at this stage.

## Layout

```
configs/default.yaml       configuration
src/sdp/config.py          configuration loader
src/sdp/seed.py            global seeding (Python, NumPy)
src/sdp/data/download.py   Zenodo search, download, md5 verification, MANIFEST
src/sdp/data/legacy.py     comparison of local legacy files with Zenodo
src/sdp/data/clean.py      per-release cleaning
src/sdp/data/audit.py      audit tables and report
scripts/prepare_data.py    step 1 entry point
tests/                     pytest suite
```

## References

- Jureczko, M., & Madeyski, L. (2010). Towards identifying software project clusters with regard to defect prediction. *PROMISE '10*.
- Sun, Z., Li, J., Sun, H., & He, L. (2018). An empirical study of public data quality problems in cross project defect prediction. arXiv:1805.10787.
