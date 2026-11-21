# xai-sdp-revision

Experimental pipeline for the revision of a research paper on explainable software defect prediction (Journal of Systems and Software).

This repository is built in steps. **Step 1** covers the data pipeline (acquisition, cleaning, auditing); **step 2** finalises the study dataset and adds the evaluation framework (preprocessing, split protocols, metrics, statistics, experiment runner); **step 3** adds the learned models (MLP, LR, RF, XGBoost), hyperparameter tuning, the main prediction experiments and ablations. No explainers are included yet.

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv            # or: uv venv --python 3.11 .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                    # makes the `sdp` package importable
```

**macOS note.** PyTorch, XGBoost (via Homebrew's `libomp`, required by the XGBoost wheel: `brew install libomp`) and scikit-learn each bring an OpenMP runtime, and two of them in one process crash with a segmentation fault. The entry points in `scripts/` and the test suite therefore re-execute themselves once with `DYLD_LIBRARY_PATH` pointing at PyTorch's `libomp`, so a single runtime is used (`src/sdp/runtime.py`). In your own scripts or notebooks, call `sdp.runtime.ensure_single_openmp_runtime()` before importing torch, xgboost or scikit-learn.

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
| ant | 268440 | [10.5281/zenodo.268440](https://doi.org/10.5281/zenodo.268440) | 1.7 (1.3–1.6: see below) |
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

**ANT 1.3–1.6** are not on Zenodo (the ANT record contains only 1.7). They are taken from the GitHub mirror of the PROMISE repository, [hjx-zhans/promise-backup](https://github.com/hjx-zhans/promise-backup), at the pinned commit `1e8fd48cbd5275dd57409f3917de4c8ae9e15add`, directory `bug-data/ant/`. File paths are resolved from the repository tree at that commit, and each file is verified against its git blob SHA-1. `MANIFEST.csv` records `source = github-mirror`, the repository URL, commit SHA, file path, md5 and retrieval date (Zenodo fields empty). To show the mirror is faithful, the pipeline also downloads the mirror's ANT 1.7 and compares it row by row with Zenodo's, and compares mirror ANT 1.3–1.6 with locally held legacy copies. It stops without substituting anything if any comparison differs; all were identical (`reports/mirror_verification.csv`). The mirror files have a different layout (a single `name` column holding the class name, no project/version columns); they are stored byte-for-byte and handled by the cleaner.

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

`reports/data_audit.md` reports, for each release, the raw rows, invalid rows, inconsistent and duplicate rows removed, final rows, defective count and ratio, and mean/median LOC. For each pair of consecutive releases it reports the classes shared by name and the rows of release n+1 that are unchanged carry-overs (identical metrics **and** label to some row of release n). It flags releases with a defective ratio below 5% or above 75%, projects with fewer than 3 usable releases, and consecutive releases with no class names in common. Nothing is excluded by the audit itself; exclusion happens in the study inclusion rule below.

## Step 2: evaluation framework

### Study dataset (`src/sdp/data/study.py`)

Releases whose defective ratio after cleaning is < 5% or > 75% are excluded; then a project is kept only if at least 3 releases remain (thresholds in `configs/default.yaml`, section `study`). Releases follow the chronological order listed explicitly under `projects` in the configuration (never string order). Result: 10 projects, 34 releases, 24 time-aware pairs (log4j excluded); see `reports/study_dataset.md`.

### Protocols (`src/sdp/eval/splits.py`, `src/sdp/eval/preprocess.py`)

- **time_aware**: train on release n, test on release n+1 of the same project (`consecutive`), or train on all included releases <= n (`cumulative`); set `evaluation.time_aware_variant`.
- **within_release**: 10 x 5-fold repeated stratified cross-validation per release (seed 42).
- Inside every training set a stratified, seeded 20% **validation** hold-out is used for threshold tuning (and later early stopping and calibration checks). Models are fitted on the remaining 80% and are not refitted. The test set is never used for fitting or tuning.
- Every split has a stable `split_id` (e.g. `ta-consecutive:ant:1.3>1.4`, `wr:ant:1.3:r01f2`).
- Features are the 20 metrics only (`class_name`, `bug_count`, `defective` never). Preprocessing (`none`, `standard`, `log_standard`) sits inside the model pipeline, so it is fitted on training data only. Original LOC is kept separately as the effort measure.

### Metrics (`src/sdp/eval/metrics.py`)

- Threshold-independent: ROC-AUC, PR-AUC (average precision).
- Threshold-dependent, at 0.5 and at the MCC-maximising validation threshold: precision, recall, F1, MCC, balanced accuracy, G-mean, false alarm rate, NECM15 = (FP + 15 FN) / N (Herbold, 2018).
- Calibration: Brier score, NLL (clipped log loss), ECE (10 equal-width bins).
- Effort-aware (LOC as effort, ranking by predicted risk density = probability / LOC): Popt (Mende & Koschke, 2010), Recall@20%LOC, IFA.
- Degenerate cases (single class, no positive predictions, zero effort) return NaN with a warning.

### Statistics (`src/sdp/eval/stats.py`)

Bootstrap 95% CIs of means (2,000 resamples, seeded); Wilcoxon signed-rank tests paired by split with Holm correction; Cliff's delta with Romano et al. magnitudes; Scott-Knott ESD (implemented here, as no maintained Python package exists; log1p transform for non-negative data, merging of adjacent groups with negligible Cohen's d).

### Running the baselines

```bash
python -m scripts.prepare_data        # data, audit and study dataset
python -m scripts.run_baselines       # all_defective, loc_ranking, majority
```

Per-experiment outputs go to `results/<experiment>/metrics.csv` (long format) and `predictions.csv` (per test instance; git-ignored). The summary with means and 95% CIs per project and baseline is `reports/baselines_summary.md`.

## Step 3: models, tuning and experiments

### Models (`src/sdp/models/`)

- **MLP** (`mlp.py`): PyTorch network built only from `nn.Linear`, `nn.ReLU(inplace=False)` and `nn.Dropout`, applied in order (as required by Captum DeepLift/DeepLiftShap); one logit output, plus `ProbabilityModel` (network + `nn.Sigmoid()`) for the explainers. `MLPClassifierTorch` is a scikit-learn style wrapper: mini-batch Adam with weight decay, `BCEWithLogitsLoss` (with `pos_weight = n_neg / n_pos` for `class_weight`), early stopping on validation loss (patience 20, max 500 epochs, best weights restored), `save`/`load`. CPU only and deterministic (`torch.use_deterministic_algorithms(True)`, one thread per worker).
- **LR** (L2, lbfgs), **RF**, **XGBoost** (`tree_method="hist"`, early stopping on the validation split) in `classic.py`.
- **Imbalance** (`imbalance.py`): `none`, `class_weight` (LR/RF `class_weight="balanced"`, XGBoost `scale_pos_weight`, MLP `pos_weight`) or `smote` (SMOTE on the 80% fitting part only; never on validation or test data). Default: `class_weight`.
- **Baselines** (`baselines.py`): `manual_up` (risk 1 / max(LOC, 1)), `manual_down` (risk LOC), both min-max normalised on training data; `all_defective`; `majority`. Baselines always see raw features.

### Tuning (`src/sdp/tuning.py`)

Optuna TPE (seeded with the model seed) maximises the mean ROC-AUC of a stratified inner 3-fold CV on the 80% fitting part of the training data only; validation and test data are never used. Search spaces and budgets are in `configs/default.yaml` (`tuning`): 40 trials per time-aware split; for within-release, 15 trials once per (release, repeat) on the first fold, reused for the other four folds. After tuning, the model is refitted on the fitting part with early stopping on the validation part. Chosen hyperparameters: `results/<experiment>/hyperparameters.csv`.

### Experiments (`configs/experiments/`, `src/sdp/experiments/engine.py`)

| Experiment | Protocol | Models | Settings | Seeds |
|---|---|---|---|---|
| `main_time_aware` | time-aware, consecutive | mlp, lr, rf, xgb + 4 baselines | MLP/LR `log_standard`, RF/XGB `standard`; `class_weight` | 0–9 |
| `main_within_release` | within-release 10 x 5-fold | same | same | 0 |
| `ablation_preprocessing` | time-aware | mlp, lr | `none` / `standard` / `log_standard` | 0–9 |
| `ablation_imbalance` | time-aware | mlp, rf | `none` / `class_weight` / `smote` (`log_standard`) | 0–9 |
| `main_time_aware_cumulative` | time-aware, cumulative | mlp, lr, rf, xgb | as main | 0–4 |

Seeds control model initialisation, tuning and SMOTE; the splits stay fixed. Jobs run in parallel with joblib (`n_jobs` in each config); every worker sets its own seeds and uses one thread. Each job's result is cached in `results/_cache/` under a hash of everything that determines it, so interrupted runs resume and units shared by experiments are computed once.

For `main_time_aware` and `ablation_preprocessing`, every fitted learned model and its fitted preprocessor are saved to `artifacts/models/<experiment>/<split>/<model>/<preprocessing>/seed_<k>/` with `meta.json` (hyperparameters, validation-tuned threshold, training time, feature order); load them with `sdp.experiments.persistence.load_artifact`. `results/` and `artifacts/` are git-ignored.

```bash
# smoke test (one project, 2 seeds, 5 trials) and runtime estimate only
python -m scripts.run_experiment configs/experiments/*.yaml --smoke --estimate-only
# full runs (resumable)
python -m scripts.run_experiment configs/experiments/main_time_aware.yaml \
    configs/experiments/ablation_preprocessing.yaml configs/experiments/ablation_imbalance.yaml \
    configs/experiments/main_time_aware_cumulative.yaml configs/experiments/main_within_release.yaml
python -m scripts.make_step3_reports
```

### Reports (`src/sdp/analysis/`)

`reports/step3_main_time_aware.md`, `step3_within_release.md`, `step3_ablation_preprocessing.md`, `step3_ablation_imbalance.md`, `step3_cumulative.md` and `step3_runtime.md` (each with CSVs). Summaries give the mean and bootstrap 95% CI over splits x seeds; Scott-Knott ESD ranks and Wilcoxon signed-rank tests (Holm) with Cliff's delta use per-split means across seeds, so each split counts once.

## Step 4: explanations and explainer agreement

Explains every test instance of the 24 `main_time_aware` splits with the persisted step 3 models (nothing is retrained): MLP and LR (`log_standard`), RF and XGBoost (`standard`), all `class_weight`. Settings: `configs/explain.yaml`.

### Conventions (`src/sdp/explain/base.py`)

- Explained output: P(defective) for every model and method (the MLP through `ProbabilityModel`), except LR-LinearSHAP, which is exact on the log-odds scale and stored with `output_scale="log_odds"`.
- Attributions are computed in each model's input space (after its fitted, element-wise preprocessor) and stored by feature name.
- Background set per (split, seed): 100 rows of the fitting part of the training data, stratified by label with the model seed (all rows if fewer), saved under `artifacts/attributions/_backgrounds/`. DeepLift and Integrated Gradients use its mean as the single baseline.
- Store: long-format Parquet, `artifacts/attributions/<model>/<method>/<split>__seed_<k>.parquet` (split, project, releases, model, method, seed, instance, class name, TP/FP/FN/TN outcome at the validation-MCC threshold, y_true, y_prob, LOC, feature, raw value, attribution, output scale). Read it with `sdp.explain.base.read_attributions` or `sdp.analysis.explanations.load_explanations`.

### Methods

| Model | Methods | Seeds |
|---|---|---|
| MLP | DeepLift, DeepLiftShap, GradientShap (50 samples, stdevs 0), IntegratedGradients (512 Gauss-Legendre steps; 64 in v1), Saliency (signed gradient), InputXGradient (`captum_methods.py`) | 0–9 |
| MLP | LIME (5000 samples, 20 features, no discretisation, sampling around the instance), KernelSHAP (1000 samples, no lasso selection) | 0–2 |
| RF, XGBoost | interventional TreeSHAP on the probability, background set as data | 0–9 |
| LR | LinearSHAP (log-odds) | 0–9 |
| LR, RF, XGBoost | LIME | 0–2 |

LIME weights are mapped to features through `explanation.as_map()[1]` (index → weight). Completeness (DeepLift, DeepLiftShap, IntegratedGradients; Captum on a float64 copy of the MLP) and additivity (TreeSHAP, LinearSHAP) are checked per instance and saved under `artifacts/attributions/_checks/`. For random forests, shap's interventional code stores thresholds as float32, so the forest given to shap has its thresholds rounded down to float32 (identical predictions on float32 inputs); otherwise additivity fails by up to 0.02. Every loaded model is checked against its stored step 3 test predictions.

```bash
# smoke test (one split, seed 0, all methods) and runtime estimate
python -m scripts.run_explanations --smoke --estimate-only
# full run (resumable; with --smoke it reduces LIME/KernelSHAP samples if the estimate exceeds 8 h)
python -m scripts.run_explanations
python -m scripts.make_step4_reports
```

### Analysis (`src/sdp/analysis/explanations.py`, `step4.py`, `figures.py`)

Absolute attributions are L1-normalised per instance, averaged within a split (seeds averaged), then over splits (each split weighted equally). Reports in `reports/`:

- `step4_global_importance.md`: rankings for all instances, outcome groups and defective classes, per project and ALL;
- `step4_direction.md`: Spearman(raw value, signed attribution) and disagreements in direction;
- `step4_size_share.md`: share of attribution on `loc` and on {loc, wmc, rfc, npm}, and loc as top-1 feature;
- `step4_agreement.md`: feature, rank, sign and signed-rank agreement@k (k = 3, 5), Spearman and Kendall (Krishna et al., 2022), within the MLP, across models (SHAP family) and LIME across models, with bootstrap CIs over instances;
- `step4_sanity_checks.md`, `step4_runtime.md`;
- draft figures (300 dpi) in `reports/figures/step4/`.

## Step 5: robustness and sensitivity of the explanations

**Fixes to step 4 (v2).** Integrated Gradients now uses 512 Gauss-Legendre steps (95th percentile completeness error 3.5e-4; 2.8e-3 with 64 steps), and LIME samples its perturbations around the explained instance (`sample_around_instance=True`). The v1 LIME results are kept in the store as `LIME_mean_sampling` (`python -m scripts.run_explanations --keep-method-as LIME LIME_mean_sampling`). The v1 step 4 reports are archived in `reports/archive/step4_v1/`, and each regenerated step 4 report opens with a "Changes since v1" section.

Studies (`src/sdp/robustness/`, settings in `configs/robustness.yaml`); each reuses the step 4 explainers, background sets, store and analysis functions:

| Study | Module | Scope |
|---|---|---|
| LIME stability and settings | `lime_study.py` | seed-0 models; up to 10 instances per outcome group per split; 10 runs (random_state 0–9); one-factor grid over `num_samples`, kernel width and sampling scheme; surrogate R² |
| Background sensitivity | `background_study.py` | seed-0 MLP (DeepLiftShap, GradientShap), RF and XGBoost (TreeSHAP); all test instances; sizes 10–all, random / k-means (weighted) / one-class schemes, 5 repeated draws as the noise floor |
| Preprocessing | `preprocessing_study.py` | `ablation_preprocessing` MLP and LR models × none / standard / log_standard × 10 seeds; DeepLiftShap, DeepLift (and an all-zero baseline DeepLift for unscaled inputs), LinearSHAP; Wilcoxon + Holm + Cliff's delta on loc and size-group shares |
| Multicollinearity | `collinearity.py` | Spearman clustering (cut at \|ρ\| = 0.7) and VIF per training release; stable clusters; AutoSpearman; grouped (cluster-level) attributions; MLP and RF retrained on the AutoSpearman metrics (`configs/experiments/autospearman_time_aware.yaml`, step 3 engine with per-split feature selection) |

```bash
# step 4 v2 (keeps the v1 LIME results), then step 5
python -m scripts.run_explanations --keep-method-as LIME LIME_mean_sampling
python -m scripts.make_step4_reports
python -m scripts.run_robustness --smoke --estimate-only   # smoke test and runtime estimate
python -m scripts.run_robustness                           # all studies (or --studies lime background ...)
python -m scripts.make_step5_reports
```

Outputs: attributions and LIME runs under `artifacts/robustness/`, `artifacts/attributions_ablation_preprocessing/<preprocessing>/` and `artifacts/attributions_autospearman_time_aware/`; retrained models under `artifacts/models/autospearman_time_aware/`; reports `reports/step5_lime_stability.md`, `step5_background_sensitivity.md`, `step5_preprocessing_explanations.md`, `step5_collinearity.md` and `step5_runtime.md` (with CSVs) and figures in `reports/figures/step5/`. Per-instance values are in `results/robustness/`.

## Layout

```
configs/default.yaml       configuration
src/sdp/config.py          configuration loader
src/sdp/seed.py            global seeding (Python, NumPy)
src/sdp/data/download.py   Zenodo search, download, md5 verification, MANIFEST
src/sdp/data/legacy.py     comparison of local legacy files with Zenodo
src/sdp/data/clean.py      per-release cleaning
src/sdp/data/audit.py      audit tables and report
src/sdp/data/study.py      study inclusion rule and release pairs
src/sdp/eval/              preprocessing, splits, metrics, statistics
src/sdp/models/            MLP, LR/RF/XGBoost, baselines, imbalance handling
src/sdp/tuning.py          Optuna tuning
src/sdp/experiments/       step 2 runner, step 3 engine, configs, persistence
src/sdp/explain/           explainers, background sets, attribution store, runner (step 4)
src/sdp/robustness/        step 5 studies (LIME, background, preprocessing, collinearity)
src/sdp/analysis/          step 3–5 aggregation, reports and figures
configs/experiments/       experiment definitions
scripts/prepare_data.py    data preparation entry point
scripts/run_baselines.py   baseline experiments entry point (step 2)
scripts/run_experiment.py  step 3 experiments (smoke test, estimate, full runs)
scripts/make_step3_reports.py  step 3 reports
scripts/run_explanations.py    step 4 explanations (smoke test, estimate, full run)
scripts/make_step4_reports.py  step 4 reports and figures
configs/explain.yaml       step 4 explanation settings
configs/robustness.yaml    step 5 study settings
scripts/run_robustness.py      step 5 studies (smoke test, estimate, full run)
scripts/make_step5_reports.py  step 5 reports and figures
tests/                     pytest suite
```

## References

- Jureczko, M., & Madeyski, L. (2010). Towards identifying software project clusters with regard to defect prediction. *PROMISE '10*.
- Herbold, S. (2018). Benchmarking cross-project defect prediction approaches with costs metrics. arXiv:1801.04107.
- Jiarpakdee, J., Tantithamthavorn, C., & Treude, C. (2018). AutoSpearman: Automatically mitigating correlated software metrics for interpreting defect models. *ICSME 2018*, 92–103.
- Krishna, S., Han, T., Gu, A., Pombra, J., Jabbari, S., Wu, S., & Lakkaraju, H. (2022). The disagreement problem in explainable machine learning: A practitioner's perspective. arXiv:2202.01602.
- Mende, T., & Koschke, R. (2010). Effort-aware defect prediction models. *CSMR 2010*, 107–116.
- Romano, J., Kromrey, J. D., Coraggio, J., & Skowronek, J. (2006). Appropriate statistics for ordinal level data. *Florida Association of Institutional Research*.
- Scott, A. J., & Knott, M. (1974). A cluster analysis method for grouping means in the analysis of variance. *Biometrics*, 30(3), 507–512.
- Sun, Z., Li, J., Sun, H., & He, L. (2018). An empirical study of public data quality problems in cross project defect prediction. arXiv:1805.10787.
- Tantithamthavorn, C., McIntosh, S., Hassan, A. E., & Matsumoto, K. (2017). An empirical comparison of model validation techniques for defect prediction models. *IEEE TSE*, 43(1), 1–18.
