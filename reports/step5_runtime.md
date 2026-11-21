# Step 5 runtime

## Machine

| item | value |
|---|---|
| CPU | Apple M5 |
| Logical cores | 10 |
| Performance / efficiency cores | 4 / 6 |
| RAM (GB) | 24.00 |
| OS | Darwin 27.0.1 (arm64) |
| Python | 3.11.15 |
| numpy | 2.4.6 |
| pandas | 3.1.1 |
| scikit-learn | 1.9.1 |
| scipy | 1.17.1 |
| torch | 2.15.0 |
| xgboost | 3.2.0 |
| optuna | 5.1.0 |
| imbalanced-learn | 0.14.2 |
| captum | 0.9.0 |
| joblib | 1.6.0 |

## Step 4 recomputation (v2)

Integrated Gradients (512 steps) and LIME (instance sampling) recomputed with the step 4 runner: 19.4 min wall, 456 of 960 jobs recomputed.

## Smoke test and estimate

Smoke split `ta-consecutive:xerces:init>1.2` (seed 0). The AutoSpearman retraining is estimated from the step 3 timings (seeds 0–2, tuning time × 25/40).

| study | smoke wall (s) | estimated CPU h | actual wall (min) |
|---|---|---|---|
| lime | 37.7 | 0.30 | 3.0 |
| background | 9.3 | 0.74 | 12.3 |
| preprocessing | 3.2 | 0.14 | 2.9 |
| autospearman |  | 0.77 | 4.7 |

Estimated total: 1.95 CPU h, ~0.28 h wall with n_jobs=10 (assumed parallel efficiency 0.7).

Reductions: none (estimate below 8 h).

## Full run

| study | units | CPU h |
|---|---|---|
| lime | 96 | 0.48 |
| background | 1008 | 2.03 |
| preprocessing | 2400 | 0.43 |
| autospearman retraining | 144 | 0.76 |

Wall times (min): lime 3.0, background 12.3, preprocessing 2.9, autospearman_training 4.7, autospearman_explanations 1.6.
