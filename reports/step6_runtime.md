# Step 6 runtime

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

## Smoke test, estimate and full run

Smoke split `ta-consecutive:xerces:init>1.2` (seed 0; ROAR/KAR with one seed, scaled to three). Estimated full run: 2.22 CPU h, ~0.32 h wall with n_jobs=10 (assumed parallel efficiency 0.7).

| part | smoke wall (s) | estimated CPU h | actual CPU h | actual wall (min) |
|---|---|---|---|---|
| curves | 27.2 | 0.25 | 0.61 | 4.9 |
| train | 4.1 | 0.15 | 0.56 | 3.5 |
| roar | 20.7 | 1.07 | 2.25 | 13.6 |
| sanity | 60.2 | 0.74 | 4.02 | 25.9 |

Reductions: none (estimate below 8 h).
