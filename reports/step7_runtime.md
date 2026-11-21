# Step 7 runtime

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

## Permuted-label refit (data-only reference)

96 models refitted (seed 0, persisted hyperparameters, no re-tuning) and 336 explanation sets stored in 4.9 min wall (10 workers). Reproduction of step 6: maximum absolute differences — AUC 0.0e+00, Spearman 0.0e+00, FA@5 0.0e+00 over 336 (split, model, method) units. The refit was run directly, without its own smoke test: it repeats the step 6 data-randomisation refit, whose cost was known (step 6 runtime report), plus storing the explanations.

## Smoke test (project ant) and full run of the analyses

Smoke test: 4 splits, 39 s; estimated full run 3.9 min. Full run: 1.4 min (single process). No reductions.

| part | smoke (s) | full (s) |
|---|---|---|
| load | 29.6 | 28.8 |
| cross_version | 2.3 | 11.1 |
| unchanged | 0.2 | 1.1 |
| seeds | 2.3 | 14.2 |
| projects | 0.0 | 0.5 |
| drift | 5.0 | 29.8 |
