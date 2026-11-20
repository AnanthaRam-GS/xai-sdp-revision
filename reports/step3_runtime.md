# Step 3 runtime

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

## Experiments

CPU hours are summed over jobs (single-threaded workers). Jobs reused from the cache were computed by an earlier experiment with identical settings (e.g. the `log_standard` MLP/LR units of the preprocessing ablation), so their CPU time is listed under both experiments but spent once.

| experiment | jobs | jobs_reused_from_cache | n_jobs | wall_time_h | tuning_cpu_h | fit_cpu_h |
|---|---|---|---|---|---|---|
| main_time_aware | 1056 | 16 | 10 | 0.43 | 4.25 | 0.04 |
| main_within_release | 2720 | 0 | 10 | 0.25 | 2.11 | 0.30 |
| ablation_preprocessing | 1440 | 480 | 10 | 0.36 | 5.07 | 0.07 |
| ablation_imbalance | 1440 | 240 | 10 | 1.02 | 11.43 | 0.12 |
| main_time_aware_cumulative | 480 | 0 | 10 | 0.23 | 2.29 | 0.02 |

Total wall time: 2.29 h. Total CPU time of unique units: 22.63 h.

## Per model (unique units over all experiments)

`units` = fitted (split, setting, seed) units; `mean_tuning_s` = mean time of one tuning run (Optuna study); `mean_fit_s` = mean time of the final fit on the fitting part.

| model | units | tuning_cpu_h | fit_cpu_h | mean_tuning_s | mean_fit_s |
|---|---|---|---|---|---|
| mlp | 3020 | 9.05 | 0.27 | 19.63 | 0.32 |
| lr | 2540 | 1.30 | 0.02 | 3.96 | 0.02 |
| rf | 2780 | 11.21 | 0.21 | 28.42 | 0.27 |
| xgb | 2060 | 0.54 | 0.02 | 2.80 | 0.04 |
| manual_up | 1724 | 0.00 | 0.00 |  | 0.00 |
| manual_down | 1724 | 0.00 | 0.00 |  | 0.00 |
| all_defective | 1724 | 0.00 | 0.00 |  | 0.00 |
| majority | 1724 | 0.00 | 0.00 |  | 0.00 |

## Smoke test and estimate

Smoke test (`main_time_aware` on ant, 2 seeds, 5 trials): 7.7 s. Estimated before the full run: 19.5 CPU h, ~4.0 h wall with n_jobs=10. The estimate was below 8 h, so the time-aware tuning budget was kept at 40 trials.

