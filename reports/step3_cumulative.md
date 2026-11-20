# Consecutive vs cumulative time-aware training

Robustness check for the four learned models with seeds [0, 1, 2, 3, 4]. Consecutive results are the `main_time_aware` runs with the same seeds; cumulative training uses all included releases up to n. Both are tested on release n+1; pairs are matched on (project, test release). The first pair of each project has identical training data in both variants (10 of 24 pairs).

Summaries: mean and percentile-bootstrap 95% CI (2,000 resamples) over all splits x seeds. Statistical comparisons use the per-split mean across seeds (each split counts once). `@0.5` = 0.5 threshold; `@tuned` = MCC-maximising threshold chosen on the validation split. Lower is better for Brier, ECE, IFA and NECM15; higher for all other metrics.

## mlp

| metric | consecutive | cumulative |
|---|---|---|
| AUC | 0.685 [0.667, 0.705] | 0.704 [0.687, 0.722] |
| MCC@tuned | 0.229 [0.204, 0.256] | 0.251 [0.227, 0.275] |
| Brier | 0.235 [0.226, 0.245] | 0.230 [0.222, 0.239] |
| Popt | 0.770 [0.750, 0.788] | 0.770 [0.749, 0.790] |

## lr

| metric | consecutive | cumulative |
|---|---|---|
| AUC | 0.688 [0.669, 0.706] | 0.704 [0.686, 0.721] |
| MCC@tuned | 0.239 [0.212, 0.264] | 0.241 [0.217, 0.265] |
| Brier | 0.231 [0.224, 0.238] | 0.228 [0.222, 0.235] |
| Popt | 0.770 [0.750, 0.789] | 0.775 [0.755, 0.795] |

## rf

| metric | consecutive | cumulative |
|---|---|---|
| AUC | 0.676 [0.655, 0.696] | 0.702 [0.682, 0.721] |
| MCC@tuned | 0.248 [0.223, 0.271] | 0.248 [0.225, 0.270] |
| Brier | 0.236 [0.223, 0.251] | 0.225 [0.216, 0.235] |
| Popt | 0.738 [0.717, 0.760] | 0.764 [0.742, 0.786] |

## xgb

| metric | consecutive | cumulative |
|---|---|---|
| AUC | 0.675 [0.655, 0.694] | 0.694 [0.677, 0.712] |
| MCC@tuned | 0.225 [0.202, 0.248] | 0.247 [0.222, 0.271] |
| Brier | 0.233 [0.221, 0.247] | 0.228 [0.218, 0.237] |
| Popt | 0.740 [0.718, 0.762] | 0.760 [0.738, 0.782] |

## Wilcoxon (Holm across the four models per metric) and Cliff's delta

`cliffs_delta` > 0 means consecutive has larger values.

| metric | model | a | b | n | mean_a | mean_b | p_value | p_holm | cliffs_delta | magnitude |
|---|---|---|---|---|---|---|---|---|---|---|
| AUC | mlp | consecutive | cumulative | 24 | 0.685 | 0.704 | 0.158 | 0.280 | -0.153 | small |
| AUC | lr | consecutive | cumulative | 24 | 0.688 | 0.704 | 0.056 | 0.167 | -0.094 | negligible |
| AUC | rf | consecutive | cumulative | 24 | 0.676 | 0.702 | 0.022 | 0.088 | -0.153 | small |
| AUC | xgb | consecutive | cumulative | 24 | 0.675 | 0.694 | 0.140 | 0.280 | -0.094 | negligible |
| MCC@tuned | mlp | consecutive | cumulative | 24 | 0.229 | 0.251 | 0.234 | 0.703 | -0.118 | negligible |
| MCC@tuned | lr | consecutive | cumulative | 24 | 0.239 | 0.241 | 0.875 | 1.000 | -0.042 | negligible |
| MCC@tuned | rf | consecutive | cumulative | 24 | 0.248 | 0.248 | 0.925 | 1.000 | 0.031 | negligible |
| MCC@tuned | xgb | consecutive | cumulative | 24 | 0.225 | 0.247 | 0.026 | 0.103 | -0.094 | negligible |
| Brier | mlp | consecutive | cumulative | 24 | 0.235 | 0.230 | 0.875 | 1.000 | 0.104 | negligible |
| Brier | lr | consecutive | cumulative | 24 | 0.231 | 0.228 | 0.730 | 1.000 | 0.035 | negligible |
| Brier | rf | consecutive | cumulative | 24 | 0.236 | 0.225 | 0.221 | 0.884 | 0.056 | negligible |
| Brier | xgb | consecutive | cumulative | 24 | 0.233 | 0.228 | 0.865 | 1.000 | 0.040 | negligible |
| Popt | mlp | consecutive | cumulative | 24 | 0.770 | 0.770 | 0.975 | 0.975 | -0.010 | negligible |
| Popt | lr | consecutive | cumulative | 24 | 0.770 | 0.775 | 0.064 | 0.128 | -0.056 | negligible |
| Popt | rf | consecutive | cumulative | 24 | 0.738 | 0.764 | 0.008 | 0.031 | -0.132 | negligible |
| Popt | xgb | consecutive | cumulative | 24 | 0.740 | 0.760 | 0.030 | 0.091 | -0.080 | negligible |

