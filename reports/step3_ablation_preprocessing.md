# Ablation: preprocessing

Experiment `ablation_preprocessing`.

- Splits: 24; seeds per learned model: 10; rule-based baselines: 1 (deterministic).
- Models (preprocessing, imbalance): `mlp` (none, class_weight), `lr` (none, class_weight).
- Optuna trials per tuning run: 40 (chosen hyperparameters in `results/<experiment>/hyperparameters.csv`).

Summaries: mean and percentile-bootstrap 95% CI (2,000 resamples) over all splits x seeds. Statistical comparisons use the per-split mean across seeds (each split counts once). `@0.5` = 0.5 threshold; `@tuned` = MCC-maximising threshold chosen on the validation split. Lower is better for Brier, ECE, IFA and NECM15; higher for all other metrics.

## mlp

| metric | none | standard | log_standard |
|---|---|---|---|
| AUC | 0.657 [0.642, 0.672] | 0.663 [0.649, 0.677] | 0.686 [0.673, 0.699] |
| MCC@tuned | 0.212 [0.192, 0.231] | 0.213 [0.193, 0.232] | 0.234 [0.215, 0.251] |
| Brier | 0.279 [0.269, 0.290] | 0.239 [0.232, 0.246] | 0.236 [0.229, 0.242] |
| Popt | 0.748 [0.734, 0.762] | 0.767 [0.754, 0.781] | 0.769 [0.756, 0.782] |

## lr

| metric | none | standard | log_standard |
|---|---|---|---|
| AUC | 0.671 [0.656, 0.685] | 0.670 [0.656, 0.684] | 0.687 [0.673, 0.701] |
| MCC@tuned | 0.212 [0.193, 0.231] | 0.217 [0.198, 0.236] | 0.240 [0.221, 0.257] |
| Brier | 0.234 [0.228, 0.241] | 0.234 [0.229, 0.239] | 0.232 [0.227, 0.237] |
| Popt | 0.764 [0.751, 0.777] | 0.770 [0.756, 0.783] | 0.770 [0.755, 0.784] |

## Pairwise comparisons between preprocessing settings

Wilcoxon signed-rank on per-split means across seeds, Holm correction within each (model, metric) family; `cliffs_delta` > 0 means setting `a` has larger values.

| model | metric | a | b | n | mean_a | mean_b | p_value | p_holm | cliffs_delta | magnitude |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp | AUC | log_standard | none | 24 | 0.686 | 0.657 | 0.003 | 0.006 | 0.170 | small |
| mlp | AUC | log_standard | standard | 24 | 0.686 | 0.663 | <0.001 | <0.001 | 0.139 | negligible |
| mlp | AUC | none | standard | 24 | 0.657 | 0.663 | 0.303 | 0.303 | -0.069 | negligible |
| mlp | MCC@tuned | log_standard | none | 24 | 0.234 | 0.212 | 0.136 | 0.271 | 0.087 | negligible |
| mlp | MCC@tuned | log_standard | standard | 24 | 0.234 | 0.213 | <0.001 | 0.002 | 0.090 | negligible |
| mlp | MCC@tuned | none | standard | 24 | 0.212 | 0.213 | 0.944 | 0.944 | -0.017 | negligible |
| mlp | Brier | log_standard | none | 24 | 0.236 | 0.279 | <0.001 | <0.001 | -0.410 | medium |
| mlp | Brier | log_standard | standard | 24 | 0.236 | 0.239 | 0.197 | 0.197 | -0.035 | negligible |
| mlp | Brier | none | standard | 24 | 0.279 | 0.239 | <0.001 | <0.001 | 0.406 | medium |
| mlp | Popt | log_standard | none | 24 | 0.769 | 0.748 | 0.005 | 0.014 | 0.115 | negligible |
| mlp | Popt | log_standard | standard | 24 | 0.769 | 0.767 | 0.509 | 0.509 | -0.003 | negligible |
| mlp | Popt | none | standard | 24 | 0.748 | 0.767 | 0.006 | 0.014 | -0.090 | negligible |
| lr | AUC | log_standard | none | 24 | 0.687 | 0.671 | 0.060 | 0.121 | 0.104 | negligible |
| lr | AUC | log_standard | standard | 24 | 0.687 | 0.670 | 0.003 | 0.009 | 0.104 | negligible |
| lr | AUC | none | standard | 24 | 0.671 | 0.670 | 0.790 | 0.790 | 0.003 | negligible |
| lr | MCC@tuned | log_standard | none | 24 | 0.240 | 0.212 | <0.001 | 0.003 | 0.118 | negligible |
| lr | MCC@tuned | log_standard | standard | 24 | 0.240 | 0.217 | 0.013 | 0.025 | 0.097 | negligible |
| lr | MCC@tuned | none | standard | 24 | 0.212 | 0.217 | 0.456 | 0.456 | 0.000 | negligible |
| lr | Brier | log_standard | none | 24 | 0.232 | 0.234 | 0.966 | 1.000 | -0.028 | negligible |
| lr | Brier | log_standard | standard | 24 | 0.232 | 0.234 | 0.584 | 1.000 | -0.069 | negligible |
| lr | Brier | none | standard | 24 | 0.234 | 0.234 | 0.422 | 1.000 | -0.024 | negligible |
| lr | Popt | log_standard | none | 24 | 0.770 | 0.764 | 0.390 | 1.000 | 0.042 | negligible |
| lr | Popt | log_standard | standard | 24 | 0.770 | 0.770 | 0.768 | 1.000 | -0.003 | negligible |
| lr | Popt | none | standard | 24 | 0.764 | 0.770 | 0.705 | 1.000 | -0.049 | negligible |

