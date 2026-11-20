# Ablation: class-imbalance handling

Experiment `ablation_imbalance`.

- Splits: 24; seeds per learned model: 10; rule-based baselines: 1 (deterministic).
- Models (preprocessing, imbalance): `mlp` (log_standard, none), `rf` (log_standard, none).
- Optuna trials per tuning run: 40 (chosen hyperparameters in `results/<experiment>/hyperparameters.csv`).

Summaries: mean and percentile-bootstrap 95% CI (2,000 resamples) over all splits x seeds. Statistical comparisons use the per-split mean across seeds (each split counts once). `@0.5` = 0.5 threshold; `@tuned` = MCC-maximising threshold chosen on the validation split. Lower is better for Brier, ECE, IFA and NECM15; higher for all other metrics.

## mlp

| metric | none | class_weight | smote |
|---|---|---|---|
| AUC | 0.679 [0.666, 0.692] | 0.686 [0.673, 0.699] | 0.682 [0.669, 0.694] |
| MCC@tuned | 0.218 [0.200, 0.235] | 0.234 [0.215, 0.251] | 0.222 [0.205, 0.238] |
| Brier | 0.230 [0.218, 0.243] | 0.236 [0.229, 0.242] | 0.236 [0.228, 0.245] |
| Popt | 0.754 [0.740, 0.768] | 0.769 [0.756, 0.782] | 0.745 [0.730, 0.760] |

## rf

| metric | none | class_weight | smote |
|---|---|---|---|
| AUC | 0.677 [0.662, 0.692] | 0.676 [0.661, 0.691] | 0.670 [0.655, 0.684] |
| MCC@tuned | 0.238 [0.220, 0.255] | 0.244 [0.226, 0.261] | 0.228 [0.209, 0.245] |
| Brier | 0.227 [0.215, 0.239] | 0.236 [0.227, 0.245] | 0.234 [0.224, 0.244] |
| Popt | 0.728 [0.710, 0.746] | 0.740 [0.724, 0.756] | 0.730 [0.713, 0.748] |

## Pairwise comparisons between imbalance settings

Wilcoxon signed-rank on per-split means across seeds, Holm correction within each (model, metric) family; `cliffs_delta` > 0 means setting `a` has larger values.

| model | metric | a | b | n | mean_a | mean_b | p_value | p_holm | cliffs_delta | magnitude |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp | AUC | class_weight | none | 24 | 0.686 | 0.679 | 0.264 | 0.529 | 0.062 | negligible |
| mlp | AUC | class_weight | smote | 24 | 0.686 | 0.682 | 0.084 | 0.252 | 0.059 | negligible |
| mlp | AUC | none | smote | 24 | 0.679 | 0.682 | 0.747 | 0.747 | 0.017 | negligible |
| mlp | MCC@tuned | class_weight | none | 24 | 0.234 | 0.218 | 0.101 | 0.202 | 0.062 | negligible |
| mlp | MCC@tuned | class_weight | smote | 24 | 0.234 | 0.222 | 0.037 | 0.110 | 0.073 | negligible |
| mlp | MCC@tuned | none | smote | 24 | 0.218 | 0.222 | 0.812 | 0.812 | 0.003 | negligible |
| mlp | Brier | class_weight | none | 24 | 0.236 | 0.230 | 0.390 | 0.869 | 0.177 | small |
| mlp | Brier | class_weight | smote | 24 | 0.236 | 0.236 | 0.989 | 0.989 | 0.021 | negligible |
| mlp | Brier | none | smote | 24 | 0.230 | 0.236 | 0.290 | 0.869 | -0.128 | negligible |
| mlp | Popt | class_weight | none | 24 | 0.769 | 0.754 | 0.208 | 0.208 | 0.090 | negligible |
| mlp | Popt | class_weight | smote | 24 | 0.769 | 0.745 | 0.001 | 0.004 | 0.139 | negligible |
| mlp | Popt | none | smote | 24 | 0.754 | 0.745 | 0.011 | 0.021 | 0.073 | negligible |
| rf | AUC | class_weight | none | 24 | 0.676 | 0.677 | 0.790 | 0.790 | -0.024 | negligible |
| rf | AUC | class_weight | smote | 24 | 0.676 | 0.670 | 0.101 | 0.303 | 0.045 | negligible |
| rf | AUC | none | smote | 24 | 0.677 | 0.670 | 0.107 | 0.303 | 0.062 | negligible |
| rf | MCC@tuned | class_weight | none | 24 | 0.244 | 0.238 | 0.768 | 0.768 | -0.007 | negligible |
| rf | MCC@tuned | class_weight | smote | 24 | 0.244 | 0.228 | 0.143 | 0.430 | 0.069 | negligible |
| rf | MCC@tuned | none | smote | 24 | 0.238 | 0.228 | 0.331 | 0.661 | 0.080 | negligible |
| rf | Brier | class_weight | none | 24 | 0.236 | 0.227 | 0.218 | 0.506 | 0.128 | negligible |
| rf | Brier | class_weight | smote | 24 | 0.236 | 0.234 | 0.360 | 0.506 | 0.021 | negligible |
| rf | Brier | none | smote | 24 | 0.227 | 0.234 | 0.169 | 0.506 | -0.111 | negligible |
| rf | Popt | class_weight | none | 24 | 0.740 | 0.728 | 0.565 | 1.000 | 0.031 | negligible |
| rf | Popt | class_weight | smote | 24 | 0.740 | 0.730 | 0.107 | 0.322 | 0.045 | negligible |
| rf | Popt | none | smote | 24 | 0.728 | 0.730 | 0.623 | 1.000 | 0.014 | negligible |

