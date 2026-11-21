# Step 8: calibration

Learned models of `main_time_aware` (all 24 splits × 10 seeds). Platt scaling (logistic regression on the logit of the predicted probability) and isotonic regression are fitted on each model's validation split only and applied to its test predictions; the models are not refitted. MCC@tuned uses a threshold re-tuned on the calibrated validation probabilities. Per-split means over seeds; mean and 95% bootstrap CI over splits. ECE: 10 equal-width bins.

## Summary

| model | method | ECE | BRIER | NLL | AUC | MCC@tuned |
|---|---|---|---|---|---|---|
| MLP | uncalibrated | 0.221 [0.182, 0.261] | 0.236 [0.217, 0.256] | 0.686 [0.636, 0.743] | 0.686 [0.644, 0.725] | 0.234 [0.177, 0.288] |
| MLP | platt | 0.196 [0.139, 0.258] | 0.230 [0.192, 0.271] | 0.703 [0.599, 0.823] | 0.676 [0.635, 0.716] | 0.225 [0.168, 0.282] |
| MLP | isotonic | 0.218 [0.161, 0.275] | 0.247 [0.205, 0.292] | 3.298 [2.054, 4.779] | 0.671 [0.632, 0.708] | 0.234 [0.177, 0.288] |
| LR | uncalibrated | 0.227 [0.191, 0.267] | 0.232 [0.218, 0.248] | 0.667 [0.632, 0.707] | 0.687 [0.642, 0.729] | 0.240 [0.183, 0.296] |
| LR | platt | 0.186 [0.128, 0.249] | 0.227 [0.188, 0.267] | 0.686 [0.581, 0.804] | 0.687 [0.642, 0.729] | 0.240 [0.183, 0.296] |
| LR | isotonic | 0.210 [0.155, 0.268] | 0.244 [0.201, 0.289] | 3.335 [2.002, 4.926] | 0.673 [0.631, 0.714] | 0.240 [0.183, 0.296] |
| RF | uncalibrated | 0.220 [0.182, 0.261] | 0.235 [0.208, 0.266] | 0.798 [0.639, 1.027] | 0.676 [0.626, 0.722] | 0.247 [0.193, 0.299] |
| RF | platt | 0.196 [0.141, 0.256] | 0.233 [0.195, 0.273] | 0.749 [0.611, 0.927] | 0.653 [0.597, 0.707] | 0.233 [0.175, 0.289] |
| RF | isotonic | 0.220 [0.166, 0.276] | 0.248 [0.207, 0.289] | 3.460 [2.380, 4.798] | 0.665 [0.623, 0.706] | 0.247 [0.193, 0.299] |
| XGB | uncalibrated | 0.213 [0.171, 0.259] | 0.234 [0.206, 0.265] | 0.703 [0.616, 0.815] | 0.675 [0.630, 0.717] | 0.228 [0.178, 0.276] |
| XGB | platt | 0.195 [0.137, 0.256] | 0.232 [0.194, 0.272] | 0.705 [0.604, 0.825] | 0.652 [0.597, 0.703] | 0.212 [0.153, 0.269] |
| XGB | isotonic | 0.217 [0.161, 0.276] | 0.248 [0.207, 0.289] | 3.086 [2.127, 4.123] | 0.666 [0.626, 0.704] | 0.228 [0.178, 0.276] |

Notes:

* Platt scaling is monotone and leaves the ranking (AUC) unchanged when its fitted slope is positive. In 25 of 960 (split, model, seed) units the slope on the validation split is negative (RF 10, XGB 10, MLP 5): there the model ranks the validation classes worse than chance, Platt reverses the ranking, and the test AUC changes. Excluding these units, the mean change in test AUC is 4.7e-07.
* Isotonic regression produces probabilities of exactly 0 or 1 in sparse regions; one confident mistake then costs up to −log(1e−15) ≈ 34.5 in NLL, which explains the high isotonic NLL. Its ties also change the AUC slightly.

## Tests (calibrated vs uncalibrated; Wilcoxon signed-rank, Holm within model, Cliff's delta)

| model | metric | a | b | n | mean_a | mean_b | p_value | p_holm | cliffs_delta | magnitude |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP | ece | platt | uncalibrated | 24 | 0.196 | 0.221 | 0.277 | 1.000 | -0.281 | small |
| MLP | brier | platt | uncalibrated | 24 | 0.230 | 0.236 | 0.390 | 1.000 | -0.174 | small |
| MLP | nll | platt | uncalibrated | 24 | 0.703 | 0.686 | 0.705 | 1.000 | -0.118 | negligible |
| MLP | auc | platt | uncalibrated | 24 | 0.676 | 0.686 | 0.043 | 0.345 | -0.068 | negligible |
| MLP | mcc_tuned | platt | uncalibrated | 24 | 0.225 | 0.234 | 0.180 | 1.000 | -0.031 | negligible |
| MLP | ece | isotonic | uncalibrated | 24 | 0.218 | 0.221 | 0.855 | 1.000 | -0.142 | negligible |
| MLP | brier | isotonic | uncalibrated | 24 | 0.247 | 0.236 | 0.855 | 1.000 | -0.101 | negligible |
| MLP | nll | isotonic | uncalibrated | 24 | 3.298 | 0.686 | <0.001 | <0.001 | 0.868 | large |
| MLP | auc | isotonic | uncalibrated | 24 | 0.671 | 0.686 | <0.001 | <0.001 | -0.118 | negligible |
| MLP | mcc_tuned | isotonic | uncalibrated | 24 | 0.234 | 0.234 | 1.000 | 1.000 | 0.000 | negligible |
| LR | ece | platt | uncalibrated | 24 | 0.186 | 0.227 | 0.101 | 0.809 | -0.375 | medium |
| LR | brier | platt | uncalibrated | 24 | 0.227 | 0.232 | 0.345 | 1.000 | -0.233 | small |
| LR | nll | platt | uncalibrated | 24 | 0.686 | 0.667 | 0.877 | 1.000 | -0.135 | negligible |
| LR | auc | platt | uncalibrated | 24 | 0.687 | 0.687 | 1.000 | 1.000 | 0.000 | negligible |
| LR | mcc_tuned | platt | uncalibrated | 24 | 0.240 | 0.240 | 1.000 | 1.000 | 0.000 | negligible |
| LR | ece | isotonic | uncalibrated | 24 | 0.210 | 0.227 | 0.422 | 1.000 | -0.250 | small |
| LR | brier | isotonic | uncalibrated | 24 | 0.244 | 0.232 | 0.944 | 1.000 | -0.132 | negligible |
| LR | nll | isotonic | uncalibrated | 24 | 3.335 | 0.667 | <0.001 | <0.001 | 0.826 | large |
| LR | auc | isotonic | uncalibrated | 24 | 0.673 | 0.687 | <0.001 | <0.001 | -0.104 | negligible |
| LR | mcc_tuned | isotonic | uncalibrated | 24 | 0.240 | 0.240 | 1.000 | 1.000 | 0.000 | negligible |
| RF | ece | platt | uncalibrated | 24 | 0.196 | 0.220 | 0.188 | 1.000 | -0.247 | small |
| RF | brier | platt | uncalibrated | 24 | 0.233 | 0.235 | 0.603 | 1.000 | -0.083 | negligible |
| RF | nll | platt | uncalibrated | 24 | 0.749 | 0.798 | 0.331 | 1.000 | -0.080 | negligible |
| RF | auc | platt | uncalibrated | 24 | 0.653 | 0.676 | 0.273 | 1.000 | -0.073 | negligible |
| RF | mcc_tuned | platt | uncalibrated | 24 | 0.233 | 0.247 | 0.180 | 1.000 | -0.059 | negligible |
| RF | ece | isotonic | uncalibrated | 24 | 0.220 | 0.220 | 1.000 | 1.000 | -0.115 | negligible |
| RF | brier | isotonic | uncalibrated | 24 | 0.248 | 0.235 | 0.360 | 1.000 | 0.000 | negligible |
| RF | nll | isotonic | uncalibrated | 24 | 3.460 | 0.798 | <0.001 | <0.001 | 0.809 | large |
| RF | auc | isotonic | uncalibrated | 24 | 0.665 | 0.676 | 0.003 | 0.023 | -0.097 | negligible |
| RF | mcc_tuned | isotonic | uncalibrated | 24 | 0.247 | 0.247 | 1.000 | 1.000 | 0.000 | negligible |
| XGB | ece | platt | uncalibrated | 24 | 0.195 | 0.213 | 0.331 | 1.000 | -0.215 | small |
| XGB | brier | platt | uncalibrated | 24 | 0.232 | 0.234 | 0.473 | 1.000 | -0.080 | negligible |
| XGB | nll | platt | uncalibrated | 24 | 0.705 | 0.703 | 0.812 | 1.000 | -0.031 | negligible |
| XGB | auc | platt | uncalibrated | 24 | 0.652 | 0.675 | 0.317 | 1.000 | -0.068 | negligible |
| XGB | mcc_tuned | platt | uncalibrated | 24 | 0.212 | 0.228 | 0.317 | 1.000 | -0.054 | negligible |
| XGB | ece | isotonic | uncalibrated | 24 | 0.217 | 0.213 | 0.812 | 1.000 | -0.059 | negligible |
| XGB | brier | isotonic | uncalibrated | 24 | 0.248 | 0.234 | 0.277 | 1.000 | -0.003 | negligible |
| XGB | nll | isotonic | uncalibrated | 24 | 3.086 | 0.703 | <0.001 | <0.001 | 0.861 | large |
| XGB | auc | isotonic | uncalibrated | 24 | 0.666 | 0.675 | 0.006 | 0.053 | -0.073 | negligible |
| XGB | mcc_tuned | isotonic | uncalibrated | 24 | 0.228 | 0.228 | 1.000 | 1.000 | 0.000 | negligible |

## Reliability diagram data before calibration (pooled over splits and seeds)

Cells: mean predicted probability / observed defective rate (instances).

| bin | LR | MLP | RF | XGB |
|---|---|---|---|---|
| [0.0, 0.1) | 0.06 / 0.12 (4196) | 0.04 / 0.11 (9999) | 0.05 / 0.21 (13122) | 0.05 / 0.22 (14408) |
| [0.1, 0.2) | 0.15 / 0.18 (7107) | 0.15 / 0.20 (8687) | 0.15 / 0.19 (13586) | 0.15 / 0.19 (15095) |
| [0.2, 0.3) | 0.25 / 0.22 (10165) | 0.25 / 0.24 (10710) | 0.25 / 0.20 (12274) | 0.25 / 0.20 (12284) |
| [0.3, 0.4) | 0.35 / 0.20 (13798) | 0.35 / 0.25 (11449) | 0.35 / 0.22 (10774) | 0.35 / 0.23 (10022) |
| [0.4, 0.5) | 0.45 / 0.26 (19057) | 0.45 / 0.26 (12722) | 0.45 / 0.26 (9744) | 0.45 / 0.27 (8317) |
| [0.5, 0.6) | 0.54 / 0.37 (16446) | 0.55 / 0.34 (11787) | 0.55 / 0.35 (8058) | 0.55 / 0.37 (7730) |
| [0.6, 0.7) | 0.65 / 0.42 (9861) | 0.65 / 0.40 (9796) | 0.65 / 0.42 (7581) | 0.65 / 0.43 (7705) |
| [0.7, 0.8) | 0.75 / 0.50 (6743) | 0.75 / 0.45 (8719) | 0.75 / 0.52 (7762) | 0.75 / 0.51 (8263) |
| [0.8, 0.9) | 0.84 / 0.59 (4648) | 0.85 / 0.55 (6499) | 0.85 / 0.55 (7289) | 0.85 / 0.54 (7289) |
| [0.9, 1.0) | 0.94 / 0.60 (2089) | 0.94 / 0.63 (3742) | 0.94 / 0.60 (3920) | 0.93 / 0.62 (2997) |
