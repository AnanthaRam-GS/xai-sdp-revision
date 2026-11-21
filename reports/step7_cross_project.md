# Step 7: cross-project stability

Project-level importance = mean of the seed-averaged global importance over the project's splits. Mean of the 45 off-diagonal values of the 10 × 10 matrix with a 95% bootstrap CI over project pairs; data-only = the same for the permuted-label models (seed 0).

| explanation | Spearman (mean off-diagonal) | Spearman data-only | RBO (mean off-diagonal) | RBO data-only |
|---|---|---|---|---|
| MLP-DeepLift | 0.089 [0.020, 0.154] | 0.123 | 0.455 [0.431, 0.479] | 0.491 |
| MLP-DeepLiftShap | 0.087 [0.009, 0.158] | 0.058 | 0.456 [0.429, 0.483] | 0.468 |
| MLP-GradientShap | 0.079 [0.001, 0.151] | 0.045 | 0.451 [0.424, 0.477] | 0.461 |
| MLP-IntegratedGradients | 0.082 [0.012, 0.148] | 0.119 | 0.452 [0.428, 0.477] | 0.490 |
| MLP-Saliency | 0.072 [-0.000, 0.145] | 0.093 | 0.462 [0.436, 0.487] | 0.471 |
| MLP-InputXGradient | 0.101 [0.032, 0.166] | 0.154 | 0.456 [0.433, 0.479] | 0.489 |
| MLP-LIME | 0.055 [-0.017, 0.126] | 0.089 | 0.452 [0.428, 0.476] | 0.477 |
| MLP-KernelSHAP | 0.073 [-0.006, 0.147] | 0.043 | 0.460 [0.435, 0.487] | 0.463 |
| LR-LinearSHAP | 0.065 [-0.010, 0.136] | 0.004 | 0.454 [0.428, 0.483] | 0.437 |
| LR-LIME | 0.084 [0.009, 0.156] | -0.021 | 0.474 [0.442, 0.506] | 0.437 |
| RF-TreeSHAP | 0.456 [0.384, 0.521] | 0.479 | 0.536 [0.507, 0.564] | 0.551 |
| RF-LIME | 0.443 [0.377, 0.507] | 0.339 | 0.553 [0.522, 0.585] | 0.505 |
| XGB-TreeSHAP | 0.377 [0.313, 0.436] | 0.491 | 0.520 [0.495, 0.545] | 0.564 |
| XGB-LIME | 0.378 [0.315, 0.437] | 0.386 | 0.516 [0.485, 0.547] | 0.523 |
| random null (mean / 95th pct) | 0.001 / 0.373 |  | 0.439 / 0.592 |  |

## Metrics in the top-5 of at least 7 of the 10 projects

| explanation | in the top-5 of ≥ 7 of 10 projects |
|---|---|
| MLP-DeepLift | none |
| MLP-DeepLiftShap | none |
| MLP-GradientShap | none |
| MLP-IntegratedGradients | none |
| MLP-Saliency | none |
| MLP-InputXGradient | none |
| MLP-LIME | none |
| MLP-KernelSHAP | none |
| LR-LinearSHAP | loc (7) |
| LR-LIME | loc (7) |
| RF-TreeSHAP | rfc (8), loc (8) |
| RF-LIME | loc (9), amc (7) |
| XGB-TreeSHAP | loc (8) |
| XGB-LIME | loc (7) |

Full matrices: `step7_cross_project_matrices.csv`.
