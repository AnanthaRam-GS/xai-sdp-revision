# Step 7: seed stability and prediction multiplicity

All pairs of training seeds of each split (45 pairs for 10 seeds; 3 for LIME and KernelSHAP). Global: the stability metrics between the two seeds' importance vectors; instance: per-instance Spearman of |attribution|, feature agreement@5 and sign agreement@5, averaged over the test instances. Per-split means over seed pairs, then mean and 95% bootstrap CI over the 24 splits.

## Seed stability per explanation

| explanation | seeds | global ρ | global RBO | global top-5 Jaccard | instance ρ | instance FA@5 | instance SA@5 |
|---|---|---|---|---|---|---|---|
| MLP-DeepLift | 10 | 0.549 [0.472, 0.617] | 0.673 [0.632, 0.713] | 0.446 [0.388, 0.503] | 0.595 [0.564, 0.623] | 0.610 [0.584, 0.633] | 0.590 [0.556, 0.620] |
| MLP-DeepLiftShap | 10 | 0.577 [0.502, 0.645] | 0.683 [0.644, 0.721] | 0.472 [0.416, 0.525] | 0.601 [0.561, 0.636] | 0.637 [0.609, 0.663] | 0.625 [0.590, 0.656] |
| MLP-GradientShap | 10 | 0.575 [0.501, 0.643] | 0.684 [0.646, 0.722] | 0.467 [0.412, 0.521] | 0.547 [0.512, 0.579] | 0.604 [0.579, 0.629] | 0.592 [0.560, 0.621] |
| MLP-IntegratedGradients | 10 | 0.544 [0.468, 0.610] | 0.670 [0.631, 0.709] | 0.443 [0.387, 0.498] | 0.597 [0.566, 0.624] | 0.611 [0.586, 0.635] | 0.591 [0.557, 0.621] |
| MLP-Saliency | 10 | 0.527 [0.453, 0.595] | 0.664 [0.622, 0.704] | 0.425 [0.373, 0.475] | 0.365 [0.309, 0.416] | 0.486 [0.444, 0.525] | 0.481 [0.436, 0.523] |
| MLP-InputXGradient | 10 | 0.562 [0.479, 0.633] | 0.676 [0.632, 0.717] | 0.447 [0.385, 0.508] | 0.585 [0.556, 0.611] | 0.596 [0.572, 0.617] | 0.571 [0.538, 0.600] |
| MLP-LIME | 3 | 0.570 [0.470, 0.660] | 0.683 [0.628, 0.735] | 0.468 [0.380, 0.557] | 0.491 [0.406, 0.570] | 0.560 [0.496, 0.621] | 0.557 [0.490, 0.620] |
| MLP-KernelSHAP | 3 | 0.612 [0.525, 0.690] | 0.701 [0.656, 0.745] | 0.496 [0.423, 0.572] | 0.639 [0.588, 0.685] | 0.666 [0.629, 0.703] | 0.653 [0.607, 0.696] |
| LR-LinearSHAP | 10 | 0.792 [0.752, 0.832] | 0.815 [0.788, 0.840] | 0.676 [0.629, 0.720] | 0.824 [0.796, 0.851] | 0.793 [0.770, 0.818] | 0.781 [0.750, 0.811] |
| LR-LIME | 3 | 0.786 [0.713, 0.852] | 0.818 [0.777, 0.855] | 0.670 [0.591, 0.745] | 0.784 [0.711, 0.850] | 0.769 [0.710, 0.821] | 0.766 [0.704, 0.820] |
| RF-TreeSHAP | 10 | 0.923 [0.912, 0.932] | 0.879 [0.865, 0.893] | 0.772 [0.727, 0.819] | 0.806 [0.788, 0.823] | 0.785 [0.768, 0.804] | 0.784 [0.766, 0.803] |
| RF-LIME | 3 | 0.896 [0.877, 0.914] | 0.864 [0.837, 0.886] | 0.769 [0.708, 0.825] | 0.801 [0.774, 0.824] | 0.809 [0.782, 0.832] | 0.807 [0.777, 0.831] |
| XGB-TreeSHAP | 10 | 0.833 [0.810, 0.855] | 0.820 [0.802, 0.836] | 0.644 [0.605, 0.685] | 0.727 [0.704, 0.751] | 0.701 [0.676, 0.724] | 0.697 [0.671, 0.721] |
| XGB-LIME | 3 | 0.808 [0.769, 0.845] | 0.804 [0.775, 0.831] | 0.620 [0.554, 0.685] | 0.724 [0.692, 0.755] | 0.724 [0.694, 0.750] | 0.723 [0.693, 0.750] |

## Prediction multiplicity across the 10 seeds (mean over splits)

| model | share of instances with ≥ 2 different labels | mean SD of p | mean pairwise Cohen's κ |
|---|---|---|---|
| MLP | 0.353 | 0.072 | 0.678 |
| LR | 0.237 | 0.056 | 0.781 |
| RF | 0.262 | 0.053 | 0.783 |
| XGB | 0.332 | 0.069 | 0.676 |

## Model comparison (per-split values; Wilcoxon signed-rank, Holm within measure, Cliff's delta)

| measure | a | b | n | mean_a | mean_b | p_value | p_holm | cliffs_delta | magnitude |
|---|---|---|---|---|---|---|---|---|---|
| global ρ | MLP-DeepLiftShap | LR-LinearSHAP | 24 | 0.577 | 0.792 | <0.001 | <0.001 | -0.743 | large |
| global ρ | MLP-DeepLiftShap | RF-TreeSHAP | 24 | 0.577 | 0.923 | <0.001 | <0.001 | -1.000 | large |
| global ρ | MLP-DeepLiftShap | XGB-TreeSHAP | 24 | 0.577 | 0.833 | <0.001 | <0.001 | -0.951 | large |
| global ρ | LR-LinearSHAP | RF-TreeSHAP | 24 | 0.792 | 0.923 | <0.001 | <0.001 | -0.753 | large |
| global ρ | LR-LinearSHAP | XGB-TreeSHAP | 24 | 0.792 | 0.833 | 0.084 | 0.084 | -0.222 | small |
| global ρ | RF-TreeSHAP | XGB-TreeSHAP | 24 | 0.923 | 0.833 | <0.001 | <0.001 | 0.840 | large |
| instance ρ | MLP-DeepLiftShap | LR-LinearSHAP | 24 | 0.601 | 0.824 | <0.001 | <0.001 | -0.986 | large |
| instance ρ | MLP-DeepLiftShap | RF-TreeSHAP | 24 | 0.601 | 0.806 | <0.001 | <0.001 | -0.997 | large |
| instance ρ | MLP-DeepLiftShap | XGB-TreeSHAP | 24 | 0.601 | 0.727 | <0.001 | <0.001 | -0.781 | large |
| instance ρ | LR-LinearSHAP | RF-TreeSHAP | 24 | 0.824 | 0.806 | 0.345 | 0.345 | 0.167 | small |
| instance ρ | LR-LinearSHAP | XGB-TreeSHAP | 24 | 0.824 | 0.727 | <0.001 | 0.001 | 0.674 | large |
| instance ρ | RF-TreeSHAP | XGB-TreeSHAP | 24 | 0.806 | 0.727 | <0.001 | <0.001 | 0.691 | large |
| label disagreement share | MLP | LR | 24 | 0.353 | 0.237 | 0.002 | 0.013 | 0.465 | medium |
| label disagreement share | MLP | RF | 24 | 0.353 | 0.262 | 0.037 | 0.147 | 0.372 | medium |
| label disagreement share | MLP | XGB | 24 | 0.353 | 0.332 | 0.584 | 1.000 | 0.040 | negligible |
| label disagreement share | LR | RF | 24 | 0.237 | 0.262 | 0.855 | 1.000 | -0.167 | small |
| label disagreement share | LR | XGB | 24 | 0.237 | 0.332 | 0.136 | 0.407 | -0.372 | medium |
| label disagreement share | RF | XGB | 24 | 0.262 | 0.332 | 0.009 | 0.044 | -0.299 | small |
| mean Cohen's κ | MLP | LR | 24 | 0.678 | 0.781 | <0.001 | 0.005 | -0.497 | large |
| mean Cohen's κ | MLP | RF | 24 | 0.678 | 0.783 | 0.002 | 0.009 | -0.514 | large |
| mean Cohen's κ | MLP | XGB | 24 | 0.678 | 0.676 | 0.747 | 1.000 | 0.028 | negligible |
| mean Cohen's κ | LR | RF | 24 | 0.781 | 0.783 | 0.705 | 1.000 | 0.049 | negligible |
| mean Cohen's κ | LR | XGB | 24 | 0.781 | 0.676 | 0.004 | 0.012 | 0.483 | large |
| mean Cohen's κ | RF | XGB | 24 | 0.783 | 0.676 | <0.001 | 0.002 | 0.507 | large |

## Sources of variation

One row per source; values are mean Spearman ρ (higher = more stable). *Instance* rows compare the explanations of the same instance; *global* rows compare importance vectors. The step 5 rows use their own samples: LIME runs = 10 runs with different random states on up to 40 instances per split (seed-0 models); background draws = 5 independent stratified backgrounds of 100 rows (seed-0 models, all instances).

| source | level | MLP-DeepLiftShap | MLP-KernelSHAP | MLP-IntegratedGradients | MLP-LIME | MLP-Saliency | LR-LinearSHAP | RF-TreeSHAP | XGB-TreeSHAP |
|---|---|---|---|---|---|---|---|---|---|
| training seed | global | 0.577 | 0.612 | 0.544 | 0.570 | 0.527 | 0.792 | 0.923 | 0.833 |
| training seed | instance | 0.601 | 0.639 | 0.597 | 0.491 | 0.365 | 0.824 | 0.806 | 0.727 |
| LIME random state (step 5) | instance |  |  |  | 0.991 |  |  |  |  |
| background draw (step 5) | instance | 0.963 |  |  |  |  |  | 0.972 | 0.974 |
| release (consecutive splits) | global | 0.162 | 0.158 | 0.171 | 0.046 | 0.038 | 0.271 | 0.512 | 0.521 |
| project | global | 0.087 | 0.073 | 0.082 | 0.055 | 0.072 | 0.065 | 0.456 | 0.377 |
| random null (95th pct) | global | 0.373 | 0.373 | 0.373 | 0.373 | 0.373 | 0.373 | 0.373 | 0.373 |
