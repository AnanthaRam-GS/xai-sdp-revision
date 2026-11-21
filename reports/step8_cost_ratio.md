# Step 8: NECM cost-ratio sweep

NECM_c = (FP + c·FN)/N on the test releases of the 24 time-aware splits (`main_time_aware` predictions), seeds averaged per split; mean and 95% bootstrap CI over splits. Thresholds: 0.5; the validation-MCC threshold stored in step 3 (all models); and for the learned models the cost-aware threshold minimising NECM_c on the validation split. Validation predictions come from the persisted step 3 models by inference only; they reproduce the stored thresholds and test predictions exactly (maximum differences 0.0e+00 and 0.0e+00 over 960 models).

Note: the Majority baseline outputs a constant score per split (the training defective rate). Its validation-MCC threshold equals that score and predictions are p ≥ t, so at the tuned threshold it flags every class and coincides with all_defective; at 0.5 it flags none (step 3 behaviour, unchanged).

## NECM at the validation-MCC threshold

| model | c = 1 | c = 2 | c = 3 | c = 5 | c = 7.5 | c = 10 | c = 15 | c = 20 |
|---|---|---|---|---|---|---|---|---|
| MLP | 0.347 [0.306, 0.389] | 0.493 [0.427, 0.568] | 0.639 [0.537, 0.759] | 0.931 [0.748, 1.151] | 1.296 [1.015, 1.644] | 1.661 [1.271, 2.134] | 2.391 [1.789, 3.113] | 3.121 [2.303, 4.095] |
| LR | 0.342 [0.304, 0.381] | 0.470 [0.410, 0.541] | 0.598 [0.507, 0.709] | 0.854 [0.696, 1.056] | 1.173 [0.920, 1.490] | 1.493 [1.144, 1.929] | 2.132 [1.591, 2.794] | 2.772 [2.038, 3.666] |
| RF | 0.351 [0.306, 0.397] | 0.487 [0.416, 0.569] | 0.624 [0.513, 0.759] | 0.897 [0.701, 1.146] | 1.238 [0.942, 1.633] | 1.579 [1.176, 2.123] | 2.262 [1.637, 3.097] | 2.944 [2.096, 4.063] |
| XGB | 0.361 [0.315, 0.407] | 0.510 [0.437, 0.591] | 0.659 [0.549, 0.794] | 0.957 [0.759, 1.210] | 1.329 [1.015, 1.729] | 1.702 [1.271, 2.245] | 2.447 [1.786, 3.277] | 3.193 [2.300, 4.311] |
| ManualUp | 0.584 [0.504, 0.660] | 0.671 [0.597, 0.739] | 0.758 [0.668, 0.846] | 0.932 [0.780, 1.096] | 1.150 [0.900, 1.426] | 1.368 [1.017, 1.767] | 1.804 [1.247, 2.448] | 2.239 [1.477, 3.131] |
| ManualDown | 0.335 [0.282, 0.397] | 0.485 [0.398, 0.594] | 0.635 [0.506, 0.803] | 0.935 [0.704, 1.228] | 1.310 [0.952, 1.762] | 1.686 [1.197, 2.294] | 2.436 [1.682, 3.350] | 3.187 [2.171, 4.410] |
| all_defective | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] |
| majority | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] |

## NECM at the cost-aware threshold (learned models)

| model | c = 1 | c = 2 | c = 3 | c = 5 | c = 7.5 | c = 10 | c = 15 | c = 20 |
|---|---|---|---|---|---|---|---|---|
| MLP | 0.329 [0.277, 0.381] | 0.490 [0.412, 0.582] | 0.606 [0.512, 0.718] | 0.745 [0.630, 0.895] | 0.906 [0.739, 1.142] | 1.030 [0.816, 1.342] | 1.228 [0.924, 1.681] | 1.455 [1.046, 2.066] |
| LR | 0.326 [0.275, 0.378] | 0.482 [0.407, 0.573] | 0.611 [0.506, 0.734] | 0.708 [0.592, 0.860] | 0.856 [0.692, 1.077] | 0.976 [0.766, 1.282] | 1.137 [0.858, 1.568] | 1.349 [0.958, 1.943] |
| RF | 0.328 [0.271, 0.386] | 0.478 [0.401, 0.565] | 0.604 [0.509, 0.723] | 0.792 [0.662, 0.965] | 0.950 [0.778, 1.181] | 1.087 [0.852, 1.408] | 1.360 [0.997, 1.869] | 1.594 [1.089, 2.282] |
| XGB | 0.333 [0.274, 0.394] | 0.494 [0.418, 0.582] | 0.604 [0.512, 0.725] | 0.785 [0.645, 0.982] | 0.947 [0.759, 1.206] | 1.049 [0.824, 1.357] | 1.203 [0.903, 1.612] | 1.377 [0.998, 1.864] |

## NECM at the 0.5 threshold

| model | c = 1 | c = 2 | c = 3 | c = 5 | c = 7.5 | c = 10 | c = 15 | c = 20 |
|---|---|---|---|---|---|---|---|---|
| MLP | 0.366 [0.332, 0.403] | 0.486 [0.430, 0.552] | 0.606 [0.519, 0.712] | 0.846 [0.694, 1.037] | 1.146 [0.911, 1.445] | 1.446 [1.130, 1.852] | 2.046 [1.556, 2.663] | 2.647 [1.979, 3.479] |
| LR | 0.364 [0.328, 0.400] | 0.484 [0.424, 0.551] | 0.605 [0.513, 0.713] | 0.846 [0.687, 1.039] | 1.148 [0.902, 1.450] | 1.449 [1.114, 1.858] | 2.052 [1.539, 2.675] | 2.655 [1.963, 3.490] |
| RF | 0.341 [0.299, 0.382] | 0.473 [0.407, 0.554] | 0.606 [0.502, 0.735] | 0.871 [0.687, 1.106] | 1.202 [0.920, 1.565] | 1.533 [1.147, 2.020] | 2.194 [1.613, 2.937] | 2.856 [2.070, 3.858] |
| XGB | 0.339 [0.297, 0.380] | 0.475 [0.409, 0.555] | 0.611 [0.511, 0.738] | 0.882 [0.708, 1.110] | 1.222 [0.949, 1.574] | 1.562 [1.185, 2.037] | 2.241 [1.664, 2.963] | 2.921 [2.141, 3.887] |
| ManualUp | 0.334 [0.264, 0.410] | 0.645 [0.507, 0.800] | 0.957 [0.750, 1.191] | 1.581 [1.235, 1.975] | 2.360 [1.839, 2.954] | 3.140 [2.445, 3.933] | 4.699 [3.658, 5.889] | 6.259 [4.870, 7.844] |
| ManualDown | 0.310 [0.244, 0.389] | 0.609 [0.474, 0.768] | 0.907 [0.703, 1.146] | 1.504 [1.161, 1.904] | 2.249 [1.733, 2.852] | 2.995 [2.306, 3.800] | 4.487 [3.455, 5.695] | 5.978 [4.604, 7.590] |
| all_defective | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] | 0.686 [0.605, 0.757] |
| majority | 0.360 [0.271, 0.448] | 0.572 [0.460, 0.681] | 0.784 [0.622, 0.949] | 1.208 [0.918, 1.500] | 1.739 [1.286, 2.209] | 2.269 [1.650, 2.924] | 3.329 [2.378, 4.342] | 4.390 [3.087, 5.763] |

## Break-even cost ratio

The largest cost ratio c below which the model's mean NECM is lower than both all_defective's and ManualUp's (at their validation-MCC thresholds) at every smaller c (grid step 0.05, up to 30). `never` = not even at c = 0.05.

| model | cost-aware threshold | MCC-tuned threshold |
|---|---|---|
| LR | 4.55 | 3.70 |
| MLP | 4.15 | 3.35 |
| RF | 3.90 | 3.50 |
| XGB | 3.95 | 3.20 |

## Tests at c ∈ {1, 5, 15}

Per-split NECM (seeds averaged), Wilcoxon signed-rank, Holm within each c, Cliff's delta (negative = the learned model has the lower cost).

| c | a | b | n | mean_a | mean_b | p_value | p_holm | cliffs_delta | magnitude |
|---|---|---|---|---|---|---|---|---|---|
| 1 | MLP (val_mcc) | all_defective | 24 | 0.347 | 0.686 | <0.001 | <0.001 | -0.809 | large |
| 1 | MLP (val_mcc) | ManualUp | 24 | 0.347 | 0.584 | <0.001 | 0.002 | -0.674 | large |
| 1 | MLP (cost_aware) | all_defective | 24 | 0.329 | 0.686 | <0.001 | <0.001 | -0.837 | large |
| 1 | MLP (cost_aware) | ManualUp | 24 | 0.329 | 0.584 | <0.001 | <0.001 | -0.698 | large |
| 1 | LR (val_mcc) | all_defective | 24 | 0.342 | 0.686 | <0.001 | <0.001 | -0.840 | large |
| 1 | LR (val_mcc) | ManualUp | 24 | 0.342 | 0.584 | <0.001 | 0.001 | -0.708 | large |
| 1 | LR (cost_aware) | all_defective | 24 | 0.326 | 0.686 | <0.001 | <0.001 | -0.851 | large |
| 1 | LR (cost_aware) | ManualUp | 24 | 0.326 | 0.584 | <0.001 | <0.001 | -0.708 | large |
| 1 | RF (val_mcc) | all_defective | 24 | 0.351 | 0.686 | <0.001 | <0.001 | -0.833 | large |
| 1 | RF (val_mcc) | ManualUp | 24 | 0.351 | 0.584 | <0.001 | 0.001 | -0.674 | large |
| 1 | RF (cost_aware) | all_defective | 24 | 0.328 | 0.686 | <0.001 | <0.001 | -0.837 | large |
| 1 | RF (cost_aware) | ManualUp | 24 | 0.328 | 0.584 | <0.001 | <0.001 | -0.691 | large |
| 1 | XGB (val_mcc) | all_defective | 24 | 0.361 | 0.686 | <0.001 | <0.001 | -0.812 | large |
| 1 | XGB (val_mcc) | ManualUp | 24 | 0.361 | 0.584 | 0.003 | 0.003 | -0.658 | large |
| 1 | XGB (cost_aware) | all_defective | 24 | 0.333 | 0.686 | <0.001 | <0.001 | -0.844 | large |
| 1 | XGB (cost_aware) | ManualUp | 24 | 0.333 | 0.584 | <0.001 | 0.001 | -0.694 | large |
| 5 | MLP (val_mcc) | all_defective | 24 | 0.931 | 0.686 | 0.331 | 1.000 | 0.188 | small |
| 5 | MLP (val_mcc) | ManualUp | 24 | 0.931 | 0.932 | 0.623 | 1.000 | -0.101 | negligible |
| 5 | MLP (cost_aware) | all_defective | 24 | 0.745 | 0.686 | 0.877 | 1.000 | -0.062 | negligible |
| 5 | MLP (cost_aware) | ManualUp | 24 | 0.745 | 0.932 | 0.089 | 1.000 | -0.340 | medium |
| 5 | LR (val_mcc) | all_defective | 24 | 0.854 | 0.686 | 0.456 | 1.000 | 0.118 | negligible |
| 5 | LR (val_mcc) | ManualUp | 24 | 0.854 | 0.932 | 0.331 | 1.000 | -0.191 | small |
| 5 | LR (cost_aware) | all_defective | 24 | 0.708 | 0.686 | 0.422 | 1.000 | -0.122 | negligible |
| 5 | LR (cost_aware) | ManualUp | 24 | 0.708 | 0.932 | 0.027 | 0.431 | -0.385 | medium |
| 5 | RF (val_mcc) | all_defective | 24 | 0.897 | 0.686 | 0.584 | 1.000 | 0.135 | negligible |
| 5 | RF (val_mcc) | ManualUp | 24 | 0.897 | 0.932 | 0.456 | 1.000 | -0.142 | negligible |
| 5 | RF (cost_aware) | all_defective | 24 | 0.792 | 0.686 | 0.643 | 1.000 | 0.104 | negligible |
| 5 | RF (cost_aware) | ManualUp | 24 | 0.792 | 0.932 | 0.229 | 1.000 | -0.240 | small |
| 5 | XGB (val_mcc) | all_defective | 24 | 0.957 | 0.686 | 0.390 | 1.000 | 0.181 | small |
| 5 | XGB (val_mcc) | ManualUp | 24 | 0.957 | 0.932 | 0.768 | 1.000 | -0.101 | negligible |
| 5 | XGB (cost_aware) | all_defective | 24 | 0.785 | 0.686 | 0.833 | 1.000 | 0.056 | negligible |
| 5 | XGB (cost_aware) | ManualUp | 24 | 0.785 | 0.932 | 0.178 | 1.000 | -0.288 | small |
| 15 | MLP (val_mcc) | all_defective | 24 | 2.391 | 0.686 | <0.001 | <0.001 | 0.823 | large |
| 15 | MLP (val_mcc) | ManualUp | 24 | 2.391 | 1.804 | 0.229 | 1.000 | 0.267 | small |
| 15 | MLP (cost_aware) | all_defective | 24 | 1.228 | 0.686 | 0.002 | 0.021 | 0.573 | large |
| 15 | MLP (cost_aware) | ManualUp | 24 | 1.228 | 1.804 | 0.169 | 1.000 | -0.208 | small |
| 15 | LR (val_mcc) | all_defective | 24 | 2.132 | 0.686 | <0.001 | <0.001 | 0.788 | large |
| 15 | LR (val_mcc) | ManualUp | 24 | 2.132 | 1.804 | 0.456 | 1.000 | 0.240 | small |
| 15 | LR (cost_aware) | all_defective | 24 | 1.137 | 0.686 | 0.005 | 0.048 | 0.479 | large |
| 15 | LR (cost_aware) | ManualUp | 24 | 1.137 | 1.804 | 0.089 | 0.715 | -0.267 | small |
| 15 | RF (val_mcc) | all_defective | 24 | 2.262 | 0.686 | <0.001 | <0.001 | 0.847 | large |
| 15 | RF (val_mcc) | ManualUp | 24 | 2.262 | 1.804 | 0.360 | 1.000 | 0.222 | small |
| 15 | RF (cost_aware) | all_defective | 24 | 1.360 | 0.686 | 0.002 | 0.021 | 0.576 | large |
| 15 | RF (cost_aware) | ManualUp | 24 | 1.360 | 1.804 | 0.456 | 1.000 | -0.163 | small |
| 15 | XGB (val_mcc) | all_defective | 24 | 2.447 | 0.686 | <0.001 | <0.001 | 0.806 | large |
| 15 | XGB (val_mcc) | ManualUp | 24 | 2.447 | 1.804 | 0.208 | 1.000 | 0.250 | small |
| 15 | XGB (cost_aware) | all_defective | 24 | 1.203 | 0.686 | 0.004 | 0.035 | 0.497 | large |
| 15 | XGB (cost_aware) | ManualUp | 24 | 1.203 | 1.804 | 0.160 | 1.000 | -0.260 | small |
