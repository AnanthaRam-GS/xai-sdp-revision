# Step 6: deletion and insertion curves

Seed-0 models of `main_time_aware`, all test instances of the 24 splits. Features are removed (deletion) or added back (insertion) in each explanation's per-instance order of |attribution| (ties by feature order), k = 0..20. Removal is *marginal*: the removed features take the values of 50 rows of the step 4 background set (seeded selection) and the model's P(defective) is averaged over them. The curves track confidence in the model's own prediction at its validation-tuned threshold (p if predicted defective, 1 − p otherwise). Areas: trapezoid rule over k/20. Random = mean over 10 seeded random orderings per instance. Gains: deletion AUC_random − AUC and insertion AUC − AUC_random (higher is better for both). Mean and 95% bootstrap CI over instances.

## Main results

| explanation | instances | deletion AUC ↓ | random | deletion gain ↑ | insertion AUC ↑ | random  | insertion gain ↑ |
|---|---|---|---|---|---|---|---|
| MLP-DeepLift | 9411 | 0.554 [0.552, 0.556] | 0.605 | 0.051 [0.050, 0.053] | 0.654 [0.651, 0.657] | 0.605 | 0.049 [0.048, 0.051] |
| MLP-DeepLiftShap | 9411 | 0.545 [0.543, 0.547] | 0.605 | 0.060 [0.059, 0.062] | 0.661 [0.658, 0.664] | 0.605 | 0.056 [0.055, 0.057] |
| MLP-GradientShap | 9411 | 0.547 [0.545, 0.549] | 0.605 | 0.058 [0.057, 0.060] | 0.659 [0.656, 0.662] | 0.605 | 0.054 [0.053, 0.055] |
| MLP-IntegratedGradients | 9411 | 0.553 [0.552, 0.555] | 0.605 | 0.052 [0.051, 0.053] | 0.655 [0.652, 0.658] | 0.605 | 0.050 [0.049, 0.051] |
| MLP-Saliency | 9411 | 0.564 [0.562, 0.566] | 0.605 | 0.041 [0.040, 0.043] | 0.646 [0.643, 0.649] | 0.605 | 0.041 [0.040, 0.042] |
| MLP-InputXGradient | 9411 | 0.555 [0.553, 0.557] | 0.605 | 0.050 [0.049, 0.052] | 0.653 [0.650, 0.656] | 0.605 | 0.048 [0.047, 0.049] |
| MLP-LIME | 9411 | 0.559 [0.558, 0.561] | 0.605 | 0.046 [0.045, 0.047] | 0.651 [0.648, 0.654] | 0.605 | 0.046 [0.045, 0.047] |
| MLP-KernelSHAP | 9411 | 0.544 [0.543, 0.546] | 0.605 | 0.061 [0.059, 0.062] | 0.661 [0.658, 0.664] | 0.605 | 0.056 [0.055, 0.058] |
| LR-LinearSHAP | 9411 | 0.538 [0.537, 0.540] | 0.586 | 0.047 [0.046, 0.049] | 0.636 [0.633, 0.639] | 0.585 | 0.051 [0.049, 0.053] |
| LR-LIME | 9411 | 0.541 [0.540, 0.543] | 0.586 | 0.044 [0.043, 0.046] | 0.634 [0.631, 0.637] | 0.585 | 0.049 [0.047, 0.051] |
| RF-TreeSHAP | 9411 | 0.567 [0.565, 0.570] | 0.630 | 0.063 [0.062, 0.064] | 0.694 [0.692, 0.697] | 0.630 | 0.064 [0.063, 0.065] |
| RF-LIME | 9411 | 0.589 [0.587, 0.592] | 0.630 | 0.041 [0.040, 0.042] | 0.674 [0.671, 0.676] | 0.630 | 0.043 [0.042, 0.044] |
| XGB-TreeSHAP | 9411 | 0.564 [0.561, 0.567] | 0.631 | 0.067 [0.065, 0.068] | 0.694 [0.690, 0.698] | 0.631 | 0.063 [0.061, 0.065] |
| XGB-LIME | 9411 | 0.585 [0.582, 0.588] | 0.631 | 0.046 [0.044, 0.047] | 0.675 [0.671, 0.678] | 0.631 | 0.044 [0.042, 0.045] |

## Scott-Knott ESD ranks within each model (per-split means; 1 = best)

Ranked on the gains over random: within a model every explanation shares the same random reference, so the gains order the methods exactly as the AUCs do, but without the between-split variation of the random level (Scott-Knott ESD is not paired by split; on the raw AUCs all methods of a model fall into one group).

| explanation | deletion gain | insertion gain |
|---|---|---|
| MLP-DeepLift | 2 | 1 |
| MLP-DeepLiftShap | 1 | 1 |
| MLP-GradientShap | 1 | 1 |
| MLP-IntegratedGradients | 2 | 1 |
| MLP-Saliency | 2 | 1 |
| MLP-InputXGradient | 2 | 1 |
| MLP-LIME | 2 | 1 |
| MLP-KernelSHAP | 1 | 1 |
| LR-LinearSHAP | 1 | 1 |
| LR-LIME | 1 | 1 |
| RF-TreeSHAP | 1 | 1 |
| RF-LIME | 2 | 2 |
| XGB-TreeSHAP | 1 | 1 |
| XGB-LIME | 2 | 2 |

## Scott-Knott ESD ranks across all 14 explanations (gain over random)

| explanation | deletion gain | insertion gain |
|---|---|---|
| MLP-DeepLift | 2 | 2 |
| MLP-DeepLiftShap | 1 | 1 |
| MLP-GradientShap | 1 | 1 |
| MLP-IntegratedGradients | 2 | 2 |
| MLP-Saliency | 2 | 2 |
| MLP-InputXGradient | 2 | 2 |
| MLP-LIME | 2 | 2 |
| MLP-KernelSHAP | 1 | 1 |
| LR-LinearSHAP | 2 | 2 |
| LR-LIME | 2 | 2 |
| RF-TreeSHAP | 1 | 1 |
| RF-LIME | 2 | 2 |
| XGB-TreeSHAP | 1 | 1 |
| XGB-LIME | 2 | 2 |

## Gains by outcome group (mean over instances)

| explanation | deletion_gain TP | deletion_gain FP | deletion_gain FN | deletion_gain TN | insertion_gain TP | insertion_gain FP | insertion_gain FN | insertion_gain TN |
|---|---|---|---|---|---|---|---|---|
| MLP-DeepLift | 0.075 | 0.072 | 0.024 | 0.043 | 0.074 | 0.071 | 0.023 | 0.041 |
| MLP-DeepLiftShap | 0.095 | 0.087 | 0.027 | 0.048 | 0.089 | 0.081 | 0.026 | 0.044 |
| MLP-GradientShap | 0.092 | 0.085 | 0.026 | 0.046 | 0.086 | 0.079 | 0.025 | 0.042 |
| MLP-IntegratedGradients | 0.077 | 0.073 | 0.024 | 0.043 | 0.075 | 0.071 | 0.024 | 0.041 |
| MLP-Saliency | 0.054 | 0.052 | 0.020 | 0.040 | 0.051 | 0.051 | 0.021 | 0.039 |
| MLP-InputXGradient | 0.071 | 0.068 | 0.025 | 0.044 | 0.069 | 0.067 | 0.024 | 0.041 |
| MLP-LIME | 0.067 | 0.062 | 0.020 | 0.040 | 0.066 | 0.060 | 0.021 | 0.041 |
| MLP-KernelSHAP | 0.097 | 0.087 | 0.027 | 0.048 | 0.090 | 0.083 | 0.026 | 0.044 |
| LR-LinearSHAP | 0.082 | 0.072 | 0.018 | 0.032 | 0.086 | 0.076 | 0.022 | 0.035 |
| LR-LIME | 0.072 | 0.054 | 0.022 | 0.036 | 0.078 | 0.060 | 0.025 | 0.040 |
| RF-TreeSHAP | 0.097 | 0.090 | 0.035 | 0.049 | 0.098 | 0.089 | 0.037 | 0.050 |
| RF-LIME | 0.050 | 0.052 | 0.024 | 0.039 | 0.051 | 0.053 | 0.027 | 0.041 |
| XGB-TreeSHAP | 0.110 | 0.073 | 0.038 | 0.058 | 0.110 | 0.072 | 0.035 | 0.052 |
| XGB-LIME | 0.064 | 0.044 | 0.024 | 0.047 | 0.064 | 0.046 | 0.025 | 0.042 |

## Deletion gain per project

| explanation | deletion_gain ant | deletion_gain camel | deletion_gain ivy | deletion_gain jedit | deletion_gain lucene | deletion_gain poi | deletion_gain synapse | deletion_gain velocity | deletion_gain xalan | deletion_gain xerces |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP-DeepLift | 0.072 | 0.032 | 0.055 | 0.050 | 0.042 | 0.048 | 0.036 | 0.046 | 0.054 | 0.070 |
| MLP-DeepLiftShap | 0.081 | 0.035 | 0.068 | 0.064 | 0.050 | 0.065 | 0.032 | 0.055 | 0.057 | 0.088 |
| MLP-GradientShap | 0.079 | 0.033 | 0.066 | 0.062 | 0.050 | 0.062 | 0.031 | 0.053 | 0.057 | 0.084 |
| MLP-IntegratedGradients | 0.072 | 0.033 | 0.055 | 0.052 | 0.043 | 0.049 | 0.036 | 0.046 | 0.055 | 0.071 |
| MLP-Saliency | 0.069 | 0.029 | 0.039 | 0.035 | 0.029 | 0.031 | 0.027 | 0.035 | 0.043 | 0.056 |
| MLP-InputXGradient | 0.075 | 0.031 | 0.050 | 0.050 | 0.043 | 0.044 | 0.037 | 0.041 | 0.053 | 0.067 |
| MLP-LIME | 0.069 | 0.029 | 0.045 | 0.042 | 0.033 | 0.046 | 0.028 | 0.044 | 0.045 | 0.065 |
| MLP-KernelSHAP | 0.084 | 0.034 | 0.068 | 0.064 | 0.052 | 0.067 | 0.033 | 0.056 | 0.058 | 0.086 |
| LR-LinearSHAP | 0.063 | 0.030 | 0.038 | 0.059 | 0.023 | 0.050 | 0.038 | 0.014 | 0.068 | 0.045 |
| LR-LIME | 0.060 | 0.030 | 0.026 | 0.059 | 0.016 | 0.052 | 0.030 | 0.012 | 0.060 | 0.046 |
| RF-TreeSHAP | 0.070 | 0.047 | 0.073 | 0.066 | 0.050 | 0.059 | 0.049 | 0.071 | 0.075 | 0.071 |
| RF-LIME | 0.060 | 0.022 | 0.059 | 0.048 | 0.033 | 0.031 | 0.026 | 0.033 | 0.044 | 0.051 |
| XGB-TreeSHAP | 0.063 | 0.049 | 0.050 | 0.062 | 0.052 | 0.085 | 0.083 | 0.051 | 0.088 | 0.076 |
| XGB-LIME | 0.056 | 0.024 | 0.037 | 0.047 | 0.030 | 0.059 | 0.062 | 0.023 | 0.059 | 0.047 |

## Insertion gain per project

| explanation | insertion_gain ant | insertion_gain camel | insertion_gain ivy | insertion_gain jedit | insertion_gain lucene | insertion_gain poi | insertion_gain synapse | insertion_gain velocity | insertion_gain xalan | insertion_gain xerces |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP-DeepLift | 0.064 | 0.038 | 0.048 | 0.050 | 0.040 | 0.047 | 0.032 | 0.044 | 0.054 | 0.063 |
| MLP-DeepLiftShap | 0.069 | 0.041 | 0.061 | 0.060 | 0.049 | 0.061 | 0.027 | 0.049 | 0.056 | 0.074 |
| MLP-GradientShap | 0.067 | 0.038 | 0.059 | 0.058 | 0.048 | 0.059 | 0.027 | 0.048 | 0.055 | 0.071 |
| MLP-IntegratedGradients | 0.064 | 0.040 | 0.048 | 0.051 | 0.040 | 0.048 | 0.032 | 0.044 | 0.055 | 0.063 |
| MLP-Saliency | 0.065 | 0.031 | 0.033 | 0.038 | 0.029 | 0.030 | 0.024 | 0.035 | 0.044 | 0.054 |
| MLP-InputXGradient | 0.065 | 0.035 | 0.043 | 0.050 | 0.040 | 0.042 | 0.033 | 0.039 | 0.053 | 0.062 |
| MLP-LIME | 0.066 | 0.036 | 0.038 | 0.045 | 0.034 | 0.045 | 0.025 | 0.045 | 0.046 | 0.059 |
| MLP-KernelSHAP | 0.070 | 0.041 | 0.061 | 0.060 | 0.050 | 0.063 | 0.028 | 0.050 | 0.057 | 0.074 |
| LR-LinearSHAP | 0.058 | 0.044 | 0.037 | 0.063 | 0.021 | 0.056 | 0.039 | 0.014 | 0.071 | 0.052 |
| LR-LIME | 0.056 | 0.046 | 0.027 | 0.062 | 0.016 | 0.059 | 0.033 | 0.012 | 0.064 | 0.056 |
| RF-TreeSHAP | 0.069 | 0.049 | 0.072 | 0.069 | 0.052 | 0.060 | 0.048 | 0.070 | 0.074 | 0.072 |
| RF-LIME | 0.061 | 0.025 | 0.059 | 0.050 | 0.037 | 0.037 | 0.026 | 0.034 | 0.045 | 0.052 |
| XGB-TreeSHAP | 0.057 | 0.045 | 0.049 | 0.059 | 0.049 | 0.085 | 0.086 | 0.049 | 0.086 | 0.064 |
| XGB-LIME | 0.047 | 0.023 | 0.036 | 0.043 | 0.030 | 0.061 | 0.065 | 0.023 | 0.059 | 0.046 |

## Sensitivity: mean removal (deletion only)

Removed features replaced by the background mean instead of averaging over background rows.

| explanation | deletion AUC (marginal) | deletion AUC (mean removal) | gain (marginal) | gain (mean removal) |
|---|---|---|---|---|
| MLP-DeepLift | 0.554 | 0.550 | 0.051 | 0.061 |
| MLP-DeepLiftShap | 0.545 | 0.541 | 0.060 | 0.071 |
| MLP-GradientShap | 0.547 | 0.543 | 0.058 | 0.068 |
| MLP-IntegratedGradients | 0.553 | 0.550 | 0.052 | 0.061 |
| MLP-Saliency | 0.564 | 0.562 | 0.041 | 0.049 |
| MLP-InputXGradient | 0.555 | 0.551 | 0.050 | 0.061 |
| MLP-LIME | 0.559 | 0.559 | 0.046 | 0.052 |
| MLP-KernelSHAP | 0.544 | 0.540 | 0.061 | 0.072 |
| LR-LinearSHAP | 0.538 | 0.542 | 0.047 | 0.050 |
| LR-LIME | 0.541 | 0.546 | 0.044 | 0.046 |
| RF-TreeSHAP | 0.567 | 0.554 | 0.063 | 0.069 |
| RF-LIME | 0.589 | 0.570 | 0.041 | 0.054 |
| XGB-TreeSHAP | 0.564 | 0.573 | 0.067 | 0.067 |
| XGB-LIME | 0.585 | 0.591 | 0.046 | 0.050 |
