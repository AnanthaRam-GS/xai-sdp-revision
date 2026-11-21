# Step 4: explainer agreement

Agreement metrics after Krishna et al. (2022), "The Disagreement Problem in Explainable Machine Learning", computed per test instance and pair of explanations of the same instance:

* feature agreement@k: share of the top-k features (by |attribution|) common to both;
* rank agreement@k: share of top-k positions holding the same feature in both;
* sign agreement@k: share of common top-k features with the same attribution sign;
* signed-rank agreement@k: same feature at the same position with the same sign;
* Spearman and Kendall (tau-b) rank correlation of |attribution| over all 20 features.

Ties in the top-k ordering are broken by the canonical feature order; rank correlations use average ranks. Seeds 0–2 (where all methods exist); each instance's agreement is averaged over the three seeds, and the mean and 95% percentile bootstrap CI (2000 resamples) are over instances. The CIs treat instances as independent; instances of one split share one model, so they understate between-split variation (per-project values are in the CSV). Saliency is compared as |gradient| for magnitudes and by the gradient's sign for the sign-based metrics.

**Note.** DeepLift and DeepLiftShap share the same propagation rules (DeepLiftShap averages DeepLift over the background baselines instead of using their mean), so their high agreement is expected and is not independent evidence. The same holds, to a lesser degree, for IntegratedGradients vs DeepLift (same baseline) and for InputXGradient vs Saliency (same gradient).

**Output scale.** Every explanation explains P(defective) except LR-LinearSHAP, which is exact on the log-odds scale (exact interventional SHAP on the probability scale is not linear). LR-LIME explains the probability. Because attributions are L1-normalised per instance before aggregation, and the comparisons are rank-based, the scale difference does not affect them.

**LIME weights are slopes.** With `discretize_continuous=False` (and LIME's default `sample_around_instance=False`), LIME fits a weighted linear surrogate to samples drawn around the training mean, and its weights are that surrogate's coefficients (per standardised unit). They are slopes, like Saliency's gradients, not contributions relative to a reference (slope × (x − reference)) as for the SHAP family, DeepLift, Integrated Gradients and InputXGradient. They also vary little between instances: the median coefficient of variation across the test instances of a split is about 0.1–0.2 for LIME and above 1 for TreeSHAP, DeepLiftShap and Saliency (checked on ant 1.5→1.6, seed 0). LIME's sign therefore reflects the slope, not whether the instance's value raises its prediction.
This explains why LIME's sign agreement is highest with Saliency (both are slopes) and low with the contribution-type methods.

## Within MLP, k = 5 (mean over instances, ALL)

### feature_agreement@5

| MLP | DeepLift | DeepLiftShap | GradientShap | IntegratedGradients | Saliency | InputXGradient | LIME | KernelSHAP |
|---|---|---|---|---|---|---|---|---|
| DeepLift | 1.000 | 0.801 | 0.772 | 0.972 | 0.590 | 0.861 | 0.518 | 0.776 |
| DeepLiftShap | 0.801 | 1.000 | 0.865 | 0.802 | 0.554 | 0.757 | 0.574 | 0.919 |
| GradientShap | 0.772 | 0.865 | 1.000 | 0.774 | 0.549 | 0.734 | 0.565 | 0.852 |
| IntegratedGradients | 0.972 | 0.802 | 0.774 | 1.000 | 0.585 | 0.849 | 0.521 | 0.779 |
| Saliency | 0.590 | 0.554 | 0.549 | 0.585 | 1.000 | 0.615 | 0.629 | 0.538 |
| InputXGradient | 0.861 | 0.757 | 0.734 | 0.849 | 0.615 | 1.000 | 0.490 | 0.733 |
| LIME | 0.518 | 0.574 | 0.565 | 0.521 | 0.629 | 0.490 | 1.000 | 0.589 |
| KernelSHAP | 0.776 | 0.919 | 0.852 | 0.779 | 0.538 | 0.733 | 0.589 | 1.000 |

### rank_agreement@5

| MLP | DeepLift | DeepLiftShap | GradientShap | IntegratedGradients | Saliency | InputXGradient | LIME | KernelSHAP |
|---|---|---|---|---|---|---|---|---|
| DeepLift | 1.000 | 0.414 | 0.371 | 0.871 | 0.166 | 0.521 | 0.134 | 0.370 |
| DeepLiftShap | 0.414 | 1.000 | 0.523 | 0.416 | 0.153 | 0.350 | 0.162 | 0.665 |
| GradientShap | 0.371 | 0.523 | 1.000 | 0.374 | 0.151 | 0.322 | 0.158 | 0.491 |
| IntegratedGradients | 0.871 | 0.416 | 0.374 | 1.000 | 0.163 | 0.493 | 0.135 | 0.373 |
| Saliency | 0.166 | 0.153 | 0.151 | 0.163 | 1.000 | 0.178 | 0.215 | 0.147 |
| InputXGradient | 0.521 | 0.350 | 0.322 | 0.493 | 0.178 | 1.000 | 0.124 | 0.315 |
| LIME | 0.134 | 0.162 | 0.158 | 0.135 | 0.215 | 0.124 | 1.000 | 0.168 |
| KernelSHAP | 0.370 | 0.665 | 0.491 | 0.373 | 0.147 | 0.315 | 0.168 | 1.000 |

### sign_agreement@5

| MLP | DeepLift | DeepLiftShap | GradientShap | IntegratedGradients | Saliency | InputXGradient | LIME | KernelSHAP |
|---|---|---|---|---|---|---|---|---|
| DeepLift | 1.000 | 0.801 | 0.771 | 0.972 | 0.277 | 0.860 | 0.247 | 0.775 |
| DeepLiftShap | 0.801 | 1.000 | 0.865 | 0.802 | 0.259 | 0.756 | 0.275 | 0.919 |
| GradientShap | 0.771 | 0.865 | 1.000 | 0.773 | 0.258 | 0.732 | 0.271 | 0.852 |
| IntegratedGradients | 0.972 | 0.802 | 0.773 | 1.000 | 0.275 | 0.848 | 0.249 | 0.778 |
| Saliency | 0.277 | 0.259 | 0.258 | 0.275 | 1.000 | 0.288 | 0.629 | 0.251 |
| InputXGradient | 0.860 | 0.756 | 0.732 | 0.848 | 0.288 | 1.000 | 0.231 | 0.730 |
| LIME | 0.247 | 0.275 | 0.271 | 0.249 | 0.629 | 0.231 | 1.000 | 0.283 |
| KernelSHAP | 0.775 | 0.919 | 0.852 | 0.778 | 0.251 | 0.730 | 0.283 | 1.000 |

### signed_rank_agreement@5

| MLP | DeepLift | DeepLiftShap | GradientShap | IntegratedGradients | Saliency | InputXGradient | LIME | KernelSHAP |
|---|---|---|---|---|---|---|---|---|
| DeepLift | 1.000 | 0.414 | 0.371 | 0.871 | 0.076 | 0.521 | 0.066 | 0.369 |
| DeepLiftShap | 0.414 | 1.000 | 0.523 | 0.416 | 0.068 | 0.350 | 0.079 | 0.665 |
| GradientShap | 0.371 | 0.523 | 1.000 | 0.374 | 0.069 | 0.322 | 0.076 | 0.491 |
| IntegratedGradients | 0.871 | 0.416 | 0.374 | 1.000 | 0.075 | 0.492 | 0.067 | 0.373 |
| Saliency | 0.076 | 0.068 | 0.069 | 0.075 | 1.000 | 0.081 | 0.215 | 0.066 |
| InputXGradient | 0.521 | 0.350 | 0.322 | 0.492 | 0.081 | 1.000 | 0.060 | 0.314 |
| LIME | 0.066 | 0.079 | 0.076 | 0.067 | 0.215 | 0.060 | 1.000 | 0.082 |
| KernelSHAP | 0.369 | 0.665 | 0.491 | 0.373 | 0.066 | 0.314 | 0.082 | 1.000 |

### spearman

| MLP | DeepLift | DeepLiftShap | GradientShap | IntegratedGradients | Saliency | InputXGradient | LIME | KernelSHAP |
|---|---|---|---|---|---|---|---|---|
| DeepLift | 1.000 | 0.799 | 0.763 | 0.988 | 0.633 | 0.900 | 0.458 | 0.775 |
| DeepLiftShap | 0.799 | 1.000 | 0.899 | 0.799 | 0.525 | 0.745 | 0.566 | 0.947 |
| GradientShap | 0.763 | 0.899 | 1.000 | 0.765 | 0.512 | 0.714 | 0.549 | 0.881 |
| IntegratedGradients | 0.988 | 0.799 | 0.765 | 1.000 | 0.621 | 0.888 | 0.463 | 0.778 |
| Saliency | 0.633 | 0.525 | 0.512 | 0.621 | 1.000 | 0.707 | 0.579 | 0.490 |
| InputXGradient | 0.900 | 0.745 | 0.714 | 0.888 | 0.707 | 1.000 | 0.404 | 0.721 |
| LIME | 0.458 | 0.566 | 0.549 | 0.463 | 0.579 | 0.404 | 1.000 | 0.610 |
| KernelSHAP | 0.775 | 0.947 | 0.881 | 0.778 | 0.490 | 0.721 | 0.610 | 1.000 |

### kendall

| MLP | DeepLift | DeepLiftShap | GradientShap | IntegratedGradients | Saliency | InputXGradient | LIME | KernelSHAP |
|---|---|---|---|---|---|---|---|---|
| DeepLift | 1.000 | 0.649 | 0.608 | 0.956 | 0.480 | 0.778 | 0.333 | 0.620 |
| DeepLiftShap | 0.649 | 1.000 | 0.770 | 0.649 | 0.386 | 0.590 | 0.422 | 0.852 |
| GradientShap | 0.608 | 0.770 | 1.000 | 0.610 | 0.375 | 0.558 | 0.408 | 0.745 |
| IntegratedGradients | 0.956 | 0.649 | 0.610 | 1.000 | 0.468 | 0.759 | 0.337 | 0.624 |
| Saliency | 0.480 | 0.386 | 0.375 | 0.468 | 1.000 | 0.557 | 0.433 | 0.357 |
| InputXGradient | 0.778 | 0.590 | 0.558 | 0.759 | 0.557 | 1.000 | 0.291 | 0.564 |
| LIME | 0.333 | 0.422 | 0.408 | 0.337 | 0.433 | 0.291 | 1.000 | 0.459 |
| KernelSHAP | 0.620 | 0.852 | 0.745 | 0.624 | 0.357 | 0.564 | 0.459 | 1.000 |

## Cross-model: MLP-DeepLiftShap, RF-TreeSHAP, XGB-TreeSHAP, LR-LinearSHAP (ALL)

Different models explain the same instances; agreement here mixes explainer and model disagreement. Mean [95% CI].

| pair | n | feature_agreement@5 | rank_agreement@5 | sign_agreement@5 | signed_rank_agreement@5 | spearman | kendall |
|---|---|---|---|---|---|---|---|
| MLP-DeepLiftShap vs LR-LinearSHAP | 9411 | 0.592 [0.589, 0.595] | 0.190 [0.187, 0.192] | 0.568 [0.565, 0.571] | 0.184 [0.182, 0.187] | 0.558 [0.555, 0.560] | 0.416 [0.414, 0.418] |
| MLP-DeepLiftShap vs RF-TreeSHAP | 9411 | 0.468 [0.465, 0.471] | 0.129 [0.127, 0.131] | 0.450 [0.447, 0.453] | 0.125 [0.123, 0.127] | 0.350 [0.346, 0.354] | 0.251 [0.249, 0.254] |
| MLP-DeepLiftShap vs XGB-TreeSHAP | 9411 | 0.447 [0.444, 0.450] | 0.126 [0.124, 0.128] | 0.428 [0.425, 0.432] | 0.122 [0.120, 0.124] | 0.307 [0.303, 0.311] | 0.220 [0.218, 0.223] |
| RF-TreeSHAP vs LR-LinearSHAP | 9411 | 0.464 [0.461, 0.468] | 0.128 [0.125, 0.130] | 0.421 [0.417, 0.424] | 0.118 [0.115, 0.120] | 0.338 [0.333, 0.342] | 0.244 [0.240, 0.247] |
| RF-TreeSHAP vs XGB-TreeSHAP | 9411 | 0.659 [0.656, 0.662] | 0.254 [0.251, 0.257] | 0.653 [0.650, 0.656] | 0.253 [0.250, 0.255] | 0.659 [0.656, 0.661] | 0.507 [0.505, 0.510] |
| XGB-TreeSHAP vs LR-LinearSHAP | 9411 | 0.425 [0.422, 0.428] | 0.121 [0.119, 0.123] | 0.392 [0.389, 0.396] | 0.114 [0.112, 0.116] | 0.281 [0.277, 0.285] | 0.202 [0.199, 0.205] |

## LIME across the four models (ALL)

| pair | n | feature_agreement@5 | rank_agreement@5 | sign_agreement@5 | signed_rank_agreement@5 | spearman | kendall |
|---|---|---|---|---|---|---|---|
| LR-LIME vs RF-LIME | 9411 | 0.539 [0.535, 0.543] | 0.156 [0.154, 0.159] | 0.529 [0.525, 0.533] | 0.154 [0.151, 0.156] | 0.387 [0.381, 0.394] | 0.286 [0.281, 0.291] |
| LR-LIME vs XGB-LIME | 9411 | 0.504 [0.502, 0.507] | 0.174 [0.172, 0.176] | 0.493 [0.490, 0.496] | 0.170 [0.168, 0.172] | 0.363 [0.358, 0.368] | 0.265 [0.262, 0.269] |
| MLP-LIME vs LR-LIME | 9411 | 0.537 [0.535, 0.540] | 0.168 [0.165, 0.170] | 0.535 [0.533, 0.537] | 0.166 [0.163, 0.168] | 0.514 [0.512, 0.517] | 0.380 [0.378, 0.381] |
| MLP-LIME vs RF-LIME | 9411 | 0.496 [0.492, 0.500] | 0.183 [0.180, 0.186] | 0.493 [0.489, 0.497] | 0.183 [0.180, 0.186] | 0.409 [0.404, 0.413] | 0.295 [0.292, 0.298] |
| MLP-LIME vs XGB-LIME | 9411 | 0.503 [0.500, 0.506] | 0.161 [0.159, 0.163] | 0.498 [0.494, 0.501] | 0.160 [0.158, 0.162] | 0.392 [0.387, 0.396] | 0.283 [0.280, 0.286] |
| RF-LIME vs XGB-LIME | 9411 | 0.703 [0.701, 0.705] | 0.302 [0.299, 0.305] | 0.702 [0.700, 0.704] | 0.302 [0.299, 0.305] | 0.689 [0.686, 0.692] | 0.532 [0.530, 0.534] |

## Within MLP, all pairs (ALL), k = 5 with 95% CI

| pair | n | feature_agreement@5 | rank_agreement@5 | sign_agreement@5 | signed_rank_agreement@5 | spearman | kendall |
|---|---|---|---|---|---|---|---|
| MLP-DeepLift vs MLP-DeepLiftShap | 9411 | 0.801 [0.799, 0.803] | 0.414 [0.410, 0.417] | 0.801 [0.799, 0.803] | 0.414 [0.410, 0.417] | 0.799 [0.797, 0.801] | 0.649 [0.646, 0.651] |
| MLP-DeepLift vs MLP-GradientShap | 9411 | 0.772 [0.769, 0.774] | 0.371 [0.368, 0.374] | 0.771 [0.768, 0.773] | 0.371 [0.368, 0.374] | 0.763 [0.760, 0.765] | 0.608 [0.605, 0.610] |
| MLP-DeepLift vs MLP-InputXGradient | 9411 | 0.861 [0.859, 0.863] | 0.521 [0.517, 0.524] | 0.860 [0.859, 0.862] | 0.521 [0.517, 0.524] | 0.900 [0.899, 0.902] | 0.778 [0.777, 0.780] |
| MLP-DeepLift vs MLP-IntegratedGradients | 9411 | 0.972 [0.971, 0.973] | 0.871 [0.868, 0.874] | 0.972 [0.971, 0.973] | 0.871 [0.868, 0.874] | 0.988 [0.988, 0.988] | 0.956 [0.955, 0.956] |
| MLP-DeepLift vs MLP-KernelSHAP | 9411 | 0.776 [0.774, 0.778] | 0.370 [0.366, 0.373] | 0.775 [0.773, 0.777] | 0.369 [0.366, 0.373] | 0.775 [0.773, 0.777] | 0.620 [0.618, 0.623] |
| MLP-DeepLift vs MLP-LIME | 9411 | 0.518 [0.515, 0.521] | 0.134 [0.132, 0.136] | 0.247 [0.244, 0.251] | 0.066 [0.065, 0.068] | 0.458 [0.455, 0.462] | 0.333 [0.330, 0.336] |
| MLP-DeepLift vs MLP-Saliency | 9411 | 0.590 [0.587, 0.593] | 0.166 [0.163, 0.168] | 0.277 [0.273, 0.281] | 0.076 [0.074, 0.078] | 0.633 [0.631, 0.636] | 0.480 [0.477, 0.482] |
| MLP-DeepLiftShap vs MLP-GradientShap | 9411 | 0.865 [0.863, 0.867] | 0.523 [0.520, 0.527] | 0.865 [0.863, 0.867] | 0.523 [0.520, 0.527] | 0.899 [0.897, 0.900] | 0.770 [0.769, 0.772] |
| MLP-DeepLiftShap vs MLP-InputXGradient | 9411 | 0.757 [0.755, 0.760] | 0.350 [0.347, 0.354] | 0.756 [0.753, 0.758] | 0.350 [0.346, 0.353] | 0.745 [0.743, 0.748] | 0.590 [0.588, 0.593] |
| MLP-DeepLiftShap vs MLP-IntegratedGradients | 9411 | 0.802 [0.800, 0.804] | 0.416 [0.412, 0.419] | 0.802 [0.800, 0.804] | 0.416 [0.412, 0.419] | 0.799 [0.797, 0.802] | 0.649 [0.647, 0.652] |
| MLP-DeepLiftShap vs MLP-KernelSHAP | 9411 | 0.919 [0.917, 0.920] | 0.665 [0.662, 0.669] | 0.919 [0.917, 0.920] | 0.665 [0.662, 0.669] | 0.947 [0.946, 0.948] | 0.852 [0.851, 0.854] |
| MLP-DeepLiftShap vs MLP-LIME | 9411 | 0.574 [0.572, 0.577] | 0.162 [0.159, 0.164] | 0.275 [0.271, 0.279] | 0.079 [0.077, 0.081] | 0.566 [0.563, 0.569] | 0.422 [0.419, 0.424] |
| MLP-DeepLiftShap vs MLP-Saliency | 9411 | 0.554 [0.552, 0.557] | 0.153 [0.151, 0.156] | 0.259 [0.255, 0.262] | 0.068 [0.067, 0.070] | 0.525 [0.522, 0.529] | 0.386 [0.384, 0.389] |
| MLP-GradientShap vs MLP-InputXGradient | 9411 | 0.734 [0.731, 0.736] | 0.322 [0.319, 0.326] | 0.732 [0.730, 0.735] | 0.322 [0.319, 0.325] | 0.714 [0.712, 0.717] | 0.558 [0.555, 0.560] |
| MLP-GradientShap vs MLP-IntegratedGradients | 9411 | 0.774 [0.772, 0.776] | 0.374 [0.371, 0.377] | 0.773 [0.771, 0.775] | 0.374 [0.370, 0.377] | 0.765 [0.762, 0.767] | 0.610 [0.608, 0.613] |
| MLP-GradientShap vs MLP-KernelSHAP | 9411 | 0.852 [0.850, 0.853] | 0.491 [0.488, 0.495] | 0.852 [0.850, 0.853] | 0.491 [0.488, 0.495] | 0.881 [0.880, 0.883] | 0.745 [0.743, 0.747] |
| MLP-GradientShap vs MLP-LIME | 9411 | 0.565 [0.562, 0.568] | 0.158 [0.156, 0.161] | 0.271 [0.267, 0.275] | 0.076 [0.074, 0.078] | 0.549 [0.546, 0.552] | 0.408 [0.405, 0.410] |
| MLP-GradientShap vs MLP-Saliency | 9411 | 0.549 [0.546, 0.551] | 0.151 [0.149, 0.153] | 0.258 [0.254, 0.262] | 0.069 [0.067, 0.070] | 0.512 [0.509, 0.515] | 0.375 [0.373, 0.377] |
| MLP-InputXGradient vs MLP-KernelSHAP | 9411 | 0.733 [0.731, 0.735] | 0.315 [0.311, 0.318] | 0.730 [0.728, 0.733] | 0.314 [0.310, 0.317] | 0.721 [0.718, 0.723] | 0.564 [0.562, 0.566] |
| MLP-InputXGradient vs MLP-LIME | 9411 | 0.490 [0.487, 0.493] | 0.124 [0.122, 0.126] | 0.231 [0.228, 0.235] | 0.060 [0.059, 0.062] | 0.404 [0.401, 0.408] | 0.291 [0.288, 0.294] |
| MLP-IntegratedGradients vs MLP-InputXGradient | 9411 | 0.849 [0.847, 0.851] | 0.493 [0.489, 0.496] | 0.848 [0.847, 0.850] | 0.492 [0.489, 0.496] | 0.888 [0.886, 0.889] | 0.759 [0.758, 0.761] |
| MLP-IntegratedGradients vs MLP-KernelSHAP | 9411 | 0.779 [0.777, 0.781] | 0.373 [0.370, 0.377] | 0.778 [0.776, 0.780] | 0.373 [0.369, 0.376] | 0.778 [0.776, 0.780] | 0.624 [0.622, 0.626] |
| MLP-IntegratedGradients vs MLP-LIME | 9411 | 0.521 [0.518, 0.524] | 0.135 [0.132, 0.137] | 0.249 [0.245, 0.253] | 0.067 [0.065, 0.069] | 0.463 [0.460, 0.467] | 0.337 [0.334, 0.340] |
| MLP-IntegratedGradients vs MLP-Saliency | 9411 | 0.585 [0.583, 0.588] | 0.163 [0.160, 0.165] | 0.275 [0.271, 0.279] | 0.075 [0.073, 0.077] | 0.621 [0.618, 0.624] | 0.468 [0.465, 0.470] |
| MLP-LIME vs MLP-KernelSHAP | 9411 | 0.589 [0.587, 0.592] | 0.168 [0.166, 0.171] | 0.283 [0.279, 0.287] | 0.082 [0.080, 0.084] | 0.610 [0.607, 0.613] | 0.459 [0.457, 0.462] |
| MLP-Saliency vs MLP-InputXGradient | 9411 | 0.615 [0.612, 0.617] | 0.178 [0.175, 0.180] | 0.288 [0.283, 0.292] | 0.081 [0.079, 0.083] | 0.707 [0.705, 0.710] | 0.557 [0.554, 0.559] |
| MLP-Saliency vs MLP-KernelSHAP | 9411 | 0.538 [0.535, 0.541] | 0.147 [0.145, 0.149] | 0.251 [0.248, 0.255] | 0.066 [0.064, 0.068] | 0.490 [0.487, 0.494] | 0.357 [0.355, 0.360] |
| MLP-Saliency vs MLP-LIME | 9411 | 0.629 [0.627, 0.632] | 0.215 [0.212, 0.217] | 0.629 [0.626, 0.632] | 0.215 [0.212, 0.217] | 0.579 [0.575, 0.582] | 0.433 [0.430, 0.436] |

## k = 3 (ALL)

| pair | feature_agreement@3 | rank_agreement@3 | sign_agreement@3 | signed_rank_agreement@3 |
|---|---|---|---|---|
| MLP-DeepLift vs MLP-DeepLiftShap | 0.778 [0.775, 0.781] | 0.524 [0.519, 0.528] | 0.778 [0.775, 0.781] | 0.524 [0.519, 0.528] |
| MLP-DeepLift vs MLP-GradientShap | 0.746 [0.743, 0.749] | 0.477 [0.473, 0.482] | 0.746 [0.743, 0.749] | 0.477 [0.473, 0.481] |
| MLP-DeepLift vs MLP-InputXGradient | 0.844 [0.842, 0.847] | 0.626 [0.622, 0.631] | 0.844 [0.842, 0.847] | 0.626 [0.622, 0.631] |
| MLP-DeepLift vs MLP-IntegratedGradients | 0.969 [0.968, 0.970] | 0.911 [0.908, 0.914] | 0.969 [0.968, 0.970] | 0.911 [0.908, 0.914] |
| MLP-DeepLift vs MLP-KernelSHAP | 0.746 [0.743, 0.749] | 0.474 [0.469, 0.478] | 0.745 [0.742, 0.748] | 0.473 [0.469, 0.478] |
| MLP-DeepLift vs MLP-LIME | 0.411 [0.407, 0.415] | 0.167 [0.164, 0.170] | 0.202 [0.198, 0.206] | 0.084 [0.081, 0.086] |
| MLP-DeepLift vs MLP-Saliency | 0.493 [0.490, 0.497] | 0.210 [0.206, 0.213] | 0.230 [0.226, 0.234] | 0.097 [0.095, 0.100] |
| MLP-DeepLiftShap vs MLP-GradientShap | 0.848 [0.846, 0.851] | 0.634 [0.629, 0.638] | 0.848 [0.846, 0.851] | 0.634 [0.629, 0.638] |
| MLP-DeepLiftShap vs MLP-InputXGradient | 0.728 [0.725, 0.731] | 0.452 [0.447, 0.456] | 0.728 [0.724, 0.731] | 0.451 [0.447, 0.456] |
| MLP-DeepLiftShap vs MLP-IntegratedGradients | 0.779 [0.776, 0.782] | 0.525 [0.521, 0.530] | 0.779 [0.776, 0.782] | 0.525 [0.521, 0.530] |
| MLP-DeepLiftShap vs MLP-KernelSHAP | 0.907 [0.905, 0.909] | 0.758 [0.754, 0.762] | 0.907 [0.905, 0.909] | 0.758 [0.754, 0.762] |
| MLP-DeepLiftShap vs MLP-LIME | 0.472 [0.468, 0.476] | 0.206 [0.202, 0.209] | 0.228 [0.224, 0.232] | 0.100 [0.098, 0.103] |
| MLP-DeepLiftShap vs MLP-Saliency | 0.464 [0.460, 0.467] | 0.196 [0.192, 0.199] | 0.212 [0.208, 0.216] | 0.087 [0.084, 0.089] |
| MLP-GradientShap vs MLP-InputXGradient | 0.704 [0.700, 0.707] | 0.421 [0.417, 0.425] | 0.703 [0.700, 0.706] | 0.421 [0.417, 0.425] |
| MLP-GradientShap vs MLP-IntegratedGradients | 0.748 [0.745, 0.751] | 0.480 [0.476, 0.485] | 0.748 [0.745, 0.751] | 0.480 [0.476, 0.485] |
| MLP-GradientShap vs MLP-KernelSHAP | 0.832 [0.830, 0.835] | 0.604 [0.600, 0.609] | 0.832 [0.830, 0.835] | 0.604 [0.600, 0.609] |
| MLP-GradientShap vs MLP-LIME | 0.468 [0.464, 0.471] | 0.203 [0.199, 0.206] | 0.225 [0.221, 0.229] | 0.098 [0.095, 0.101] |
| MLP-GradientShap vs MLP-Saliency | 0.460 [0.456, 0.463] | 0.193 [0.190, 0.196] | 0.211 [0.207, 0.215] | 0.087 [0.085, 0.089] |
| MLP-InputXGradient vs MLP-KernelSHAP | 0.697 [0.694, 0.700] | 0.410 [0.405, 0.414] | 0.696 [0.693, 0.700] | 0.410 [0.405, 0.414] |
| MLP-InputXGradient vs MLP-LIME | 0.385 [0.381, 0.389] | 0.154 [0.151, 0.157] | 0.187 [0.183, 0.190] | 0.075 [0.073, 0.078] |
| MLP-IntegratedGradients vs MLP-InputXGradient | 0.830 [0.828, 0.833] | 0.600 [0.596, 0.604] | 0.830 [0.828, 0.833] | 0.600 [0.595, 0.604] |
| MLP-IntegratedGradients vs MLP-KernelSHAP | 0.748 [0.745, 0.751] | 0.478 [0.473, 0.482] | 0.748 [0.745, 0.751] | 0.478 [0.473, 0.482] |
| MLP-IntegratedGradients vs MLP-LIME | 0.414 [0.410, 0.418] | 0.168 [0.165, 0.171] | 0.203 [0.199, 0.207] | 0.085 [0.082, 0.087] |
| MLP-IntegratedGradients vs MLP-Saliency | 0.489 [0.485, 0.492] | 0.206 [0.202, 0.209] | 0.228 [0.223, 0.232] | 0.096 [0.093, 0.098] |
| MLP-LIME vs MLP-KernelSHAP | 0.487 [0.484, 0.491] | 0.215 [0.211, 0.219] | 0.235 [0.231, 0.239] | 0.104 [0.101, 0.107] |
| MLP-Saliency vs MLP-InputXGradient | 0.519 [0.515, 0.522] | 0.224 [0.220, 0.228] | 0.242 [0.238, 0.246] | 0.104 [0.101, 0.107] |
| MLP-Saliency vs MLP-KernelSHAP | 0.448 [0.444, 0.451] | 0.187 [0.184, 0.190] | 0.204 [0.200, 0.208] | 0.083 [0.081, 0.085] |
| MLP-Saliency vs MLP-LIME | 0.563 [0.560, 0.567] | 0.281 [0.277, 0.285] | 0.563 [0.559, 0.567] | 0.281 [0.277, 0.285] |
| MLP-DeepLiftShap vs LR-LinearSHAP | 0.520 [0.516, 0.524] | 0.247 [0.243, 0.251] | 0.508 [0.504, 0.512] | 0.242 [0.238, 0.246] |
| MLP-DeepLiftShap vs RF-TreeSHAP | 0.405 [0.401, 0.409] | 0.172 [0.168, 0.175] | 0.398 [0.394, 0.402] | 0.170 [0.166, 0.173] |
| MLP-DeepLiftShap vs XGB-TreeSHAP | 0.387 [0.383, 0.391] | 0.165 [0.162, 0.168] | 0.380 [0.376, 0.384] | 0.162 [0.159, 0.166] |
| RF-TreeSHAP vs LR-LinearSHAP | 0.395 [0.391, 0.400] | 0.166 [0.162, 0.170] | 0.372 [0.368, 0.377] | 0.157 [0.154, 0.161] |
| RF-TreeSHAP vs XGB-TreeSHAP | 0.628 [0.624, 0.632] | 0.340 [0.336, 0.344] | 0.626 [0.622, 0.630] | 0.339 [0.335, 0.344] |
| XGB-TreeSHAP vs LR-LinearSHAP | 0.362 [0.358, 0.367] | 0.159 [0.155, 0.162] | 0.346 [0.342, 0.350] | 0.153 [0.150, 0.157] |
| LR-LIME vs RF-LIME | 0.474 [0.471, 0.478] | 0.218 [0.215, 0.222] | 0.460 [0.456, 0.463] | 0.214 [0.210, 0.217] |
| LR-LIME vs XGB-LIME | 0.465 [0.462, 0.469] | 0.236 [0.233, 0.239] | 0.454 [0.450, 0.457] | 0.231 [0.228, 0.234] |
| MLP-LIME vs LR-LIME | 0.500 [0.497, 0.503] | 0.239 [0.235, 0.242] | 0.500 [0.497, 0.503] | 0.239 [0.235, 0.242] |
| MLP-LIME vs RF-LIME | 0.460 [0.456, 0.463] | 0.246 [0.242, 0.250] | 0.459 [0.455, 0.463] | 0.246 [0.242, 0.250] |
| MLP-LIME vs XGB-LIME | 0.454 [0.450, 0.458] | 0.227 [0.224, 0.230] | 0.454 [0.450, 0.458] | 0.227 [0.224, 0.230] |
| RF-LIME vs XGB-LIME | 0.684 [0.681, 0.686] | 0.403 [0.399, 0.407] | 0.683 [0.680, 0.686] | 0.402 [0.398, 0.406] |

## Per project: feature agreement@5

### Cross-model

| project | MLP-DeepLiftShap vs LR-LinearSHAP | MLP-DeepLiftShap vs RF-TreeSHAP | MLP-DeepLiftShap vs XGB-TreeSHAP | RF-TreeSHAP vs LR-LinearSHAP | RF-TreeSHAP vs XGB-TreeSHAP | XGB-TreeSHAP vs LR-LinearSHAP |
|---|---|---|---|---|---|---|
| ant | 0.639 | 0.482 | 0.468 | 0.488 | 0.628 | 0.429 |
| camel | 0.561 | 0.404 | 0.432 | 0.441 | 0.651 | 0.425 |
| ivy | 0.615 | 0.491 | 0.442 | 0.550 | 0.719 | 0.489 |
| jedit | 0.583 | 0.514 | 0.491 | 0.514 | 0.659 | 0.469 |
| lucene | 0.656 | 0.507 | 0.443 | 0.562 | 0.656 | 0.475 |
| poi | 0.619 | 0.464 | 0.434 | 0.419 | 0.706 | 0.390 |
| synapse | 0.592 | 0.407 | 0.310 | 0.510 | 0.524 | 0.438 |
| velocity | 0.647 | 0.484 | 0.468 | 0.470 | 0.698 | 0.469 |
| xalan | 0.538 | 0.507 | 0.491 | 0.421 | 0.670 | 0.391 |
| xerces | 0.548 | 0.436 | 0.390 | 0.373 | 0.671 | 0.353 |

### LIME across models

| project | LR-LIME vs RF-LIME | LR-LIME vs XGB-LIME | MLP-LIME vs LR-LIME | MLP-LIME vs RF-LIME | MLP-LIME vs XGB-LIME | RF-LIME vs XGB-LIME |
|---|---|---|---|---|---|---|
| ant | 0.751 | 0.562 | 0.584 | 0.610 | 0.554 | 0.626 |
| camel | 0.305 | 0.386 | 0.466 | 0.361 | 0.432 | 0.709 |
| ivy | 0.615 | 0.537 | 0.492 | 0.616 | 0.549 | 0.685 |
| jedit | 0.631 | 0.595 | 0.588 | 0.536 | 0.557 | 0.716 |
| lucene | 0.587 | 0.574 | 0.600 | 0.498 | 0.500 | 0.658 |
| poi | 0.464 | 0.465 | 0.508 | 0.283 | 0.341 | 0.687 |
| synapse | 0.595 | 0.502 | 0.452 | 0.349 | 0.294 | 0.695 |
| velocity | 0.548 | 0.536 | 0.615 | 0.746 | 0.720 | 0.818 |
| xalan | 0.628 | 0.535 | 0.550 | 0.640 | 0.615 | 0.780 |
| xerces | 0.283 | 0.430 | 0.551 | 0.373 | 0.465 | 0.693 |

Per-project values for every pair and metric (with CIs) are in `step4_agreement.csv`.
