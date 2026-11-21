# Step 4: share of attribution on size

## Changes since v1

v1 is archived in `reports/archive/step4_v1/`. v2 changes two settings: Integrated Gradients now uses 512 Gauss-Legendre steps (v1: 64), and LIME samples its perturbations around the explained instance (`sample_around_instance=True`; v1 sampled around the training mean). The v1 LIME results are kept in the store as `LIME_mean_sampling`. Everything else is unchanged.

LIME size shares (ALL), v1 → v2:

| model | loc share v1 → v2 | size-group share v1 → v2 | % loc top-1 v1 → v2 |
|---|---|---|---|
| MLP | 0.066 → 0.064 | 0.239 → 0.232 | 9.7 → 10.7 |
| LR | 0.079 → 0.079 | 0.275 → 0.275 | 11.2 → 11.2 |
| RF | 0.114 → 0.119 | 0.334 → 0.332 | 23.4 → 24.3 |
| XGB | 0.105 → 0.110 | 0.305 → 0.308 | 18.7 → 18.1 |

LIME coefficient of variation across instances (median over splits of the median over features of std/mean|w| across the test instances of a split, seed 0):

| model | v1 (mean sampling) | v2 (instance sampling) |
|---|---|---|
| MLP | 0.076 | 0.582 |
| LR | 0.024 | 0.184 |
| RF | 0.229 | 0.733 |
| XGB | 0.250 | 0.679 |

For LR the v2 change leaves the normalised weights essentially unchanged: LR is linear on the log-odds scale, so its probability slopes differ between instances mainly by the common factor p(1 − p), which the per-instance normalisation removes.


Share of the total normalised |attribution| assigned to `loc` alone and to the size group {loc, wmc, rfc, npm}, and the percentage of test instances whose top-1 feature (largest |attribution|) is `loc`. Under equal attribution each feature would get 0.05 and the size group 0.20.

Absolute attributions are normalised per instance to unit L1 norm, averaged over the test instances of a split (and over the seeds of that split), then over splits; every split has the same weight. `ALL` = all 24 time-aware splits.

**Output scale.** Every explanation explains P(defective) except LR-LinearSHAP, which is exact on the log-odds scale (exact interventional SHAP on the probability scale is not linear). LR-LIME explains the probability. Because attributions are L1-normalised per instance before aggregation, and the comparisons are rank-based, the scale difference does not affect them.

## Overall (ALL)

| explanation | loc share | size-group share | % instances with loc top-1 |
|---|---|---|---|
| MLP-DeepLift | 0.055 | 0.204 | 5.5 |
| MLP-DeepLiftShap | 0.060 | 0.220 | 6.5 |
| MLP-GradientShap | 0.059 | 0.218 | 6.4 |
| MLP-IntegratedGradients | 0.055 | 0.204 | 5.5 |
| MLP-Saliency | 0.060 | 0.224 | 8.2 |
| MLP-InputXGradient | 0.054 | 0.200 | 5.3 |
| MLP-LIME | 0.064 | 0.232 | 10.7 |
| MLP-KernelSHAP | 0.061 | 0.219 | 7.4 |
| LR-LIME | 0.079 | 0.275 | 11.2 |
| LR-LinearSHAP | 0.068 | 0.241 | 9.2 |
| RF-LIME | 0.119 | 0.332 | 24.3 |
| RF-TreeSHAP | 0.106 | 0.317 | 18.2 |
| XGB-LIME | 0.110 | 0.308 | 18.1 |
| XGB-TreeSHAP | 0.100 | 0.295 | 13.8 |

## loc share per project

| project | MLP-DeepLift | MLP-DeepLiftShap | MLP-GradientShap | MLP-IntegratedGradients | MLP-Saliency | MLP-InputXGradient | MLP-LIME | MLP-KernelSHAP | LR-LIME | LR-LinearSHAP | RF-LIME | RF-TreeSHAP | XGB-LIME | XGB-TreeSHAP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ant | 0.060 | 0.065 | 0.063 | 0.061 | 0.066 | 0.057 | 0.071 | 0.067 | 0.075 | 0.068 | 0.143 | 0.126 | 0.076 | 0.078 |
| camel | 0.038 | 0.042 | 0.042 | 0.037 | 0.042 | 0.037 | 0.040 | 0.035 | 0.121 | 0.099 | 0.027 | 0.046 | 0.026 | 0.043 |
| ivy | 0.068 | 0.079 | 0.079 | 0.067 | 0.074 | 0.067 | 0.097 | 0.099 | 0.081 | 0.079 | 0.220 | 0.188 | 0.233 | 0.181 |
| jedit | 0.044 | 0.046 | 0.044 | 0.044 | 0.050 | 0.044 | 0.046 | 0.040 | 0.058 | 0.038 | 0.105 | 0.089 | 0.062 | 0.078 |
| lucene | 0.058 | 0.065 | 0.064 | 0.058 | 0.062 | 0.058 | 0.059 | 0.062 | 0.084 | 0.076 | 0.088 | 0.101 | 0.113 | 0.093 |
| poi | 0.050 | 0.057 | 0.055 | 0.050 | 0.060 | 0.052 | 0.069 | 0.062 | 0.084 | 0.066 | 0.096 | 0.064 | 0.126 | 0.080 |
| synapse | 0.042 | 0.044 | 0.044 | 0.042 | 0.048 | 0.043 | 0.047 | 0.041 | 0.048 | 0.043 | 0.089 | 0.120 | 0.045 | 0.079 |
| velocity | 0.086 | 0.090 | 0.088 | 0.086 | 0.084 | 0.083 | 0.101 | 0.102 | 0.088 | 0.086 | 0.119 | 0.094 | 0.141 | 0.118 |
| xalan | 0.080 | 0.087 | 0.087 | 0.081 | 0.085 | 0.078 | 0.093 | 0.088 | 0.152 | 0.125 | 0.225 | 0.178 | 0.275 | 0.222 |
| xerces | 0.024 | 0.024 | 0.025 | 0.024 | 0.031 | 0.027 | 0.023 | 0.016 | 0.014 | 0.015 | 0.074 | 0.067 | 0.052 | 0.067 |

## Size-group share per project

| project | MLP-DeepLift | MLP-DeepLiftShap | MLP-GradientShap | MLP-IntegratedGradients | MLP-Saliency | MLP-InputXGradient | MLP-LIME | MLP-KernelSHAP | LR-LIME | LR-LinearSHAP | RF-LIME | RF-TreeSHAP | XGB-LIME | XGB-TreeSHAP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ant | 0.206 | 0.221 | 0.219 | 0.206 | 0.224 | 0.201 | 0.229 | 0.219 | 0.266 | 0.244 | 0.359 | 0.346 | 0.280 | 0.277 |
| camel | 0.255 | 0.283 | 0.280 | 0.254 | 0.268 | 0.243 | 0.297 | 0.281 | 0.492 | 0.432 | 0.302 | 0.322 | 0.319 | 0.307 |
| ivy | 0.223 | 0.241 | 0.240 | 0.223 | 0.234 | 0.222 | 0.252 | 0.255 | 0.245 | 0.254 | 0.412 | 0.386 | 0.481 | 0.416 |
| jedit | 0.209 | 0.229 | 0.223 | 0.210 | 0.230 | 0.201 | 0.237 | 0.226 | 0.278 | 0.216 | 0.411 | 0.357 | 0.350 | 0.350 |
| lucene | 0.213 | 0.234 | 0.233 | 0.214 | 0.229 | 0.210 | 0.245 | 0.241 | 0.300 | 0.265 | 0.441 | 0.404 | 0.412 | 0.364 |
| poi | 0.203 | 0.226 | 0.222 | 0.203 | 0.237 | 0.200 | 0.242 | 0.222 | 0.267 | 0.221 | 0.274 | 0.252 | 0.267 | 0.245 |
| synapse | 0.180 | 0.181 | 0.180 | 0.180 | 0.197 | 0.179 | 0.195 | 0.175 | 0.186 | 0.165 | 0.198 | 0.235 | 0.155 | 0.182 |
| velocity | 0.171 | 0.179 | 0.179 | 0.171 | 0.192 | 0.171 | 0.212 | 0.195 | 0.198 | 0.190 | 0.311 | 0.305 | 0.338 | 0.332 |
| xalan | 0.247 | 0.264 | 0.263 | 0.247 | 0.265 | 0.241 | 0.264 | 0.247 | 0.391 | 0.307 | 0.368 | 0.337 | 0.364 | 0.321 |
| xerces | 0.129 | 0.135 | 0.136 | 0.130 | 0.152 | 0.133 | 0.141 | 0.127 | 0.139 | 0.136 | 0.211 | 0.214 | 0.144 | 0.177 |

## % instances with loc as top-1 feature, per project

| project | MLP-DeepLift | MLP-DeepLiftShap | MLP-GradientShap | MLP-IntegratedGradients | MLP-Saliency | MLP-InputXGradient | MLP-LIME | MLP-KernelSHAP | LR-LIME | LR-LinearSHAP | RF-LIME | RF-TreeSHAP | XGB-LIME | XGB-TreeSHAP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ant | 7.9 | 9.9 | 9.8 | 8.5 | 13.5 | 6.8 | 15.1 | 10.1 | 11.1 | 6.9 | 27.0 | 19.3 | 8.3 | 5.3 |
| camel | 1.5 | 2.4 | 2.2 | 1.5 | 0.6 | 1.4 | 0.1 | 0.4 | 16.7 | 17.9 | 0.0 | 2.1 | 0.0 | 1.1 |
| ivy | 5.6 | 6.6 | 6.6 | 5.3 | 10.3 | 6.0 | 21.9 | 12.9 | 0.0 | 3.7 | 62.2 | 37.8 | 27.4 | 24.9 |
| jedit | 2.4 | 1.4 | 1.3 | 2.2 | 2.3 | 2.2 | 3.5 | 1.6 | 11.1 | 3.6 | 20.2 | 8.5 | 1.2 | 5.6 |
| lucene | 3.6 | 5.2 | 5.1 | 3.5 | 5.2 | 4.0 | 1.5 | 1.7 | 0.0 | 7.6 | 9.3 | 11.5 | 12.2 | 6.8 |
| poi | 5.0 | 4.8 | 4.7 | 4.7 | 7.8 | 5.5 | 7.9 | 5.9 | 9.2 | 7.5 | 17.5 | 9.1 | 34.1 | 13.4 |
| synapse | 3.3 | 3.3 | 3.4 | 3.4 | 3.9 | 3.7 | 3.5 | 3.2 | 0.0 | 1.3 | 13.7 | 25.5 | 1.6 | 5.9 |
| velocity | 12.9 | 16.2 | 16.1 | 13.0 | 17.7 | 11.7 | 28.5 | 24.2 | 15.6 | 12.8 | 20.4 | 26.0 | 33.9 | 35.5 |
| xalan | 11.7 | 14.9 | 14.6 | 11.5 | 18.6 | 11.7 | 25.3 | 15.1 | 50.0 | 36.4 | 75.8 | 50.0 | 72.1 | 49.9 |
| xerces | 0.3 | 0.3 | 0.4 | 0.4 | 0.0 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.1 | 2.1 |
