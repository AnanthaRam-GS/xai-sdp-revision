# Step 6: sanity checks (Adebayo et al., 2018)

Similarity with the original explanation per test instance (seed 0, all test instances): Spearman correlation of |attribution| (ρ) and feature agreement@5; means over instances, then over splits. A method is flagged when its mean ρ stays above 0.5 after full model randomisation (MLP) or after label permutation.

## Pass / flag

| explanation | ρ after full model randomisation | ρ after label permutation | permuted-label test AUC | result |
|---|---|---|---|---|
| MLP-DeepLift | 0.384 | 0.386 | 0.490 | pass |
| MLP-DeepLiftShap | 0.282 | 0.289 | 0.490 | pass |
| MLP-GradientShap | 0.256 | 0.296 | 0.490 | pass |
| MLP-IntegratedGradients | 0.384 | 0.386 | 0.490 | pass |
| MLP-Saliency | 0.002 | 0.009 | 0.490 | pass |
| MLP-InputXGradient | 0.386 | 0.390 | 0.490 | pass |
| MLP-LIME | 0.001 | -0.003 | 0.490 | pass |
| MLP-KernelSHAP | 0.293 | 0.300 | 0.490 | pass |
| LR-LinearSHAP | n/a (not an MLP) | 0.355 | 0.471 | pass |
| LR-LIME | n/a (not an MLP) | -0.031 | 0.471 | pass |
| RF-TreeSHAP | n/a (not an MLP) | 0.237 | 0.484 | pass |
| RF-LIME | n/a (not an MLP) | 0.179 | 0.484 | pass |
| XGB-TreeSHAP | n/a (not an MLP) | 0.254 | 0.487 | pass |
| XGB-LIME | n/a (not an MLP) | 0.263 | 0.487 | pass |

## Cascading model-parameter randomisation (MLP)

Linear layers re-initialised (PyTorch default initialisation, seeded) cumulatively from the output layer towards the input. The MLPs have different depths (number of Linear layers per split: 2: 11, 3: 6, 4: 7), so `top 2/3 layers` exist only for deeper networks; `all layers` is full randomisation for every split.

| explanation | output ρ / FA@5 | top 2 layers ρ / FA@5 | top 3 layers ρ / FA@5 | all layers ρ / FA@5 |
|---|---|---|---|---|
| MLP-DeepLift | 0.604 / 0.610 | 0.489 / 0.523 | 0.450 / 0.468 | 0.384 / 0.429 |
| MLP-DeepLiftShap | 0.515 / 0.574 | 0.365 / 0.475 | 0.350 / 0.456 | 0.282 / 0.411 |
| MLP-GradientShap | 0.505 / 0.561 | 0.333 / 0.457 | 0.320 / 0.437 | 0.256 / 0.400 |
| MLP-IntegratedGradients | 0.600 / 0.605 | 0.487 / 0.522 | 0.447 / 0.463 | 0.384 / 0.427 |
| MLP-Saliency | 0.369 / 0.494 | 0.184 / 0.365 | 0.046 / 0.276 | 0.002 / 0.254 |
| MLP-InputXGradient | 0.586 / 0.601 | 0.497 / 0.523 | 0.434 / 0.458 | 0.386 / 0.432 |
| MLP-LIME | 0.391 / 0.495 | 0.157 / 0.358 | 0.075 / 0.299 | 0.001 / 0.250 |
| MLP-KernelSHAP | 0.519 / 0.571 | 0.374 / 0.481 | 0.378 / 0.465 | 0.293 / 0.414 |

## Independent single-layer randomisation (MLP), Spearman

| explanation | first (input) layer | other hidden | output |
|---|---|---|---|
| MLP-DeepLift | 0.375 | 0.475 | 0.589 |
| MLP-DeepLiftShap | 0.259 | 0.383 | 0.515 |
| MLP-GradientShap | 0.239 | 0.340 | 0.467 |
| MLP-IntegratedGradients | 0.373 | 0.475 | 0.587 |
| MLP-Saliency | -0.000 | 0.167 | 0.340 |
| MLP-InputXGradient | 0.387 | 0.492 | 0.569 |
| MLP-LIME | -0.018 | 0.195 | 0.369 |
| MLP-KernelSHAP | 0.267 | 0.395 | 0.517 |

## Data randomisation (permuted training labels)

Each model refitted on the training release with permuted labels (seed 0, class counts unchanged) and persisted hyperparameters; the permuted-label models' test AUC should be about 0.5.

| explanation | Spearman | FA@5 | permuted-label test AUC |
|---|---|---|---|
| MLP-DeepLift | 0.386 | 0.437 | 0.490 |
| MLP-DeepLiftShap | 0.289 | 0.412 | 0.490 |
| MLP-GradientShap | 0.296 | 0.413 | 0.490 |
| MLP-IntegratedGradients | 0.386 | 0.438 | 0.490 |
| MLP-Saliency | 0.009 | 0.266 | 0.490 |
| MLP-InputXGradient | 0.390 | 0.437 | 0.490 |
| MLP-LIME | -0.003 | 0.264 | 0.490 |
| MLP-KernelSHAP | 0.300 | 0.417 | 0.490 |
| LR-LinearSHAP | 0.355 | 0.415 | 0.471 |
| LR-LIME | -0.031 | 0.242 | 0.471 |
| RF-TreeSHAP | 0.237 | 0.321 | 0.484 |
| RF-LIME | 0.179 | 0.305 | 0.484 |
| XGB-TreeSHAP | 0.254 | 0.319 | 0.487 |
| XGB-LIME | 0.263 | 0.357 | 0.487 |

Note: after full model randomisation or label permutation, the contribution-type MLP explanations (DeepLift, DeepLiftShap, GradientShap, Integrated Gradients, InputXGradient, KernelSHAP) keep part of their ranking (ρ 0.29–0.39 after label permutation), whereas the slope-type ones (Saliency, LIME) drop to ρ ≈ 0. Contributions are (approximately) slope × (x − reference): the input term (x − reference) is the same for any model and carries ranking information by itself, which is the known weakness of input-multiplied attributions in these tests (Adebayo et al., 2018). None of the methods exceeds the 0.5 flag, but the remaining similarity should be read as partly model-independent.
