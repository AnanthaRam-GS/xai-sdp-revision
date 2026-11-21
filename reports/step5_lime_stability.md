# Step 5: LIME stability and settings sensitivity

Seed-0 models of `main_time_aware`; per split up to 10 test instances per outcome group (groups of the seed-0 MLP; the same instances for all four models): 934 instances (TP 240, FP 238, FN 222, TN 234). Main setting: `discretize_continuous=False`, `sample_around_instance=True`, 5000 samples, default kernel width (0.75 × √20), 20 features. The sample is in `results/robustness/lime_instance_sample.csv`; per-instance values are in `results/robustness/lime_*_instances.csv`.

**LIME weights are slopes.** LIME (`discretize_continuous=False`, `sample_around_instance=True`) fits a weighted linear surrogate to perturbations drawn around the explained instance; its weights are that surrogate's coefficients (per standardised unit), i.e. local slopes like Saliency's gradients, not contributions relative to a reference (slope × (x − reference)) as for the SHAP family, DeepLift, Integrated Gradients and InputXGradient. LIME's sign therefore says whether the prediction rises with the feature near the instance, not whether the instance's value raises its prediction. The step 4 v1 setting (`sample_around_instance=False`, kept as `LIME_mean_sampling`) sampled around the training mean and gave almost the same weights for every instance.

## Run-to-run variability of the main setting (10 runs, random_state 0–9)

Per instance: mean pairwise Spearman (|weights|, 20 features) and feature agreement@5 over the 45 run pairs; coefficient of variation std/|mean| of each feature's weight across runs (mean over features with a non-zero mean; median shown too because weights near 0 inflate it); whether the top-1 feature differs between any two runs; mean surrogate R². Averaged over instances.

| model | instances | Spearman | FA@5 | mean CV | median CV | % top-1 changes | R² |
|---|---|---|---|---|---|---|---|
| MLP | 934 | 0.991 | 0.966 | 0.278 | 0.031 | 10.278 | 0.850 |
| LR | 934 | 0.995 | 0.981 | 0.092 | 0.022 | 11.456 | 0.918 |
| RF | 934 | 0.943 | 0.953 | 2.237 | 0.154 | 11.884 | 0.559 |
| XGB | 934 | 0.920 | 0.947 | 4.923 | 0.197 | 10.707 | 0.533 |

Run-to-run Spearman by outcome group:

| model | FN | FP | TN | TP |
|---|---|---|---|---|
| MLP | 0.991 | 0.992 | 0.989 | 0.992 |
| LR | 0.994 | 0.996 | 0.995 | 0.996 |
| RF | 0.951 | 0.941 | 0.946 | 0.938 |
| XGB | 0.931 | 0.918 | 0.919 | 0.911 |

## Settings grid vs the main setting (random_state 0)

One factor at a time; ρ = Spearman with the main setting's run 0, FA@5 = feature agreement@5, R² = local surrogate R² (`explanation.score`, a local-fidelity indicator). `main` shows the main setting's own R².

| setting | MLP ρ | MLP FA@5 | MLP R² | LR ρ | LR FA@5 | LR R² | RF ρ | RF FA@5 | RF R² | XGB ρ | XGB FA@5 | XGB R² |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | 1.000 | 1.000 | 0.850 | 1.000 | 1.000 | 0.919 | 1.000 | 1.000 | 0.559 | 1.000 | 1.000 | 0.534 |
| num_samples=1000 | 0.981 | 0.943 | 0.853 | 0.990 | 0.965 | 0.920 | 0.895 | 0.925 | 0.561 | 0.857 | 0.915 | 0.538 |
| num_samples=2000 | 0.987 | 0.962 | 0.851 | 0.993 | 0.977 | 0.919 | 0.921 | 0.936 | 0.560 | 0.894 | 0.936 | 0.536 |
| num_samples=10000 | 0.992 | 0.967 | 0.850 | 0.996 | 0.986 | 0.918 | 0.952 | 0.957 | 0.559 | 0.933 | 0.958 | 0.533 |
| kernel_width=0.5×√20 | 0.997 | 0.988 | 0.855 | 0.999 | 0.996 | 0.921 | 0.989 | 0.979 | 0.559 | 0.986 | 0.977 | 0.533 |
| kernel_width=1.0×√20 | 0.999 | 0.996 | 0.849 | 1.000 | 0.998 | 0.918 | 0.997 | 0.992 | 0.559 | 0.996 | 0.993 | 0.534 |
| kernel_width=2.0×√20 | 0.998 | 0.989 | 0.847 | 0.999 | 0.996 | 0.917 | 0.993 | 0.985 | 0.559 | 0.990 | 0.985 | 0.534 |
| sampling=mean | 0.772 | 0.756 | 0.850 | 0.997 | 0.987 | 0.954 | 0.728 | 0.740 | 0.577 | 0.755 | 0.751 | 0.542 |
| sampling=quartile | 0.470 | 0.518 | 0.351 | 0.686 | 0.623 | 0.377 | 0.508 | 0.578 | 0.342 | 0.551 | 0.597 | 0.325 |

## Instance dependence (coefficient of variation across instances)

Median over splits of the median over features of std/mean|w| across all test instances of a split (seed 0, full step 4 store). Higher = explanations differ more between instances.

| model | v1 mean sampling (LIME_mean_sampling) | v2 instance sampling (LIME) |
|---|---|---|
| MLP | 0.076 | 0.582 |
| LR | 0.024 | 0.184 |
| RF | 0.229 | 0.733 |
| XGB | 0.250 | 0.679 |
