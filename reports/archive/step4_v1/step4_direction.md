# Step 4: direction of feature effects

For each (model, method, feature): the Spearman correlation between the raw metric value and the signed attribution over the test instances of a split and seed, averaged over seeds and then over splits (ALL). `+` / `−` = mean correlation ≥ 0.1 / ≤ −0.1, `0` otherwise. Rows are ordered by mean importance over all explanations. Saliency uses the signed gradient. Full results (per project, share of splits with a positive correlation) are in `step4_direction.csv`.

**Output scale.** Every explanation explains P(defective) except LR-LinearSHAP, which is exact on the log-odds scale (exact interventional SHAP on the probability scale is not linear). LR-LIME explains the probability. Because attributions are L1-normalised per instance before aggregation, and the comparisons are rank-based, the scale difference does not affect them.

**LIME weights are slopes.** With `discretize_continuous=False` (and LIME's default `sample_around_instance=False`), LIME fits a weighted linear surrogate to samples drawn around the training mean, and its weights are that surrogate's coefficients (per standardised unit). They are slopes, like Saliency's gradients, not contributions relative to a reference (slope × (x − reference)) as for the SHAP family, DeepLift, Integrated Gradients and InputXGradient. They also vary little between instances: the median coefficient of variation across the test instances of a split is about 0.1–0.2 for LIME and above 1 for TreeSHAP, DeepLiftShap and Saliency (checked on ant 1.5→1.6, seed 0). LIME's sign therefore reflects the slope, not whether the instance's value raises its prediction.
Consequently, the near-zero Spearman correlations of LIME and Saliency below say that their slope hardly changes with the feature's value; they are not evidence of a different direction. Direction disagreements involving LIME or Saliency should be read with this in mind.

## Spearman(raw value, signed attribution), ALL

| feature | MLP-DeepLift | MLP-DeepLiftShap | MLP-GradientShap | MLP-IntegratedGradients | MLP-Saliency | MLP-InputXGradient | MLP-LIME | MLP-KernelSHAP | LR-LIME | LR-LinearSHAP | RF-LIME | RF-TreeSHAP | XGB-LIME | XGB-TreeSHAP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loc | + +0.57 | + +0.59 | + +0.58 | + +0.57 | 0 -0.05 | + +0.52 | 0 -0.08 | + +0.60 | + +0.19 | + +0.38 | 0 -0.03 | + +0.48 | 0 -0.04 | + +0.31 |
| rfc | + +0.73 | + +0.76 | + +0.74 | + +0.73 | 0 +0.08 | + +0.67 | + +0.13 | + +0.79 | + +0.23 | + +0.80 | 0 -0.01 | + +0.60 | 0 -0.01 | + +0.48 |
| ce | + +0.49 | + +0.51 | + +0.50 | + +0.49 | 0 +0.02 | + +0.48 | 0 +0.07 | + +0.52 | + +0.15 | + +0.50 | 0 -0.07 | + +0.39 | 0 -0.07 | + +0.31 |
| cam | − -0.29 | − -0.31 | − -0.31 | − -0.29 | 0 -0.00 | − -0.27 | + +0.10 | − -0.31 | + +0.17 | − -0.42 | + +0.16 | − -0.38 | + +0.17 | − -0.16 |
| cbo | + +0.41 | + +0.45 | + +0.44 | + +0.41 | 0 -0.03 | + +0.41 | + +0.12 | + +0.48 | 0 +0.10 | + +0.40 | 0 -0.04 | + +0.40 | 0 -0.03 | + +0.29 |
| amc | + +0.54 | + +0.57 | + +0.56 | + +0.55 | 0 +0.01 | + +0.48 | 0 -0.05 | + +0.52 | + +0.15 | + +0.57 | 0 -0.05 | + +0.41 | 0 +0.03 | + +0.30 |
| dam | + +0.16 | + +0.16 | + +0.16 | + +0.16 | 0 -0.01 | + +0.15 | 0 -0.01 | + +0.15 | + +0.12 | + +0.19 | − -0.13 | 0 +0.09 | 0 -0.05 | + +0.12 |
| lcom | + +0.50 | + +0.54 | + +0.53 | + +0.51 | − -0.12 | + +0.46 | 0 -0.04 | + +0.49 | + +0.11 | + +0.52 | − -0.12 | + +0.53 | 0 -0.06 | + +0.28 |
| npm | + +0.54 | + +0.59 | + +0.57 | + +0.54 | − -0.16 | + +0.48 | − -0.15 | + +0.54 | 0 +0.05 | + +0.69 | − -0.13 | + +0.43 | − -0.12 | + +0.27 |
| wmc | + +0.38 | + +0.40 | + +0.39 | + +0.38 | 0 -0.04 | + +0.36 | − -0.11 | + +0.37 | 0 +0.09 | + +0.25 | 0 -0.07 | + +0.47 | 0 +0.02 | + +0.24 |
| mfa | − -0.31 | − -0.32 | − -0.32 | − -0.31 | − -0.12 | − -0.28 | 0 -0.06 | − -0.36 | 0 +0.04 | − -0.53 | 0 -0.08 | − -0.23 | 0 +0.01 | − -0.12 |
| ca | + +0.18 | + +0.21 | + +0.21 | + +0.18 | 0 -0.08 | + +0.20 | 0 -0.05 | + +0.21 | 0 +0.04 | + +0.27 | 0 -0.02 | + +0.14 | 0 -0.03 | + +0.17 |
| dit | − -0.13 | − -0.14 | − -0.15 | − -0.13 | − -0.12 | − -0.14 | − -0.14 | − -0.12 | 0 +0.01 | − -0.12 | − -0.12 | − -0.11 | 0 -0.05 | − -0.18 |
| avg_cc | 0 +0.01 | 0 +0.02 | 0 +0.02 | 0 +0.01 | 0 +0.05 | 0 +0.03 | + +0.13 | 0 +0.03 | + +0.13 | 0 -0.07 | 0 +0.00 | + +0.24 | 0 +0.08 | + +0.14 |
| cbm | + +0.12 | + +0.14 | + +0.14 | + +0.12 | 0 +0.01 | + +0.12 | 0 +0.06 | + +0.12 | 0 +0.01 | + +0.48 | 0 +0.02 | 0 -0.03 | 0 +0.02 | 0 +0.09 |
| lcom3 | + +0.15 | + +0.17 | + +0.17 | + +0.15 | − -0.19 | + +0.15 | − -0.18 | + +0.35 | 0 +0.04 | + +0.18 | − -0.29 | + +0.14 | − -0.25 | + +0.19 |
| max_cc | + +0.33 | + +0.35 | + +0.35 | + +0.34 | 0 +0.06 | + +0.29 | + +0.13 | + +0.38 | + +0.17 | + +0.49 | 0 -0.00 | + +0.36 | 0 +0.03 | + +0.26 |
| moa | + +0.26 | + +0.28 | + +0.29 | + +0.26 | 0 -0.01 | + +0.26 | 0 +0.02 | + +0.26 | 0 +0.07 | + +0.44 | 0 +0.02 | + +0.26 | 0 +0.03 | + +0.22 |
| ic | 0 -0.06 | 0 -0.07 | 0 -0.07 | 0 -0.06 | 0 -0.02 | 0 -0.05 | 0 +0.03 | 0 -0.07 | 0 +0.02 | − -0.17 | 0 -0.03 | 0 -0.09 | 0 -0.05 | 0 -0.06 |
| noc | 0 +0.00 | 0 +0.01 | 0 +0.02 | 0 +0.01 | 0 -0.07 | 0 +0.01 | 0 -0.04 | 0 +0.00 | 0 +0.02 | + +0.23 | 0 -0.01 | + +0.12 | 0 -0.01 | + +0.20 |

## Features on which explanations disagree in direction (ALL)

A feature is listed when, within a model (or across all explanations), at least one explanation has a positive direction and another a negative one.

| feature | within | positive | negative |
|---|---|---|---|
| cam | LR | LR-LIME | LR-LinearSHAP |
| cam | MLP | MLP-LIME | MLP-DeepLift, MLP-DeepLiftShap, MLP-GradientShap, MLP-InputXGradient, MLP-IntegratedGradients, MLP-KernelSHAP |
| cam | RF | RF-LIME | RF-TreeSHAP |
| cam | XGB | XGB-LIME | XGB-TreeSHAP |
| cam | all models | LR-LIME, MLP-LIME, RF-LIME, XGB-LIME | LR-LinearSHAP, MLP-DeepLift, MLP-DeepLiftShap, MLP-GradientShap, MLP-InputXGradient, MLP-IntegratedGradients, MLP-KernelSHAP, RF-TreeSHAP, XGB-TreeSHAP |
| dam | all models | LR-LIME, LR-LinearSHAP, MLP-DeepLift, MLP-DeepLiftShap, MLP-GradientShap, MLP-InputXGradient, MLP-IntegratedGradients, MLP-KernelSHAP, XGB-TreeSHAP | RF-LIME |
| lcom | MLP | MLP-DeepLift, MLP-DeepLiftShap, MLP-GradientShap, MLP-InputXGradient, MLP-IntegratedGradients, MLP-KernelSHAP | MLP-Saliency |
| lcom | RF | RF-TreeSHAP | RF-LIME |
| lcom | all models | LR-LIME, LR-LinearSHAP, MLP-DeepLift, MLP-DeepLiftShap, MLP-GradientShap, MLP-InputXGradient, MLP-IntegratedGradients, MLP-KernelSHAP, RF-TreeSHAP, XGB-TreeSHAP | MLP-Saliency, RF-LIME |
| lcom3 | MLP | MLP-DeepLift, MLP-DeepLiftShap, MLP-GradientShap, MLP-InputXGradient, MLP-IntegratedGradients, MLP-KernelSHAP | MLP-LIME, MLP-Saliency |
| lcom3 | RF | RF-TreeSHAP | RF-LIME |
| lcom3 | XGB | XGB-TreeSHAP | XGB-LIME |
| lcom3 | all models | LR-LinearSHAP, MLP-DeepLift, MLP-DeepLiftShap, MLP-GradientShap, MLP-InputXGradient, MLP-IntegratedGradients, MLP-KernelSHAP, RF-TreeSHAP, XGB-TreeSHAP | MLP-LIME, MLP-Saliency, RF-LIME, XGB-LIME |
| npm | MLP | MLP-DeepLift, MLP-DeepLiftShap, MLP-GradientShap, MLP-InputXGradient, MLP-IntegratedGradients, MLP-KernelSHAP | MLP-LIME, MLP-Saliency |
| npm | RF | RF-TreeSHAP | RF-LIME |
| npm | XGB | XGB-TreeSHAP | XGB-LIME |
| npm | all models | LR-LinearSHAP, MLP-DeepLift, MLP-DeepLiftShap, MLP-GradientShap, MLP-InputXGradient, MLP-IntegratedGradients, MLP-KernelSHAP, RF-TreeSHAP, XGB-TreeSHAP | MLP-LIME, MLP-Saliency, RF-LIME, XGB-LIME |
| wmc | MLP | MLP-DeepLift, MLP-DeepLiftShap, MLP-GradientShap, MLP-InputXGradient, MLP-IntegratedGradients, MLP-KernelSHAP | MLP-LIME |
| wmc | all models | LR-LinearSHAP, MLP-DeepLift, MLP-DeepLiftShap, MLP-GradientShap, MLP-InputXGradient, MLP-IntegratedGradients, MLP-KernelSHAP, RF-TreeSHAP, XGB-TreeSHAP | MLP-LIME |

## Sign of the mean signed (L1-normalised) attribution, ALL

The sign says whether the feature on average pushes predictions up (+) or down (−) relative to the reference; it depends on the reference (background) and is not the same as the direction above.

| feature | MLP-DeepLift | MLP-DeepLiftShap | MLP-GradientShap | MLP-IntegratedGradients | MLP-Saliency | MLP-InputXGradient | MLP-LIME | MLP-KernelSHAP | LR-LIME | LR-LinearSHAP | RF-LIME | RF-TreeSHAP | XGB-LIME | XGB-TreeSHAP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loc | − | + | + | − | + | − | + | + | + | + | + | − | + | − |
| rfc | + | − | − | + | + | + | + | − | + | + | + | − | + | − |
| ce | − | − | − | − | + | − | + | − | + | + | + | − | + | − |
| cam | − | − | − | − | − | − | − | − | − | − | − | − | − | − |
| cbo | − | − | − | − | + | − | + | − | + | − | + | − | + | − |
| amc | + | + | + | + | + | − | + | + | + | + | + | − | + | − |
| dam | − | − | − | + | + | − | + | − | + | + | + | − | + | − |
| lcom | − | − | − | − | + | − | + | − | + | − | + | − | + | − |
| npm | − | + | + | − | + | − | + | + | + | + | + | + | + | + |
| wmc | − | + | + | − | + | − | + | + | + | + | + | − | + | − |
| mfa | − | − | − | − | − | − | − | − | − | − | − | − | − | − |
| ca | − | + | + | − | + | − | + | + | + | − | + | + | + | − |
| dit | − | + | + | − | − | − | − | + | − | + | − | + | − | + |
| avg_cc | + | + | + | + | + | + | + | + | − | + | + | − | + | − |
| cbm | + | + | + | + | + | + | − | + | + | + | + | − | + | − |
| lcom3 | − | − | − | − | + | − | + | − | + | − | + | − | + | − |
| max_cc | + | − | − | + | + | + | + | − | + | − | + | − | + | − |
| moa | − | − | − | − | + | − | + | − | + | − | + | − | + | − |
| ic | + | + | + | + | − | + | − | + | − | + | − | + | − | + |
| noc | − | − | − | − | + | − | + | − | + | − | + | − | + | − |

Note: MLP-Saliency and the LIME weights are slopes, not differences from a reference; their signs are reported for completeness.
