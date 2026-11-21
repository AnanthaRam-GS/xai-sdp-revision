# Step 4: completeness and additivity checks

## Changes since v1

v1 is archived in `reports/archive/step4_v1/`. v2 changes two settings: Integrated Gradients now uses 512 Gauss-Legendre steps (v1: 64), and LIME samples its perturbations around the explained instance (`sample_around_instance=True`; v1 sampled around the training mean). The v1 LIME results are kept in the store as `LIME_mean_sampling`. Everything else is unchanged.

Integrated Gradients completeness, v1 vs v2:

| version | median | 95th pct | max | status |
|---|---|---|---|---|
| v1 (64 steps) | 4.18e-04 | 2.77e-03 | 5.17e-02 | fail |
| v2 (512 steps) | 5.22e-05 | 3.55e-04 | 5.35e-03 | pass |


Per instance: |sum(attributions) − (f(x) − f(reference))|.

* DeepLift and IntegratedGradients: reference = the background mean (single baseline);
* DeepLiftShap: reference output = the mean output over the background set;
* TreeSHAP: additivity, sum(SHAP) + expected value vs the predicted probability;
* LinearSHAP (LR): additivity on the log-odds scale (decision function).

Tolerance: the 95th percentile must not exceed 1e-03. Captum methods run on a float64 copy of the MLP.

## Summary over all splits and seeds

| explanation | check | instances | median | 95th pct | max | status |
|---|---|---|---|---|---|---|
| LR-LinearSHAP | additivity | 94110 | 1.11e-16 | 6.66e-16 | 6.55e-15 | pass |
| MLP-DeepLift | completeness | 94110 | 5.55e-17 | 2.22e-16 | 1.03e-15 | pass |
| MLP-DeepLiftShap | completeness | 94110 | 5.55e-17 | 1.53e-16 | 6.66e-16 | pass |
| MLP-IntegratedGradients | completeness | 94110 | 5.22e-05 | 3.55e-04 | 5.35e-03 | pass |
| RF-TreeSHAP | additivity | 94110 | 2.64e-09 | 1.48e-08 | 4.24e-08 | pass |
| XGB-TreeSHAP | additivity | 94110 | 2.42e-08 | 9.44e-08 | 3.77e-07 | pass |

## Splits whose 95th percentile exceeds the tolerance

Per split, pooled over seeds; the per-seed table below gives how many of the 10 seeds exceed it.

_(none)_

### Per seed

| explanation | split | seeds flagged (of 10) | worst seed 95th pct |
|---|---|---|---|
| MLP-IntegratedGradients | ta-consecutive:camel:1.2>1.4 | 2 | 2.11e-03 |
| MLP-IntegratedGradients | ta-consecutive:jedit:4.0>4.1 | 1 | 2.09e-03 |
| MLP-IntegratedGradients | ta-consecutive:poi:2.5>3.0 | 1 | 1.46e-03 |

## Notes

* IntegratedGradients approximates a path integral with Gauss-Legendre nodes; the MLP's ReLU kinks and the sigmoid make the integrand non-smooth, so a residual error remains. It shrinks roughly as 1/steps: on the worst v1 split (camel 1.2→1.4, seed 0) the 95th percentile is 1.35e-02, 3.30e-03, 8.47e-04 and 2.22e-04 with 16, 64, 256 and 1024 steps. v1 used 64 steps; v2 uses 512.
* TreeSHAP for the random forests: shap's interventional algorithm stores split thresholds as float32 while scikit-learn compares float32 inputs with float64 thresholds; where rounding a threshold crosses a data value shap took the wrong branch (additivity errors up to 0.02 in the smoke test). The forest given to shap is a copy with thresholds rounded down to float32, which partitions float32 inputs exactly as the original model; predictions are unchanged.
* Reproduction: every loaded model reproduces its stored step 3 test predictions; maximum absolute difference 0.0 over 960 (split, model, seed) units.
