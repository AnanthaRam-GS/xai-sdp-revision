# Step 4: completeness and additivity checks

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
| MLP-IntegratedGradients | completeness | 94110 | 4.18e-04 | 2.77e-03 | 5.17e-02 | WARNING: > 1e-03 |
| RF-TreeSHAP | additivity | 94110 | 2.64e-09 | 1.48e-08 | 4.24e-08 | pass |
| XGB-TreeSHAP | additivity | 94110 | 2.42e-08 | 9.44e-08 | 3.77e-07 | pass |

## Splits whose 95th percentile exceeds the tolerance

Per split, pooled over seeds; the per-seed table below gives how many of the 10 seeds exceed it.

| explanation | split | 95th pct | max |
|---|---|---|---|
| MLP-IntegratedGradients | ta-consecutive:camel:1.2>1.4 | 7.55e-03 | 5.17e-02 |
| MLP-IntegratedGradients | ta-consecutive:jedit:4.0>4.1 | 5.98e-03 | 2.70e-02 |
| MLP-IntegratedGradients | ta-consecutive:poi:2.5>3.0 | 5.88e-03 | 2.47e-02 |
| MLP-IntegratedGradients | ta-consecutive:xerces:init>1.2 | 3.35e-03 | 1.13e-02 |
| MLP-IntegratedGradients | ta-consecutive:jedit:4.1>4.2 | 3.30e-03 | 1.17e-02 |
| MLP-IntegratedGradients | ta-consecutive:xerces:1.2>1.3 | 2.88e-03 | 9.33e-03 |
| MLP-IntegratedGradients | ta-consecutive:ant:1.5>1.6 | 2.81e-03 | 8.62e-03 |
| MLP-IntegratedGradients | ta-consecutive:xalan:2.5>2.6 | 2.51e-03 | 1.04e-02 |
| MLP-IntegratedGradients | ta-consecutive:ant:1.6>1.7 | 2.46e-03 | 1.52e-02 |
| MLP-IntegratedGradients | ta-consecutive:ant:1.4>1.5 | 2.07e-03 | 5.89e-03 |
| MLP-IntegratedGradients | ta-consecutive:poi:1.5>2.0 | 2.00e-03 | 9.10e-03 |
| MLP-IntegratedGradients | ta-consecutive:velocity:1.5>1.6 | 1.92e-03 | 7.35e-03 |
| MLP-IntegratedGradients | ta-consecutive:lucene:2.0>2.2 | 1.66e-03 | 4.71e-03 |
| MLP-IntegratedGradients | ta-consecutive:camel:1.4>1.6 | 1.49e-03 | 7.53e-03 |
| MLP-IntegratedGradients | ta-consecutive:jedit:3.2>4.0 | 1.35e-03 | 5.29e-03 |
| MLP-IntegratedGradients | ta-consecutive:lucene:2.2>2.4 | 1.28e-03 | 5.57e-03 |
| MLP-IntegratedGradients | ta-consecutive:ivy:1.1>1.4 | 1.19e-03 | 2.71e-03 |
| MLP-IntegratedGradients | ta-consecutive:xalan:2.4>2.5 | 1.15e-03 | 4.06e-03 |
| MLP-IntegratedGradients | ta-consecutive:synapse:1.0>1.1 | 1.12e-03 | 2.65e-03 |
| MLP-IntegratedGradients | ta-consecutive:velocity:1.4>1.5 | 1.07e-03 | 3.29e-03 |
| MLP-IntegratedGradients | ta-consecutive:poi:2.0>2.5 | 1.06e-03 | 6.00e-03 |

### Per seed

| explanation | split | seeds flagged (of 10) | worst seed 95th pct |
|---|---|---|---|
| MLP-IntegratedGradients | ta-consecutive:camel:1.2>1.4 | 10 | 1.56e-02 |
| MLP-IntegratedGradients | ta-consecutive:jedit:4.0>4.1 | 10 | 1.36e-02 |
| MLP-IntegratedGradients | ta-consecutive:poi:2.5>3.0 | 10 | 1.08e-02 |
| MLP-IntegratedGradients | ta-consecutive:jedit:4.1>4.2 | 10 | 6.93e-03 |
| MLP-IntegratedGradients | ta-consecutive:poi:1.5>2.0 | 7 | 6.03e-03 |
| MLP-IntegratedGradients | ta-consecutive:xerces:init>1.2 | 10 | 6.00e-03 |
| MLP-IntegratedGradients | ta-consecutive:ant:1.6>1.7 | 10 | 5.79e-03 |
| MLP-IntegratedGradients | ta-consecutive:xerces:1.2>1.3 | 9 | 5.34e-03 |
| MLP-IntegratedGradients | ta-consecutive:ant:1.5>1.6 | 10 | 4.52e-03 |
| MLP-IntegratedGradients | ta-consecutive:xalan:2.5>2.6 | 10 | 3.77e-03 |
| MLP-IntegratedGradients | ta-consecutive:camel:1.4>1.6 | 6 | 3.28e-03 |
| MLP-IntegratedGradients | ta-consecutive:lucene:2.2>2.4 | 7 | 3.03e-03 |
| MLP-IntegratedGradients | ta-consecutive:ant:1.4>1.5 | 10 | 2.88e-03 |
| MLP-IntegratedGradients | ta-consecutive:velocity:1.5>1.6 | 10 | 2.75e-03 |
| MLP-IntegratedGradients | ta-consecutive:poi:2.0>2.5 | 3 | 2.40e-03 |
| MLP-IntegratedGradients | ta-consecutive:lucene:2.0>2.2 | 8 | 2.34e-03 |
| MLP-IntegratedGradients | ta-consecutive:jedit:3.2>4.0 | 7 | 2.21e-03 |
| MLP-IntegratedGradients | ta-consecutive:xalan:2.4>2.5 | 5 | 1.98e-03 |
| MLP-IntegratedGradients | ta-consecutive:velocity:1.4>1.5 | 4 | 1.73e-03 |
| MLP-IntegratedGradients | ta-consecutive:synapse:1.0>1.1 | 6 | 1.52e-03 |
| MLP-IntegratedGradients | ta-consecutive:ivy:1.1>1.4 | 8 | 1.43e-03 |
| MLP-IntegratedGradients | ta-consecutive:ant:1.3>1.4 | 3 | 1.15e-03 |

## Notes

* IntegratedGradients approximates a path integral with 64 Gauss-Legendre nodes; the MLP's ReLU kinks and the sigmoid make the integrand non-smooth, so a residual error is expected. It shrinks roughly as 1/steps: on the worst split (camel 1.2→1.4, seed 0) the 95th percentile is 1.35e-02, 3.30e-03, 8.47e-04 and 2.22e-04 with 16, 64, 256 and 1024 steps. The configured 64 steps were kept; the error is reported, not corrected.
* TreeSHAP for the random forests: shap's interventional algorithm stores split thresholds as float32 while scikit-learn compares float32 inputs with float64 thresholds; where rounding a threshold crosses a data value shap took the wrong branch (additivity errors up to 0.02 in the smoke test). The forest given to shap is a copy with thresholds rounded down to float32, which partitions float32 inputs exactly as the original model; predictions are unchanged.
* Reproduction: every loaded model reproduces its stored step 3 test predictions; maximum absolute difference 0.0 over 960 (split, model, seed) units.
