# Step 4: global feature importance

Experiment `main_time_aware` (24 splits, 9411 test instances; all test instances explained). Seeds: 10 for the gradient-based methods, TreeSHAP and LinearSHAP; 3 (0–2) for LIME and KernelSHAP.

Absolute attributions are normalised per instance to unit L1 norm, averaged over the test instances of a split (and over the seeds of that split), then over splits; every split has the same weight. `ALL` = all 24 time-aware splits.

**Output scale.** Every explanation explains P(defective) except LR-LinearSHAP, which is exact on the log-odds scale (exact interventional SHAP on the probability scale is not linear). LR-LIME explains the probability. Because attributions are L1-normalised per instance before aggregation, and the comparisons are rank-based, the scale difference does not affect them.

**LIME weights are slopes.** With `discretize_continuous=False` (and LIME's default `sample_around_instance=False`), LIME fits a weighted linear surrogate to samples drawn around the training mean, and its weights are that surrogate's coefficients (per standardised unit). They are slopes, like Saliency's gradients, not contributions relative to a reference (slope × (x − reference)) as for the SHAP family, DeepLift, Integrated Gradients and InputXGradient. They also vary little between instances: the median coefficient of variation across the test instances of a split is about 0.1–0.2 for LIME and above 1 for TreeSHAP, DeepLiftShap and Saliency (checked on ant 1.5→1.6, seed 0). LIME's sign therefore reflects the slope, not whether the instance's value raises its prediction.

Cells: feature (mean normalised |attribution|). The full rankings (all groups, all projects, with ranks) are in `step4_global_importance.csv`.

## Top-10 features, ALL splits, all test instances: MLP

| rank | MLP-DeepLift | MLP-DeepLiftShap | MLP-GradientShap | MLP-IntegratedGradients | MLP-Saliency | MLP-InputXGradient | MLP-LIME | MLP-KernelSHAP |
|---|---|---|---|---|---|---|---|---|
| 1 | dam (0.062) | ce (0.062) | ce (0.062) | dam (0.062) | rfc (0.061) | dam (0.061) | rfc (0.069) | ce (0.065) |
| 2 | ce (0.060) | rfc (0.061) | rfc (0.060) | ce (0.060) | ce (0.061) | ce (0.060) | loc (0.066) | dam (0.065) |
| 3 | mfa (0.059) | dam (0.060) | dam (0.060) | mfa (0.060) | loc (0.060) | mfa (0.059) | ce (0.065) | rfc (0.063) |
| 4 | dit (0.058) | loc (0.060) | loc (0.059) | dit (0.058) | cbo (0.054) | dit (0.058) | cbo (0.056) | loc (0.061) |
| 5 | rfc (0.056) | dit (0.056) | dit (0.056) | rfc (0.056) | amc (0.053) | rfc (0.054) | cam (0.055) | mfa (0.056) |
| 6 | loc (0.055) | lcom (0.055) | lcom (0.055) | loc (0.055) | cam (0.052) | loc (0.054) | dam (0.054) | dit (0.054) |
| 7 | lcom (0.054) | mfa (0.054) | mfa (0.054) | lcom (0.054) | npm (0.052) | lcom (0.053) | npm (0.054) | cbo (0.053) |
| 8 | cbm (0.052) | cbo (0.052) | cbo (0.052) | cbm (0.052) | lcom (0.052) | cbm (0.052) | ca (0.053) | cam (0.053) |
| 9 | cbo (0.051) | cam (0.051) | cam (0.051) | cbo (0.051) | wmc (0.050) | cbo (0.051) | amc (0.051) | lcom (0.052) |
| 10 | cam (0.050) | cbm (0.051) | cbm (0.051) | cam (0.050) | ca (0.050) | cam (0.050) | wmc (0.050) | cbm (0.051) |

## Top-10 features, ALL splits, all test instances: LR, RF, XGBoost

| rank | LR-LIME | LR-LinearSHAP | RF-LIME | RF-TreeSHAP | XGB-LIME | XGB-TreeSHAP |
|---|---|---|---|---|---|---|
| 1 | loc (0.079) | loc (0.068) | loc (0.114) | loc (0.106) | loc (0.105) | loc (0.100) |
| 2 | rfc (0.078) | rfc (0.067) | rfc (0.103) | rfc (0.098) | rfc (0.098) | rfc (0.095) |
| 3 | wmc (0.065) | ce (0.062) | amc (0.069) | cam (0.071) | amc (0.071) | cam (0.079) |
| 4 | ce (0.062) | dam (0.058) | ce (0.063) | amc (0.068) | cam (0.070) | amc (0.072) |
| 5 | cbo (0.060) | cbo (0.056) | cam (0.062) | ce (0.066) | ce (0.070) | ce (0.067) |
| 6 | npm (0.052) | cbm (0.055) | wmc (0.061) | cbo (0.065) | avg_cc (0.060) | cbo (0.063) |
| 7 | amc (0.050) | wmc (0.055) | lcom (0.061) | wmc (0.059) | lcom (0.059) | npm (0.058) |
| 8 | lcom (0.048) | mfa (0.054) | cbo (0.059) | npm (0.054) | npm (0.057) | avg_cc (0.057) |
| 9 | cbm (0.048) | lcom (0.052) | npm (0.057) | lcom (0.052) | cbo (0.054) | lcom3 (0.052) |
| 10 | max_cc (0.048) | npm (0.051) | ca (0.046) | avg_cc (0.049) | lcom3 (0.053) | dam (0.051) |

## Top-10 features, ALL splits, actual defective classes only

| rank | MLP-DeepLiftShap | MLP-LIME | LR-LIME | LR-LinearSHAP | RF-LIME | RF-TreeSHAP | XGB-LIME | XGB-TreeSHAP |
|---|---|---|---|---|---|---|---|---|
| 1 | rfc (0.065) | rfc (0.069) | loc (0.079) | rfc (0.072) | loc (0.113) | loc (0.110) | loc (0.104) | loc (0.100) |
| 2 | ce (0.064) | loc (0.065) | rfc (0.078) | loc (0.070) | rfc (0.102) | rfc (0.102) | rfc (0.097) | rfc (0.097) |
| 3 | loc (0.059) | ce (0.064) | wmc (0.065) | ce (0.066) | amc (0.068) | cam (0.070) | cam (0.070) | cam (0.078) |
| 4 | lcom (0.059) | cbo (0.056) | ce (0.062) | wmc (0.059) | ce (0.063) | ce (0.067) | amc (0.070) | amc (0.068) |
| 5 | dam (0.054) | cam (0.055) | cbo (0.060) | cbm (0.057) | cam (0.062) | amc (0.064) | ce (0.070) | ce (0.068) |
| 6 | cbm (0.052) | dam (0.054) | npm (0.052) | lcom (0.056) | lcom (0.062) | wmc (0.062) | lcom (0.060) | cbo (0.060) |
| 7 | wmc (0.052) | ca (0.054) | amc (0.050) | cbo (0.055) | wmc (0.061) | cbo (0.061) | avg_cc (0.060) | avg_cc (0.058) |
| 8 | cbo (0.052) | npm (0.053) | lcom (0.048) | npm (0.052) | cbo (0.059) | lcom (0.055) | npm (0.057) | npm (0.057) |
| 9 | dit (0.052) | amc (0.050) | cbm (0.048) | moa (0.050) | npm (0.057) | npm (0.053) | cbo (0.054) | lcom (0.053) |
| 10 | cam (0.052) | wmc (0.050) | max_cc (0.048) | dam (0.050) | ca (0.047) | avg_cc (0.049) | lcom3 (0.053) | lcom3 (0.051) |

## MLP-DeepLiftShap: top-5 features per project

| project | n_splits | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|---|
| ant | 4 | ce (0.107) | cam (0.076) | rfc (0.066) | loc (0.065) | ca (0.062) |
| camel | 2 | npm (0.123) | ic (0.072) | dam (0.063) | wmc (0.063) | mfa (0.059) |
| ivy | 2 | cbo (0.086) | rfc (0.083) | loc (0.079) | amc (0.070) | ce (0.066) |
| jedit | 3 | dam (0.133) | rfc (0.095) | mfa (0.060) | ca (0.053) | avg_cc (0.053) |
| lucene | 2 | ce (0.080) | cbo (0.076) | rfc (0.073) | lcom3 (0.065) | loc (0.065) |
| poi | 3 | cbm (0.104) | dit (0.087) | lcom (0.075) | npm (0.071) | mfa (0.063) |
| synapse | 2 | dam (0.070) | ce (0.067) | max_cc (0.062) | cbm (0.060) | cam (0.057) |
| velocity | 2 | loc (0.090) | amc (0.089) | dit (0.073) | cbo (0.063) | ce (0.063) |
| xalan | 2 | loc (0.087) | cam (0.087) | lcom (0.070) | npm (0.059) | amc (0.059) |
| xerces | 2 | cbo (0.095) | moa (0.095) | noc (0.077) | dam (0.075) | dit (0.057) |
| ALL | 24 | ce (0.062) | rfc (0.061) | dam (0.060) | loc (0.060) | dit (0.056) |

## Top-5 features per project: MLP-DeepLiftShap, RF-TreeSHAP and LIME (all models)

| project | MLP-DeepLiftShap | RF-TreeSHAP | MLP-LIME | LR-LIME | RF-LIME | XGB-LIME |
|---|---|---|---|---|---|---|
| ant | ce, cam, rfc, loc, ca | cam, loc, rfc, ce, wmc | ce, rfc, cam, loc, ca | ce, cam, rfc, ca, loc | loc, cam, rfc, ce, cbo | cam, ce, rfc, loc, amc |
| camel | npm, ic, dam, wmc, mfa | npm, rfc, ce, mfa, wmc | npm, max_cc, rfc, dam, wmc | npm, rfc, loc, wmc, avg_cc | npm, ca, lcom, wmc, lcom3 | npm, ca, rfc, avg_cc, lcom3 |
| ivy | cbo, rfc, loc, amc, ce | loc, rfc, cbo, amc, cam | cbo, loc, rfc, amc, max_cc | rfc, amc, loc, cbo, avg_cc | loc, rfc, cbo, amc, cam | loc, rfc, cbo, cam, avg_cc |
| jedit | dam, rfc, mfa, ca, avg_cc | rfc, dam, loc, wmc, avg_cc | dam, rfc, ca, avg_cc, mfa | rfc, dam, wmc, avg_cc, ca | rfc, dam, loc, wmc, max_cc | rfc, dam, avg_cc, wmc, lcom3 |
| lucene | ce, cbo, rfc, lcom3, loc | rfc, wmc, loc, cam, lcom3 | ce, rfc, cbo, wmc, loc | rfc, loc, wmc, ce, cbo | rfc, wmc, loc, cbo, npm | rfc, wmc, loc, lcom3, cbo |
| poi | cbm, dit, lcom, npm, mfa | cbm, lcom3, lcom, npm, mfa | npm, loc, dit, cbm, lcom | cbm, loc, dit, npm, lcom | lcom, lcom3, loc, amc, cbm | loc, lcom, lcom3, cbm, max_cc |
| synapse | dam, ce, max_cc, cbm, cam | amc, loc, ce, rfc, dam | ce, cbm, moa, cam, max_cc | ce, lcom, noc, max_cc, cbo | ce, loc, amc, lcom, cbo | amc, ce, lcom, mfa, max_cc |
| velocity | loc, amc, dit, cbo, ce | npm, amc, loc, dit, rfc | loc, amc, cbo, dit, ce | amc, loc, dit, ic, cbm | npm, amc, loc, ce, cbo | npm, loc, amc, cbo, dit |
| xalan | loc, cam, lcom, npm, amc | loc, lcom, amc, rfc, avg_cc | cam, loc, ca, lcom, amc | loc, wmc, mfa, rfc, cbm | loc, amc, rfc, lcom3, lcom | loc, lcom3, lcom, cam, amc |
| xerces | cbo, moa, noc, dam, dit | cbo, noc, ce, cam, loc | noc, moa, cbo, dam, cam | cbo, moa, avg_cc, max_cc, noc | noc, cam, ce, cbo, amc | noc, cam, cbo, amc, ce |

## Outcome groups: MLP-DeepLiftShap (ALL)

Outcome groups at each model's validation-MCC threshold. Cells: importance (rank within the group); `X−all` = importance in group X minus importance over all instances. Top-10 features by overall importance.

| feature | all | TP | FP | FN | TN | defective | TP−all | FP−all | FN−all | TN−all |
|---|---|---|---|---|---|---|---|---|---|---|
| ce | 0.062 (1) | 0.071 (2) | 0.062 (2) | 0.057 (5) | 0.063 (2) | 0.064 (2) | +0.009 | +0.000 | -0.005 | +0.001 |
| rfc | 0.061 (2) | 0.076 (1) | 0.065 (1) | 0.056 (6) | 0.061 (4) | 0.065 (1) | +0.015 | +0.004 | -0.005 | -0.000 |
| dam | 0.060 (3) | 0.047 (12) | 0.054 (8) | 0.060 (1) | 0.062 (3) | 0.054 (5) | -0.013 | -0.006 | +0.000 | +0.002 |
| loc | 0.060 (4) | 0.063 (4) | 0.060 (4) | 0.057 (4) | 0.065 (1) | 0.059 (3) | +0.004 | +0.000 | -0.002 | +0.005 |
| dit | 0.056 (5) | 0.046 (13) | 0.056 (5) | 0.059 (2) | 0.061 (5) | 0.052 (9) | -0.010 | +0.000 | +0.003 | +0.005 |
| lcom | 0.055 (6) | 0.064 (3) | 0.061 (3) | 0.052 (9) | 0.050 (11) | 0.059 (4) | +0.009 | +0.006 | -0.002 | -0.005 |
| mfa | 0.054 (7) | 0.040 (16) | 0.049 (12) | 0.059 (3) | 0.056 (6) | 0.049 (12) | -0.015 | -0.006 | +0.004 | +0.002 |
| cbo | 0.052 (8) | 0.057 (6) | 0.054 (7) | 0.044 (15) | 0.051 (8) | 0.052 (8) | +0.005 | +0.002 | -0.008 | -0.001 |
| cam | 0.051 (9) | 0.052 (8) | 0.052 (11) | 0.053 (8) | 0.050 (10) | 0.052 (10) | +0.001 | +0.001 | +0.001 | -0.001 |
| cbm | 0.051 (10) | 0.047 (11) | 0.053 (9) | 0.054 (7) | 0.046 (14) | 0.052 (6) | -0.004 | +0.002 | +0.003 | -0.005 |
