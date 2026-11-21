# Step 5: explanations across preprocessing settings

Persisted `ablation_preprocessing` models (step 3): MLP and LR with preprocessing none / standard / log_standard, 10 seeds, 24 time-aware splits, all test instances. MLP: DeepLiftShap (step 4 background, in each model's own input space) and DeepLift (baseline = background mean); for preprocessing `none` also DeepLift with an all-zero baseline (the original paper's setting: unscaled inputs, zero reference). LR: LinearSHAP (log-odds). Aggregation as in step 4 (L1-normalised |attribution|, instances → split (seeds averaged) → ALL / project).

## Global top-5 (ALL)

| explanation | top-5 (ALL) |
|---|---|
| MLP-DeepLiftShap · none | loc (0.344), lcom (0.182), amc (0.095), rfc (0.089), cbo (0.042) |
| MLP-DeepLiftShap · standard | dam (0.081), dit (0.072), rfc (0.070), mfa (0.070), cam (0.070) |
| MLP-DeepLiftShap · log_standard | ce (0.062), rfc (0.061), dam (0.060), loc (0.060), dit (0.056) |
| MLP-DeepLift · none | loc (0.347), lcom (0.270), rfc (0.078), amc (0.073), ca (0.033) |
| MLP-DeepLift · standard | dam (0.081), mfa (0.070), dit (0.070), cam (0.067), rfc (0.066) |
| MLP-DeepLift · log_standard | dam (0.062), ce (0.060), mfa (0.059), dit (0.058), rfc (0.056) |
| MLP-DeepLift_zero_baseline · none | loc (0.326), rfc (0.109), amc (0.104), lcom (0.096), cbo (0.065) |
| LR-LinearSHAP · none | rfc (0.196), loc (0.123), cbo (0.089), wmc (0.084), ca (0.078) |
| LR-LinearSHAP · standard | rfc (0.080), dam (0.074), mfa (0.073), ce (0.072), cam (0.069) |
| LR-LinearSHAP · log_standard | loc (0.068), rfc (0.067), ce (0.062), dam (0.058), cbo (0.056) |

## Global top-5 per project

| project | MLP-DeepLiftShap · none | MLP-DeepLiftShap · standard | MLP-DeepLiftShap · log_standard | MLP-DeepLift · none | MLP-DeepLift · standard | MLP-DeepLift · log_standard | MLP-DeepLift_zero_baseline · none | LR-LinearSHAP · none | LR-LinearSHAP · standard | LR-LinearSHAP · log_standard |
|---|---|---|---|---|---|---|---|---|---|---|
| ant | loc, lcom, amc, rfc, ca | ce, rfc, cam, mfa, loc | ce, cam, rfc, loc, ca | loc, lcom, rfc, amc, ca | ce, rfc, cam, mfa, loc | ce, cam, mfa, ca, rfc | loc, amc, rfc, lcom, wmc | rfc, loc, lcom, ca, wmc | ce, rfc, cam, loc, dam | ce, cam, ca, rfc, loc |
| camel | loc, lcom, rfc, amc, npm | ic, mfa, npm, ce, dam | npm, ic, dam, wmc, mfa | loc, lcom, rfc, cbo, ce | ic, mfa, npm, ce, cam | npm, ic, dit, dam, mfa | loc, rfc, lcom, amc, cbo | npm, wmc, cbo, ce, ca | wmc, npm, ce, cbo, ic | npm, rfc, loc, wmc, max_cc |
| ivy | loc, lcom, rfc, amc, cbo | cam, amc, max_cc, dit, ce | cbo, rfc, loc, amc, ce | lcom, loc, rfc, amc, cbo | cam, amc, max_cc, dit, rfc | cbo, rfc, loc, ce, amc | loc, lcom, rfc, cbo, amc | loc, rfc, amc, lcom, wmc | amc, avg_cc, cam, dit, max_cc | rfc, cbo, amc, loc, cam |
| jedit | loc, lcom, rfc, amc, cbo | dam, mfa, dit, rfc, avg_cc | dam, rfc, mfa, ca, avg_cc | lcom, loc, rfc, amc, ca | dam, mfa, dit, rfc, moa | dam, rfc, mfa, moa, ca | loc, rfc, amc, cbo, lcom | rfc, cbo, ca, loc, ce | dam, mfa, rfc, cbm, dit | dam, rfc, ca, cbo, avg_cc |
| lucene | loc, lcom, rfc, amc, cbo | lcom3, dam, wmc, rfc, ce | ce, cbo, rfc, lcom3, loc | loc, lcom, amc, rfc, max_cc | dam, lcom3, ce, wmc, rfc | cbo, ce, rfc, lcom3, dit | loc, rfc, amc, lcom, cbo | rfc, amc, lcom, wmc, loc | wmc, dam, rfc, lcom3, ce | ce, rfc, loc, cbo, wmc |
| poi | loc, lcom, amc, rfc, cbm | cbm, dit, mfa, npm, ic | cbm, dit, lcom, npm, mfa | loc, lcom, amc, cbm, rfc | cbm, dit, mfa, npm, ic | cbm, dit, lcom, mfa, ic | loc, lcom, rfc, amc, wmc | cbm, loc, rfc, npm, wmc | cbm, dit, ic, mfa, rfc | cbm, dit, lcom, ic, loc |
| synapse | loc, lcom, amc, rfc, cbo | ce, dam, mfa, max_cc, loc | dam, ce, max_cc, cbm, cam | loc, lcom, amc, rfc, cbo | dam, ce, mfa, max_cc, loc | dam, ce, max_cc, mfa, cam | loc, amc, rfc, cbo, lcom | loc, rfc, lcom, ce, wmc | ce, dam, mfa, cam, loc | ce, mfa, lcom, dam, max_cc |
| velocity | loc, lcom, rfc, amc, cbo | lcom3, amc, dam, dit, cam | loc, amc, dit, cbo, ce | loc, lcom, rfc, amc, cbo | lcom3, dam, amc, dit, cam | loc, amc, mfa, dam, dit | loc, rfc, cbo, amc, lcom | cbo, rfc, amc, ca, npm | amc, rfc, mfa, cbo, lcom3 | mfa, dit, loc, ic, amc |
| xalan | loc, lcom, amc, rfc, ce | rfc, cam, loc, cbm, mfa | loc, cam, lcom, npm, amc | loc, lcom, rfc, amc, ce | rfc, loc, cam, cbm, mfa | loc, cam, lcom, cbm, amc | loc, amc, rfc, lcom, ce | rfc, wmc, loc, cbm, ce | rfc, cam, mfa, dit, cbm | loc, mfa, wmc, cbm, dit |
| xerces | loc, lcom, amc, rfc, cbo | dam, dit, cam, mfa, avg_cc | cbo, moa, noc, dam, dit | loc, lcom, amc, rfc, npm | dam, cam, dit, mfa, npm | moa, cbo, noc, dam, dit | loc, lcom, amc, rfc, wmc | amc, cbo, loc, rfc, ca | cbo, dam, ce, lcom3, ca | cbo, moa, dam, lcom3, avg_cc |

## Size share (ALL)

Equal attribution would give loc 0.05 and the size group {loc, wmc, rfc, npm} 0.20.

| explanation | loc share | size-group share | % loc top-1 |
|---|---|---|---|
| MLP-DeepLiftShap · none | 0.344 | 0.503 | 58.5 |
| MLP-DeepLiftShap · standard | 0.042 | 0.197 | 2.2 |
| MLP-DeepLiftShap · log_standard | 0.060 | 0.220 | 6.5 |
| MLP-DeepLift · none | 0.347 | 0.487 | 51.8 |
| MLP-DeepLift · standard | 0.041 | 0.193 | 2.2 |
| MLP-DeepLift · log_standard | 0.055 | 0.204 | 5.5 |
| MLP-DeepLift_zero_baseline · none | 0.326 | 0.528 | 57.0 |
| LR-LinearSHAP · none | 0.123 | 0.464 | 14.3 |
| LR-LinearSHAP · standard | 0.045 | 0.222 | 2.3 |
| LR-LinearSHAP · log_standard | 0.068 | 0.241 | 9.2 |

## loc share per project

| project | MLP-DeepLiftShap · none | MLP-DeepLiftShap · standard | MLP-DeepLiftShap · log_standard | MLP-DeepLift · none | MLP-DeepLift · standard | MLP-DeepLift · log_standard | MLP-DeepLift_zero_baseline · none | LR-LinearSHAP · none | LR-LinearSHAP · standard | LR-LinearSHAP · log_standard |
|---|---|---|---|---|---|---|---|---|---|---|
| ant | 0.383 | 0.067 | 0.065 | 0.368 | 0.061 | 0.060 | 0.348 | 0.133 | 0.071 | 0.068 |
| camel | 0.257 | 0.041 | 0.042 | 0.276 | 0.040 | 0.038 | 0.273 | 0.022 | 0.034 | 0.099 |
| ivy | 0.370 | 0.026 | 0.079 | 0.351 | 0.029 | 0.068 | 0.321 | 0.243 | 0.033 | 0.079 |
| jedit | 0.331 | 0.016 | 0.046 | 0.251 | 0.017 | 0.044 | 0.314 | 0.072 | 0.021 | 0.038 |
| lucene | 0.333 | 0.035 | 0.065 | 0.348 | 0.036 | 0.058 | 0.338 | 0.107 | 0.043 | 0.076 |
| poi | 0.360 | 0.030 | 0.057 | 0.375 | 0.030 | 0.050 | 0.373 | 0.144 | 0.032 | 0.066 |
| synapse | 0.338 | 0.066 | 0.044 | 0.331 | 0.064 | 0.042 | 0.325 | 0.245 | 0.072 | 0.043 |
| velocity | 0.319 | 0.021 | 0.090 | 0.387 | 0.021 | 0.086 | 0.274 | 0.055 | 0.019 | 0.086 |
| xalan | 0.452 | 0.083 | 0.087 | 0.467 | 0.082 | 0.080 | 0.451 | 0.100 | 0.080 | 0.125 |
| xerces | 0.262 | 0.027 | 0.024 | 0.325 | 0.029 | 0.024 | 0.199 | 0.109 | 0.034 | 0.015 |

## Direction (Spearman of raw value vs signed attribution, ALL)

`+` / `−` = mean ρ ≥ 0.1 / ≤ −0.1, `0` otherwise.

| explanation | loc | rfc | amc | lcom | cam |
|---|---|---|---|---|---|
| MLP-DeepLiftShap · none | + +0.67 | 0 +0.05 | − -0.52 | + +0.46 | − -0.36 |
| MLP-DeepLiftShap · standard | + +0.48 | + +0.78 | + +0.39 | + +0.18 | − -0.46 |
| MLP-DeepLiftShap · log_standard | + +0.59 | + +0.76 | + +0.57 | + +0.54 | − -0.31 |
| MLP-DeepLift · none | + +0.56 | + +0.11 | − -0.40 | + +0.34 | − -0.36 |
| MLP-DeepLift · standard | + +0.46 | + +0.74 | + +0.37 | + +0.16 | − -0.46 |
| MLP-DeepLift · log_standard | + +0.57 | + +0.73 | + +0.54 | + +0.50 | − -0.29 |
| MLP-DeepLift_zero_baseline · none | + +0.32 | 0 +0.07 | − -0.29 | + +0.38 | − -0.36 |
| LR-LinearSHAP · none | + +0.18 | + +0.60 | + +0.16 | − -0.17 | − -0.33 |
| LR-LinearSHAP · standard | + +0.48 | + +0.82 | + +0.34 | 0 +0.09 | − -0.56 |
| LR-LinearSHAP · log_standard | + +0.38 | + +0.80 | + +0.57 | + +0.52 | − -0.42 |

## Statistical comparison of loc share and size-group share

Per-split means (seeds averaged; 24 splits), Wilcoxon signed-rank (two-sided) between preprocessing settings within each explanation family, Holm-corrected within (family, metric), Cliff's delta (a vs b; positive = a larger). `none (zero baseline)` is DeepLift with the zero baseline.

| family | metric | a | b | n | mean_a | mean_b | p_value | p_holm | cliffs_delta | magnitude |
|---|---|---|---|---|---|---|---|---|---|---|
| LR-LinearSHAP | loc share | log_standard | none | 24 | 0.068 | 0.123 | 0.046 | 0.046 | -0.212 | small |
| LR-LinearSHAP | loc share | log_standard | standard | 24 | 0.068 | 0.045 | 0.013 | 0.025 | 0.368 | medium |
| LR-LinearSHAP | loc share | none | standard | 24 | 0.123 | 0.045 | <0.001 | <0.001 | 0.493 | large |
| LR-LinearSHAP | size-group share | log_standard | none | 24 | 0.241 | 0.464 | <0.001 | <0.001 | -0.788 | large |
| LR-LinearSHAP | size-group share | log_standard | standard | 24 | 0.241 | 0.222 | 0.114 | 0.114 | 0.139 | negligible |
| LR-LinearSHAP | size-group share | none | standard | 24 | 0.464 | 0.222 | <0.001 | <0.001 | 0.865 | large |
| MLP-DeepLift | loc share | log_standard | none | 24 | 0.055 | 0.347 | <0.001 | <0.001 | -1.000 | large |
| MLP-DeepLift | loc share | log_standard | none (zero baseline) | 24 | 0.055 | 0.326 | <0.001 | <0.001 | -1.000 | large |
| MLP-DeepLift | loc share | log_standard | standard | 24 | 0.055 | 0.041 | 0.016 | 0.033 | 0.413 | medium |
| MLP-DeepLift | loc share | none | none (zero baseline) | 24 | 0.347 | 0.326 | 0.114 | 0.114 | 0.111 | negligible |
| MLP-DeepLift | loc share | none | standard | 24 | 0.347 | 0.041 | <0.001 | <0.001 | 1.000 | large |
| MLP-DeepLift | loc share | none (zero baseline) | standard | 24 | 0.326 | 0.041 | <0.001 | <0.001 | 1.000 | large |
| MLP-DeepLift | size-group share | log_standard | none | 24 | 0.204 | 0.487 | <0.001 | <0.001 | -1.000 | large |
| MLP-DeepLift | size-group share | log_standard | none (zero baseline) | 24 | 0.204 | 0.528 | <0.001 | <0.001 | -1.000 | large |
| MLP-DeepLift | size-group share | log_standard | standard | 24 | 0.204 | 0.193 | 0.390 | 0.390 | 0.153 | small |
| MLP-DeepLift | size-group share | none | none (zero baseline) | 24 | 0.487 | 0.528 | 0.002 | 0.004 | -0.295 | small |
| MLP-DeepLift | size-group share | none | standard | 24 | 0.487 | 0.193 | <0.001 | <0.001 | 0.993 | large |
| MLP-DeepLift | size-group share | none (zero baseline) | standard | 24 | 0.528 | 0.193 | <0.001 | <0.001 | 1.000 | large |
| MLP-DeepLiftShap | loc share | log_standard | none | 24 | 0.060 | 0.344 | <0.001 | <0.001 | -1.000 | large |
| MLP-DeepLiftShap | loc share | log_standard | standard | 24 | 0.060 | 0.042 | 0.007 | 0.007 | 0.444 | medium |
| MLP-DeepLiftShap | loc share | none | standard | 24 | 0.344 | 0.042 | <0.001 | <0.001 | 1.000 | large |
| MLP-DeepLiftShap | size-group share | log_standard | none | 24 | 0.220 | 0.503 | <0.001 | <0.001 | -0.997 | large |
| MLP-DeepLiftShap | size-group share | log_standard | standard | 24 | 0.220 | 0.197 | 0.089 | 0.089 | 0.233 | small |
| MLP-DeepLiftShap | size-group share | none | standard | 24 | 0.503 | 0.197 | <0.001 | <0.001 | 0.997 | large |
