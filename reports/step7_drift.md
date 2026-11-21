# Step 7: data drift and its relation to stability

Per release pair (raw metrics): mean and maximum two-sample Kolmogorov–Smirnov statistic over the 20 metrics, the change in the defective ratio, and a classifier two-sample test (random forest, 200 trees, 5-fold stratified CV AUC, seed 0; 0.5 = indistinguishable) with its top-3 feature importances. Relationships: Spearman ρ with Holm correction across all tests of a family. **The number of pairs is small (n = 24 train→test pairs, n = 14 consecutive pairs), so these correlations are descriptive and underpowered.**

## Drift between consecutive test releases

| project | comparison | mean_ks | max_ks | max_ks_feature | delta_ratio | c2st_auc | c2st_top_features |
|---|---|---|---|---|---|---|---|
| ant | ant 1.3>1.4 vs 1.4>1.5 | 0.057 | 0.104 | rfc | -0.107 | 0.607 | amc, loc, rfc |
| ant | ant 1.4>1.5 vs 1.5>1.6 | 0.039 | 0.068 | avg_cc | 0.155 | 0.507 | loc, amc, rfc |
| ant | ant 1.5>1.6 vs 1.6>1.7 | 0.048 | 0.108 | amc | -0.036 | 0.618 | amc, loc, cbo |
| camel | camel 1.2>1.4 vs 1.4>1.6 | 0.022 | 0.035 | cbm | 0.027 | 0.533 | loc, amc, mfa |
| ivy | ivy 1.1>1.4 vs 1.4>2.0 | 0.051 | 0.107 | cbo | 0.048 | 0.379 | ce, cbo, loc |
| jedit | jedit 3.2>4.0 vs 4.0>4.1 | 0.034 | 0.056 | dam | 0.011 | 0.572 | amc, loc, rfc |
| jedit | jedit 4.0>4.1 vs 4.1>4.2 | 0.043 | 0.081 | mfa | -0.124 | 0.487 | amc, loc, cbo |
| lucene | lucene 2.0>2.2 vs 2.2>2.4 | 0.057 | 0.135 | dam | 0.011 | 0.635 | loc, amc, cam |
| poi | poi 1.5>2.0 vs 2.0>2.5 | 0.034 | 0.064 | cbo | 0.507 | 0.528 | loc, amc, cam |
| poi | poi 2.0>2.5 vs 2.5>3.0 | 0.080 | 0.316 | cbm | 0.009 | 0.806 | cbm, cam, ce |
| synapse | synapse 1.0>1.1 vs 1.1>1.2 | 0.042 | 0.067 | ic | 0.074 | 0.583 | loc, amc, rfc |
| velocity | velocity 1.4>1.5 vs 1.5>1.6 | 0.032 | 0.065 | mfa | -0.312 | 0.591 | mfa, loc, amc |
| xalan | xalan 2.4>2.5 vs 2.5>2.6 | 0.018 | 0.034 | amc | -0.046 | 0.518 | amc, loc, rfc |
| xerces | xerces init>1.2 vs 1.2>1.3 | 0.016 | 0.028 | ca | 0.019 | 0.499 | loc, cam, amc |

## Drift between training and test release (time-aware splits)

| split_id | mean_ks | max_ks | max_ks_feature | delta_ratio | c2st_auc | c2st_top_features |
|---|---|---|---|---|---|---|
| ta-consecutive:ant:1.3>1.4 | 0.048 | 0.070 | lcom3 | 0.057 | 0.506 | loc, amc, rfc |
| ta-consecutive:ant:1.4>1.5 | 0.057 | 0.104 | rfc | -0.107 | 0.610 | loc, rfc, amc |
| ta-consecutive:ant:1.5>1.6 | 0.039 | 0.068 | avg_cc | 0.155 | 0.503 | loc, amc, rfc |
| ta-consecutive:ant:1.6>1.7 | 0.048 | 0.108 | amc | -0.036 | 0.620 | amc, loc, cbo |
| ta-consecutive:camel:1.2>1.4 | 0.028 | 0.067 | moa | -0.188 | 0.590 | amc, loc, cbo |
| ta-consecutive:camel:1.4>1.6 | 0.022 | 0.035 | cbm | 0.027 | 0.529 | loc, amc, mfa |
| ta-consecutive:ivy:1.1>1.4 | 0.076 | 0.245 | ce | -0.512 | 0.669 | ce, loc, cam |
| ta-consecutive:ivy:1.4>2.0 | 0.051 | 0.107 | cbo | 0.048 | 0.413 | ce, cbo, loc |
| ta-consecutive:jedit:3.2>4.0 | 0.028 | 0.038 | amc | -0.090 | 0.508 | amc, loc, rfc |
| ta-consecutive:jedit:4.0>4.1 | 0.034 | 0.056 | dam | 0.011 | 0.557 | amc, loc, rfc |
| ta-consecutive:jedit:4.1>4.2 | 0.043 | 0.081 | mfa | -0.124 | 0.502 | loc, amc, cbo |
| ta-consecutive:lucene:2.0>2.2 | 0.029 | 0.060 | amc | 0.110 | 0.550 | loc, amc, cbo |
| ta-consecutive:lucene:2.2>2.4 | 0.057 | 0.135 | dam | 0.011 | 0.643 | amc, loc, cam |
| ta-consecutive:poi:1.5>2.0 | 0.053 | 0.091 | lcom | -0.477 | 0.666 | amc, loc, lcom3 |
| ta-consecutive:poi:2.0>2.5 | 0.034 | 0.064 | cbo | 0.507 | 0.533 | loc, amc, cam |
| ta-consecutive:poi:2.5>3.0 | 0.080 | 0.316 | cbm | 0.009 | 0.803 | cbm, cam, amc |
| ta-consecutive:synapse:1.0>1.1 | 0.133 | 0.424 | dam | 0.172 | 0.885 | dam, lcom3, cam |
| ta-consecutive:synapse:1.1>1.2 | 0.042 | 0.067 | ic | 0.074 | 0.598 | loc, amc, cbo |
| ta-consecutive:velocity:1.4>1.5 | 0.072 | 0.154 | mfa | -0.066 | 0.663 | mfa, amc, loc |
| ta-consecutive:velocity:1.5>1.6 | 0.032 | 0.065 | mfa | -0.312 | 0.608 | mfa, loc, amc |
| ta-consecutive:xalan:2.4>2.5 | 0.017 | 0.043 | cbo | 0.332 | 0.537 | amc, loc, cbo |
| ta-consecutive:xalan:2.5>2.6 | 0.018 | 0.034 | amc | -0.046 | 0.513 | amc, loc, rfc |
| ta-consecutive:xerces:init>1.2 | 0.117 | 0.228 | lcom | -0.277 | 0.704 | loc, cbo, cam |
| ta-consecutive:xerces:1.2>1.3 | 0.016 | 0.028 | ca | 0.019 | 0.512 | loc, cam, amc |

## (i) Train→test drift vs test AUC (step 3, mean over seeds)

| x | y | n | rho | p_value | p_holm |
|---|---|---|---|---|---|
| mean_ks | AUC LR | 24 | +0.07 | 0.734 | 1.000 |
| mean_ks | AUC MLP | 24 | +0.12 | 0.590 | 1.000 |
| mean_ks | AUC RF | 24 | +0.06 | 0.787 | 1.000 |
| mean_ks | AUC XGB | 24 | -0.06 | 0.768 | 1.000 |
| max_ks | AUC LR | 24 | +0.13 | 0.549 | 1.000 |
| max_ks | AUC MLP | 24 | +0.17 | 0.424 | 1.000 |
| max_ks | AUC RF | 24 | +0.12 | 0.563 | 1.000 |
| max_ks | AUC XGB | 24 | +0.00 | 0.997 | 1.000 |
| c2st_auc | AUC LR | 24 | -0.19 | 0.379 | 1.000 |
| c2st_auc | AUC MLP | 24 | -0.13 | 0.546 | 1.000 |
| c2st_auc | AUC RF | 24 | -0.18 | 0.398 | 1.000 |
| c2st_auc | AUC XGB | 24 | -0.26 | 0.228 | 1.000 |
| abs_delta_ratio | AUC LR | 24 | -0.18 | 0.405 | 1.000 |
| abs_delta_ratio | AUC MLP | 24 | -0.08 | 0.710 | 1.000 |
| abs_delta_ratio | AUC RF | 24 | -0.10 | 0.654 | 1.000 |
| abs_delta_ratio | AUC XGB | 24 | -0.08 | 0.701 | 1.000 |

## (ii) Drift between consecutive test releases vs cross-version Spearman of the explanations

| x | y | n | rho | p_value | p_holm |
|---|---|---|---|---|---|
| mean_ks | MLP-DeepLiftShap | 14 | +0.02 | 0.958 | 1.000 |
| mean_ks | MLP-KernelSHAP | 14 | +0.32 | 0.267 | 1.000 |
| mean_ks | MLP-IntegratedGradients | 14 | -0.15 | 0.615 | 1.000 |
| mean_ks | MLP-LIME | 14 | +0.32 | 0.270 | 1.000 |
| mean_ks | MLP-Saliency | 14 | -0.16 | 0.584 | 1.000 |
| mean_ks | LR-LinearSHAP | 14 | +0.02 | 0.958 | 1.000 |
| mean_ks | RF-TreeSHAP | 14 | +0.14 | 0.637 | 1.000 |
| mean_ks | XGB-TreeSHAP | 14 | +0.22 | 0.455 | 1.000 |
| max_ks | MLP-DeepLiftShap | 14 | +0.14 | 0.626 | 1.000 |
| max_ks | MLP-KernelSHAP | 14 | +0.36 | 0.203 | 1.000 |
| max_ks | MLP-IntegratedGradients | 14 | -0.06 | 0.829 | 1.000 |
| max_ks | MLP-LIME | 14 | +0.39 | 0.171 | 1.000 |
| max_ks | MLP-Saliency | 14 | -0.05 | 0.864 | 1.000 |
| max_ks | LR-LinearSHAP | 14 | +0.05 | 0.876 | 1.000 |
| max_ks | RF-TreeSHAP | 14 | +0.02 | 0.958 | 1.000 |
| max_ks | XGB-TreeSHAP | 14 | +0.11 | 0.703 | 1.000 |
| c2st_auc | MLP-DeepLiftShap | 14 | -0.05 | 0.852 | 1.000 |
| c2st_auc | MLP-KernelSHAP | 14 | +0.20 | 0.503 | 1.000 |
| c2st_auc | MLP-IntegratedGradients | 14 | -0.13 | 0.670 | 1.000 |
| c2st_auc | MLP-LIME | 14 | +0.09 | 0.748 | 1.000 |
| c2st_auc | MLP-Saliency | 14 | -0.35 | 0.227 | 1.000 |
| c2st_auc | LR-LinearSHAP | 14 | -0.14 | 0.637 | 1.000 |
| c2st_auc | RF-TreeSHAP | 14 | -0.23 | 0.436 | 1.000 |
| c2st_auc | XGB-TreeSHAP | 14 | -0.16 | 0.584 | 1.000 |
| abs_delta_ratio | MLP-DeepLiftShap | 14 | -0.27 | 0.358 | 1.000 |
| abs_delta_ratio | MLP-KernelSHAP | 14 | -0.38 | 0.175 | 1.000 |
| abs_delta_ratio | MLP-IntegratedGradients | 14 | -0.01 | 0.982 | 1.000 |
| abs_delta_ratio | MLP-LIME | 14 | -0.14 | 0.637 | 1.000 |
| abs_delta_ratio | MLP-Saliency | 14 | -0.09 | 0.759 | 1.000 |
| abs_delta_ratio | LR-LinearSHAP | 14 | -0.38 | 0.185 | 1.000 |
| abs_delta_ratio | RF-TreeSHAP | 14 | -0.05 | 0.876 | 1.000 |
| abs_delta_ratio | XGB-TreeSHAP | 14 | +0.03 | 0.923 | 1.000 |
