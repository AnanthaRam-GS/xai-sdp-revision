# Step 6: ROAR and KAR

Global rankings from training-set attributions only (the fitting rows of each split; LIME on 200 stratified fitting rows), seeds 0, 1, 2. Ranking = mean L1-normalised |attribution| over those rows. References: 5 seeded random orderings (averaged), size-first (loc, rfc, wmc, npm, then random) and the RF-TreeSHAP ranking applied to the other models. ROAR drops the top-k metrics, KAR keeps only the top-k (k ∈ {1, 2, 3, 5, 8, 10, 15}); each model type is refitted with its persisted step 3 hyperparameters (no re-tuning), preprocessing, imbalance handling and early stopping, and evaluated on the test release. Area = mean test AUC over k (trapezoid rule over the evaluated k, divided by the k range); per-split means over seeds. ROAR: lower is better (removing the important metrics should hurt); KAR: higher is better.

Check: refitting with all 20 metrics (k = 0) reproduces the step 3 test AUC (max |difference| 0.00e+00 over 288 models).

## Area scores

| model | ranking | ROAR area (AUC) ↓ | KAR area (AUC) ↑ |
|---|---|---|---|
| LR | LR-LinearSHAP | 0.674 | 0.678 |
| LR | random | 0.682 | 0.675 |
| LR | RF-TreeSHAP ranking | 0.663 | 0.683 |
| LR | size-first | 0.672 | 0.694 |
| MLP | MLP-DeepLiftShap | 0.680 | 0.676 |
| MLP | MLP-IntegratedGradients | 0.681 | 0.674 |
| MLP | MLP-LIME | 0.676 | 0.677 |
| MLP | MLP-Saliency | 0.677 | 0.678 |
| MLP | random | 0.678 | 0.671 |
| MLP | RF-TreeSHAP ranking | 0.657 | 0.684 |
| MLP | size-first | 0.672 | 0.687 |
| RF | RF-TreeSHAP | 0.658 | 0.671 |
| RF | random | 0.673 | 0.660 |
| RF | size-first | 0.670 | 0.665 |
| XGB | XGB-TreeSHAP | 0.657 | 0.669 |
| XGB | random | 0.669 | 0.658 |
| XGB | RF-TreeSHAP ranking | 0.652 | 0.672 |
| XGB | size-first | 0.666 | 0.668 |

## Wilcoxon signed-rank tests on the area scores

Per model and test (24 splits); Holm within (model, test); Cliff's delta of a vs b. For ROAR a negative delta favours a (a lower area); for KAR a positive delta favours a.

| model | test | a | b | n | mean_a | mean_b | p_value | p_holm | cliffs_delta | magnitude |
|---|---|---|---|---|---|---|---|---|---|---|
| LR | KAR | LR-LinearSHAP | random | 24 | 0.678 | 0.675 | 0.121 | 0.190 | 0.035 | negligible |
| LR | KAR | LR-LinearSHAP | size-first | 24 | 0.678 | 0.694 | 0.046 | 0.147 | -0.090 | negligible |
| LR | KAR | RF-TreeSHAP ranking | random | 24 | 0.683 | 0.675 | 0.037 | 0.147 | 0.062 | negligible |
| LR | KAR | RF-TreeSHAP ranking | size-first | 24 | 0.683 | 0.694 | 0.095 | 0.190 | -0.076 | negligible |
| LR | KAR | size-first | random | 24 | 0.694 | 0.675 | <0.001 | 0.005 | 0.149 | small |
| LR | ROAR | LR-LinearSHAP | random | 24 | 0.674 | 0.682 | 0.056 | 0.126 | -0.035 | negligible |
| LR | ROAR | LR-LinearSHAP | size-first | 24 | 0.674 | 0.672 | 0.877 | 0.877 | 0.000 | negligible |
| LR | ROAR | RF-TreeSHAP ranking | random | 24 | 0.663 | 0.682 | <0.001 | 0.002 | -0.115 | negligible |
| LR | ROAR | RF-TreeSHAP ranking | size-first | 24 | 0.663 | 0.672 | 0.031 | 0.126 | -0.052 | negligible |
| LR | ROAR | size-first | random | 24 | 0.672 | 0.682 | 0.031 | 0.126 | -0.066 | negligible |
| MLP | KAR | MLP-DeepLiftShap | random | 24 | 0.676 | 0.671 | 0.009 | 0.079 | 0.052 | negligible |
| MLP | KAR | MLP-DeepLiftShap | size-first | 24 | 0.676 | 0.687 | 0.160 | 0.640 | -0.062 | negligible |
| MLP | KAR | MLP-IntegratedGradients | random | 24 | 0.674 | 0.671 | 0.009 | 0.079 | 0.042 | negligible |
| MLP | KAR | MLP-IntegratedGradients | size-first | 24 | 0.674 | 0.687 | 0.121 | 0.604 | -0.066 | negligible |
| MLP | KAR | MLP-LIME | random | 24 | 0.677 | 0.671 | 0.011 | 0.079 | 0.035 | negligible |
| MLP | KAR | MLP-LIME | size-first | 24 | 0.677 | 0.687 | 0.277 | 0.655 | -0.073 | negligible |
| MLP | KAR | MLP-Saliency | random | 24 | 0.678 | 0.671 | 0.008 | 0.079 | 0.042 | negligible |
| MLP | KAR | MLP-Saliency | size-first | 24 | 0.678 | 0.687 | 0.218 | 0.655 | -0.076 | negligible |
| MLP | KAR | RF-TreeSHAP ranking | random | 24 | 0.684 | 0.671 | 0.010 | 0.079 | 0.087 | negligible |
| MLP | KAR | RF-TreeSHAP ranking | size-first | 24 | 0.684 | 0.687 | 0.922 | 0.922 | -0.024 | negligible |
| MLP | KAR | size-first | random | 24 | 0.687 | 0.671 | 0.005 | 0.059 | 0.122 | negligible |
| MLP | ROAR | MLP-DeepLiftShap | random | 24 | 0.680 | 0.678 | 0.208 | 1.000 | -0.024 | negligible |
| MLP | ROAR | MLP-DeepLiftShap | size-first | 24 | 0.680 | 0.672 | 0.663 | 1.000 | 0.042 | negligible |
| MLP | ROAR | MLP-IntegratedGradients | random | 24 | 0.681 | 0.678 | 0.114 | 0.809 | -0.017 | negligible |
| MLP | ROAR | MLP-IntegratedGradients | size-first | 24 | 0.681 | 0.672 | 0.900 | 1.000 | 0.042 | negligible |
| MLP | ROAR | MLP-LIME | random | 24 | 0.676 | 0.678 | 0.079 | 0.709 | -0.042 | negligible |
| MLP | ROAR | MLP-LIME | size-first | 24 | 0.676 | 0.672 | 0.944 | 1.000 | 0.017 | negligible |
| MLP | ROAR | MLP-Saliency | random | 24 | 0.677 | 0.678 | 0.101 | 0.809 | -0.038 | negligible |
| MLP | ROAR | MLP-Saliency | size-first | 24 | 0.677 | 0.672 | 0.877 | 1.000 | 0.017 | negligible |
| MLP | ROAR | RF-TreeSHAP ranking | random | 24 | 0.657 | 0.678 | <0.001 | <0.001 | -0.167 | small |
| MLP | ROAR | RF-TreeSHAP ranking | size-first | 24 | 0.657 | 0.672 | 0.007 | 0.065 | -0.108 | negligible |
| MLP | ROAR | size-first | random | 24 | 0.672 | 0.678 | 0.160 | 0.960 | -0.045 | negligible |
| RF | KAR | RF-TreeSHAP | random | 24 | 0.671 | 0.660 | 0.007 | 0.022 | 0.076 | negligible |
| RF | KAR | RF-TreeSHAP | size-first | 24 | 0.671 | 0.665 | 0.705 | 0.705 | 0.024 | negligible |
| RF | KAR | size-first | random | 24 | 0.665 | 0.660 | 0.241 | 0.481 | 0.059 | negligible |
| RF | ROAR | RF-TreeSHAP | random | 24 | 0.658 | 0.673 | 0.004 | 0.007 | -0.118 | negligible |
| RF | ROAR | RF-TreeSHAP | size-first | 24 | 0.658 | 0.670 | 0.002 | 0.005 | -0.094 | negligible |
| RF | ROAR | size-first | random | 24 | 0.670 | 0.673 | 0.527 | 0.527 | -0.014 | negligible |
| XGB | KAR | XGB-TreeSHAP | random | 24 | 0.669 | 0.658 | 0.008 | 0.032 | 0.087 | negligible |
| XGB | KAR | XGB-TreeSHAP | size-first | 24 | 0.669 | 0.668 | 0.527 | 1.000 | 0.000 | negligible |
| XGB | KAR | RF-TreeSHAP ranking | random | 24 | 0.672 | 0.658 | 0.003 | 0.016 | 0.104 | negligible |
| XGB | KAR | RF-TreeSHAP ranking | size-first | 24 | 0.672 | 0.668 | 0.877 | 1.000 | 0.021 | negligible |
| XGB | KAR | size-first | random | 24 | 0.668 | 0.658 | 0.034 | 0.102 | 0.069 | negligible |
| XGB | ROAR | XGB-TreeSHAP | random | 24 | 0.657 | 0.669 | 0.018 | 0.054 | -0.097 | negligible |
| XGB | ROAR | XGB-TreeSHAP | size-first | 24 | 0.657 | 0.666 | 0.089 | 0.179 | -0.076 | negligible |
| XGB | ROAR | RF-TreeSHAP ranking | random | 24 | 0.652 | 0.669 | 0.008 | 0.040 | -0.128 | negligible |
| XGB | ROAR | RF-TreeSHAP ranking | size-first | 24 | 0.652 | 0.666 | 0.011 | 0.042 | -0.108 | negligible |
| XGB | ROAR | size-first | random | 24 | 0.666 | 0.669 | 0.317 | 0.317 | -0.003 | negligible |

## Test AUC at each k: ROAR

| model | ranking | k=0 | k=1 | k=2 | k=3 | k=5 | k=8 | k=10 | k=15 |
|---|---|---|---|---|---|---|---|---|---|
| LR | LR-LinearSHAP | 0.689 | 0.689 | 0.687 | 0.687 | 0.687 | 0.678 | 0.674 | 0.641 |
| LR | random | 0.689 | 0.688 | 0.687 | 0.687 | 0.686 | 0.685 | 0.683 | 0.670 |
| LR | RF-TreeSHAP ranking | 0.689 | 0.684 | 0.682 | 0.680 | 0.678 | 0.669 | 0.663 | 0.620 |
| LR | size-first | 0.689 | 0.688 | 0.684 | 0.684 | 0.675 | 0.676 | 0.674 | 0.647 |
| MLP | MLP-DeepLiftShap | 0.682 | 0.684 | 0.688 | 0.692 | 0.686 | 0.686 | 0.680 | 0.659 |
| MLP | MLP-IntegratedGradients | 0.682 | 0.684 | 0.689 | 0.690 | 0.685 | 0.685 | 0.676 | 0.669 |
| MLP | MLP-LIME | 0.682 | 0.682 | 0.685 | 0.692 | 0.685 | 0.677 | 0.675 | 0.655 |
| MLP | MLP-Saliency | 0.682 | 0.682 | 0.685 | 0.688 | 0.684 | 0.679 | 0.675 | 0.659 |
| MLP | random | 0.682 | 0.683 | 0.684 | 0.683 | 0.683 | 0.680 | 0.678 | 0.667 |
| MLP | RF-TreeSHAP ranking | 0.682 | 0.679 | 0.679 | 0.676 | 0.672 | 0.670 | 0.656 | 0.608 |
| MLP | size-first | 0.682 | 0.683 | 0.678 | 0.682 | 0.674 | 0.669 | 0.672 | 0.660 |
| RF | RF-TreeSHAP | 0.678 | 0.679 | 0.677 | 0.678 | 0.674 | 0.663 | 0.653 | 0.619 |
| RF | random | 0.678 | 0.678 | 0.678 | 0.678 | 0.679 | 0.675 | 0.672 | 0.659 |
| RF | size-first | 0.678 | 0.678 | 0.677 | 0.678 | 0.676 | 0.674 | 0.671 | 0.648 |
| XGB | XGB-TreeSHAP | 0.676 | 0.676 | 0.674 | 0.675 | 0.672 | 0.658 | 0.655 | 0.620 |
| XGB | random | 0.676 | 0.676 | 0.676 | 0.675 | 0.673 | 0.672 | 0.668 | 0.656 |
| XGB | RF-TreeSHAP ranking | 0.676 | 0.679 | 0.675 | 0.673 | 0.669 | 0.657 | 0.647 | 0.607 |
| XGB | size-first | 0.676 | 0.676 | 0.673 | 0.674 | 0.671 | 0.671 | 0.669 | 0.645 |

## Test AUC at each k: KAR

| model | ranking | k=1 | k=2 | k=3 | k=5 | k=8 | k=10 | k=15 | k=20 |
|---|---|---|---|---|---|---|---|---|---|
| LR | LR-LinearSHAP | 0.616 | 0.652 | 0.665 | 0.675 | 0.687 | 0.686 | 0.689 | 0.689 |
| LR | random | 0.604 | 0.648 | 0.666 | 0.677 | 0.680 | 0.684 | 0.687 | 0.689 |
| LR | RF-TreeSHAP ranking | 0.656 | 0.676 | 0.675 | 0.682 | 0.688 | 0.686 | 0.689 | 0.689 |
| LR | size-first | 0.697 | 0.703 | 0.704 | 0.698 | 0.691 | 0.688 | 0.691 | 0.689 |
| MLP | MLP-DeepLiftShap | 0.609 | 0.653 | 0.673 | 0.677 | 0.681 | 0.684 | 0.682 | 0.682 |
| MLP | MLP-IntegratedGradients | 0.599 | 0.642 | 0.663 | 0.678 | 0.681 | 0.683 | 0.683 | 0.682 |
| MLP | MLP-LIME | 0.636 | 0.656 | 0.672 | 0.678 | 0.681 | 0.683 | 0.684 | 0.682 |
| MLP | MLP-Saliency | 0.632 | 0.656 | 0.677 | 0.679 | 0.683 | 0.682 | 0.682 | 0.682 |
| MLP | random | 0.608 | 0.643 | 0.663 | 0.671 | 0.675 | 0.679 | 0.684 | 0.682 |
| MLP | RF-TreeSHAP ranking | 0.665 | 0.682 | 0.689 | 0.681 | 0.683 | 0.686 | 0.687 | 0.682 |
| MLP | size-first | 0.681 | 0.698 | 0.695 | 0.693 | 0.684 | 0.683 | 0.681 | 0.682 |
| RF | RF-TreeSHAP | 0.636 | 0.658 | 0.668 | 0.671 | 0.675 | 0.675 | 0.678 | 0.678 |
| RF | random | 0.591 | 0.624 | 0.645 | 0.657 | 0.667 | 0.672 | 0.678 | 0.678 |
| RF | size-first | 0.632 | 0.658 | 0.659 | 0.659 | 0.667 | 0.667 | 0.677 | 0.678 |
| XGB | XGB-TreeSHAP | 0.642 | 0.659 | 0.665 | 0.667 | 0.672 | 0.674 | 0.674 | 0.676 |
| XGB | random | 0.592 | 0.627 | 0.642 | 0.657 | 0.665 | 0.669 | 0.674 | 0.676 |
| XGB | RF-TreeSHAP ranking | 0.647 | 0.663 | 0.672 | 0.671 | 0.676 | 0.674 | 0.677 | 0.676 |
| XGB | size-first | 0.650 | 0.677 | 0.667 | 0.665 | 0.670 | 0.667 | 0.674 | 0.676 |

MCC@tuned and Popt area scores are in `step6_roar_kar_areas.csv`.
