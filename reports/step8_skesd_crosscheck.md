# Step 8: Scott-Knott ESD cross-check against R

R packages: ScottKnottESD 2.0.3 (current CRAN release); ScottKnottESD 1.2.2 (original (2017 procedure)), each `sk_esd(x, alpha = 0.05)`, installed into user libraries; no system-wide changes. Our implementation (`sdp.eval.stats.scott_knott_esd`) log-transforms non-negative data before ranking; R receives exactly the data our implementation ranks (R does not transform). Datasets: step 3 per-split test AUC (8 models, 24 splits); step 6 per-split negative mean rank behind the fidelity summary (14 explanations); step 7 per-comparison cross-version Spearman ρ (14 explanations, 14 comparisons). Treatments with identical values are sent once (R's partition fails on exact ties) and given their twin's group.

The two R versions implement different procedures. Version 1.x (Tantithamthavorn et al., 2017) runs the Scott-Knott test (`ScottKnott::SK`, likelihood-ratio statistic, alpha) and then merges adjacent treatments whose Cohen's d is negligible; our implementation follows this procedure but merges adjacent *groups* (pooled observations). Version 2.x partitions until the extreme treatments of a group differ by a negligible Cohen's d and does not use the significance test.

| R package | dataset | blocks | treatments | groups_ours | groups_r | same_partition | same_group_numbers | differences | identical treatments sent once |
|---|---|---|---|---|---|---|---|---|---|
| ScottKnottESD 2.0.3 | step3_auc | 24 | 8 | 3 | 4 | False | False | all_defective: ours 2, R 3; lr: ours 1, R 2; majority: ours 2, R 3; manual_up: ours 3, R 4; mlp: ours 1, R 2; rf: ours 1, R 2; xgb: ours 1, R 2 | majority = all_defective |
| ScottKnottESD 2.0.3 | step6_fidelity_rank | 24 | 14 | 4 | 9 | False | False | LR-LIME: ours 4, R 8; LR-LinearSHAP: ours 2, R 3; MLP-DeepLift: ours 3, R 5; MLP-DeepLiftShap: ours 1, R 2; MLP-GradientShap: ours 2, R 4; MLP-InputXGradient: ours 3, R 6; MLP-IntegratedGradients: ours 2, R 4; MLP-LIME: ours 3, R 5; MLP-Saliency: ours 3, R 7; RF-LIME: ours 4, R 9; RF-TreeSHAP: ours 1, R 2; XGB-LIME: ours 4, R 8 | none |
| ScottKnottESD 2.0.3 | step7_cross_version_spearman | 14 | 14 | 2 | 6 | False | False | LR-LIME: ours 2, R 4; LR-LinearSHAP: ours 2, R 3; MLP-DeepLift: ours 2, R 5; MLP-DeepLiftShap: ours 2, R 5; MLP-GradientShap: ours 2, R 5; MLP-InputXGradient: ours 2, R 4; MLP-IntegratedGradients: ours 2, R 5; MLP-KernelSHAP: ours 2, R 5; MLP-LIME: ours 2, R 6; MLP-Saliency: ours 2, R 6; RF-LIME: ours 1, R 2 | none |
| ScottKnottESD 1.2.2 | step3_auc | 24 | 8 | 3 | 3 | True | True | none | majority = all_defective |
| ScottKnottESD 1.2.2 | step6_fidelity_rank | 24 | 14 | 4 | 4 | True | True | none | none |
| ScottKnottESD 1.2.2 | step7_cross_version_spearman | 14 | 14 | 2 | 2 | True | True | none | none |
