# Step 7: stability summary

Plain-language summary of the step 7 results (details in the other `step7_*` reports).

## Key numbers

| quantity | value |
|---|---|
| cross-version Spearman (MLP-DeepLiftShap / RF-TreeSHAP) | 0.16 / 0.51 |
| data-only reference (permuted labels) | 0.09 / 0.39 |
| random null: mean / 95th percentile | 0.00 / 0.37 |
| seed stability, global Spearman (MLP-DeepLiftShap / RF-TreeSHAP) | 0.58 / 0.92 |
| cross-project Spearman (MLP-DeepLiftShap / RF-TreeSHAP) | 0.09 / 0.46 |
| test instances whose label changes between seeds (MLP / LR / RF / XGB) | 35.3% / 23.7% / 26.2% / 33.2% |

## In plain language

* **Across releases**, the global rankings are only weakly to moderately stable: mean Spearman ρ over the 14 consecutive-split comparisons ranges from 0.04 to 0.52. Only RF-TreeSHAP, XGB-TreeSHAP exceed the 95th percentile of the random null (0.37); the MLP explanations (DeepLiftShap, KernelSHAP, IntegratedGradients, LIME, Saliency) stay at ρ ≈ 0.04–0.17.
* **Much of the stability that exists does not need a defect signal**: the permuted-label models reach ρ = -0.06–0.39 (highest for the tree models, whose importances follow the metric distributions). Explanations whose mean ρ exceeds their data-only reference: MLP-DeepLiftShap, MLP-KernelSHAP, MLP-IntegratedGradients, LR-LinearSHAP, RF-TreeSHAP, XGB-TreeSHAP; after Holm correction the difference is significant (Spearman, p < 0.05) only for XGB-LIME (n = 14 comparisons).
* **Across training seeds**, global rankings are very stable for the tree models (mean ρ 0.83–0.92) and only moderately for the MLP (0.53–0.61). Predictions themselves are not stable: 24%–35% of test instances change predicted label between at least two seeds (MLP 35%).
* **Unchanged classes** (1163 classes with identical metrics and label in consecutive test releases) get explanations that agree with ρ ≈ 0.29 across versions, against ≈ 0.64 between two seeds of the same model: retraining on a new release changes the explanation of an unchanged class more than a new seed does.
* **Across projects**, project-level rankings agree weakly (mean ρ 0.05–0.46).
* **Data drift** between consecutive test releases is mostly small (mean KS 0.02–0.08; classifier two-sample AUC 0.38–0.81). Its correlation with explanation stability, and that of train→test drift with test AUC, is not significant in any test after Holm correction (small n: 14 and 24 pairs).

## The original paper's claims vs the rebuilt pipeline

| original claim | rebuilt pipeline | verdict |
|---|---|---|
| LOC is the top feature in all versions | loc is top-1 in 4 of 24 splits for MLP-DeepLiftShap and 6 for RF-TreeSHAP; in the top-5 of 38% / 67% of splits. With unscaled inputs and a zero baseline (the original setting, step 5) loc takes 0.33 of the attribution and is top-1 for 57% of instances, against 0.06 with log-standardised inputs. | not supported: an artefact of unscaled inputs |
| Spearman 0.69–0.89 across versions | MLP-DeepLiftShap: mean 0.16 (range -0.31–0.51) over 14 comparisons; RF-TreeSHAP: mean 0.51 (0.04–0.76). Data-only reference (permuted labels): 0.09 / 0.39; random null 95th pct 0.37. | not reproduced: far lower for the MLP, moderate for the tree models, and partly reached without any defect signal |
| DeepLIFT and DeepSHAP agree | MLP-DeepLift vs MLP-DeepLiftShap (step 4): feature agreement@5 0.80, Spearman 0.80; but both use the same propagation rules, so the agreement is expected and not independent evidence; agreement with LIME and across models is much lower. | true but uninformative |
