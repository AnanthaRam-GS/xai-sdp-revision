# Figure captions (draft, factual)

| file | caption | source |
|---|---|---|
| `fig_performance.pdf` | Per-split test AUC, MCC at the validation-tuned threshold and Popt of all models under time-aware evaluation (boxes and points: 24 splits, seeds averaged); diamonds mark the within-release means. | reports/step3_main_time_aware.md, reports/step3_within_release.md |
| `fig_cost_ratio.pdf` | Normalised expected cost of misclassification (NECM) against the cost ratio c for every model at its validation-MCC threshold, with 95% bootstrap confidence bands over splits; at this threshold the Majority baseline flags every class and coincides with All defective. | reports/step8_cost_ratio.md |
| `fig_calibration.pdf` | Reliability diagrams of the four learned models before calibration (test predictions pooled over splits and seeds, ten equal-width bins). | reports/step8_calibration.md |
| `fig_global_importance.pdf` | Mean L1-normalised absolute attribution of the 20 metrics for the 14 explanations (all test instances, ALL splits), with metrics grouped by the stable correlation clusters. | reports/step4_global_importance.md, reports/step5_collinearity.md |
| `fig_direction.pdf` | Spearman correlation between each metric's value and its signed attribution (ALL splits) for the eight primary explanations. | reports/step4_direction.md |
| `fig_agreement.pdf` | Mean feature agreement at k = 5 between the MLP explanations (left) and between the SHAP-family explanations of the four models (right), seeds 0–2. | reports/step4_agreement.md |
| `fig_scaling_effect.pdf` | Share of the normalised absolute attribution on LOC and on the size group {LOC, WMC, RFC, NPM} by preprocessing setting for the MLP and LR, including DeepLift with a zero baseline on unscaled inputs. | reports/step5_preprocessing_explanations.md |
| `fig_fidelity.pdf` | (a) Mean deletion curves (confidence in the model's own prediction against the number of removed metrics) for the MLP and RF explanations, with the random ordering dashed; (b) ROAR and KAR area scores of the training-set rankings against random and size-first rankings. | reports/step6_deletion_insertion.md, reports/step6_roar_kar.md |
| `fig_sanity.pdf` | Mean Spearman correlation between the original and recomputed MLP explanations as Linear layers are re-initialised cumulatively from the output layer. | reports/step6_sanity_checks.md |
| `fig_stability.pdf` | (a) Cross-version Spearman correlation of global importance per primary explanation (mean and 95% CI over 14 comparisons) with the data-only reference and the random-null band; (b) mean Spearman correlation per source of variation. | reports/step7_cross_version.md, reports/step7_seed_stability.md |
| `fig_background_sensitivity.pdf` | Mean Spearman correlation with the main background explanation against the background size, with independent draws of size 100 as reference. | reports/step5_background_sensitivity.md |
| `fig_lime_settings.pdf` | Mean Spearman correlation of LIME explanations between runs (run-to-run) and between each setting and the main setting, per model. | reports/step5_lime_stability.md |
| `fig_collinearity_dendrogram.pdf` | Hierarchical clustering (average linkage) of the 20 metrics on 1 − |Spearman ρ| for the jEdit 4.0 training release, with the cut at |ρ| = 0.7. | reports/step5_collinearity.md |
| `fig_topk_frequency.pdf` | Share of the 24 time-aware splits in which each metric is among the five most important metrics of each explanation. | reports/step7_topk_frequency.md |
