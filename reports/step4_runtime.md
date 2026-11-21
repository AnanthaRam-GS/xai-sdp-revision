# Step 4 runtime

## Changes since v1

v1 is archived in `reports/archive/step4_v1/`. v2 changes two settings: Integrated Gradients now uses 512 Gauss-Legendre steps (v1: 64), and LIME samples its perturbations around the explained instance (`sample_around_instance=True`; v1 sampled around the training mean). The v1 LIME results are kept in the store as `LIME_mean_sampling`. Everything else is unchanged.

v1 full run: 14.4 min wall. v2 run: 19.4 min wall, 504 of 960 jobs reused from the cache.


## Machine

| item | value |
|---|---|
| CPU | Apple M5 |
| Logical cores | 10 |
| Performance / efficiency cores | 4 / 6 |
| RAM (GB) | 24.00 |
| OS | Darwin 27.0.1 (arm64) |
| Python | 3.11.15 |
| numpy | 2.4.6 |
| pandas | 3.1.1 |
| scikit-learn | 1.9.1 |
| scipy | 1.17.1 |
| torch | 2.15.0 |
| xgboost | 3.2.0 |
| optuna | 5.1.0 |
| imbalanced-learn | 0.14.2 |
| captum | 0.9.0 |
| joblib | 1.6.0 |
| shap | 0.51.0 |
| lime | 0.2.0.1 |
| pyarrow | 26.0.1 |

## Smoke test and estimate

Smoke test: split `ta-consecutive:xerces:init>1.2`, seed 0, all methods (22.8 s wall).

| explanation | instances | time_s | ms_per_instance |
|---|---|---|---|
| LR-LinearSHAP | 344 | 0.24 | 0.70 |
| LR-LIME | 344 | 1.22 | 3.55 |
| XGB-TreeSHAP | 344 | 0.26 | 0.75 |
| XGB-LIME | 344 | 1.46 | 4.25 |
| MLP-DeepLift | 344 | 0.01 | 0.04 |
| MLP-DeepLiftShap | 344 | 0.75 | 2.17 |
| MLP-GradientShap | 344 | 0.10 | 0.29 |
| MLP-IntegratedGradients | 344 | 0.10 | 0.30 |
| MLP-Saliency | 344 | 0.00 | 0.00 |
| MLP-InputXGradient | 344 | 0.00 | 0.01 |
| MLP-LIME | 344 | 1.10 | 3.19 |
| MLP-KernelSHAP | 344 | 4.07 | 11.84 |
| RF-TreeSHAP | 344 | 3.00 | 8.72 |
| RF-LIME | 344 | 17.63 | 51.25 |

Estimated full run: 1.36 CPU h, ~0.40 h wall with n_jobs=10. The estimate is below 8 h, so LIME (num_samples=5000) and KernelSHAP (nsamples=1000) were run at full size; no reduction was applied.

## Full run

v2 recomputed every job whose methods or settings changed (all MLP jobs for Integrated Gradients, all seed 0–2 jobs for LIME); jobs of LR, RF and XGBoost with seeds 3–9 were unchanged and reused from the cache, so their timings are those of the v1 run.

960 jobs ((split, model, seed) units), n_jobs=10, 504 reused from the cache. Wall time 19.4 min; CPU time 3.70 h in the explainers.

| explanation | units | instances | cpu_h | ms_per_instance |
|---|---|---|---|---|
| MLP-DeepLift | 240 | 94110 | 0.001 | 0.04 |
| MLP-DeepLiftShap | 240 | 94110 | 0.173 | 6.63 |
| MLP-GradientShap | 240 | 94110 | 0.031 | 1.18 |
| MLP-IntegratedGradients | 240 | 94110 | 1.316 | 50.33 |
| MLP-Saliency | 240 | 94110 | 0.000 | 0.01 |
| MLP-InputXGradient | 240 | 94110 | 0.000 | 0.01 |
| MLP-LIME | 72 | 28233 | 0.116 | 14.80 |
| MLP-KernelSHAP | 72 | 28233 | 0.569 | 72.60 |
| LR-LIME | 72 | 28233 | 0.060 | 7.68 |
| LR-LinearSHAP | 240 | 94110 | 0.000 | 0.00 |
| RF-LIME | 72 | 28233 | 0.534 | 68.08 |
| RF-TreeSHAP | 240 | 94110 | 0.703 | 26.90 |
| XGB-LIME | 72 | 28233 | 0.118 | 15.02 |
| XGB-TreeSHAP | 240 | 94110 | 0.076 | 2.91 |
