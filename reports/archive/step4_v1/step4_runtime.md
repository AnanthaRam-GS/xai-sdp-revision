# Step 4 runtime

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

960 jobs ((split, model, seed) units), n_jobs=10, 4 reused from the smoke test. Wall time 14.4 min; CPU time 2.36 h in the explainers.

| explanation | units | instances | cpu_h | ms_per_instance |
|---|---|---|---|---|
| MLP-DeepLift | 240 | 94110 | 0.001 | 0.03 |
| MLP-DeepLiftShap | 240 | 94110 | 0.137 | 5.26 |
| MLP-GradientShap | 240 | 94110 | 0.024 | 0.94 |
| MLP-IntegratedGradients | 240 | 94110 | 0.023 | 0.88 |
| MLP-Saliency | 240 | 94110 | 0.000 | 0.01 |
| MLP-InputXGradient | 240 | 94110 | 0.000 | 0.01 |
| MLP-LIME | 72 | 28233 | 0.088 | 11.17 |
| MLP-KernelSHAP | 72 | 28233 | 0.455 | 58.08 |
| LR-LIME | 72 | 28233 | 0.068 | 8.73 |
| LR-LinearSHAP | 240 | 94110 | 0.000 | 0.01 |
| RF-LIME | 72 | 28233 | 0.638 | 81.34 |
| RF-TreeSHAP | 240 | 94110 | 0.713 | 27.26 |
| XGB-LIME | 72 | 28233 | 0.131 | 16.73 |
| XGB-TreeSHAP | 240 | 94110 | 0.078 | 2.98 |
