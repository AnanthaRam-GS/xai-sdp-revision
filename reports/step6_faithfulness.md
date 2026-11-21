# Step 6: faithfulness correlation

Bhatt et al. (2020): per instance, 50 random subsets of 5 features (seeded, shared by all explanations of the instance); for each subset the sum of the attributions and the prediction drop f(x) − f(x without the subset), with marginal removal (50 background rows) and f = P(defective). Faithfulness is the Pearson correlation over the 50 subsets (undefined when the attributions are constant). Saliency as |gradient|. Mean and 95% bootstrap CI over instances; Scott-Knott ESD on per-split means (1 = best).

## Faithfulness

| explanation | faithfulness | median | % > 0 | % undefined | rank (all 14) | rank within model |
|---|---|---|---|---|---|---|
| MLP-DeepLift | 0.924 [0.923, 0.926] | 0.946 | 100.0 | 0.0 | 2 | 1 |
| MLP-DeepLiftShap | 0.950 [0.949, 0.951] | 0.962 | 100.0 | 0.0 | 1 | 1 |
| MLP-GradientShap | 0.931 [0.930, 0.933] | 0.948 | 100.0 | 0.0 | 2 | 1 |
| MLP-IntegratedGradients | 0.924 [0.923, 0.926] | 0.946 | 100.0 | 0.0 | 2 | 1 |
| MLP-Saliency | -0.052 [-0.062, -0.043] | -0.066 | 45.3 | 0.4 | 4 | 2 |
| MLP-InputXGradient | 0.900 [0.899, 0.902] | 0.926 | 99.6 | 0.4 | 2 | 1 |
| MLP-LIME | -0.017 [-0.027, -0.008] | -0.008 | 49.5 | 0.0 | 3 | 2 |
| MLP-KernelSHAP | 0.949 [0.948, 0.950] | 0.962 | 100.0 | 0.0 | 1 | 1 |
| LR-LinearSHAP | 0.978 [0.977, 0.978] | 0.985 | 100.0 | 0.0 | 1 | 1 |
| LR-LIME | -0.002 [-0.013, 0.009] | 0.011 | 50.6 | 0.0 | 3 | 2 |
| RF-TreeSHAP | 0.952 [0.951, 0.953] | 0.966 | 100.0 | 0.0 | 1 | 1 |
| RF-LIME | -0.106 [-0.116, -0.096] | -0.107 | 43.4 | 0.0 | 5 | 2 |
| XGB-TreeSHAP | 0.956 [0.955, 0.957] | 0.967 | 100.0 | 0.0 | 1 | 1 |
| XGB-LIME | -0.090 [-0.101, -0.080] | -0.087 | 44.4 | 0.0 | 5 | 2 |

## By outcome group

| explanation | faithfulness TP | faithfulness FP | faithfulness FN | faithfulness TN |
|---|---|---|---|---|
| MLP-DeepLift | 0.923 | 0.936 | 0.932 | 0.918 |
| MLP-DeepLiftShap | 0.953 | 0.957 | 0.955 | 0.944 |
| MLP-GradientShap | 0.937 | 0.941 | 0.936 | 0.925 |
| MLP-IntegratedGradients | 0.925 | 0.936 | 0.932 | 0.918 |
| MLP-Saliency | 0.319 | 0.315 | -0.171 | -0.282 |
| MLP-InputXGradient | 0.886 | 0.904 | 0.909 | 0.901 |
| MLP-LIME | 0.316 | 0.153 | -0.015 | -0.194 |
| MLP-KernelSHAP | 0.951 | 0.956 | 0.956 | 0.944 |
| LR-LinearSHAP | 0.982 | 0.983 | 0.977 | 0.973 |
| LR-LIME | 0.369 | 0.185 | -0.011 | -0.217 |
| RF-TreeSHAP | 0.959 | 0.959 | 0.946 | 0.949 |
| RF-LIME | 0.245 | 0.175 | -0.198 | -0.306 |
| XGB-TreeSHAP | 0.966 | 0.961 | 0.956 | 0.950 |
| XGB-LIME | 0.246 | 0.120 | -0.128 | -0.289 |

## Per project

| explanation | faithfulness ant | faithfulness camel | faithfulness ivy | faithfulness jedit | faithfulness lucene | faithfulness poi | faithfulness synapse | faithfulness velocity | faithfulness xalan | faithfulness xerces |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP-DeepLift | 0.927 | 0.886 | 0.957 | 0.907 | 0.939 | 0.930 | 0.936 | 0.937 | 0.949 | 0.920 |
| MLP-DeepLiftShap | 0.939 | 0.932 | 0.973 | 0.944 | 0.956 | 0.959 | 0.956 | 0.963 | 0.964 | 0.944 |
| MLP-GradientShap | 0.924 | 0.912 | 0.959 | 0.923 | 0.937 | 0.939 | 0.936 | 0.944 | 0.948 | 0.924 |
| MLP-IntegratedGradients | 0.926 | 0.886 | 0.957 | 0.909 | 0.940 | 0.932 | 0.936 | 0.937 | 0.949 | 0.919 |
| MLP-Saliency | -0.154 | -0.042 | -0.237 | -0.108 | 0.019 | 0.021 | -0.167 | 0.136 | 0.023 | 0.015 |
| MLP-InputXGradient | 0.903 | 0.878 | 0.940 | 0.880 | 0.915 | 0.904 | 0.907 | 0.897 | 0.917 | 0.891 |
| MLP-LIME | -0.010 | -0.049 | -0.108 | -0.078 | 0.040 | 0.118 | -0.012 | -0.056 | 0.024 | -0.111 |
| MLP-KernelSHAP | 0.940 | 0.932 | 0.971 | 0.940 | 0.955 | 0.958 | 0.955 | 0.961 | 0.965 | 0.945 |
| LR-LinearSHAP | 0.977 | 0.958 | 0.992 | 0.978 | 0.995 | 0.976 | 0.989 | 0.996 | 0.981 | 0.978 |
| LR-LIME | -0.006 | -0.048 | -0.059 | 0.003 | 0.069 | 0.117 | 0.021 | -0.121 | 0.019 | -0.066 |
| RF-TreeSHAP | 0.978 | 0.917 | 0.991 | 0.951 | 0.966 | 0.934 | 0.962 | 0.953 | 0.947 | 0.969 |
| RF-LIME | -0.153 | -0.080 | -0.138 | -0.106 | -0.055 | -0.064 | -0.111 | -0.060 | -0.128 | -0.118 |
| XGB-TreeSHAP | 0.956 | 0.935 | 0.977 | 0.959 | 0.987 | 0.952 | 0.968 | 0.979 | 0.969 | 0.918 |
| XGB-LIME | -0.115 | -0.063 | -0.115 | -0.045 | -0.007 | -0.073 | -0.154 | 0.006 | -0.128 | -0.172 |

Note: LIME and Saliency weights are slopes, not contributions relative to a reference (step 4), so their subset sums are not expected to track prediction drops; the SHAP-type methods, DeepLift and Integrated Gradients are designed to.
