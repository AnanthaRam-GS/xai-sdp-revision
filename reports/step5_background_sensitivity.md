# Step 5: background-set sensitivity

Seed-0 models, all test instances of 24 splits (all 24). Methods: MLP-DeepLiftShap, MLP-GradientShap, RF-TreeSHAP, XGB-TreeSHAP (interventional). Main background: 100 fitting rows, stratified by label, draw seed 0 (as in step 4). Each variant is compared with the main background per instance; values are means over instances within a split, then over splits.

* `size=n`: stratified sample of n rows (`all` = every fitting row);
* `scheme=random`: simple random sample of 100; `scheme=kmeans`: shap.kmeans with 100 weighted centres (each centre repeated by its cluster size); `scheme=defective` / `scheme=clean`: 100 rows of one class;
* `draw=d`: stratified sample of 100 with draw seed d (draw 0 is the main background).

GradientShap is itself stochastic (50 random baselines and path points per instance). With the same background and only a different random seed, its per-instance Spearman is about 0.83–0.88 (checked on the three ant splits), so its agreement of about 0.86 with the main background is its own Monte-Carlo noise, not an effect of the background.

## Spearman with the main background

| variant | MLP-DeepLiftShap | MLP-GradientShap | RF-TreeSHAP | XGB-TreeSHAP |
|---|---|---|---|---|
| size=10 | 0.822 | 0.752 | 0.881 | 0.900 |
| size=25 | 0.906 | 0.809 | 0.935 | 0.940 |
| size=50 | 0.957 | 0.840 | 0.972 | 0.971 |
| size=200 | 0.979 | 0.858 | 0.971 | 0.976 |
| size=all | 0.979 | 0.858 | 0.968 | 0.973 |
| scheme=random | 0.964 | 0.856 | 0.968 | 0.973 |
| scheme=kmeans | 0.978 | 0.861 | 0.956 | 0.962 |
| scheme=defective | 0.744 | 0.687 | 0.646 | 0.808 |
| scheme=clean | 0.896 | 0.805 | 0.780 | 0.888 |
| draw=1 | 0.967 | 0.862 | 0.974 | 0.979 |
| draw=2 | 0.964 | 0.861 | 0.973 | 0.976 |
| draw=3 | 0.965 | 0.861 | 0.973 | 0.975 |
| draw=4 | 0.960 | 0.858 | 0.971 | 0.972 |

## Feature agreement@5 with the main background

| variant | MLP-DeepLiftShap | MLP-GradientShap | RF-TreeSHAP | XGB-TreeSHAP |
|---|---|---|---|---|
| size=10 | 0.804 | 0.756 | 0.849 | 0.840 |
| size=25 | 0.867 | 0.794 | 0.889 | 0.888 |
| size=50 | 0.922 | 0.820 | 0.935 | 0.934 |
| size=200 | 0.950 | 0.838 | 0.940 | 0.938 |
| size=all | 0.948 | 0.838 | 0.931 | 0.931 |
| scheme=random | 0.931 | 0.834 | 0.931 | 0.931 |
| scheme=kmeans | 0.949 | 0.841 | 0.919 | 0.915 |
| scheme=defective | 0.742 | 0.706 | 0.660 | 0.744 |
| scheme=clean | 0.858 | 0.792 | 0.779 | 0.826 |
| draw=1 | 0.935 | 0.841 | 0.942 | 0.944 |
| draw=2 | 0.928 | 0.841 | 0.940 | 0.940 |
| draw=3 | 0.931 | 0.840 | 0.941 | 0.939 |
| draw=4 | 0.924 | 0.837 | 0.936 | 0.935 |

## Sign agreement@5 with the main background

| variant | MLP-DeepLiftShap | MLP-GradientShap | RF-TreeSHAP | XGB-TreeSHAP |
|---|---|---|---|---|
| size=10 | 0.804 | 0.755 | 0.849 | 0.840 |
| size=25 | 0.867 | 0.794 | 0.889 | 0.888 |
| size=50 | 0.922 | 0.820 | 0.935 | 0.934 |
| size=200 | 0.950 | 0.838 | 0.940 | 0.938 |
| size=all | 0.948 | 0.838 | 0.931 | 0.931 |
| scheme=random | 0.931 | 0.834 | 0.931 | 0.931 |
| scheme=kmeans | 0.949 | 0.841 | 0.919 | 0.915 |
| scheme=defective | 0.739 | 0.701 | 0.644 | 0.736 |
| scheme=clean | 0.858 | 0.791 | 0.776 | 0.821 |
| draw=1 | 0.935 | 0.841 | 0.942 | 0.944 |
| draw=2 | 0.928 | 0.841 | 0.940 | 0.940 |
| draw=3 | 0.931 | 0.840 | 0.941 | 0.939 |
| draw=4 | 0.924 | 0.837 | 0.936 | 0.935 |

## Mean absolute difference of L1-normalised attributions

| variant | MLP-DeepLiftShap | MLP-GradientShap | RF-TreeSHAP | XGB-TreeSHAP |
|---|---|---|---|---|
| size=10 | 0.015 | 0.018 | 0.013 | 0.013 |
| size=25 | 0.010 | 0.015 | 0.008 | 0.008 |
| size=50 | 0.006 | 0.013 | 0.005 | 0.005 |
| size=200 | 0.003 | 0.012 | 0.004 | 0.004 |
| size=all | 0.003 | 0.012 | 0.005 | 0.005 |
| scheme=random | 0.005 | 0.012 | 0.005 | 0.005 |
| scheme=kmeans | 0.003 | 0.012 | 0.006 | 0.006 |
| scheme=defective | 0.019 | 0.022 | 0.027 | 0.021 |
| scheme=clean | 0.010 | 0.016 | 0.018 | 0.014 |
| draw=1 | 0.005 | 0.012 | 0.004 | 0.004 |
| draw=2 | 0.005 | 0.012 | 0.004 | 0.004 |
| draw=3 | 0.005 | 0.012 | 0.004 | 0.004 |
| draw=4 | 0.005 | 0.012 | 0.005 | 0.005 |

## Noise floor: 5 independent stratified draws of size 100

Mean pairwise agreement between the five draws (per instance, then over splits) and the mean per-feature variance of the L1-normalised attributions across draws. Differences from the main background smaller than this are within background-sampling noise.

| explanation | Spearman | FA@5 | SA@5 | MAD | variance |
|---|---|---|---|---|---|
| MLP-DeepLiftShap | 0.9631 | 0.9297 | 0.9297 | 0.0049 | 0.0000 |
| MLP-GradientShap | 0.8609 | 0.8406 | 0.8404 | 0.0119 | 0.0002 |
| RF-TreeSHAP | 0.9725 | 0.9385 | 0.9385 | 0.0044 | 0.0000 |
| XGB-TreeSHAP | 0.9743 | 0.9377 | 0.9377 | 0.0044 | 0.0000 |

## Global top-5 changes

Number of scopes (projects and ALL) whose global top-5 set or order differs from the main background, and the ALL top-5 when it differs.

| explanation | variant | scopes | top-5 set changed | top-5 order changed | ALL top-5 (main) | ALL top-5 (variant) |
|---|---|---|---|---|---|---|
| MLP-DeepLiftShap | size=10 | 11 | 5 | 9 | rfc, ce, dam, cbo, dit | same |
| MLP-DeepLiftShap | size=25 | 11 | 4 | 7 | rfc, ce, dam, cbo, dit | same |
| MLP-DeepLiftShap | size=50 | 11 | 2 | 5 | rfc, ce, dam, cbo, dit | same |
| MLP-DeepLiftShap | size=200 | 11 | 2 | 4 | rfc, ce, dam, cbo, dit | same |
| MLP-DeepLiftShap | size=all | 11 | 2 | 4 | rfc, ce, dam, cbo, dit | same |
| MLP-DeepLiftShap | scheme=random | 11 | 2 | 5 | rfc, ce, dam, cbo, dit | rfc, ce, dam, dit, cbo |
| MLP-DeepLiftShap | scheme=kmeans | 11 | 2 | 5 | rfc, ce, dam, cbo, dit | rfc, ce, dam, dit, cbo |
| MLP-DeepLiftShap | scheme=defective | 11 | 8 | 9 | rfc, ce, dam, cbo, dit | rfc, cbo, ce, lcom, dam |
| MLP-DeepLiftShap | scheme=clean | 11 | 7 | 9 | rfc, ce, dam, cbo, dit | rfc, ce, dam, dit, amc |
| MLP-DeepLiftShap | draw=1 | 11 | 3 | 4 | rfc, ce, dam, cbo, dit | same |
| MLP-DeepLiftShap | draw=2 | 11 | 2 | 7 | rfc, ce, dam, cbo, dit | rfc, ce, dam, dit, cbo |
| MLP-DeepLiftShap | draw=3 | 11 | 3 | 4 | rfc, ce, dam, cbo, dit | rfc, ce, dam, dit, cbo |
| MLP-DeepLiftShap | draw=4 | 11 | 1 | 4 | rfc, ce, dam, cbo, dit | rfc, ce, dam, dit, cbo |
| MLP-GradientShap | size=10 | 11 | 5 | 9 | rfc, ce, dam, cbo, dit | same |
| MLP-GradientShap | size=25 | 11 | 4 | 8 | rfc, ce, dam, cbo, dit | same |
| MLP-GradientShap | size=50 | 11 | 1 | 7 | rfc, ce, dam, cbo, dit | same |
| MLP-GradientShap | size=200 | 11 | 0 | 2 | rfc, ce, dam, cbo, dit | same |
| MLP-GradientShap | size=all | 11 | 0 | 2 | rfc, ce, dam, cbo, dit | same |
| MLP-GradientShap | scheme=random | 11 | 1 | 3 | rfc, ce, dam, cbo, dit | same |
| MLP-GradientShap | scheme=kmeans | 11 | 0 | 2 | rfc, ce, dam, cbo, dit | same |
| MLP-GradientShap | scheme=defective | 11 | 9 | 9 | rfc, ce, dam, cbo, dit | rfc, cbo, ce, lcom, dam |
| MLP-GradientShap | scheme=clean | 11 | 5 | 9 | rfc, ce, dam, cbo, dit | rfc, ce, dam, dit, cam |
| MLP-GradientShap | draw=1 | 11 | 2 | 5 | rfc, ce, dam, cbo, dit | same |
| MLP-GradientShap | draw=2 | 11 | 2 | 5 | rfc, ce, dam, cbo, dit | same |
| MLP-GradientShap | draw=3 | 11 | 1 | 3 | rfc, ce, dam, cbo, dit | same |
| MLP-GradientShap | draw=4 | 11 | 0 | 2 | rfc, ce, dam, cbo, dit | same |
| RF-TreeSHAP | size=10 | 11 | 7 | 10 | rfc, loc, amc, cam, cbo | rfc, loc, amc, cam, wmc |
| RF-TreeSHAP | size=25 | 11 | 4 | 8 | rfc, loc, amc, cam, cbo | same |
| RF-TreeSHAP | size=50 | 11 | 3 | 7 | rfc, loc, amc, cam, cbo | same |
| RF-TreeSHAP | size=200 | 11 | 3 | 5 | rfc, loc, amc, cam, cbo | same |
| RF-TreeSHAP | size=all | 11 | 2 | 5 | rfc, loc, amc, cam, cbo | same |
| RF-TreeSHAP | scheme=random | 11 | 2 | 5 | rfc, loc, amc, cam, cbo | same |
| RF-TreeSHAP | scheme=kmeans | 11 | 3 | 8 | rfc, loc, amc, cam, cbo | loc, rfc, amc, cam, cbo |
| RF-TreeSHAP | scheme=defective | 11 | 7 | 10 | rfc, loc, amc, cam, cbo | loc, rfc, cam, amc, wmc |
| RF-TreeSHAP | scheme=clean | 11 | 8 | 9 | rfc, loc, amc, cam, cbo | rfc, loc, amc, cam, ce |
| RF-TreeSHAP | draw=1 | 11 | 2 | 3 | rfc, loc, amc, cam, cbo | same |
| RF-TreeSHAP | draw=2 | 11 | 3 | 7 | rfc, loc, amc, cam, cbo | same |
| RF-TreeSHAP | draw=3 | 11 | 2 | 4 | rfc, loc, amc, cam, cbo | same |
| RF-TreeSHAP | draw=4 | 11 | 3 | 4 | rfc, loc, amc, cam, cbo | same |
| XGB-TreeSHAP | size=10 | 11 | 4 | 10 | loc, rfc, cam, amc, ce | rfc, loc, cam, amc, ce |
| XGB-TreeSHAP | size=25 | 11 | 4 | 8 | loc, rfc, cam, amc, ce | same |
| XGB-TreeSHAP | size=50 | 11 | 3 | 4 | loc, rfc, cam, amc, ce | same |
| XGB-TreeSHAP | size=200 | 11 | 1 | 6 | loc, rfc, cam, amc, ce | same |
| XGB-TreeSHAP | size=all | 11 | 1 | 6 | loc, rfc, cam, amc, ce | same |
| XGB-TreeSHAP | scheme=random | 11 | 1 | 6 | loc, rfc, cam, amc, ce | same |
| XGB-TreeSHAP | scheme=kmeans | 11 | 2 | 6 | loc, rfc, cam, amc, ce | same |
| XGB-TreeSHAP | scheme=defective | 11 | 5 | 10 | loc, rfc, cam, amc, ce | same |
| XGB-TreeSHAP | scheme=clean | 11 | 4 | 8 | loc, rfc, cam, amc, ce | same |
| XGB-TreeSHAP | draw=1 | 11 | 1 | 4 | loc, rfc, cam, amc, ce | same |
| XGB-TreeSHAP | draw=2 | 11 | 2 | 3 | loc, rfc, cam, amc, ce | same |
| XGB-TreeSHAP | draw=3 | 11 | 3 | 5 | loc, rfc, cam, amc, ce | same |
| XGB-TreeSHAP | draw=4 | 11 | 1 | 4 | loc, rfc, cam, amc, ce | same |

## loc share (ALL)

| variant | MLP-DeepLiftShap | MLP-GradientShap | RF-TreeSHAP | XGB-TreeSHAP |
|---|---|---|---|---|
| size=100 | 0.051 | 0.050 | 0.100 | 0.096 |
| size=10 | 0.050 | 0.048 | 0.099 | 0.094 |
| size=25 | 0.050 | 0.049 | 0.099 | 0.095 |
| size=50 | 0.051 | 0.050 | 0.099 | 0.095 |
| size=200 | 0.052 | 0.050 | 0.100 | 0.096 |
| size=all | 0.052 | 0.050 | 0.101 | 0.098 |
| scheme=random | 0.052 | 0.050 | 0.101 | 0.098 |
| scheme=kmeans | 0.051 | 0.050 | 0.102 | 0.099 |
| scheme=defective | 0.045 | 0.045 | 0.111 | 0.103 |
| scheme=clean | 0.055 | 0.054 | 0.103 | 0.099 |
| draw=1 | 0.051 | 0.050 | 0.099 | 0.096 |
| draw=2 | 0.052 | 0.050 | 0.099 | 0.097 |
| draw=3 | 0.052 | 0.051 | 0.100 | 0.097 |
| draw=4 | 0.051 | 0.050 | 0.100 | 0.097 |

## Size-group share (ALL)

| variant | MLP-DeepLiftShap | MLP-GradientShap | RF-TreeSHAP | XGB-TreeSHAP |
|---|---|---|---|---|
| size=100 | 0.215 | 0.212 | 0.317 | 0.289 |
| size=10 | 0.217 | 0.215 | 0.324 | 0.294 |
| size=25 | 0.214 | 0.212 | 0.317 | 0.288 |
| size=50 | 0.215 | 0.213 | 0.318 | 0.288 |
| size=200 | 0.215 | 0.212 | 0.316 | 0.289 |
| size=all | 0.215 | 0.212 | 0.317 | 0.291 |
| scheme=random | 0.214 | 0.211 | 0.317 | 0.291 |
| scheme=kmeans | 0.214 | 0.212 | 0.319 | 0.293 |
| scheme=defective | 0.204 | 0.203 | 0.333 | 0.294 |
| scheme=clean | 0.221 | 0.219 | 0.316 | 0.288 |
| draw=1 | 0.215 | 0.212 | 0.317 | 0.289 |
| draw=2 | 0.216 | 0.214 | 0.316 | 0.289 |
| draw=3 | 0.215 | 0.213 | 0.316 | 0.289 |
| draw=4 | 0.215 | 0.212 | 0.318 | 0.290 |
