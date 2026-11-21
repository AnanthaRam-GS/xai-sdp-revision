# Step 7: cross-version stability of explanations

Global importance g = mean L1-normalised |attribution| over all test instances of a split, averaged over the available seeds (10; LIME and KernelSHAP 3). Each of the 14 comparisons pairs consecutive time-aware splits of a project (e.g. ant 1.3→1.4 vs 1.4→1.5). For each metric: the value with a 95% CI from resampling the test instances of both splits (1,000 resamples); the **data-only reference** is the same metric between the permuted-label models (step 6 data randomisation, seed 0), which carry no defect signal; the **random null** is the metric between two independent random rankings (10,000 draws). Seed SD is the standard deviation over seeds of the same-seed comparison.

## Summary per explanation (mean over the 14 comparisons, 95% CI over comparisons)

| explanation | comparisons | Spearman ρ | Spearman ρ data-only | Kendall τ-b | Kendall τ-b data-only | top-5 Jaccard | top-5 Jaccard data-only | top-3 Jaccard | top-3 Jaccard data-only | RBO (p = 0.9) | RBO (p = 0.9) data-only | seed SD (ρ) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP-DeepLift | 14 | 0.179 [0.069, 0.274] | 0.120 | 0.122 [0.044, 0.191] | 0.086 | 0.231 [0.164, 0.294] | 0.216 | 0.250 [0.150, 0.386] | 0.100 | 0.522 [0.463, 0.578] | 0.454 | 0.201 |
| MLP-DeepLiftShap | 14 | 0.162 [0.047, 0.266] | 0.094 | 0.112 [0.032, 0.191] | 0.077 | 0.210 [0.142, 0.274] | 0.202 | 0.257 [0.143, 0.407] | 0.121 | 0.510 [0.453, 0.566] | 0.464 | 0.185 |
| MLP-GradientShap | 14 | 0.148 [0.036, 0.249] | 0.088 | 0.104 [0.024, 0.181] | 0.065 | 0.210 [0.142, 0.274] | 0.175 | 0.243 [0.121, 0.393] | 0.107 | 0.509 [0.451, 0.567] | 0.459 | 0.184 |
| MLP-IntegratedGradients | 14 | 0.171 [0.064, 0.266] | 0.115 | 0.120 [0.042, 0.191] | 0.084 | 0.243 [0.172, 0.312] | 0.206 | 0.250 [0.150, 0.386] | 0.100 | 0.523 [0.462, 0.580] | 0.451 | 0.201 |
| MLP-Saliency | 14 | 0.038 [-0.089, 0.144] | 0.070 | 0.027 [-0.053, 0.095] | 0.058 | 0.179 [0.113, 0.243] | 0.177 | 0.179 [0.086, 0.286] | 0.121 | 0.482 [0.420, 0.543] | 0.461 | 0.208 |
| MLP-InputXGradient | 14 | 0.218 [0.125, 0.306] | 0.204 | 0.153 [0.088, 0.217] | 0.155 | 0.239 [0.181, 0.297] | 0.179 | 0.243 [0.128, 0.393] | 0.157 | 0.534 [0.476, 0.590] | 0.492 | 0.196 |
| MLP-LIME | 14 | 0.046 [-0.112, 0.178] | 0.110 | 0.039 [-0.071, 0.136] | 0.095 | 0.189 [0.123, 0.251] | 0.170 | 0.171 [0.086, 0.264] | 0.121 | 0.487 [0.428, 0.544] | 0.464 | 0.178 |
| MLP-KernelSHAP | 14 | 0.158 [0.029, 0.261] | 0.086 | 0.120 [0.028, 0.199] | 0.067 | 0.241 [0.173, 0.307] | 0.158 | 0.193 [0.100, 0.300] | 0.121 | 0.510 [0.453, 0.563] | 0.468 | 0.171 |
| LR-LinearSHAP | 14 | 0.271 [0.106, 0.413] | -0.058 | 0.198 [0.077, 0.299] | -0.035 | 0.236 [0.186, 0.284] | 0.193 | 0.257 [0.143, 0.371] | 0.143 | 0.550 [0.484, 0.608] | 0.443 | 0.142 |
| LR-LIME | 14 | 0.238 [0.075, 0.380] | -0.082 | 0.170 [0.049, 0.278] | -0.051 | 0.221 [0.156, 0.287] | 0.162 | 0.207 [0.114, 0.293] | 0.064 | 0.507 [0.450, 0.559] | 0.419 | 0.123 |
| RF-TreeSHAP | 14 | 0.512 [0.396, 0.609] | 0.390 | 0.379 [0.286, 0.460] | 0.286 | 0.313 [0.209, 0.419] | 0.254 | 0.221 [0.136, 0.307] | 0.129 | 0.589 [0.526, 0.648] | 0.525 | 0.084 |
| RF-LIME | 14 | 0.459 [0.352, 0.557] | 0.235 | 0.333 [0.249, 0.410] | 0.169 | 0.327 [0.249, 0.409] | 0.215 | 0.271 [0.157, 0.386] | 0.114 | 0.575 [0.520, 0.635] | 0.480 | 0.097 |
| XGB-TreeSHAP | 14 | 0.521 [0.446, 0.595] | 0.325 | 0.372 [0.316, 0.429] | 0.240 | 0.261 [0.199, 0.322] | 0.223 | 0.229 [0.100, 0.379] | 0.129 | 0.568 [0.522, 0.614] | 0.489 | 0.107 |
| XGB-LIME | 14 | 0.508 [0.440, 0.577] | 0.272 | 0.379 [0.326, 0.437] | 0.193 | 0.338 [0.263, 0.423] | 0.200 | 0.171 [0.093, 0.264] | 0.071 | 0.568 [0.529, 0.613] | 0.479 | 0.105 |
| random null (mean / 95th pct) |  | 0.001 / 0.373 |  | 0.001 / 0.263 |  | 0.155 / 0.429 |  | 0.094 / 0.200 |  | 0.439 / 0.592 |  |  |

## Per comparison: Spearman ρ [instance-bootstrap CI] / data-only reference (primary explanations)

| comparison | MLP-DeepLiftShap | MLP-KernelSHAP | MLP-IntegratedGradients | MLP-LIME | MLP-Saliency | LR-LinearSHAP | RF-TreeSHAP | XGB-TreeSHAP |
|---|---|---|---|---|---|---|---|---|
| ant 1.3>1.4 vs 1.4>1.5 | 0.05 [-0.02, 0.16] / -0.12 | 0.15 [0.05, 0.21] / -0.02 | 0.09 [-0.03, 0.19] / 0.02 | 0.05 [0.00, 0.12] / -0.30 | -0.00 [-0.07, 0.06] / -0.40 | 0.17 [0.09, 0.23] / -0.15 | 0.68 [0.65, 0.74] / 0.00 | 0.63 [0.59, 0.65] / 0.31 |
| ant 1.4>1.5 vs 1.5>1.6 | -0.31 [-0.36, -0.25] / -0.21 | -0.48 [-0.56, -0.42] / -0.24 | -0.33 [-0.41, -0.28] / 0.12 | -0.70 [-0.72, -0.67] / -0.43 | -0.55 [-0.59, -0.49] / -0.28 | 0.06 [-0.05, 0.12] / -0.61 | 0.32 [0.29, 0.39] / 0.12 | 0.45 [0.41, 0.48] / 0.42 |
| ant 1.5>1.6 vs 1.6>1.7 | 0.35 [0.24, 0.37] / 0.00 | 0.15 [0.07, 0.22] / -0.17 | 0.23 [0.18, 0.31] / -0.22 | 0.04 [0.00, 0.12] / 0.01 | 0.15 [0.11, 0.19] / -0.12 | 0.66 [0.62, 0.71] / 0.16 | 0.62 [0.60, 0.66] / 0.19 | 0.56 [0.53, 0.57] / 0.29 |
| camel 1.2>1.4 vs 1.4>1.6 | 0.41 [0.35, 0.47] / 0.14 | 0.32 [0.28, 0.36] / 0.16 | 0.48 [0.42, 0.54] / 0.01 | 0.03 [-0.01, 0.09] / 0.31 | 0.10 [0.05, 0.17] / 0.07 | 0.26 [0.23, 0.28] / 0.42 | 0.66 [0.60, 0.67] / 0.79 | 0.76 [0.72, 0.79] / 0.35 |
| ivy 1.1>1.4 vs 1.4>2.0 | 0.31 [0.25, 0.39] / 0.14 | 0.30 [0.21, 0.38] / 0.13 | 0.32 [0.22, 0.42] / 0.17 | 0.29 [0.22, 0.31] / 0.08 | 0.25 [0.20, 0.32] / 0.12 | 0.27 [0.13, 0.37] / 0.16 | 0.65 [0.64, 0.66] / 0.51 | 0.61 [0.55, 0.65] / 0.42 |
| jedit 3.2>4.0 vs 4.0>4.1 | 0.09 [0.02, 0.16] / 0.05 | -0.17 [-0.20, -0.07] / 0.03 | 0.20 [0.07, 0.29] / 0.06 | -0.15 [-0.20, -0.12] / 0.13 | 0.04 [-0.01, 0.12] / 0.41 | 0.53 [0.49, 0.58] / -0.30 | 0.63 [0.59, 0.69] / 0.33 | 0.45 [0.40, 0.47] / 0.52 |
| jedit 4.0>4.1 vs 4.1>4.2 | 0.51 [0.44, 0.55] / 0.02 | 0.28 [0.21, 0.35] / 0.00 | 0.33 [0.24, 0.46] / 0.00 | 0.34 [0.27, 0.36] / 0.15 | 0.30 [0.23, 0.34] / 0.10 | 0.64 [0.57, 0.69] / -0.35 | 0.76 [0.69, 0.79] / 0.26 | 0.72 [0.71, 0.76] / 0.16 |
| lucene 2.0>2.2 vs 2.2>2.4 | 0.11 [0.02, 0.16] / 0.02 | 0.38 [0.28, 0.42] / 0.03 | 0.01 [-0.08, 0.13] / -0.11 | 0.39 [0.32, 0.42] / 0.02 | 0.03 [-0.02, 0.08] / -0.26 | -0.07 [-0.17, -0.01] / 0.15 | 0.62 [0.59, 0.69] / 0.29 | 0.55 [0.51, 0.58] / 0.67 |
| poi 1.5>2.0 vs 2.0>2.5 | 0.09 [0.03, 0.16] / 0.91 | 0.11 [-0.00, 0.20] / 0.92 | 0.14 [0.05, 0.27] / 0.93 | -0.15 [-0.17, -0.12] / 0.96 | -0.11 [-0.17, -0.07] / 0.92 | 0.22 [0.14, 0.26] / 0.12 | 0.51 [0.46, 0.57] / 0.73 | 0.40 [0.36, 0.45] / 0.14 |
| poi 2.0>2.5 vs 2.5>3.0 | 0.34 [0.30, 0.38] / 0.14 | 0.34 [0.26, 0.37] / 0.16 | 0.22 [0.19, 0.36] / 0.01 | 0.15 [0.12, 0.18] / -0.04 | 0.09 [0.06, 0.20] / -0.13 | 0.60 [0.56, 0.64] / -0.03 | 0.37 [0.31, 0.42] / 0.45 | 0.38 [0.34, 0.41] / 0.21 |
| synapse 1.0>1.1 vs 1.1>1.2 | -0.09 [-0.24, -0.03] / -0.05 | 0.26 [0.14, 0.37] / 0.04 | 0.04 [-0.11, 0.16] / 0.12 | -0.13 [-0.15, -0.05] / 0.09 | -0.29 [-0.32, -0.21] / 0.02 | -0.41 [-0.47, -0.33] / -0.09 | 0.42 [0.38, 0.46] / 0.31 | 0.63 [0.60, 0.65] / 0.19 |
| velocity 1.4>1.5 vs 1.5>1.6 | 0.16 [0.03, 0.21] / 0.24 | 0.21 [0.13, 0.29] / 0.28 | 0.42 [0.34, 0.54] / 0.25 | 0.16 [0.12, 0.21] / 0.58 | 0.18 [0.13, 0.22] / 0.59 | 0.00 [-0.10, 0.07] / -0.22 | 0.04 [0.00, 0.15] / 0.58 | 0.23 [0.18, 0.26] / 0.53 |
| xalan 2.4>2.5 vs 2.5>2.6 | 0.14 [0.05, 0.19] / 0.09 | 0.18 [0.10, 0.22] / 0.05 | 0.12 [0.03, 0.16] / 0.11 | 0.32 [0.30, 0.35] / 0.15 | 0.23 [0.18, 0.25] / 0.04 | 0.51 [0.47, 0.54] / 0.13 | 0.69 [0.66, 0.71] / 0.22 | 0.53 [0.47, 0.57] / 0.12 |
| xerces init>1.2 vs 1.2>1.3 | 0.11 [0.05, 0.17] / -0.04 | 0.18 [0.10, 0.22] / -0.17 | 0.13 [0.01, 0.18] / 0.13 | -0.00 [-0.07, 0.07] / -0.15 | 0.12 [0.04, 0.17] / -0.09 | 0.35 [0.30, 0.42] / -0.18 | 0.20 [0.16, 0.26] / 0.67 | 0.39 [0.36, 0.43] / 0.23 |

## Per comparison: RBO [instance-bootstrap CI] / data-only reference (primary explanations)

| comparison | MLP-DeepLiftShap | MLP-KernelSHAP | MLP-IntegratedGradients | MLP-LIME | MLP-Saliency | LR-LinearSHAP | RF-TreeSHAP | XGB-TreeSHAP |
|---|---|---|---|---|---|---|---|---|
| ant 1.3>1.4 vs 1.4>1.5 | 0.51 [0.43, 0.63] / 0.38 | 0.52 [0.47, 0.64] / 0.39 | 0.52 [0.43, 0.65] / 0.39 | 0.51 [0.46, 0.53] / 0.33 | 0.43 [0.42, 0.47] / 0.32 | 0.56 [0.51, 0.58] / 0.40 | 0.53 [0.51, 0.58] / 0.38 | 0.49 [0.48, 0.50] / 0.48 |
| ant 1.4>1.5 vs 1.5>1.6 | 0.30 [0.29, 0.31] / 0.42 | 0.29 [0.28, 0.31] / 0.40 | 0.30 [0.29, 0.32] / 0.41 | 0.27 [0.27, 0.28] / 0.32 | 0.28 [0.27, 0.29] / 0.43 | 0.44 [0.40, 0.45] / 0.33 | 0.44 [0.43, 0.46] / 0.52 | 0.54 [0.51, 0.54] / 0.61 |
| ant 1.5>1.6 vs 1.6>1.7 | 0.47 [0.42, 0.49] / 0.40 | 0.45 [0.41, 0.51] / 0.38 | 0.50 [0.49, 0.52] / 0.36 | 0.39 [0.37, 0.43] / 0.41 | 0.41 [0.39, 0.42] / 0.35 | 0.53 [0.51, 0.56] / 0.55 | 0.59 [0.59, 0.61] / 0.51 | 0.51 [0.50, 0.52] / 0.48 |
| camel 1.2>1.4 vs 1.4>1.6 | 0.65 [0.63, 0.68] / 0.47 | 0.65 [0.62, 0.66] / 0.46 | 0.71 [0.69, 0.75] / 0.43 | 0.60 [0.59, 0.62] / 0.47 | 0.63 [0.59, 0.64] / 0.41 | 0.65 [0.54, 0.66] / 0.69 | 0.74 [0.71, 0.74] / 0.76 | 0.66 [0.63, 0.72] / 0.55 |
| ivy 1.1>1.4 vs 1.4>2.0 | 0.55 [0.53, 0.60] / 0.45 | 0.60 [0.53, 0.62] / 0.43 | 0.52 [0.50, 0.60] / 0.51 | 0.55 [0.54, 0.55] / 0.48 | 0.56 [0.55, 0.63] / 0.49 | 0.49 [0.43, 0.58] / 0.44 | 0.65 [0.65, 0.66] / 0.48 | 0.59 [0.54, 0.60] / 0.48 |
| jedit 3.2>4.0 vs 4.0>4.1 | 0.65 [0.64, 0.68] / 0.40 | 0.61 [0.52, 0.64] / 0.43 | 0.66 [0.63, 0.69] / 0.42 | 0.65 [0.64, 0.66] / 0.48 | 0.67 [0.66, 0.68] / 0.54 | 0.75 [0.67, 0.79] / 0.34 | 0.69 [0.68, 0.72] / 0.44 | 0.62 [0.61, 0.62] / 0.46 |
| jedit 4.0>4.1 vs 4.1>4.2 | 0.72 [0.69, 0.83] / 0.55 | 0.65 [0.63, 0.76] / 0.49 | 0.73 [0.64, 0.76] / 0.48 | 0.63 [0.61, 0.64] / 0.50 | 0.62 [0.61, 0.64] / 0.48 | 0.68 [0.60, 0.70] / 0.35 | 0.71 [0.68, 0.73] / 0.47 | 0.72 [0.69, 0.72] / 0.47 |
| lucene 2.0>2.2 vs 2.2>2.4 | 0.47 [0.42, 0.49] / 0.41 | 0.58 [0.49, 0.64] / 0.56 | 0.44 [0.40, 0.48] / 0.40 | 0.54 [0.51, 0.54] / 0.45 | 0.47 [0.46, 0.50] / 0.39 | 0.41 [0.38, 0.44] / 0.44 | 0.66 [0.58, 0.68] / 0.42 | 0.51 [0.50, 0.51] / 0.60 |
| poi 1.5>2.0 vs 2.0>2.5 | 0.52 [0.46, 0.53] / 0.86 | 0.44 [0.39, 0.47] / 0.87 | 0.55 [0.50, 0.57] / 0.83 | 0.34 [0.34, 0.37] / 0.90 | 0.38 [0.36, 0.39] / 0.91 | 0.50 [0.42, 0.52] / 0.54 | 0.52 [0.48, 0.54] / 0.56 | 0.46 [0.43, 0.53] / 0.41 |
| poi 2.0>2.5 vs 2.5>3.0 | 0.45 [0.44, 0.46] / 0.44 | 0.48 [0.44, 0.49] / 0.42 | 0.49 [0.45, 0.51] / 0.40 | 0.46 [0.46, 0.46] / 0.38 | 0.40 [0.39, 0.42] / 0.42 | 0.63 [0.62, 0.65] / 0.47 | 0.52 [0.50, 0.54] / 0.49 | 0.42 [0.41, 0.45] / 0.41 |
| synapse 1.0>1.1 vs 1.1>1.2 | 0.37 [0.34, 0.39] / 0.35 | 0.42 [0.40, 0.45] / 0.37 | 0.39 [0.35, 0.44] / 0.35 | 0.34 [0.33, 0.35] / 0.35 | 0.33 [0.32, 0.35] / 0.34 | 0.32 [0.31, 0.38] / 0.38 | 0.41 [0.40, 0.42] / 0.50 | 0.62 [0.55, 0.62] / 0.37 |
| velocity 1.4>1.5 vs 1.5>1.6 | 0.57 [0.45, 0.59] / 0.50 | 0.55 [0.53, 0.67] / 0.54 | 0.59 [0.52, 0.74] / 0.44 | 0.56 [0.54, 0.58] / 0.60 | 0.58 [0.53, 0.59] / 0.53 | 0.42 [0.40, 0.44] / 0.38 | 0.43 [0.42, 0.45] / 0.62 | 0.47 [0.46, 0.49] / 0.66 |
| xalan 2.4>2.5 vs 2.5>2.6 | 0.45 [0.41, 0.45] / 0.42 | 0.42 [0.40, 0.43] / 0.40 | 0.45 [0.40, 0.46] / 0.39 | 0.47 [0.46, 0.47] / 0.44 | 0.47 [0.46, 0.47] / 0.42 | 0.73 [0.64, 0.73] / 0.48 | 0.74 [0.71, 0.74] / 0.42 | 0.68 [0.56, 0.70] / 0.43 |
| xerces init>1.2 vs 1.2>1.3 | 0.46 [0.45, 0.53] / 0.47 | 0.47 [0.46, 0.52] / 0.42 | 0.46 [0.43, 0.48] / 0.49 | 0.51 [0.50, 0.54] / 0.39 | 0.52 [0.51, 0.55] / 0.40 | 0.58 [0.58, 0.70] / 0.39 | 0.61 [0.58, 0.63] / 0.81 | 0.66 [0.64, 0.67] / 0.43 |

## Scott-Knott ESD ranks (per-comparison values; 1 = most stable)

| explanation | Spearman | RBO |
|---|---|---|
| MLP-DeepLift | 2 | 1 |
| MLP-DeepLiftShap | 2 | 1 |
| MLP-GradientShap | 2 | 1 |
| MLP-IntegratedGradients | 2 | 1 |
| MLP-Saliency | 2 | 1 |
| MLP-InputXGradient | 2 | 1 |
| MLP-LIME | 2 | 1 |
| MLP-KernelSHAP | 2 | 1 |
| LR-LinearSHAP | 2 | 1 |
| LR-LIME | 2 | 1 |
| RF-TreeSHAP | 1 | 1 |
| RF-LIME | 1 | 1 |
| XGB-TreeSHAP | 1 | 1 |
| XGB-LIME | 1 | 1 |

## Each explanation vs its data-only reference (Wilcoxon signed-rank over comparisons, Holm, Cliff's delta)

| metric | a | b | n | mean_a | mean_b | p_value | p_holm | cliffs_delta | magnitude |
|---|---|---|---|---|---|---|---|---|---|
| Spearman ρ | MLP-DeepLift | data-only reference | 14 | 0.179 | 0.120 | 0.217 | 1.000 | 0.347 | medium |
| Spearman ρ | MLP-DeepLiftShap | data-only reference | 14 | 0.162 | 0.094 | 0.091 | 0.815 | 0.362 | medium |
| Spearman ρ | MLP-GradientShap | data-only reference | 14 | 0.148 | 0.088 | 0.135 | 1.000 | 0.306 | small |
| Spearman ρ | MLP-IntegratedGradients | data-only reference | 14 | 0.171 | 0.115 | 0.173 | 1.000 | 0.372 | medium |
| Spearman ρ | MLP-Saliency | data-only reference | 14 | 0.038 | 0.070 | 0.855 | 1.000 | 0.138 | negligible |
| Spearman ρ | MLP-InputXGradient | data-only reference | 14 | 0.218 | 0.204 | 0.268 | 1.000 | 0.184 | small |
| Spearman ρ | MLP-LIME | data-only reference | 14 | 0.046 | 0.110 | 0.626 | 1.000 | 0.010 | negligible |
| Spearman ρ | MLP-KernelSHAP | data-only reference | 14 | 0.158 | 0.086 | 0.194 | 1.000 | 0.469 | medium |
| Spearman ρ | LR-LinearSHAP | data-only reference | 14 | 0.271 | -0.058 | 0.013 | 0.161 | 0.622 | large |
| Spearman ρ | LR-LIME | data-only reference | 14 | 0.238 | -0.082 | 0.011 | 0.140 | 0.561 | large |
| Spearman ρ | RF-TreeSHAP | data-only reference | 14 | 0.512 | 0.390 | 0.268 | 1.000 | 0.306 | small |
| Spearman ρ | RF-LIME | data-only reference | 14 | 0.459 | 0.235 | 0.017 | 0.166 | 0.439 | medium |
| Spearman ρ | XGB-TreeSHAP | data-only reference | 14 | 0.521 | 0.325 | 0.013 | 0.161 | 0.633 | large |
| Spearman ρ | XGB-LIME | data-only reference | 14 | 0.508 | 0.272 | 0.003 | 0.043 | 0.684 | large |
| RBO (p = 0.9) | MLP-DeepLift | data-only reference | 14 | 0.522 | 0.454 | 0.078 | 0.549 | 0.480 | large |
| RBO (p = 0.9) | MLP-DeepLiftShap | data-only reference | 14 | 0.510 | 0.464 | 0.078 | 0.549 | 0.408 | medium |
| RBO (p = 0.9) | MLP-GradientShap | data-only reference | 14 | 0.509 | 0.459 | 0.078 | 0.549 | 0.429 | medium |
| RBO (p = 0.9) | MLP-IntegratedGradients | data-only reference | 14 | 0.523 | 0.451 | 0.068 | 0.544 | 0.500 | large |
| RBO (p = 0.9) | MLP-Saliency | data-only reference | 14 | 0.482 | 0.461 | 0.153 | 0.549 | 0.173 | small |
| RBO (p = 0.9) | MLP-InputXGradient | data-only reference | 14 | 0.534 | 0.492 | 0.268 | 0.549 | 0.265 | small |
| RBO (p = 0.9) | MLP-LIME | data-only reference | 14 | 0.487 | 0.464 | 0.104 | 0.549 | 0.265 | small |
| RBO (p = 0.9) | MLP-KernelSHAP | data-only reference | 14 | 0.510 | 0.468 | 0.058 | 0.544 | 0.418 | medium |
| RBO (p = 0.9) | LR-LinearSHAP | data-only reference | 14 | 0.550 | 0.443 | 0.035 | 0.459 | 0.500 | large |
| RBO (p = 0.9) | LR-LIME | data-only reference | 14 | 0.507 | 0.419 | 0.049 | 0.544 | 0.469 | medium |
| RBO (p = 0.9) | RF-TreeSHAP | data-only reference | 14 | 0.589 | 0.525 | 0.217 | 0.549 | 0.337 | medium |
| RBO (p = 0.9) | RF-LIME | data-only reference | 14 | 0.575 | 0.480 | 0.030 | 0.414 | 0.490 | large |
| RBO (p = 0.9) | XGB-TreeSHAP | data-only reference | 14 | 0.568 | 0.489 | 0.049 | 0.544 | 0.510 | large |
| RBO (p = 0.9) | XGB-LIME | data-only reference | 14 | 0.568 | 0.479 | 0.042 | 0.502 | 0.622 | large |

## MLP-DeepLiftShap vs RF-TreeSHAP vs XGB-TreeSHAP vs LR-LinearSHAP

| metric | a | b | n | mean_a | mean_b | p_value | p_holm | cliffs_delta | magnitude |
|---|---|---|---|---|---|---|---|---|---|
| Spearman ρ | MLP-DeepLiftShap | RF-TreeSHAP | 14 | 0.162 | 0.512 | <0.001 | 0.003 | -0.750 | large |
| Spearman ρ | MLP-DeepLiftShap | XGB-TreeSHAP | 14 | 0.162 | 0.521 | <0.001 | <0.001 | -0.867 | large |
| Spearman ρ | MLP-DeepLiftShap | LR-LinearSHAP | 14 | 0.162 | 0.271 | 0.173 | 0.345 | -0.270 | small |
| Spearman ρ | RF-TreeSHAP | XGB-TreeSHAP | 14 | 0.512 | 0.521 | 0.855 | 0.855 | 0.051 | negligible |
| Spearman ρ | RF-TreeSHAP | LR-LinearSHAP | 14 | 0.512 | 0.271 | 0.011 | 0.043 | 0.510 | large |
| Spearman ρ | XGB-TreeSHAP | LR-LinearSHAP | 14 | 0.521 | 0.271 | 0.017 | 0.050 | 0.500 | large |
| RBO (p = 0.9) | MLP-DeepLiftShap | RF-TreeSHAP | 14 | 0.510 | 0.589 | 0.009 | 0.051 | -0.378 | medium |
| RBO (p = 0.9) | MLP-DeepLiftShap | XGB-TreeSHAP | 14 | 0.510 | 0.568 | 0.217 | 0.866 | -0.316 | small |
| RBO (p = 0.9) | MLP-DeepLiftShap | LR-LinearSHAP | 14 | 0.510 | 0.550 | 0.326 | 0.866 | -0.163 | small |
| RBO (p = 0.9) | RF-TreeSHAP | XGB-TreeSHAP | 14 | 0.589 | 0.568 | 0.268 | 0.866 | 0.122 | negligible |
| RBO (p = 0.9) | RF-TreeSHAP | LR-LinearSHAP | 14 | 0.589 | 0.550 | 0.119 | 0.594 | 0.184 | small |
| RBO (p = 0.9) | XGB-TreeSHAP | LR-LinearSHAP | 14 | 0.568 | 0.550 | 0.583 | 0.866 | 0.102 | negligible |

## Direction stability

Sign of the step 4 direction statistic (Spearman of raw value vs signed attribution, seeds averaged) in every split of a project; share of the 20 features whose sign is the same in all splits (mean over projects). Flipping features are listed in `step7_cross_version_direction_flips.csv`.

| explanation | same sign in all splits (mean over projects) | worst project |
|---|---|---|
| MLP-DeepLift | 0.580 | 0.300 |
| MLP-DeepLiftShap | 0.580 | 0.350 |
| MLP-GradientShap | 0.585 | 0.350 |
| MLP-IntegratedGradients | 0.575 | 0.300 |
| MLP-Saliency | 0.500 | 0.250 |
| MLP-InputXGradient | 0.590 | 0.400 |
| MLP-LIME | 0.470 | 0.200 |
| MLP-KernelSHAP | 0.605 | 0.300 |
| LR-LinearSHAP | 0.607 | 0.400 |
| LR-LIME | 0.470 | 0.250 |
| RF-TreeSHAP | 0.583 | 0.400 |
| RF-LIME | 0.535 | 0.150 |
| XGB-TreeSHAP | 0.527 | 0.400 |
| XGB-LIME | 0.515 | 0.250 |

Most frequent flips (primary explanations; number of projects in which the feature flips):

| explanation | feature | projects |
|---|---|---|
| LR-LinearSHAP | avg_cc | 9 |
| LR-LinearSHAP | lcom3 | 7 |
| LR-LinearSHAP | dit | 6 |
| MLP-DeepLiftShap | moa | 8 |
| MLP-DeepLiftShap | cam | 7 |
| MLP-DeepLiftShap | ic | 7 |
| MLP-IntegratedGradients | dit | 7 |
| MLP-IntegratedGradients | ic | 7 |
| MLP-IntegratedGradients | moa | 7 |
| MLP-KernelSHAP | dit | 8 |
| MLP-KernelSHAP | ic | 7 |
| MLP-KernelSHAP | moa | 7 |
| MLP-LIME | ic | 9 |
| MLP-LIME | cam | 7 |
| MLP-LIME | cbm | 7 |
| MLP-Saliency | amc | 7 |
| MLP-Saliency | cbm | 7 |
| MLP-Saliency | ic | 7 |
| RF-TreeSHAP | dit | 7 |
| RF-TreeSHAP | moa | 7 |
| RF-TreeSHAP | ic | 6 |
| XGB-TreeSHAP | dit | 8 |
| XGB-TreeSHAP | amc | 7 |
| XGB-TreeSHAP | ce | 7 |

## Cluster level (step 5 stable clusters: {wmc, npm}; {dit, mfa}; {rfc, loc}; {ic, cbm}; {max_cc, avg_cc}; other metrics singletons)

Attributions summed within each cluster per instance, then the same analysis; the random null is for 15 items.

| explanation | comparisons | Spearman ρ | Spearman ρ data-only | Kendall τ-b | Kendall τ-b data-only | top-5 Jaccard | top-5 Jaccard data-only | top-3 Jaccard | top-3 Jaccard data-only | RBO (p = 0.9) | RBO (p = 0.9) data-only | seed SD (ρ) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP-DeepLift | 14 | 0.398 [0.238, 0.544] | 0.188 | 0.299 [0.178, 0.415] | 0.146 | 0.438 [0.319, 0.574] | 0.291 | 0.300 [0.207, 0.393] | 0.214 | 0.657 [0.592, 0.723] | 0.583 | 0.175 |
| MLP-DeepLiftShap | 14 | 0.341 [0.177, 0.488] | 0.085 | 0.249 [0.128, 0.362] | 0.078 | 0.367 [0.256, 0.480] | 0.240 | 0.279 [0.186, 0.371] | 0.193 | 0.623 [0.568, 0.681] | 0.558 | 0.172 |
| MLP-GradientShap | 14 | 0.345 [0.184, 0.490] | 0.091 | 0.254 [0.137, 0.365] | 0.086 | 0.367 [0.256, 0.480] | 0.248 | 0.279 [0.186, 0.371] | 0.193 | 0.626 [0.567, 0.686] | 0.559 | 0.170 |
| MLP-IntegratedGradients | 14 | 0.399 [0.229, 0.556] | 0.191 | 0.297 [0.162, 0.425] | 0.137 | 0.438 [0.319, 0.574] | 0.291 | 0.300 [0.207, 0.393] | 0.214 | 0.658 [0.592, 0.724] | 0.579 | 0.179 |
| MLP-Saliency | 14 | 0.330 [0.187, 0.453] | 0.228 | 0.224 [0.125, 0.314] | 0.181 | 0.370 [0.278, 0.467] | 0.291 | 0.200 [0.129, 0.279] | 0.229 | 0.624 [0.568, 0.683] | 0.598 | 0.198 |
| MLP-InputXGradient | 14 | 0.458 [0.335, 0.568] | 0.347 | 0.340 [0.248, 0.426] | 0.259 | 0.468 [0.345, 0.602] | 0.355 | 0.300 [0.207, 0.393] | 0.264 | 0.660 [0.602, 0.717] | 0.615 | 0.169 |
| MLP-LIME | 14 | 0.205 [0.068, 0.330] | 0.119 | 0.148 [0.049, 0.237] | 0.101 | 0.314 [0.236, 0.399] | 0.210 | 0.193 [0.143, 0.250] | 0.193 | 0.612 [0.563, 0.657] | 0.561 | 0.190 |
| MLP-KernelSHAP | 14 | 0.302 [0.160, 0.424] | 0.091 | 0.216 [0.106, 0.310] | 0.076 | 0.324 [0.249, 0.403] | 0.253 | 0.264 [0.164, 0.364] | 0.179 | 0.624 [0.571, 0.678] | 0.559 | 0.130 |
| LR-LinearSHAP | 14 | 0.327 [0.142, 0.484] | 0.066 | 0.241 [0.106, 0.358] | 0.045 | 0.363 [0.256, 0.470] | 0.239 | 0.321 [0.221, 0.414] | 0.179 | 0.653 [0.596, 0.705] | 0.560 | 0.129 |
| LR-LIME | 14 | 0.181 [0.021, 0.329] | 0.061 | 0.137 [0.029, 0.239] | 0.048 | 0.259 [0.206, 0.314] | 0.232 | 0.264 [0.164, 0.357] | 0.121 | 0.615 [0.557, 0.671] | 0.545 | 0.119 |
| RF-TreeSHAP | 14 | 0.547 [0.431, 0.648] | 0.403 | 0.405 [0.313, 0.495] | 0.288 | 0.414 [0.314, 0.549] | 0.336 | 0.229 [0.129, 0.329] | 0.229 | 0.674 [0.615, 0.732] | 0.629 | 0.094 |
| RF-LIME | 14 | 0.473 [0.366, 0.573] | 0.177 | 0.337 [0.260, 0.415] | 0.120 | 0.394 [0.281, 0.523] | 0.278 | 0.236 [0.157, 0.321] | 0.171 | 0.663 [0.609, 0.719] | 0.560 | 0.108 |
| XGB-TreeSHAP | 14 | 0.511 [0.423, 0.603] | 0.283 | 0.393 [0.327, 0.464] | 0.208 | 0.369 [0.277, 0.461] | 0.264 | 0.236 [0.157, 0.321] | 0.207 | 0.655 [0.605, 0.701] | 0.572 | 0.117 |
| XGB-LIME | 14 | 0.481 [0.391, 0.569] | 0.230 | 0.341 [0.263, 0.423] | 0.158 | 0.373 [0.271, 0.474] | 0.263 | 0.221 [0.136, 0.314] | 0.121 | 0.638 [0.591, 0.688] | 0.571 | 0.125 |
| random null (mean / 95th pct) |  | 0.004 / 0.436 |  | 0.003 / 0.314 |  | 0.215 / 0.429 |  | 0.132 / 0.500 |  | 0.530 / 0.687 |  |  |

Scott-Knott ESD ranks at cluster level:

| explanation | Spearman | RBO |
|---|---|---|
| MLP-DeepLift | 1 | 1 |
| MLP-DeepLiftShap | 2 | 1 |
| MLP-GradientShap | 2 | 1 |
| MLP-IntegratedGradients | 1 | 1 |
| MLP-Saliency | 2 | 1 |
| MLP-InputXGradient | 1 | 1 |
| MLP-LIME | 2 | 1 |
| MLP-KernelSHAP | 2 | 1 |
| LR-LinearSHAP | 2 | 1 |
| LR-LIME | 2 | 1 |
| RF-TreeSHAP | 1 | 1 |
| RF-LIME | 1 | 1 |
| XGB-TreeSHAP | 1 | 1 |
| XGB-LIME | 1 | 1 |

Cluster level vs data-only reference:

| metric | a | b | n | mean_a | mean_b | p_value | p_holm | cliffs_delta | magnitude |
|---|---|---|---|---|---|---|---|---|---|
| Spearman ρ | MLP-DeepLift | data-only reference | 14 | 0.398 | 0.188 | 0.153 | 0.918 | 0.388 | medium |
| Spearman ρ | MLP-DeepLiftShap | data-only reference | 14 | 0.341 | 0.085 | 0.068 | 0.676 | 0.469 | medium |
| Spearman ρ | MLP-GradientShap | data-only reference | 14 | 0.345 | 0.091 | 0.068 | 0.676 | 0.490 | large |
| Spearman ρ | MLP-IntegratedGradients | data-only reference | 14 | 0.399 | 0.191 | 0.194 | 0.969 | 0.357 | medium |
| Spearman ρ | MLP-Saliency | data-only reference | 14 | 0.330 | 0.228 | 0.241 | 0.969 | 0.214 | small |
| Spearman ρ | MLP-InputXGradient | data-only reference | 14 | 0.458 | 0.347 | 0.268 | 0.969 | 0.235 | small |
| Spearman ρ | MLP-LIME | data-only reference | 14 | 0.205 | 0.119 | 0.362 | 0.969 | 0.245 | small |
| Spearman ρ | MLP-KernelSHAP | data-only reference | 14 | 0.302 | 0.091 | 0.102 | 0.820 | 0.607 | large |
| Spearman ρ | LR-LinearSHAP | data-only reference | 14 | 0.327 | 0.066 | 0.042 | 0.461 | 0.469 | medium |
| Spearman ρ | LR-LIME | data-only reference | 14 | 0.181 | 0.061 | 0.217 | 0.969 | 0.281 | small |
| Spearman ρ | RF-TreeSHAP | data-only reference | 14 | 0.547 | 0.403 | 0.102 | 0.820 | 0.408 | medium |
| Spearman ρ | RF-LIME | data-only reference | 14 | 0.473 | 0.177 | 0.009 | 0.103 | 0.653 | large |
| Spearman ρ | XGB-TreeSHAP | data-only reference | 14 | 0.511 | 0.283 | 0.002 | 0.032 | 0.612 | large |
| Spearman ρ | XGB-LIME | data-only reference | 14 | 0.481 | 0.230 | 0.005 | 0.068 | 0.571 | large |
| RBO (p = 0.9) | MLP-DeepLift | data-only reference | 14 | 0.657 | 0.583 | 0.296 | 1.000 | 0.296 | small |
| RBO (p = 0.9) | MLP-DeepLiftShap | data-only reference | 14 | 0.623 | 0.558 | 0.217 | 1.000 | 0.357 | medium |
| RBO (p = 0.9) | MLP-GradientShap | data-only reference | 14 | 0.626 | 0.559 | 0.194 | 1.000 | 0.388 | medium |
| RBO (p = 0.9) | MLP-IntegratedGradients | data-only reference | 14 | 0.658 | 0.579 | 0.173 | 1.000 | 0.337 | medium |
| RBO (p = 0.9) | MLP-Saliency | data-only reference | 14 | 0.624 | 0.598 | 0.463 | 1.000 | 0.173 | small |
| RBO (p = 0.9) | MLP-InputXGradient | data-only reference | 14 | 0.660 | 0.615 | 0.358 | 1.000 | 0.265 | small |
| RBO (p = 0.9) | MLP-LIME | data-only reference | 14 | 0.612 | 0.561 | 0.217 | 1.000 | 0.327 | small |
| RBO (p = 0.9) | MLP-KernelSHAP | data-only reference | 14 | 0.624 | 0.559 | 0.173 | 1.000 | 0.388 | medium |
| RBO (p = 0.9) | LR-LinearSHAP | data-only reference | 14 | 0.653 | 0.560 | 0.025 | 0.344 | 0.449 | medium |
| RBO (p = 0.9) | LR-LIME | data-only reference | 14 | 0.615 | 0.545 | 0.030 | 0.384 | 0.388 | medium |
| RBO (p = 0.9) | RF-TreeSHAP | data-only reference | 14 | 0.674 | 0.629 | 0.358 | 1.000 | 0.173 | small |
| RBO (p = 0.9) | RF-LIME | data-only reference | 14 | 0.663 | 0.560 | 0.091 | 0.996 | 0.510 | large |
| RBO (p = 0.9) | XGB-TreeSHAP | data-only reference | 14 | 0.655 | 0.572 | 0.030 | 0.384 | 0.480 | large |
| RBO (p = 0.9) | XGB-LIME | data-only reference | 14 | 0.638 | 0.571 | 0.153 | 1.000 | 0.418 | medium |
