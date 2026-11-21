# Step 6: fidelity summary

Per explanation (seed-0 models; ROAR/KAR seeds averaged): mean gains over random for deletion and insertion, mean faithfulness correlation, ROAR gain (random area − method area) and KAR gain (method area − random area) where training-set rankings exist, and the sanity-check result. Higher is better for every number. Overall ordering: per split, each explanation is ranked within each metric (1 = best among the explanations that have it), the ranks are averaged, and Scott-Knott ESD groups the per-split mean ranks (group 1 = best). Gains over random are comparable across models; the absolute curves are not.

| explanation | deletion gain | insertion gain | faithfulness | ROAR gain | KAR gain | sanity | mean rank | SK-ESD group |
|---|---|---|---|---|---|---|---|---|
| MLP-KernelSHAP | 0.062 | 0.057 | 0.951 | n/a | n/a | pass | 4.01 | 1 |
| XGB-TreeSHAP | 0.066 | 0.063 | 0.960 | 0.013 | 0.011 | pass | 4.03 | 1 |
| RF-TreeSHAP | 0.063 | 0.064 | 0.958 | 0.015 | 0.011 | pass | 4.28 | 1 |
| MLP-DeepLiftShap | 0.061 | 0.056 | 0.952 | -0.002 | 0.005 | pass | 4.47 | 1 |
| LR-LinearSHAP | 0.045 | 0.047 | 0.981 | 0.008 | 0.002 | pass | 5.36 | 2 |
| MLP-IntegratedGradients | 0.053 | 0.050 | 0.929 | -0.002 | 0.003 | pass | 5.95 | 2 |
| MLP-GradientShap | 0.059 | 0.054 | 0.934 | n/a | n/a | pass | 6.10 | 2 |
| MLP-DeepLift | 0.052 | 0.050 | 0.929 | n/a | n/a | pass | 7.49 | 3 |
| MLP-LIME | 0.046 | 0.046 | -0.022 | 0.002 | 0.006 | pass | 7.61 | 3 |
| MLP-InputXGradient | 0.051 | 0.048 | 0.903 | n/a | n/a | pass | 8.24 | 3 |
| MLP-Saliency | 0.041 | 0.040 | -0.055 | 0.002 | 0.007 | pass | 8.51 | 3 |
| XGB-LIME | 0.046 | 0.044 | -0.082 | n/a | n/a | pass | 9.99 | 4 |
| LR-LIME | 0.042 | 0.045 | -0.004 | n/a | n/a | pass | 10.26 | 4 |
| RF-LIME | 0.042 | 0.043 | -0.101 | n/a | n/a | pass | 10.57 | 4 |

## Which explainers look most faithful

* **MLP**: deletion favours KernelSHAP, insertion favours KernelSHAP, faithfulness favours DeepLiftShap. ROAR: DeepLiftShap, IntegratedGradients, Saliency, LIME not distinguishable from random. KAR: for DeepLiftShap, IntegratedGradients, Saliency, LIME, keeping only the top-ranked metrics beats keeping random ones (Holm p < 0.05).
* **LR**: deletion favours LinearSHAP, insertion favours LinearSHAP, faithfulness favours LinearSHAP. ROAR: LinearSHAP not distinguishable from random. KAR: LinearSHAP not distinguishable from random.
* **RF**: deletion favours TreeSHAP, insertion favours TreeSHAP, faithfulness favours TreeSHAP. ROAR: for TreeSHAP, removing the top-ranked metrics hurts the refitted model more than random removal (Holm p < 0.05). KAR: for TreeSHAP, keeping only the top-ranked metrics beats keeping random ones (Holm p < 0.05).
* **XGB**: deletion favours TreeSHAP, insertion favours TreeSHAP, faithfulness favours TreeSHAP. ROAR: for TreeSHAP, removing the top-ranked metrics hurts the refitted model more than random removal (Holm p < 0.05). KAR: for TreeSHAP, keeping only the top-ranked metrics beats keeping random ones (Holm p < 0.05).

These are fidelity results for the explanations of these models on these data. Note that gains over random depend on how concentrated the model's dependence on a few metrics is, so they should be compared within a model rather than across models.
