# Step 5: multicollinearity

Diagnostics on the fitting rows of each of the 24 training releases (raw metrics): Spearman correlation, hierarchical clustering (average linkage) on 1 − |ρ| cut at |ρ| = 0.7, and VIF on log1p-transformed metrics. Stable clusters link two metrics that are in the same cluster in ≥ 75% of the training releases (connected components of these links).

## Stable clusters

{wmc, npm}; {dit, mfa}; {rfc, loc}; {ic, cbm}; {max_cc, avg_cc}. All other metrics are singletons (10). The size cluster (containing loc) is {rfc, loc}.

Pairs of metrics clustered together in at least half of the training releases:

| metric a | metric b | share of releases |
|---|---|---|
| dit | mfa | 1.00 |
| ic | cbm | 1.00 |
| max_cc | avg_cc | 1.00 |
| wmc | npm | 0.92 |
| rfc | loc | 0.92 |
| wmc | rfc | 0.62 |
| lcom3 | dam | 0.62 |
| rfc | npm | 0.58 |
| wmc | loc | 0.54 |
| wmc | cam | 0.54 |
| npm | loc | 0.50 |

## Clusters and VIF per training release

| split (training release) | n_fit | clusters (|ρ| ≥ 0.7) | VIF ≥ 5 | AutoSpearman selected |
|---|---|---|---|---|
| ta-consecutive:ant:1.3>1.4 | 100 | {wmc, rfc, npm, loc}; {dit, mfa, ic, cbm}; {max_cc, avg_cc} | wmc (43.6), rfc (30.7), loc (26.6), amc (18.2), ic (14.6), dit (14.3), cbo (13.6), npm (12.7), cbm (12.3), mfa (10.9), ca (8.9), ce (8.5), max_cc (6.7), lcom (5.6), avg_cc (5.6) | 13 |
| ta-consecutive:ant:1.4>1.5 | 140 | {wmc, rfc, npm, loc}; {dit, mfa}; {lcom3, dam}; {ic, cbm}; {max_cc, avg_cc} | wmc (44.0), loc (34.1), rfc (33.3), amc (22.2), npm (12.6), cbo (10.3), ic (9.5), dit (8.8), cbm (8.8), ca (8.3), mfa (8.2), max_cc (7.6), ce (7.1), lcom (6.2), avg_cc (6.1) | 12 |
| ta-consecutive:ant:1.5>1.6 | 232 | {wmc, rfc, npm, loc, cam}; {dit, mfa}; {cbo, ce}; {lcom3, dam}; {ic, cbm}; {max_cc, avg_cc} | loc (57.2), wmc (35.2), amc (25.4), rfc (18.1), cbm (10.3), cbo (10.3), dit (9.5), ic (9.3), npm (8.8), mfa (8.4), ce (7.9), ca (7.1), max_cc (6.8), avg_cc (5.6) | 10 |
| ta-consecutive:ant:1.6>1.7 | 277 | {wmc, rfc, npm, loc, cam}; {dit, mfa}; {cbo, ce}; {lcom3, dam}; {ic, cbm}; {max_cc, avg_cc} | loc (64.9), wmc (35.8), amc (28.0), rfc (17.4), cbm (10.2), ic (9.9), cbo (9.4), dit (8.6), npm (8.3), mfa (8.2), ce (7.3), max_cc (7.2), ca (6.5), avg_cc (5.4) | 11 |
| ta-consecutive:camel:1.2>1.4 | 446 | {wmc, rfc, npm, loc, cam}; {dit, mfa}; {cbo, ce}; {lcom3, dam}; {ic, cbm}; {max_cc, avg_cc} | loc (108.5), wmc (74.5), amc (46.8), rfc (24.2), npm (11.7), cbo (11.1), ic (8.8), cbm (8.4), ca (7.2), mfa (7.1), lcom (6.6), ce (6.4), max_cc (5.2) | 10 |
| ta-consecutive:camel:1.4>1.6 | 641 | {wmc, rfc, lcom, npm, loc, cam}; {dit, mfa}; {cbo, ce}; {lcom3, dam}; {ic, cbm}; {max_cc, avg_cc} | loc (127.4), wmc (86.0), amc (49.7), rfc (31.0), npm (11.6), cbo (10.9), ic (9.2), cbm (8.3), lcom (7.3), ca (6.9), ce (6.8), mfa (6.3), dam (5.4), max_cc (5.1) | 10 |
| ta-consecutive:ivy:1.1>1.4 | 85 | {wmc, lcom, npm}; {dit, mfa}; {rfc, ce, loc, amc}; {lcom3, dam}; {ic, cbm}; {max_cc, avg_cc} | loc (1481.7), amc (702.8), wmc (449.4), rfc (42.6), npm (23.4), ic (20.0), cbm (18.5), dit (17.7), mfa (13.6), ce (10.1), ca (9.6), max_cc (9.0), cbo (8.7), avg_cc (8.6), dam (8.5), lcom3 (6.6), lcom (5.6) | 11 |
| ta-consecutive:ivy:1.4>2.0 | 188 | {wmc, lcom, npm, cam}; {dit, mfa}; {rfc, loc, amc}; {lcom3, dam}; {ic, cbm}; {max_cc, avg_cc} | loc (309.1), amc (151.3), wmc (129.1), rfc (29.5), npm (17.1), ic (11.4), dit (10.8), cbm (10.3), mfa (8.9), lcom (8.3), avg_cc (5.9), max_cc (5.9), dam (5.0) | 12 |
| ta-consecutive:jedit:3.2>4.0 | 214 | {wmc, lcom, cam}; {dit, mfa}; {rfc, loc}; {lcom3, dam}; {ic, cbm}; {max_cc, avg_cc} | loc (490.7), amc (200.6), wmc (188.9), rfc (18.8), max_cc (14.3), cbo (12.6), ic (10.5), avg_cc (10.0), cbm (9.9), ca (9.3), ce (6.8), lcom (6.1), mfa (5.8), dit (5.3) | 12 |
| ta-consecutive:jedit:4.0>4.1 | 240 | {wmc, npm}; {dit, mfa}; {rfc, loc}; {ic, cbm}; {max_cc, avg_cc} | loc (490.9), amc (209.9), wmc (174.6), rfc (17.8), max_cc (13.8), cbo (12.1), ic (12.0), cbm (11.5), avg_cc (10.3), ca (8.5), ce (7.2), mfa (6.2), dit (5.7) | 13 |
| ta-consecutive:jedit:4.1>4.2 | 246 | {wmc, rfc, loc, cam}; {dit, mfa}; {cbo, ca}; {ic, cbm}; {max_cc, avg_cc} | loc (731.9), amc (287.6), wmc (243.5), rfc (20.0), max_cc (14.8), cbo (11.1), avg_cc (10.8), ic (8.9), cbm (8.7), ca (7.8), ce (6.6), mfa (6.6), dit (5.8), lcom (5.2) | 13 |
| ta-consecutive:lucene:2.0>2.2 | 152 | {wmc, rfc, npm, loc, cam}; {dit, mfa}; {lcom3, dam}; {ic, cbm}; {max_cc, avg_cc} | loc (556.3), amc (251.5), wmc (180.8), rfc (11.6), ic (10.4), cbm (9.5), ca (8.7), max_cc (8.3), cbo (7.9), mfa (7.7), avg_cc (7.6), dit (6.7), ce (6.2) | 12 |
| ta-consecutive:lucene:2.2>2.4 | 191 | {wmc, rfc, npm, loc, cam}; {dit, mfa}; {lcom3, dam}; {ic, cbm}; {max_cc, avg_cc} | loc (758.4), amc (331.6), wmc (243.7), rfc (13.0), mfa (9.6), cbo (9.6), ca (9.0), dit (8.2), max_cc (7.9), cbm (7.4), ic (7.3), avg_cc (6.4), ce (6.4) | 11 |
| ta-consecutive:poi:1.5>2.0 | 162 | {wmc, rfc, npm, loc}; {dit, mfa}; {lcom3, dam}; {ic, cbm}; {max_cc, avg_cc} | wmc (48.6), rfc (35.6), loc (33.8), amc (17.1), npm (17.0), mfa (13.3), dit (12.8), ic (9.9), cbm (9.7), avg_cc (9.4), max_cc (9.2), cbo (7.8), ce (6.4), ca (5.3) | 11 |
| ta-consecutive:poi:2.0>2.5 | 225 | {wmc, rfc, npm, loc}; {dit, mfa}; {ic, cbm}; {max_cc, avg_cc} | loc (56.6), wmc (47.1), rfc (27.0), amc (24.7), npm (11.9), mfa (11.4), cbm (11.3), ic (11.2), dit (10.1), cbo (9.4), max_cc (8.2), avg_cc (6.8), ca (6.7), ce (5.4) | 12 |
| ta-consecutive:poi:2.5>3.0 | 280 | {wmc, lcom, npm}; {dit, mfa}; {rfc, loc, amc}; {ic, cbm}; {max_cc, avg_cc} | loc (57.0), wmc (49.7), amc (28.6), rfc (20.0), npm (16.6), mfa (9.6), ic (9.5), cbm (9.3), cbo (9.3), dit (8.9), max_cc (6.9), ca (6.5), avg_cc (6.0), ce (5.3) | 12 |
| ta-consecutive:synapse:1.0>1.1 | 122 | {wmc, lcom, npm}; {dit, mfa}; {rfc, ce, loc}; {ic, cbm}; {amc, max_cc, avg_cc} | loc (125.6), wmc (85.7), amc (79.6), cbm (40.4), rfc (40.1), ic (34.6), npm (16.5), mfa (14.3), dit (12.8), max_cc (11.5), lcom (11.0), avg_cc (9.6), ca (8.8), ce (8.5), dam (7.5), lcom3 (7.2), cbo (7.0) | 11 |
| ta-consecutive:synapse:1.1>1.2 | 170 | {wmc, npm, cam}; {dit, mfa}; {rfc, ce, loc}; {lcom3, dam}; {ic, cbm}; {amc, max_cc, avg_cc} | loc (169.8), amc (128.0), wmc (103.1), rfc (31.5), mfa (15.5), npm (14.4), dit (14.2), cbm (13.4), ic (12.7), max_cc (12.5), avg_cc (9.1), ce (8.4), cbo (8.4), dam (7.2), lcom3 (7.2), ca (6.5), lcom (5.8) | 11 |
| ta-consecutive:velocity:1.4>1.5 | 143 | {wmc, npm}; {dit, mfa}; {rfc, loc, amc}; {lcom3, dam}; {ic, cbm}; {max_cc, avg_cc} | loc (27.8), wmc (24.0), rfc (19.5), ic (16.3), amc (16.2), cbm (15.7), max_cc (10.1), dit (8.9), cbo (8.6), mfa (8.2), avg_cc (8.0), ce (7.9), npm (7.8), ca (6.4) | 12 |
| ta-consecutive:velocity:1.5>1.6 | 158 | {wmc, npm}; {dit, mfa}; {rfc, loc, amc}; {lcom3, dam}; {ic, cbm}; {max_cc, avg_cc} | loc (30.8), wmc (22.3), rfc (20.1), amc (17.3), ic (11.9), cbm (10.2), max_cc (9.6), avg_cc (8.9), npm (8.9), mfa (8.8), cbo (8.1), ce (7.1), dit (6.9), ca (5.4), lcom3 (5.3) | 12 |
| ta-consecutive:xalan:2.4>2.5 | 555 | {wmc, rfc, npm, loc, cam}; {dit, mfa}; {cbo, ce}; {ic, cbm}; {max_cc, avg_cc} | loc (24.5), wmc (24.4), rfc (16.1), amc (13.1), cbo (10.7), ic (9.0), max_cc (9.0), cbm (8.0), ce (7.4), dit (7.4), mfa (7.3), npm (7.3), avg_cc (7.2), ca (6.6) | 11 |
| ta-consecutive:xalan:2.5>2.6 | 592 | {wmc, rfc, npm, loc, cam}; {dit, mfa}; {cbo, ce}; {ic, cbm}; {max_cc, avg_cc} | loc (37.6), wmc (33.8), amc (21.5), rfc (15.4), npm (11.0), cbo (10.4), max_cc (8.9), ic (8.8), cbm (8.1), ce (7.7), dit (7.6), avg_cc (7.4), mfa (7.4), ca (6.2) | 11 |
| ta-consecutive:xerces:init>1.2 | 116 | {wmc, rfc, npm, cam}; {dit, mfa}; {cbo, ca}; {loc, amc}; {ic, cbm}; {max_cc, avg_cc} | loc (53.7), wmc (43.1), amc (30.6), rfc (22.4), cbo (22.3), cbm (15.0), ic (14.2), mfa (11.7), ca (11.6), dit (10.1), ce (9.6), max_cc (8.9), npm (7.4), avg_cc (6.5) | 13 |
| ta-consecutive:xerces:1.2>1.3 | 275 | {wmc, rfc, lcom, npm}; {dit, mfa}; {cbo, ca}; {lcom3, dam}; {loc, amc}; {ic, cbm}; {max_cc, avg_cc} | loc (53.7), amc (38.0), wmc (32.5), cbo (23.1), rfc (21.0), mfa (15.2), dit (13.7), ca (12.8), ce (11.7), ic (8.7), cbm (8.7), max_cc (8.1), npm (6.1), avg_cc (5.7), lcom (5.4) | 10 |

## AutoSpearman (Jiarpakdee et al., 2018)

Correlation step: while a pair has |ρ| ≥ 0.7, drop the metric of the most correlated pair with the higher mean |ρ| to the remaining metrics; then drop the metric with the largest VIF while it is ≥ 5 (VIF on log1p metrics). Selection frequency over the 24 training releases:

| metric | kept in % of training releases |
|---|---|
| avg_cc | 100 |
| noc | 100 |
| moa | 100 |
| lcom3 | 96 |
| ca | 92 |
| lcom | 88 |
| amc | 88 |
| ce | 79 |
| cam | 75 |
| npm | 75 |
| mfa | 54 |
| ic | 54 |
| dit | 46 |
| dam | 42 |
| cbo | 33 |
| cbm | 29 |
| max_cc | 0 |
| wmc | 0 |
| rfc | 0 |
| loc | 0 |

Selected metrics per training release:

| split | selected | removed (step) |
|---|---|---|
| ta-consecutive:ant:1.3>1.4 | noc, lcom, ca, ce, npm, lcom3, dam, moa, mfa, cam, cbm, amc, avg_cc | ic (spearman), rfc (spearman), wmc (spearman), max_cc (spearman), dit (spearman), loc (vif), cbo (vif) |
| ta-consecutive:ant:1.4>1.5 | noc, lcom, ca, ce, npm, lcom3, moa, mfa, cam, ic, amc, avg_cc | cbm (spearman), rfc (spearman), wmc (spearman), max_cc (spearman), dit (spearman), dam (spearman), loc (spearman), cbo (vif) |
| ta-consecutive:ant:1.5>1.6 | dit, noc, cbo, lcom, ca, lcom3, moa, cam, amc, avg_cc | cbm (spearman), wmc (spearman), rfc (spearman), max_cc (spearman), mfa (spearman), loc (spearman), dam (spearman), npm (spearman), ce (spearman), ic (spearman) |
| ta-consecutive:ant:1.6>1.7 | noc, cbo, lcom, ca, npm, lcom3, moa, mfa, cbm, amc, avg_cc | ic (spearman), rfc (spearman), max_cc (spearman), wmc (spearman), dit (spearman), loc (spearman), dam (spearman), cam (spearman), ce (spearman) |
| ta-consecutive:camel:1.2>1.4 | dit, noc, cbo, ca, npm, lcom3, moa, ic, amc, avg_cc | cbm (spearman), rfc (spearman), mfa (spearman), wmc (spearman), max_cc (spearman), dam (spearman), ce (spearman), loc (spearman), lcom (spearman), cam (spearman) |
| ta-consecutive:camel:1.4>1.6 | dit, noc, cbo, lcom, ca, lcom3, moa, cbm, amc, avg_cc | ic (spearman), rfc (spearman), mfa (spearman), wmc (spearman), max_cc (spearman), dam (spearman), ce (spearman), npm (spearman), loc (spearman), cam (spearman) |
| ta-consecutive:ivy:1.1>1.4 | noc, cbo, lcom, ce, lcom3, moa, mfa, cam, ic, amc, avg_cc | cbm (spearman), dit (spearman), rfc (spearman), wmc (spearman), max_cc (spearman), loc (spearman), dam (spearman), ca (vif), npm (vif) |
| ta-consecutive:ivy:1.4>2.0 | noc, cbo, lcom, ca, ce, lcom3, moa, mfa, cam, ic, amc, avg_cc | cbm (spearman), dit (spearman), rfc (spearman), wmc (spearman), max_cc (spearman), loc (spearman), npm (spearman), dam (spearman) |
| ta-consecutive:jedit:3.2>4.0 | dit, noc, lcom, ca, ce, npm, lcom3, moa, cam, ic, amc, avg_cc | cbm (spearman), max_cc (spearman), mfa (spearman), rfc (spearman), wmc (spearman), loc (spearman), dam (spearman), cbo (vif) |
| ta-consecutive:jedit:4.0>4.1 | dit, noc, lcom, ca, ce, npm, lcom3, dam, moa, cam, ic, amc, avg_cc | cbm (spearman), mfa (spearman), max_cc (spearman), rfc (spearman), wmc (spearman), loc (spearman), cbo (vif) |
| ta-consecutive:jedit:4.1>4.2 | noc, lcom, ca, ce, npm, lcom3, dam, moa, mfa, cam, ic, amc, avg_cc | cbm (spearman), max_cc (spearman), dit (spearman), rfc (spearman), wmc (spearman), loc (spearman), cbo (spearman) |
| ta-consecutive:lucene:2.0>2.2 | noc, cbo, lcom, ce, npm, lcom3, moa, mfa, cam, cbm, amc, avg_cc | ic (spearman), max_cc (spearman), dit (spearman), rfc (spearman), wmc (spearman), loc (spearman), dam (spearman), ca (vif) |
| ta-consecutive:lucene:2.2>2.4 | dit, noc, lcom, ca, ce, npm, lcom3, moa, ic, amc, avg_cc | cbm (spearman), mfa (spearman), rfc (spearman), max_cc (spearman), wmc (spearman), loc (spearman), cam (spearman), dam (spearman), cbo (vif) |
| ta-consecutive:poi:1.5>2.0 | dit, noc, lcom, ca, ce, npm, lcom3, moa, cam, amc, avg_cc | wmc (spearman), loc (spearman), cbm (spearman), mfa (spearman), max_cc (spearman), rfc (spearman), ic (spearman), dam (spearman), cbo (vif) |
| ta-consecutive:poi:2.0>2.5 | noc, lcom, ca, ce, lcom3, dam, moa, mfa, cam, ic, amc, avg_cc | wmc (spearman), cbm (spearman), loc (spearman), dit (spearman), max_cc (spearman), rfc (spearman), npm (spearman), cbo (vif) |
| ta-consecutive:poi:2.5>3.0 | noc, lcom, ca, ce, lcom3, dam, moa, mfa, cam, cbm, amc, avg_cc | wmc (spearman), ic (spearman), rfc (spearman), dit (spearman), max_cc (spearman), loc (spearman), npm (spearman), cbo (vif) |
| ta-consecutive:synapse:1.0>1.1 | noc, cbo, ca, npm, lcom3, dam, moa, mfa, cam, ic, avg_cc | cbm (spearman), dit (spearman), rfc (spearman), max_cc (spearman), wmc (spearman), loc (spearman), amc (spearman), ce (vif), lcom (vif) |
| ta-consecutive:synapse:1.1>1.2 | dit, noc, lcom, ca, ce, npm, lcom3, moa, cam, ic, avg_cc | cbm (spearman), mfa (spearman), rfc (spearman), max_cc (spearman), wmc (spearman), dam (spearman), amc (spearman), cbo (spearman), loc (spearman) |
| ta-consecutive:velocity:1.4>1.5 | noc, lcom, ca, ce, npm, lcom3, moa, mfa, cam, cbm, amc, avg_cc | ic (spearman), dit (spearman), wmc (spearman), max_cc (spearman), rfc (spearman), dam (spearman), loc (spearman), cbo (vif) |
| ta-consecutive:velocity:1.5>1.6 | noc, lcom, ca, ce, npm, dam, moa, mfa, cam, cbm, amc, avg_cc | ic (spearman), wmc (spearman), dit (spearman), rfc (spearman), max_cc (spearman), lcom3 (spearman), loc (spearman), cbo (vif) |
| ta-consecutive:xalan:2.4>2.5 | dit, noc, lcom, ca, ce, npm, lcom3, dam, moa, amc, avg_cc | cbm (spearman), wmc (spearman), max_cc (spearman), mfa (spearman), rfc (spearman), cbo (spearman), loc (spearman), cam (spearman), ic (spearman) |
| ta-consecutive:xalan:2.5>2.6 | dit, noc, lcom, ca, ce, npm, lcom3, dam, moa, amc, avg_cc | wmc (spearman), cbm (spearman), max_cc (spearman), mfa (spearman), rfc (spearman), loc (spearman), cbo (spearman), cam (spearman), ic (spearman) |
| ta-consecutive:xerces:init>1.2 | noc, lcom, ca, ce, npm, lcom3, dam, moa, mfa, cam, ic, amc, avg_cc | dit (spearman), cbm (spearman), loc (spearman), rfc (spearman), max_cc (spearman), wmc (spearman), cbo (spearman) |
| ta-consecutive:xerces:1.2>1.3 | dit, noc, ca, ce, npm, lcom3, moa, cam, ic, avg_cc | cbm (spearman), mfa (spearman), wmc (spearman), loc (spearman), max_cc (spearman), cbo (spearman), dam (spearman), lcom (spearman), rfc (spearman), amc (spearman) |

## Grouped attributions (stable clusters, step 4 v2 explanations)

Signed attributions are summed within each stable cluster per instance; the step 4 analysis is then run on clusters. Cluster-level importances are not comparable one-to-one with feature-level ones (a cluster collects the attribution of several metrics).

| explanation | feature-level top-5 | cluster-level top-5 |
|---|---|---|
| MLP-DeepLift | dam, ce, mfa, dit, rfc | rfc+loc (0.109), dit+mfa (0.101), ic+cbm (0.091), wmc+npm (0.084), max_cc+avg_cc (0.075) |
| MLP-DeepLiftShap | ce, rfc, dam, loc, dit | rfc+loc (0.117), dit+mfa (0.094), wmc+npm (0.091), ic+cbm (0.087), max_cc+avg_cc (0.075) |
| MLP-GradientShap | ce, rfc, dam, loc, dit | rfc+loc (0.116), dit+mfa (0.094), wmc+npm (0.090), ic+cbm (0.087), max_cc+avg_cc (0.076) |
| MLP-IntegratedGradients | dam, ce, mfa, dit, rfc | rfc+loc (0.109), dit+mfa (0.101), ic+cbm (0.091), wmc+npm (0.084), max_cc+avg_cc (0.075) |
| MLP-Saliency | rfc, ce, loc, cbo, amc | rfc+loc (0.120), wmc+npm (0.092), max_cc+avg_cc (0.080), ic+cbm (0.079), dit+mfa (0.079) |
| MLP-InputXGradient | dam, ce, mfa, dit, rfc | rfc+loc (0.106), dit+mfa (0.101), ic+cbm (0.092), wmc+npm (0.083), max_cc+avg_cc (0.076) |
| MLP-LIME | rfc, loc, ce, cam, cbo | rfc+loc (0.129), wmc+npm (0.091), dit+mfa (0.079), max_cc+avg_cc (0.076), ic+cbm (0.075) |
| MLP-KernelSHAP | ce, dam, rfc, loc, mfa | rfc+loc (0.122), dit+mfa (0.095), wmc+npm (0.086), ic+cbm (0.084), max_cc+avg_cc (0.073) |
| LR-LIME | loc, rfc, wmc, ce, cbo | rfc+loc (0.149), wmc+npm (0.095), cbo (0.074), ce (0.073), max_cc+avg_cc (0.071) |
| LR-LinearSHAP | loc, rfc, ce, dam, cbo | rfc+loc (0.128), wmc+npm (0.089), dit+mfa (0.086), ic+cbm (0.076), max_cc+avg_cc (0.074) |
| RF-LIME | loc, rfc, amc, lcom, cbo | rfc+loc (0.214), wmc+npm (0.113), max_cc+avg_cc (0.083), amc (0.073), cbo (0.068) |
| RF-TreeSHAP | loc, rfc, cam, amc, ce | rfc+loc (0.195), wmc+npm (0.108), max_cc+avg_cc (0.082), cam (0.074), amc (0.071) |
| XGB-LIME | loc, rfc, amc, ce, cam | rfc+loc (0.202), wmc+npm (0.092), max_cc+avg_cc (0.090), amc (0.078), ce (0.075) |
| XGB-TreeSHAP | loc, rfc, cam, amc, ce | rfc+loc (0.181), wmc+npm (0.093), max_cc+avg_cc (0.084), cam (0.084), amc (0.078) |

Size share: feature-level loc and size-group shares vs the share of the loc cluster (rfc+loc).

| explanation | loc share | size-group share | loc-cluster share | % loc top-1 | % loc-cluster top-1 |
|---|---|---|---|---|---|
| MLP-DeepLift | 0.055 | 0.204 | 0.109 | 5.5 | 18.4 |
| MLP-DeepLiftShap | 0.060 | 0.220 | 0.117 | 6.5 | 21.9 |
| MLP-GradientShap | 0.059 | 0.218 | 0.116 | 6.4 | 21.5 |
| MLP-IntegratedGradients | 0.055 | 0.204 | 0.109 | 5.5 | 18.5 |
| MLP-Saliency | 0.060 | 0.224 | 0.120 | 8.2 | 31.3 |
| MLP-InputXGradient | 0.054 | 0.200 | 0.106 | 5.3 | 17.3 |
| MLP-LIME | 0.064 | 0.232 | 0.129 | 10.7 | 38.2 |
| MLP-KernelSHAP | 0.061 | 0.219 | 0.122 | 7.4 | 24.1 |
| LR-LIME | 0.079 | 0.275 | 0.149 | 11.2 | 42.4 |
| LR-LinearSHAP | 0.068 | 0.241 | 0.128 | 9.2 | 26.3 |
| RF-LIME | 0.119 | 0.332 | 0.214 | 24.3 | 47.6 |
| RF-TreeSHAP | 0.106 | 0.317 | 0.195 | 18.2 | 40.1 |
| XGB-LIME | 0.110 | 0.308 | 0.202 | 18.1 | 42.1 |
| XGB-TreeSHAP | 0.100 | 0.295 | 0.181 | 13.8 | 32.6 |

Cross-model agreement (seeds 0–2, mean over instances), feature level vs cluster level:

| pair | feature FA@5 | feature FA@3 | feature Spearman | cluster FA@5 | cluster FA@3 | cluster Spearman |
|---|---|---|---|---|---|---|
| MLP-DeepLiftShap vs RF-TreeSHAP | 0.468 | 0.405 | 0.350 | 0.555 | 0.488 | 0.420 |
| MLP-DeepLiftShap vs XGB-TreeSHAP | 0.447 | 0.387 | 0.307 | 0.540 | 0.464 | 0.379 |
| MLP-DeepLiftShap vs LR-LinearSHAP | 0.592 | 0.520 | 0.558 | 0.651 | 0.583 | 0.577 |
| RF-TreeSHAP vs XGB-TreeSHAP | 0.659 | 0.628 | 0.659 | 0.711 | 0.673 | 0.689 |
| RF-TreeSHAP vs LR-LinearSHAP | 0.464 | 0.395 | 0.338 | 0.538 | 0.464 | 0.400 |
| XGB-TreeSHAP vs LR-LinearSHAP | 0.425 | 0.362 | 0.281 | 0.507 | 0.429 | 0.354 |
| MLP-LIME vs LR-LIME | 0.498 | 0.435 | 0.407 | 0.588 | 0.539 | 0.433 |
| MLP-LIME vs RF-LIME | 0.465 | 0.401 | 0.341 | 0.555 | 0.503 | 0.410 |
| MLP-LIME vs XGB-LIME | 0.464 | 0.413 | 0.327 | 0.556 | 0.488 | 0.387 |
| LR-LIME vs RF-LIME | 0.498 | 0.436 | 0.350 | 0.522 | 0.502 | 0.394 |
| LR-LIME vs XGB-LIME | 0.476 | 0.425 | 0.314 | 0.495 | 0.482 | 0.360 |
| RF-LIME vs XGB-LIME | 0.683 | 0.653 | 0.671 | 0.733 | 0.712 | 0.705 |

## AutoSpearman retraining

MLP (log_standard) and RF (standard) retrained on the AutoSpearman-selected metrics of each training release with the step 3 engine (time-aware consecutive, class_weight, seeds 0, 1, 2, 25 Optuna trials; persisted under `artifacts/models/autospearman_time_aware/`). The full-feature models are the `main_time_aware` models with the same seeds (tuned with 40 trials, so the comparison also differs in tuning budget). Per-split means over seeds; Wilcoxon signed-rank, Holm within model, Cliff's delta (AutoSpearman vs all metrics).

| model | metric | a | b | n | mean_a | mean_b | p_value | p_holm | cliffs_delta | magnitude |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP | AUC | AutoSpearman | all 20 metrics | 24 | 0.680 | 0.682 | 0.927 | 1.000 | -0.002 | negligible |
| MLP | MCC@tuned | AutoSpearman | all 20 metrics | 24 | 0.228 | 0.227 | 0.855 | 1.000 | -0.017 | negligible |
| MLP | Popt | AutoSpearman | all 20 metrics | 24 | 0.768 | 0.768 | 0.944 | 1.000 | -0.010 | negligible |
| RF | AUC | AutoSpearman | all 20 metrics | 24 | 0.678 | 0.678 | 0.584 | 0.584 | 0.010 | negligible |
| RF | MCC@tuned | AutoSpearman | all 20 metrics | 24 | 0.224 | 0.251 | 0.016 | 0.049 | -0.163 | small |
| RF | Popt | AutoSpearman | all 20 metrics | 24 | 0.755 | 0.740 | 0.290 | 0.579 | 0.062 | negligible |

Explanations (MLP-DeepLiftShap, RF-TreeSHAP; seeds 0–2). Metrics removed by AutoSpearman get attribution 0; shares are relative to the metrics the model uses.

| explanation | features | top-5 (ALL) | loc share | size-group share | % loc top-1 |
|---|---|---|---|---|---|
| MLP-DeepLiftShap | AutoSpearman | amc (0.096), lcom (0.095), ce (0.084), ca (0.078), avg_cc (0.074) | 0.000 | 0.068 | 0.0 |
| MLP-DeepLiftShap | all 20 metrics | ce (0.064), rfc (0.063), dam (0.063), loc (0.061), mfa (0.056) | 0.061 | 0.220 | 7.1 |
| RF-TreeSHAP | AutoSpearman | amc (0.125), lcom (0.108), ce (0.105), avg_cc (0.099), cam (0.096) | 0.000 | 0.078 | 0.0 |
| RF-TreeSHAP | all 20 metrics | loc (0.107), rfc (0.096), cam (0.072), amc (0.069), cbo (0.066) | 0.107 | 0.316 | 18.2 |
