# Data audit

Generated 2026-11-20 16:57 UTC by `python -m scripts.prepare_data`.

Source: Jureczko CK-metric datasets (SEACRAFT / tera-PROMISE) on Zenodo. Cleaning was applied to each release separately: invalid rows dropped, then inconsistent groups removed, then duplicates removed (Sun et al., 2018).

## Data sources

| project | zenodo_record_id | doi | releases |
|---|---|---|---|
| ant | 268440 | [10.5281/zenodo.268440](https://doi.org/10.5281/zenodo.268440) | 1.7 |
| camel | 268442 | [10.5281/zenodo.268442](https://doi.org/10.5281/zenodo.268442) | 1.0, 1.2, 1.4, 1.6 |
| ivy | 322436 | [10.5281/zenodo.322436](https://doi.org/10.5281/zenodo.322436) | 1.1, 1.4, 2.0 |
| jedit | 268458 | [10.5281/zenodo.268458](https://doi.org/10.5281/zenodo.268458) | 3.2, 4.0, 4.1, 4.2, 4.3 |
| log4j | 268451 | [10.5281/zenodo.268451](https://doi.org/10.5281/zenodo.268451) | 1.0, 1.1, 1.2 |
| lucene | 322458 | [10.5281/zenodo.322458](https://doi.org/10.5281/zenodo.322458) | 2.0, 2.2, 2.4 |
| poi | 322443 | [10.5281/zenodo.322443](https://doi.org/10.5281/zenodo.322443) | 1.5, 2.0, 2.5, 3.0 |
| synapse | 322449 | [10.5281/zenodo.322449](https://doi.org/10.5281/zenodo.322449) | 1.0, 1.1, 1.2 |
| velocity | 322455 | [10.5281/zenodo.322455](https://doi.org/10.5281/zenodo.322455) | 1.4, 1.5, 1.6 |
| xalan | 268436 | [10.5281/zenodo.268436](https://doi.org/10.5281/zenodo.268436) | 2.4, 2.5, 2.6, 2.7 |
| xerces | 268474 | [10.5281/zenodo.268474](https://doi.org/10.5281/zenodo.268474) | init, 1.2, 1.3, 1.4 |

## Differences from the expected release list

- **ant**: expected but **not available on Zenodo**: 1.3, 1.4, 1.5, 1.6

## Legacy files vs Zenodo

| project | version | status | legacy_rows | zenodo_rows | byte_identical | identical_ignoring_line_endings | differing_rows |
|---|---|---|---|---|---|---|---|
| ant | 1.3 | no Zenodo counterpart |  |  |  |  |  |
| ant | 1.4 | no Zenodo counterpart |  |  |  |  |  |
| ant | 1.5 | no Zenodo counterpart |  |  |  |  |  |
| ant | 1.6 | no Zenodo counterpart |  |  |  |  |  |
| ant | 1.7 | identical | 745 | 745 | False | True | 0 |

## Per-release summary

| project | version | raw_rows | invalid_rows | inconsistent_rows_removed | duplicate_rows_removed | final_rows | defective_count | defective_ratio_pct | mean_loc | median_loc |
|---|---|---|---|---|---|---|---|---|---|---|
| ant | 1.7 | 745 | 0 | 0 | 21 | 724 | 166 | 22.93 | 287.30 | 148.00 |
| camel | 1.0 | 339 | 0 | 0 | 12 | 327 | 13 | 3.98 | 102.80 | 65.00 |
| camel | 1.2 | 608 | 0 | 21 | 29 | 558 | 205 | 36.74 | 118.00 | 66.50 |
| camel | 1.4 | 872 | 0 | 0 | 70 | 802 | 144 | 17.96 | 121.50 | 65.00 |
| camel | 1.6 | 965 | 0 | 7 | 80 | 878 | 181 | 20.62 | 127.90 | 66.00 |
| ivy | 1.1 | 111 | 0 | 2 | 2 | 107 | 62 | 57.94 | 254.80 | 114.00 |
| ivy | 1.4 | 241 | 0 | 0 | 5 | 236 | 16 | 6.78 | 250.80 | 88.50 |
| ivy | 2.0 | 352 | 0 | 0 | 7 | 345 | 40 | 11.59 | 254.20 | 88.00 |
| jedit | 3.2 | 272 | 0 | 0 | 4 | 268 | 90 | 33.58 | 480.20 | 174.00 |
| jedit | 4.0 | 306 | 0 | 2 | 3 | 301 | 74 | 24.58 | 480.40 | 173.00 |
| jedit | 4.1 | 312 | 0 | 0 | 4 | 308 | 79 | 25.65 | 496.30 | 176.50 |
| jedit | 4.2 | 367 | 0 | 0 | 4 | 363 | 48 | 13.22 | 469.50 | 206.00 |
| jedit | 4.3 | 492 | 0 | 12 | 6 | 474 | 7 | 1.48 | 426.60 | 185.50 |
| log4j | 1.0 | 135 | 0 | 0 | 0 | 135 | 34 | 25.19 | 159.60 | 105.00 |
| log4j | 1.1 | 109 | 0 | 0 | 0 | 109 | 37 | 33.94 | 182.90 | 133.00 |
| log4j | 1.2 | 205 | 0 | 0 | 3 | 202 | 187 | 92.57 | 189.00 | 125.50 |
| lucene | 2.0 | 195 | 0 | 2 | 2 | 191 | 90 | 47.12 | 263.50 | 133.00 |
| lucene | 2.2 | 247 | 0 | 4 | 4 | 239 | 139 | 58.16 | 263.30 | 130.00 |
| lucene | 2.4 | 340 | 0 | 0 | 4 | 336 | 199 | 59.23 | 305.00 | 124.00 |
| poi | 1.5 | 237 | 0 | 23 | 11 | 203 | 122 | 60.10 | 256.40 | 171.00 |
| poi | 2.0 | 314 | 0 | 2 | 30 | 282 | 35 | 12.41 | 319.80 | 158.50 |
| poi | 2.5 | 385 | 0 | 4 | 31 | 350 | 221 | 63.14 | 332.90 | 172.50 |
| poi | 3.0 | 442 | 0 | 9 | 35 | 398 | 255 | 64.07 | 316.60 | 153.00 |
| synapse | 1.0 | 157 | 0 | 0 | 4 | 153 | 16 | 10.46 | 186.70 | 142.00 |
| synapse | 1.1 | 222 | 0 | 2 | 7 | 213 | 59 | 27.70 | 197.00 | 145.00 |
| synapse | 1.2 | 256 | 0 | 0 | 11 | 245 | 86 | 35.10 | 217.40 | 145.00 |
| velocity | 1.4 | 196 | 0 | 0 | 17 | 179 | 132 | 73.74 | 280.60 | 92.00 |
| velocity | 1.5 | 214 | 0 | 0 | 16 | 198 | 133 | 67.17 | 265.20 | 92.00 |
| velocity | 1.6 | 229 | 0 | 0 | 18 | 211 | 76 | 36.02 | 267.60 | 91.00 |
| xalan | 2.4 | 723 | 0 | 0 | 29 | 694 | 110 | 15.85 | 320.40 | 136.50 |
| xalan | 2.5 | 803 | 0 | 5 | 58 | 740 | 363 | 49.05 | 343.30 | 134.50 |
| xalan | 2.6 | 885 | 0 | 43 | 118 | 724 | 322 | 44.48 | 360.20 | 148.50 |
| xalan | 2.7 | 909 | 0 | 0 | 169 | 740 | 732 | 98.92 | 368.60 | 156.00 |
| xerces | init | 162 | 0 | 5 | 11 | 146 | 65 | 44.52 | 621.00 | 97.50 |
| xerces | 1.2 | 440 | 0 | 33 | 63 | 344 | 58 | 16.86 | 459.00 | 82.00 |
| xerces | 1.3 | 453 | 0 | 2 | 89 | 362 | 68 | 18.78 | 458.50 | 81.50 |
| xerces | 1.4 | 588 | 0 | 7 | 95 | 486 | 376 | 77.37 | 287.30 | 67.00 |

Totals: 14828 raw rows; 0 invalid, 185 inconsistent and 1072 duplicate rows removed; 13571 rows remain, of which 5040 are defective.

## Overlap between consecutive releases

`shared_classes`: classes (by `class_name`) present in both releases. `unchanged_rows`: rows of release n+1 whose 20 metric values **and** label are identical to some row of release n (unchanged carry-over).

| project | release_n | release_n1 | classes_n | classes_n1 | shared_classes | shared_pct_of_n | shared_pct_of_n1 | rows_n1 | unchanged_rows | unchanged_pct_of_n1 |
|---|---|---|---|---|---|---|---|---|---|---|
| camel | 1.0 | 1.2 | 327 | 558 | 247 | 75.54 | 44.27 | 558 | 58 | 10.39 |
| camel | 1.2 | 1.4 | 558 | 802 | 508 | 91.04 | 63.34 | 802 | 123 | 15.34 |
| camel | 1.4 | 1.6 | 802 | 878 | 762 | 95.01 | 86.79 | 878 | 255 | 29.04 |
| ivy | 1.1 | 1.4 | 107 | 236 | 104 | 97.20 | 44.07 | 236 | 8 | 3.39 |
| ivy | 1.4 | 2.0 | 236 | 345 | 0 | 0.00 | 0.00 | 345 | 22 | 6.38 |
| jedit | 3.2 | 4.0 | 268 | 301 | 260 | 97.01 | 86.38 | 301 | 60 | 19.93 |
| jedit | 4.0 | 4.1 | 301 | 308 | 284 | 94.35 | 92.21 | 308 | 64 | 20.78 |
| jedit | 4.1 | 4.2 | 308 | 363 | 284 | 92.21 | 78.24 | 363 | 59 | 16.25 |
| jedit | 4.2 | 4.3 | 363 | 474 | 220 | 60.61 | 46.41 | 474 | 63 | 13.29 |
| log4j | 1.0 | 1.1 | 135 | 109 | 98 | 72.59 | 89.91 | 109 | 46 | 42.20 |
| log4j | 1.1 | 1.2 | 109 | 202 | 102 | 93.58 | 50.50 | 202 | 5 | 2.48 |
| lucene | 2.0 | 2.2 | 191 | 239 | 187 | 97.91 | 78.24 | 239 | 35 | 14.64 |
| lucene | 2.2 | 2.4 | 239 | 336 | 227 | 94.98 | 67.56 | 336 | 28 | 8.33 |
| poi | 1.5 | 2.0 | 203 | 282 | 183 | 90.15 | 64.89 | 282 | 59 | 20.92 |
| poi | 2.0 | 2.5 | 282 | 350 | 269 | 95.39 | 76.86 | 350 | 88 | 25.14 |
| poi | 2.5 | 3.0 | 350 | 398 | 334 | 95.43 | 83.92 | 398 | 52 | 13.07 |
| synapse | 1.0 | 1.1 | 153 | 213 | 147 | 96.08 | 69.01 | 213 | 24 | 11.27 |
| synapse | 1.1 | 1.2 | 213 | 245 | 207 | 97.18 | 84.49 | 245 | 52 | 21.22 |
| velocity | 1.4 | 1.5 | 179 | 198 | 136 | 75.98 | 68.69 | 198 | 5 | 2.53 |
| velocity | 1.5 | 1.6 | 198 | 211 | 183 | 92.42 | 86.73 | 211 | 41 | 19.43 |
| xalan | 2.4 | 2.5 | 694 | 740 | 644 | 92.80 | 87.03 | 740 | 127 | 17.16 |
| xalan | 2.5 | 2.6 | 740 | 724 | 685 | 92.57 | 94.61 | 724 | 250 | 34.53 |
| xalan | 2.6 | 2.7 | 724 | 740 | 678 | 93.65 | 91.62 | 740 | 61 | 8.24 |
| xerces | init | 1.2 | 146 | 344 | 98 | 67.12 | 28.49 | 344 | 12 | 3.49 |
| xerces | 1.2 | 1.3 | 344 | 362 | 330 | 95.93 | 91.16 | 362 | 204 | 56.35 |
| xerces | 1.3 | 1.4 | 362 | 486 | 218 | 60.22 | 44.86 | 486 | 83 | 17.08 |

## Flags (nothing excluded yet)

### Releases with a defective ratio < 5% or > 75%

| project | version | final_rows | defective_ratio_pct |
|---|---|---|---|
| camel | 1.0 | 327 | 3.98 |
| jedit | 4.3 | 474 | 1.48 |
| log4j | 1.2 | 202 | 92.57 |
| xalan | 2.7 | 740 | 98.92 |
| xerces | 1.4 | 486 | 77.37 |

### Projects with fewer than 3 usable releases

- **ant**: 1 usable release(s)

### Consecutive releases with no class names in common

Class tracking by `class_name` is impossible across these pairs (for example after a package rename).

| project | release_n | release_n1 |
|---|---|---|
| ivy | 1.4 | 2.0 |
