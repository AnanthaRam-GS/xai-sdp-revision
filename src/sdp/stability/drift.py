"""Data drift between releases and its relationship to stability (step 7, section 6).

For a pair of releases (raw metrics):

* the two-sample Kolmogorov–Smirnov statistic per metric (mean and maximum over the 20);
* the change in the defective ratio;
* a classifier two-sample test: a random forest separating the two releases, evaluated by
  its 5-fold cross-validated AUC (seeded; ≈ 0.5 when the releases are alike), with its
  impurity-based feature importances (fitted on all rows). The two releases are not
  independent samples: most classes appear in both, identical or slightly changed. With
  ordinary folds a class's other version sits in the training fold with the opposite
  label and the AUC drops far below 0.5. Folds are therefore grouped by class name
  (``StratifiedGroupKFold``; ivy's ``fr.jayasoft.ivy.`` prefix mapped to ``org.apache.ivy.``),
  so both versions of a class are always in the same fold; without names, identical
  metric vectors are grouped instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from sdp.eval.preprocess import FEATURES
from sdp.eval.stats import holm

DRIFT_MEASURES = ["mean_ks", "max_ks", "c2st_auc", "abs_delta_ratio"]


def ks_drift(A: pd.DataFrame, B: pd.DataFrame) -> pd.Series:
    return pd.Series({f: ks_2samp(A[f].to_numpy(float), B[f].to_numpy(float)).statistic for f in FEATURES})


def classifier_two_sample(A: pd.DataFrame, B: pd.DataFrame, seed: int = 0, n_estimators: int = 200,
                          names_a=None, names_b=None) -> tuple[float, pd.Series]:  # fmt: skip
    X = pd.concat([A[FEATURES], B[FEATURES]], ignore_index=True).to_numpy(float)
    y = np.r_[np.zeros(len(A), int), np.ones(len(B), int)]
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=1, min_samples_leaf=2)
    if names_a is not None and names_b is not None:
        keys = pd.Series(list(names_a) + list(names_b), dtype=str)
    else:
        keys = pd.Series([row.tobytes() for row in np.ascontiguousarray(X)])
    groups = pd.factorize(keys)[0]
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(rf, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    rf.fit(X, y)
    return float(roc_auc_score(y, p)), pd.Series(rf.feature_importances_, index=FEATURES)


def drift_pair(A: pd.DataFrame, B: pd.DataFrame, label_col: str = "defective", seed: int = 0,
               names_a=None, names_b=None) -> dict:  # fmt: skip
    ks = ks_drift(A, B)
    auc, imp = classifier_two_sample(A, B, seed, names_a=names_a, names_b=names_b)
    ra, rb = A[label_col].mean(), B[label_col].mean()
    return {"mean_ks": float(ks.mean()), "max_ks": float(ks.max()), "max_ks_feature": ks.idxmax(),
            "delta_ratio": float(rb - ra), "abs_delta_ratio": float(abs(rb - ra)), "c2st_auc": auc,
            "c2st_top_features": ", ".join(imp.sort_values(ascending=False).index[:3]), "ks": ks, "importance": imp}  # fmt: skip


def correlate(frame: pd.DataFrame, xs: list[str], ys: list[str], group: str | None = None) -> pd.DataFrame:
    """Spearman ρ and p for every (x, y) (within each value of ``group``); Holm across all tests."""
    rows = []
    groups = frame.groupby(group) if group else [(None, frame)]
    for key, g in groups:
        for x in xs:
            for y in ys:
                d = g[[x, y]].dropna()
                rho, p = spearmanr(d[x], d[y]) if len(d) > 2 else (np.nan, np.nan)
                rows.append(({group: key} if group else {}) | {"x": x, "y": y, "n": len(d), "rho": rho, "p_value": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"].to_numpy())
    return out
