"""The generated numbers file ``paper/numbers.tex`` and its consistency check.

Every number the text is likely to quote is one ``\\newcommand`` with a letters-only name,
followed by a comment naming its source file, the row selection, the column and the
aggregation. The registry below is the single definition: :func:`write_numbers` formats
each entry from its source, and :func:`check_numbers` re-reads every source, recomputes
each value and compares it, as an exact string, with the macro in ``numbers.tex``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LEARNED = ["mlp", "lr", "rf", "xgb"]
MODELS = LEARNED + ["manual_up", "manual_down", "all_defective", "majority"]
MODEL_TAG = {"mlp": "mlp", "lr": "lr", "rf": "rf", "xgb": "xgb", "manual_up": "manualup", "manual_down": "manualdown",
             "all_defective": "alldefective", "majority": "majority"}  # fmt: skip
MODEL_UP = {"mlp": "MLP", "lr": "LR", "rf": "RF", "xgb": "XGB"}
PRIMARY = ["MLP-DeepLiftShap", "MLP-KernelSHAP", "MLP-IntegratedGradients", "MLP-LIME", "MLP-Saliency",
           "LR-LinearSHAP", "RF-TreeSHAP", "XGB-TreeSHAP"]  # fmt: skip
MLP_METHODS = ["DeepLift", "DeepLiftShap", "GradientShap", "IntegratedGradients", "Saliency", "InputXGradient", "LIME", "KernelSHAP"]


def expl_tag(label: str) -> str:
    """'RF-TreeSHAP' → 'RFtreeshap' (letters only)."""
    model, method = label.split("-", 1)
    return model + re.sub(r"[^A-Za-z]", "", method).lower()


@dataclass(frozen=True)
class Number:
    name: str
    source: str  # path relative to the repository root
    select: tuple = ()  # ((column, value), ...)
    column: str = ""
    agg: str = "first"  # first | mean | sum | count | nunique | split_mean | ratio
    fmt: str = "3"  # "3", "2", "pct2" (× 100, two decimals) or "int"
    extra: dict = field(default_factory=dict, hash=False, compare=False)


def _fmt(v: float, fmt: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a" if not (isinstance(v, float) and np.isinf(v)) else "never"
    if fmt == "int":
        return f"{int(round(v))}"
    if fmt == "pct2":
        return f"{100 * v:.2f}"
    return f"{v:.{int(fmt)}f}"


def compute(n: Number, root: Path) -> str:
    t = pd.read_csv(Path(root) / n.source)
    for col, val in n.select:
        t = t[t[col].astype(str) == str(val)]
    if t.empty:
        raise KeyError(f"{n.name}: no row in {n.source} for {dict(n.select)}")
    if n.agg == "first":
        v = pd.to_numeric(t[n.column], errors="coerce").iloc[0]
    elif n.agg == "mean":
        v = pd.to_numeric(t[n.column], errors="coerce").mean()
    elif n.agg == "sum":
        v = pd.to_numeric(t[n.column], errors="coerce").sum()
    elif n.agg == "count":
        v = float(len(t))
    elif n.agg == "nunique":
        v = float(t[n.column].nunique())
    elif n.agg == "split_mean":  # mean over splits of the mean over seeds
        v = t.groupby("split_id")[n.column].mean().mean()
    elif n.agg == "ratio":  # sum(column) / sum(extra["denominator"])
        v = t[n.column].sum() / t[n.extra["denominator"]].sum()
    else:
        raise ValueError(n.agg)
    return _fmt(float(v), n.fmt)


def registry() -> list[Number]:
    N: list[Number] = []
    ds, pairs = "reports/study_dataset.csv", "reports/study_pairs.csv"
    N += [Number("NumProjects", ds, (("included", True),), "project", "nunique", "int"),
          Number("NumReleases", ds, (("included", True),), "version", "count", "int"),
          Number("NumExcludedReleases", ds, (("included", False),), "version", "count", "int"),
          Number("NumClasses", ds, (("included", True),), "final_rows", "sum", "int"),
          Number("DefectivePct", ds, (("included", True),), "defective_count", "ratio", "pct2", {"denominator": "final_rows"}),
          Number("NumPairs", pairs, (), "project", "count", "int")]  # fmt: skip
    metric_tag = {"AUC": "AUC", "MCC@tuned": "MCC", "Popt": "Popt", "PR-AUC": "PRAUC"}
    for m in MODELS:
        for metric in ("AUC", "MCC@tuned", "Popt", "PR-AUC"):
            sel = (("model", m), ("metric", metric))
            base = f"{metric_tag[metric]}{MODEL_TAG[m]}TA"
            N += [Number(base, "reports/step3_main_time_aware.csv", sel, "mean"),
                  Number(base + "Lo", "reports/step3_main_time_aware.csv", sel, "ci_low"),
                  Number(base + "Hi", "reports/step3_main_time_aware.csv", sel, "ci_high")]  # fmt: skip
        N.append(Number(f"AUC{MODEL_TAG[m]}WR", "reports/step3_within_release.csv", (("model", m), ("metric", "AUC")), "mean"))
    for m in ("mlp", "lr"):
        for prep, tag in (("none", "None"), ("standard", "Standard"), ("log_standard", "LogStandard")):
            N.append(Number(f"AUC{m}Prep{tag}", "reports/step3_ablation_preprocessing.csv",
                            (("model", m), ("preprocessing", prep), ("metric", "AUC")), "mean"))  # fmt: skip
    for m in ("mlp", "rf"):
        for imb, tag in (("none", "None"), ("class_weight", "ClassWeight"), ("smote", "Smote")):
            N.append(Number(f"AUC{m}Imb{tag}", "reports/step3_ablation_imbalance.csv",
                            (("model", m), ("imbalance", imb), ("metric", "AUC")), "mean"))  # fmt: skip
    pre = "reports/step5_preprocessing_explanations_size_share.csv"
    for model, method, prep, tag in (("mlp", "DeepLiftShap", "none", "MLPnone"), ("mlp", "DeepLiftShap", "standard", "MLPstandard"),
                                     ("mlp", "DeepLiftShap", "log_standard", "MLPlogstandard"),
                                     ("mlp", "DeepLift_zero_baseline", "none", "MLPzerobaseline"),
                                     ("lr", "LinearSHAP", "none", "LRnone"), ("lr", "LinearSHAP", "log_standard", "LRlogstandard")):  # fmt: skip
        sel = (("model", model), ("method", method), ("preprocessing", prep), ("scope", "ALL"))
        N += [Number(f"LOCshare{tag}", pre, sel, "loc_share"), Number(f"Sizeshare{tag}", pre, sel, "size_group_share"),
              Number(f"LOCtop{tag}", pre, sel, "loc_top1_pct", fmt="2")]  # fmt: skip
    for e in PRIMARY + ["RF-LIME", "XGB-LIME", "LR-LIME"]:
        sel = (("explanation", e), ("scope", "ALL"))
        N += [Number(f"LOCshare{expl_tag(e)}", "reports/step4_size_share.csv", sel, "loc_share"),
              Number(f"Sizeshare{expl_tag(e)}", "reports/step4_size_share.csv", sel, "size_group_share"),
              Number(f"LOCtop{expl_tag(e)}", "reports/step4_size_share.csv", sel, "loc_top1_pct", fmt="2")]  # fmt: skip
    ag = "reports/step4_agreement.csv"
    for a, b, tag in (("MLP-DeepLift", "MLP-DeepLiftShap", "DLvsDLS"), ("MLP-DeepLiftShap", "MLP-KernelSHAP", "DLSvsKS"),
                      ("MLP-DeepLiftShap", "MLP-LIME", "DLSvsLIME"), ("MLP-Saliency", "MLP-LIME", "SALvsLIME"),
                      ("MLP-DeepLiftShap", "RF-TreeSHAP", "DLSvsRF"), ("MLP-DeepLiftShap", "LR-LinearSHAP", "DLSvsLR"),
                      ("RF-TreeSHAP", "XGB-TreeSHAP", "RFvsXGB"), ("RF-LIME", "XGB-LIME", "RFLIMEvsXGBLIME")):  # fmt: skip
        for metric, mtag in (("feature_agreement@5", "FAfive"), ("sign_agreement@5", "SAfive"), ("spearman", "Spearman")):
            N.append(Number(f"{mtag}{tag}", ag, (("a", a), ("b", b), ("scope", "ALL"), ("metric", metric)), "mean"))
    for e in PRIMARY:
        sel = (("explanation", e),)
        N += [Number(f"DelGain{expl_tag(e)}", "reports/step6_summary.csv", sel, "deletion gain"),
              Number(f"InsGain{expl_tag(e)}", "reports/step6_summary.csv", sel, "insertion gain"),
              Number(f"Faith{expl_tag(e)}", "reports/step6_summary.csv", sel, "faithfulness"),
              Number(f"SanityData{expl_tag(e)}", "reports/step6_sanity_checks.csv", sel, "ρ after label permutation")]  # fmt: skip
    for k in MLP_METHODS:
        N.append(Number(f"SanityModel{expl_tag('MLP-' + k)}", "reports/step6_sanity_checks.csv", (("explanation", f"MLP-{k}"),),
                        "ρ after full model randomisation"))  # fmt: skip
    cv, src = "reports/step7_cross_version.csv", "reports/step7_seed_stability_sources.csv"
    for e in PRIMARY:
        N += [Number(f"XVspearman{expl_tag(e)}", cv, (("explanation", e), ("metric", "spearman")), "value", "mean"),
              Number(f"XVdataonly{expl_tag(e)}", cv, (("explanation", e), ("metric", "spearman")), "data_only", "mean"),
              Number(f"XVrbo{expl_tag(e)}", cv, (("explanation", e), ("metric", "rbo")), "value", "mean"),
              Number(f"SeedRho{expl_tag(e)}", src, (("source", "training seed"), ("level", "global")), e),
              Number(f"ProjRho{expl_tag(e)}", src, (("source", "project"), ("level", "global")), e)]  # fmt: skip
    N.append(Number("NullSpearmanMean", "paper/data/stability_null.csv", (("level", "feature"), ("metric", "spearman")), "null_mean"))
    N.append(Number("NullSpearmanPninetyfive", "paper/data/stability_null.csv", (("level", "feature"), ("metric", "spearman")), "null_p95"))
    for m in LEARNED:
        N.append(Number(f"Multiplicity{MODEL_UP[m]}", "reports/step7_seed_stability_multiplicity.csv", (("model", m),),
                        "disagreement_share", "mean", "pct2"))  # fmt: skip
    uc = "reports/step7_unchanged_classes_counts.csv"
    N += [Number("NumUnchangedClasses", uc, (), "unchanged", "sum", "int"),
          Number("IvyMappedMatches", uc, (("project", "ivy"),), "matched_via_mapping", "sum", "int"),
          Number("IvyMappedUnchanged", uc, (("project", "ivy"),), "unchanged_via_mapping", "sum", "int")]  # fmt: skip
    for m in LEARNED:
        for mode, tag in (("val_mcc", "tuned"), ("cost_aware", "costaware")):
            N.append(Number(f"BreakEven{MODEL_UP[m]}{tag}", "reports/step8_cost_ratio_break_even.csv",
                            (("model", m), ("mode", mode)), "break_even_c", fmt="2"))  # fmt: skip
        for method, tag in (("uncalibrated", "Raw"), ("platt", "Platt"), ("isotonic", "Isotonic")):
            N.append(Number(f"ECE{m}{tag}", "reports/step8_calibration.csv", (("model", m), ("method", method)), "ece", "split_mean"))
    for step, tag in (("step3_experiments", "StepThree"), ("step4_explanations_v2", "StepFour"), ("step5_robustness", "StepFive"),
                      ("step6_fidelity", "StepSix"), ("step7_stability", "StepSeven"), ("total", "Total")):  # fmt: skip
        N.append(Number(f"Runtime{tag}Hours", "paper/data/runtime_totals.csv", (("step", step),), "wall_hours", fmt="2"))
    names = [n.name for n in N]
    assert len(names) == len(set(names)), "duplicate macro names"
    assert all(re.fullmatch(r"[A-Za-z]+", x) for x in names), "macro names must be letters only"
    return N


def write_numbers(root: Path, path: Path) -> int:
    lines = ["% Generated by sdp.paper.numbers (python -m scripts.make_paper_assets); do not edit by hand.",
             "% Each macro: value, then its source, row selection, column and aggregation.", ""]  # fmt: skip
    for n in registry():
        sel = ", ".join(f"{c}={v}" for c, v in n.select) or "all rows"
        lines.append(f"\\newcommand{{\\{n.name}}}{{{compute(n, root)}}}% {n.source}; {sel}; {n.column or '-'} ({n.agg})")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(registry())


MACRO = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}")


def check_numbers(root: Path, path: Path) -> list[str]:
    """Problems found (empty list = every macro matches its source)."""
    found = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = MACRO.match(line)
        if m:
            found[m.group(1)] = m.group(2)
    problems = []
    reg = {n.name: n for n in registry()}
    for name, n in reg.items():
        if name not in found:
            problems.append(f"{name}: missing from {path}")
            continue
        try:
            expected = compute(n, root)
        except Exception as exc:  # noqa: BLE001 - report every failure
            problems.append(f"{name}: cannot recompute ({exc})")
            continue
        if found[name] != expected:
            problems.append(f"{name}: numbers.tex has {found[name]}, source gives {expected} ({n.source})")
    for name in set(found) - set(reg):
        problems.append(f"{name}: not defined in the registry")
    return problems
