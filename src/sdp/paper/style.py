"""Shared figure style for the paper (Elsevier 5p, two columns).

* Okabe–Ito colour-blind-safe palette; fixed colours per model (MLP, LR, RF, XGBoost) and
  greys for the rule-based baselines; fixed colours per explainer family.
* Serif font matching Times, 8 pt; widths 90 mm (single column) and 190 mm (double).
* Every figure is saved as a vector PDF (TrueType fonts embedded) and a 300-dpi PNG.
* Metric labels come from :data:`METRIC_LABEL` (upper case, e.g. LOC, RFC, AVG_CC).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sdp.eval.preprocess import FEATURES  # noqa: E402

MM = 1 / 25.4
SINGLE = 90 * MM
DOUBLE = 190 * MM

OKABE_ITO = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
             "blue": "#0072B2", "vermilion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}  # fmt: skip
MODEL_COLOUR = {"mlp": OKABE_ITO["blue"], "lr": OKABE_ITO["orange"], "rf": OKABE_ITO["green"], "xgb": OKABE_ITO["vermilion"],
                "manual_up": "#4D4D4D", "manual_down": "#808080", "all_defective": "#A6A6A6", "majority": "#CCCCCC"}  # fmt: skip
MODEL_NAME = {"mlp": "MLP", "lr": "LR", "rf": "RF", "xgb": "XGB", "manual_up": "ManualUp", "manual_down": "ManualDown",
              "all_defective": "All defective", "majority": "Majority"}  # fmt: skip
MODEL_ORDER = ["mlp", "lr", "rf", "xgb", "manual_up", "manual_down", "all_defective", "majority"]
# explainer families: SHAP-type, DeepLift-type, gradient-only, surrogate
METHOD_COLOUR = {"DeepLiftShap": OKABE_ITO["blue"], "KernelSHAP": OKABE_ITO["sky"], "GradientShap": "#7FB8DA",
                 "TreeSHAP": OKABE_ITO["green"], "LinearSHAP": OKABE_ITO["orange"], "DeepLift": OKABE_ITO["purple"],
                 "IntegratedGradients": "#9E5A87", "InputXGradient": "#E2A6C6", "Saliency": OKABE_ITO["yellow"],
                 "LIME": OKABE_ITO["vermilion"]}  # fmt: skip
METRIC_LABEL = {f: f.upper() for f in FEATURES}
GREY = "#999999"
LIGHT = "#E5E5E5"


def apply() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "figure.titlesize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "lines.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })  # fmt: skip


def explanation_label(label: str) -> str:
    """'MLP-DeepLiftShap' → 'MLP · DeepLiftShap' (readable in narrow figures)."""
    return label.replace("-", " · ", 1)


def save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    fig.savefig(paths[0], metadata={"CreationDate": None, "ModDate": None})  # reproducible PDF bytes
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths
