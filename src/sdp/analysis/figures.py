"""Draft figures for step 4 (PNG, 300 dpi).

Every tick label, cell and bar is placed from the *labels of the data frame passed
in* (feature names, explanation names); nothing is labelled by list position.
Colours: a single-hue (blue, light → dark) sequential ramp for magnitudes, and two
fixed categorical hues for the size-share bars.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
SERIES = ["#2a78d6", "#eb6834"]  # categorical slots 1 and 2
INK, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": "#898781", "axes.labelcolor": INK, "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED, "axes.titlesize": 9, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})  # fmt: skip


def _save(fig, path: Path | None, dpi: int = 300) -> None:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")


def heatmap(matrix: pd.DataFrame, title: str, colorbar_label: str, path: Path | None = None, fmt: str = "{:.2f}",
            vmin: float | None = None, vmax: float | None = None):  # fmt: skip
    """Annotated heatmap; rows and columns are labelled from ``matrix.index`` / ``.columns``."""
    values = matrix.to_numpy(dtype=float)
    h, w = values.shape
    fig, ax = plt.subplots(figsize=(0.55 * w + 2.2, 0.3 * h + 1.4))
    vmin = np.nanmin(values) if vmin is None else vmin
    vmax = np.nanmax(values) if vmax is None else vmax
    im = ax.imshow(values, cmap=SEQUENTIAL, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(w), labels=[str(c) for c in matrix.columns], rotation=45, ha="right")
    ax.set_yticks(range(h), labels=[str(i) for i in matrix.index])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    mid = vmin + 0.55 * (vmax - vmin)
    for i, row in enumerate(matrix.index):
        for j, col in enumerate(matrix.columns):
            v = matrix.loc[row, col]  # looked up by label, not by position
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6,
                        color="#ffffff" if v > mid else INK)  # fmt: skip
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(colorbar_label)
    cb.outline.set_visible(False)
    ax.set_title(title, loc="left", color=INK)
    _save(fig, path)
    return fig, ax


def size_share_bars(table: pd.DataFrame, title: str, path: Path | None = None):
    """Horizontal grouped bars: one group per explanation (``table.index``), one bar per
    column (e.g. loc share, size-group share), labelled from the column names."""
    labels = [str(i) for i in table.index][::-1]
    cols = list(table.columns)
    y = np.arange(len(labels))
    height = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(labels) + 1.0))
    for k, col in enumerate(cols):
        vals = [table.loc[name, col] for name in table.index][::-1]
        offs = y + (len(cols) - 1 - 2 * k) * height / 2
        ax.barh(offs, vals, height=height * 0.9, color=SERIES[k % len(SERIES)], label=str(col),
                edgecolor=SURFACE, linewidth=0.5)  # fmt: skip
    ax.set_yticks(y, labels=labels)
    ax.set_xlabel("share of normalised |attribution|")
    ax.set_xlim(0, float(np.nanmax(table.to_numpy())) * 1.15)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(title, loc="left", color=INK)
    _save(fig, path)
    return fig, ax
