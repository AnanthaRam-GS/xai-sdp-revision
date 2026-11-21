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


def box_panels(data: dict[str, dict[str, np.ndarray]], title: str, ylabel: str, path: Path | None = None):
    """One panel per key of ``data`` (e.g. a model); within a panel one box per inner key
    (e.g. a LIME setting), labelled from the keys."""
    panels = list(data)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels) + 0.6, 3.6), sharey=True, squeeze=False)
    for ax, panel in zip(axes[0], panels):
        names = list(data[panel])
        values = [np.asarray(data[panel][n], dtype=float) for n in names]
        values = [v[np.isfinite(v)] for v in values]
        bp = ax.boxplot(values, vert=True, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops={"color": INK, "linewidth": 1.0})  # fmt: skip
        for patch in bp["boxes"]:
            patch.set_facecolor(SERIES[0])
            patch.set_alpha(0.75)
            patch.set_edgecolor(SURFACE)
        ax.set_xticks(range(1, len(names) + 1), labels=names, rotation=60, ha="right")
        ax.set_title(panel, color=INK)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel(ylabel)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=9)
    _save(fig, path)
    return fig, axes[0]


def line_chart(table: pd.DataFrame, title: str, xlabel: str, ylabel: str, path: Path | None = None,
               ylim: tuple[float, float] | None = None):  # fmt: skip
    """One line per column of ``table``; x positions are ``table.index`` (labels kept)."""
    palette = SERIES + ["#1baf7a", "#eda100"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(table.index))
    for k, col in enumerate(table.columns):
        ax.plot(x, table[col].to_numpy(dtype=float), marker="o", markersize=4, linewidth=1.6,
                color=palette[k % len(palette)], label=str(col))  # fmt: skip
    ax.set_xticks(x, labels=[str(i) for i in table.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(title, loc="left", color=INK)
    _save(fig, path)
    return fig, ax


def single_bars(values: pd.Series, title: str, xlabel: str, path: Path | None = None, xlim=None):
    """Horizontal bars labelled from ``values.index`` (first item on top)."""
    labels = [str(i) for i in values.index][::-1]
    vals = [float(values.loc[i]) for i in values.index][::-1]
    fig, ax = plt.subplots(figsize=(4.8, 0.26 * len(labels) + 1.0))
    ax.barh(np.arange(len(labels)), vals, height=0.7, color=SERIES[0], edgecolor=SURFACE)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", color=INK)
    _save(fig, path)
    return fig, ax


def dendrogram_plot(Z, labels: list[str], title: str, cut: float | None = None, path: Path | None = None):
    """Dendrogram of a linkage matrix with leaves labelled from ``labels``; optional cut line."""
    from scipy.cluster.hierarchy import dendrogram

    from scipy.cluster.hierarchy import set_link_color_palette

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    set_link_color_palette(SERIES)
    dendrogram(Z, labels=list(labels), ax=ax, color_threshold=cut if cut is not None else 0,
               above_threshold_color="#898781", leaf_rotation=90, leaf_font_size=7)  # fmt: skip
    set_link_color_palette(None)
    if cut is not None:
        ax.axhline(cut, color=SERIES[1], linewidth=1.0, linestyle="--")
    ax.set_ylabel("1 − |Spearman ρ|")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, loc="left", color=INK)
    _save(fig, path)
    return fig, ax


CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def line_panels(panels: dict[str, pd.DataFrame], title: str, xlabel: str, ylabel: str, path: Path | None = None,
                dashed=("random",), ncols: int | None = None, sharey: bool = True, panel_legends: bool = False):  # fmt: skip
    """Small multiples: one panel per key; within a panel one line per column of the
    frame (x = its index), labelled from the column names. Columns in ``dashed`` are drawn
    as dashed grey reference lines. Colours follow a fixed column order across panels, or,
    with ``panel_legends``, the column order within each panel (each panel then has its
    own legend, so no colour is reused within a legend)."""
    names = list(dict.fromkeys(c for t in panels.values() for c in t.columns if c not in dashed))
    colour = {n: CATEGORICAL[i % len(CATEGORICAL)] for i, n in enumerate(names)}
    keys = list(panels)
    ncols = ncols or len(keys)
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols + 0.4, 2.8 * nrows + 0.6), sharey=sharey, squeeze=False)
    for ax, key in zip(axes.flat, keys):
        t = panels[key]
        x = np.arange(len(t.index))
        own = [c for c in t.columns if c not in dashed]
        for col in t.columns:
            ref = col in dashed
            c = "#898781" if ref else (CATEGORICAL[own.index(col) % len(CATEGORICAL)] if panel_legends else colour[col])
            ax.plot(x, t[col].to_numpy(dtype=float), linestyle="--" if ref else "-", linewidth=1.3 if ref else 1.5,
                    color=c, label=str(col), marker=None if ref else "o", markersize=2.5)  # fmt: skip
        if panel_legends:
            ax.legend(frameon=False, fontsize=5.5, loc="best", handlelength=1.6)
        step = max(1, len(x) // 6)
        ax.set_xticks(x[::step], labels=[str(i) for i in t.index[::step]])
        ax.set_title(str(key), color=INK)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=9)
    if panel_legends:
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        _save(fig, path)
        return fig, axes
    handles, labels = [], []
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.tight_layout(rect=(0, 0.08 + 0.05 * (len(labels) > 5), 1, 0.96))
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(0.5, 0.0))
    _save(fig, path)
    return fig, axes
