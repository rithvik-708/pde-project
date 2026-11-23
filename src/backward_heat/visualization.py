"""Publication-quality Matplotlib figures (no mathematics here, only plotting).

All figures share one visual identity (``apply_style``): a calm off-white canvas,
a diverging colormap for signed temperature fields, a perceptually uniform one
for error magnitudes, and the same categorical colours for the four cut-off rules
in every figure.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

if os.environ.get("BACKWARD_HEAT_INTERACTIVE") != "1":
    matplotlib.use("Agg")  # headless by default; demo.py opts into an interactive backend

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

FloatArray = NDArray[np.float64]

# ----------------------------------------------------------------------------- palette
INK = "#1f2430"
MUTED = "#6b7280"
GRID = "#e5e7eb"
CANVAS = "#fbfaf7"
FIELD_CMAP = "RdBu_r"  # signed temperature
ERROR_CMAP = "magma_r"  # |error|, light = small
EXACT_COLOR = "#111827"
RULE_COLORS = {"M1": "#2563eb", "M2": "#16a34a", "M3": "#ea580c", "M4": "#9333ea"}
#: distinct dash patterns so that coinciding curves (same kept modes) stay visible
RULE_STYLES = {"M1": dict(ls="-", lw=3.4, alpha=0.55), "M2": dict(ls="--", lw=2.2), "M3": dict(ls=":", lw=2.6), "M4": dict(ls="-", lw=2.0)}
NOISE_COLORS = ["#0ea5e9", "#f59e0b", "#e11d48", "#10b981", "#6366f1"]
RULE_LABELS = {
    "M1": r"$M^1_\varepsilon$ (Remark 1)",
    "M2": r"$M^2_\varepsilon$ (Thm 5)",
    "M3": r"$M^3_\varepsilon$ (Thm 6)",
    "M4": r"$M^4_\varepsilon$ (Thm 7)",
}


def apply_style() -> None:
    """Install the project-wide rcParams."""
    plt.rcParams.update(
        {
            "figure.facecolor": CANVAS,
            "axes.facecolor": CANVAS,
            "savefig.facecolor": CANVAS,
            "figure.dpi": 110,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "mathtext.fontset": "cm",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlepad": 9,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "lines.linewidth": 2.0,
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
        }
    )


apply_style()


def save_figure(fig: Figure, path_stem: Path, pdf: bool = True) -> list[Path]:
    """Save as high-resolution PNG (and PDF); returns the written paths."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    written = [path_stem.with_suffix(".png")]
    fig.savefig(written[0])
    if pdf:
        written.append(path_stem.with_suffix(".pdf"))
        fig.savefig(written[1])
    plt.close(fig)
    return written


def _subtitle(fig: Figure, text: str) -> None:
    fig.text(0.5, 0.955, text, ha="center", va="top", color=MUTED, fontsize=10)


# ----------------------------------------------------------------------------- 2-D fields
def field_panels(
    x: FloatArray,
    fields: Sequence[FloatArray],
    titles: Sequence[str],
    suptitle: str,
    subtitle: str = "",
    shared_scale: bool = True,
    error_last: bool = False,
) -> Figure:
    """Row of heat-maps with contour lines; optional last panel = error magnitude."""
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.25), constrained_layout=True)
    axes = np.atleast_1d(axes)
    signed = fields[:-1] if error_last else fields
    vmax = max(float(np.max(np.abs(f))) for f in signed) or 1.0
    for idx, (ax, F, title) in enumerate(zip(axes, fields, titles)):
        ax.grid(False)
        is_err = error_last and idx == n - 1
        if is_err:
            im = ax.pcolormesh(x, x, F.T, cmap=ERROR_CMAP, shading="gouraud", rasterized=True)
        else:
            lim = vmax if shared_scale else float(np.max(np.abs(F))) or 1.0
            im = ax.pcolormesh(
                x, x, F.T, cmap=FIELD_CMAP, norm=mcolors.TwoSlopeNorm(0.0, -lim, lim),
                shading="gouraud", rasterized=True,
            )
            ax.contour(x, x, F.T, levels=9, colors=INK, linewidths=0.45, alpha=0.45)
        ax.set_title(title)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_aspect("equal")
        ax.set_xticks([0, np.pi / 2, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
        ax.set_yticks([0, np.pi / 2, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8.5, colors=MUTED)
        if is_err:
            cb.formatter.set_powerlimits((-2, 2))
            cb.update_ticks()
    fig.suptitle(suptitle)
    if subtitle:
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))
        _subtitle(fig, subtitle)
    return fig


def surface_panels(
    x: FloatArray,
    fields: Sequence[FloatArray],
    titles: Sequence[str],
    suptitle: str,
    subtitle: str = "",
) -> Figure:
    """3-D surfaces (counterpart of the paper's Figure 1), common z-range and colours."""
    n = len(fields)
    fig = plt.figure(figsize=(5.0 * n, 4.6))
    X, Y = np.meshgrid(x, x, indexing="ij")
    lim = max(float(np.max(np.abs(F))) for F in fields)
    norm = mcolors.TwoSlopeNorm(0.0, -lim, lim)
    for k, (F, title) in enumerate(zip(fields, titles)):
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        ax.plot_surface(
            X, Y, F, cmap=FIELD_CMAP, norm=norm, rstride=2, cstride=2,
            linewidth=0, antialiased=True, shade=True,
        )
        ax.set_zlim(-lim * 1.05, lim * 1.05)
        ax.set_xlabel(r"$x$", labelpad=-2)
        ax.set_ylabel(r"$y$", labelpad=-2)
        ax.set_zlabel(r"$u$", labelpad=-2)
        ax.tick_params(labelsize=8, pad=-1)
        ax.view_init(elev=26, azim=-58)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor(CANVAS)
            pane.set_edgecolor(GRID)
        ax.set_title(title, pad=0)
    fig.suptitle(suptitle, y=0.99)
    if subtitle:
        fig.text(0.5, 0.925, subtitle, ha="center", color=MUTED, fontsize=10)
    fig.subplots_adjust(left=0.0, right=0.98, wspace=0.05, top=0.86, bottom=0.04)
    return fig


# ----------------------------------------------------------------------------- 1-D curves
def section_cuts(
    x: FloatArray,
    exact_rows: Sequence[FloatArray],
    recon_rows: dict[str, Sequence[FloatArray]],
    row_labels: Sequence[str],
    suptitle: str,
    subtitle: str = "",
) -> Figure:
    """Section-cut grid (rows: y-levels, columns: cut-off rules) -- paper Figure 2."""
    rules = list(recon_rows)
    fig, axes = plt.subplots(
        len(row_labels), len(rules), figsize=(5.2 * len(rules), 2.9 * len(row_labels)),
        sharex=True, constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    stride = max(1, len(x) // 40)
    for r, label in enumerate(row_labels):
        for c, rule in enumerate(rules):
            ax = axes[r, c]
            ax.plot(x, exact_rows[r], color=EXACT_COLOR, lw=1.4, label=r"exact $u$")
            ax.plot(x[::stride], exact_rows[r][::stride], "o", ms=3.2, color=EXACT_COLOR)
            ax.plot(x, recon_rows[rule][r], color=RULE_COLORS.get(rule, "C1"), lw=2.2,
                    label=r"$u^\varepsilon$, " + RULE_LABELS.get(rule, rule))
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_title(f"{label}", fontsize=11)
            if c == 0:
                ax.set_ylabel(r"$u(x, y_0, 0)$")
            if r == len(row_labels) - 1:
                ax.set_xlabel(r"$x$")
                ax.set_xticks([0, np.pi / 2, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
            if r == 0:
                ax.legend(loc="upper right")
    fig.suptitle(suptitle)
    if subtitle:
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.95))
        _subtitle(fig, subtitle)
    return fig


def error_vs_time(
    times: Sequence[float],
    series: dict[str, Sequence[float]],
    colors: dict[str, str],
    title: str,
    published: dict[str, Sequence[float]] | None = None,
    ylabel: str = "error (paper metric)",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Error as a function of t, with optional published markers."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 4))
    for key, vals in series.items():
        ax.plot(times, vals, color=colors[key], label=RULE_LABELS.get(key, key), **RULE_STYLES.get(key, {}))
        if published and key in published:
            ax.plot(times, published[key], "x", color=colors[key], ms=7, mew=1.6)
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def log_line_plot(
    x: Sequence[float] | FloatArray,
    series: dict[str, FloatArray],
    colors: Iterable[str],
    xlabel: str,
    ylabel: str,
    title: str,
    logx: bool = True,
    logy: bool = True,
    ax: plt.Axes | None = None,
    styles: dict[str, dict] | None = None,
) -> plt.Axes:
    """Generic multi-series line plot on (semi-)log axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.3))
    for (key, vals), color in zip(series.items(), colors):
        kw = {"color": color, "label": key}
        if styles and key in styles:
            kw.update(styles[key])
        ax.plot(x, vals, **kw)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def coefficient_spectrum(
    eigenvalues: FloatArray,
    magnitudes: dict[str, FloatArray],
    colors: Sequence[str],
    cutoffs: dict[str, float] | None,
    title: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """|coefficient| versus eigenvalue lambda = m^2 + n^2 (scatter, log-scale)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.3))
    lam = eigenvalues.ravel()
    for (label, mag), color in zip(magnitudes.items(), colors):
        vals = np.abs(mag).ravel()
        keep = vals > 0
        ax.scatter(lam[keep], vals[keep], s=9, color=color, alpha=0.75, label=label, edgecolors="none")
    if cutoffs:
        for name, M in cutoffs.items():
            ax.axvline(M, color=RULE_COLORS.get(name, MUTED), ls="--", lw=1.2)
            ax.text(M, 0.985, f" {name}", transform=ax.get_xaxis_transform(), ha="right", va="top",
                    rotation=90, color=RULE_COLORS.get(name, MUTED), fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"eigenvalue $\lambda_{mn} = m^2 + n^2$")
    ax.set_ylabel("|coefficient|")
    ax.set_title(title)
    ax.legend(loc="lower left", markerscale=2)
    return ax


def add_footer(fig: Figure, text: str) -> None:
    """Small provenance note at the bottom of a figure."""
    fig.text(0.995, 0.003, text, ha="right", va="bottom", fontsize=7.5, color=MUTED)


def field_grid(
    x: FloatArray,
    rows: Sequence[Sequence[FloatArray]],
    row_labels: Sequence[str],
    col_titles: Sequence[str],
    kinds: Sequence[str],
    suptitle: str,
    subtitle: str = "",
) -> Figure:
    """Grid of heat-maps; ``kinds[c]`` is 'field' (signed, own symmetric scale) or 'error' (magnitude)."""
    n_r, n_c = len(rows), len(col_titles)
    fig, axes = plt.subplots(n_r, n_c, figsize=(3.7 * n_c, 3.35 * n_r), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for r, row in enumerate(rows):
        for c, F in enumerate(row):
            ax = axes[r, c]
            ax.grid(False)
            if kinds[c] == "error":
                im = ax.pcolormesh(x, x, F.T, cmap=ERROR_CMAP, shading="gouraud", rasterized=True)
            else:
                lim = float(np.max(np.abs(F))) or 1.0
                im = ax.pcolormesh(x, x, F.T, cmap=FIELD_CMAP, norm=mcolors.TwoSlopeNorm(0.0, -lim, lim),
                                   shading="gouraud" if kinds[c] == "field" else "nearest", rasterized=True)
            ax.set_aspect("equal")
            ax.set_xticks([0, np.pi], ["0", r"$\pi$"])
            ax.set_yticks([0, np.pi], ["0", r"$\pi$"])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(row_labels[r] + "\n" + r"$y$")
            if r == n_r - 1:
                ax.set_xlabel(r"$x$")
            cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=8, colors=MUTED)
            cb.formatter.set_powerlimits((-2, 2))
            cb.update_ticks()
    fig.suptitle(suptitle)
    if subtitle:
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.955))
        fig.text(0.5, 0.965, subtitle, ha="center", va="top", color=MUTED, fontsize=10)
    return fig
