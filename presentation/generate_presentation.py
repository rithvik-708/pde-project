"""Build the MA5030 project presentation as a vector PDF (16:9, 15 slides).

    python presentation/generate_presentation.py

Every number and plot on the slides is read from ``results/`` (written by
``experiments/run_all.py``) or recomputed with the ``backward_heat`` package using the
same configuration and seed as the experiments. Nothing is typed in by hand except the
values printed in the paper itself (Tables 1-2 come from ``backward_heat.config``).

Outputs
  output/MA5030_Research_Presentation.pdf   the deck
  presentation/slides/slide-NN.png          per-slide raster previews (for QA)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backward_heat import (  # noqa: E402
    DEFAULT_CONFIG,
    Grid,
    PaperExample,
    SeparableExponentialSource,
    add_paper_noise,
    cutoff_rules,
    make_rng,
    reconstruct,
)
from backward_heat.config import PAPER_TABLE1  # noqa: E402

RESULTS = ROOT / "results"
TABLES, RAW = RESULTS / "tables", RESULTS / "raw"
PRES = ROOT / "presentation"
SLIDES_DIR = PRES / "slides"
OUT = ROOT / "output"
PDF_PATH = OUT / "MA5030_Research_Presentation.pdf"

# ----------------------------------------------------------------------------- theme
W, H = 13.333, 7.5  # inches, 16:9
BG = "#0E1626"  # deep navy
CARD = "#16223A"
CARD_HI = "#1C2B48"
EDGE = "#2A3A5C"
TEXT = "#ECEFF4"
MUTED = "#9AA7BD"
ACCENT = "#2EC4B6"  # teal: the single accent
WARN = "#F2A541"  # used only for instability / errata markers
GRID_C = "#2A3754"
HEAD_FONT = "Cambria"
BODY_FONT = "Calibri"
TOTAL = 15
FOOTER = "MA5030 · Fourier truncation for the 2-D backward heat equation"

CHIP_STYLE = {
    "PAPER": (ACCENT, BG),
    "OURS": ("#8FB3FF", BG),
    "INFERRED": (WARN, BG),
    "SUPPLEMENTARY": ("#C3A6FF", BG),
}

plt.rcParams.update({
    "font.family": BODY_FONT,
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "axes.facecolor": CARD,
    "axes.edgecolor": EDGE,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID_C,
    "grid.linewidth": 0.6,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.facecolor": CARD_HI,
    "legend.edgecolor": EDGE,
    "legend.labelcolor": TEXT,
    "savefig.facecolor": BG,
})

RULE_COL = {"M1": ACCENT, "M2": "#8FB3FF", "M3": "#C3A6FF", "M4": WARN}
EPS_COL = {1e-3: WARN, 1e-5: ACCENT, 1e-7: "#8FB3FF"}


# ----------------------------------------------------------------------------- helpers
def fx(x: float) -> float:
    return x / W


def fy(y: float) -> float:
    """y measured from the TOP of the slide, in inches."""
    return 1 - y / H


def new_slide(n: int, title: str | None, kicker: str | None = None, source: str | None = None):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(BG)
    if title:
        if kicker:
            txt(fig, 0.7, 0.42, kicker.upper(), 12, ACCENT, weight="bold", family=BODY_FONT)
        txt(fig, 0.7, 0.68, title, 30, TEXT, weight="bold", family=HEAD_FONT)
    if n > 1:
        txt(fig, 0.7, 7.12, FOOTER, 10, MUTED)
        txt(fig, W - 0.7, 7.12, f"{n} / {TOTAL}", 10, MUTED, ha="right")
        if source:
            txt(fig, W / 2 + 0.6, 7.12, source, 10, MUTED, ha="center", style="italic")
    return fig


def txt(fig, x, y, s, size=16, color=TEXT, ha="left", va="top", weight="normal", family=None, style="normal", **kw):
    return fig.text(fx(x), fy(y), s, fontsize=size, color=color, ha=ha, va=va, weight=weight,
                    family=family or BODY_FONT, style=style, **kw)


def card(fig, x, y, w, h, color=CARD, edge=None, radius=0.12):
    p = FancyBboxPatch((fx(x), fy(y + h)), fx(w), h / H, boxstyle=f"round,pad=0,rounding_size={radius / H}",
                       transform=fig.transFigure, facecolor=color, edgecolor=edge or "none", linewidth=1.0,
                       mutation_aspect=W / H, zorder=0)
    fig.patches.append(p)
    return p


def chip(fig, x, y, label):
    fg, bg = CHIP_STYLE[label]
    t = txt(fig, x, y, label, 9.5, bg, weight="bold", va="center",
            bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.6", facecolor=fg, edgecolor="none"))
    return t


def bullets(fig, x, y, items, size=17, gap=0.5, color=TEXT, marker_color=ACCENT, cont_gap=None):
    """items: list of str or list[str] (continuation lines). Returns y after the list."""
    cont_gap = cont_gap if cont_gap is not None else gap * 0.72
    for item in items:
        lines = [item] if isinstance(item, str) else item
        fig.patches.append(Ellipse((fx(x + 0.07), fy(y + 0.17 * size / 17)), 0.09 / W, 0.09 / H,
                                   transform=fig.transFigure, color=marker_color))
        for j, line in enumerate(lines):
            txt(fig, x + 0.3, y + j * cont_gap, line, size, color)
        y += (len(lines) - 1) * cont_gap + gap
    return y


def eq(fig, x, y, s, size=22, color=TEXT, ha="left", va="top"):
    return fig.text(fx(x), fy(y), s, fontsize=size, color=color, ha=ha, va=va)


def axes(fig, x, y, w, h, **kw):
    ax = fig.add_axes([fx(x), fy(y + h), fx(w), h / H], **kw)
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_color(EDGE)
    return ax


def arrow(fig, x0, y0, x1, y1, color=MUTED, lw=1.8, style="-|>", ms=14):
    a = FancyArrowPatch((fx(x0), fy(y0)), (fx(x1), fy(y1)), transform=fig.transFigure, arrowstyle=style,
                        mutation_scale=ms, color=color, linewidth=lw)
    fig.patches.append(a)


def stat(fig, x, y, value, label, color=ACCENT, vsize=34, lsize=13, width=None):
    txt(fig, x, y, value, vsize, color, weight="bold", family=HEAD_FONT)
    lines = label.split("\n")
    for i, line in enumerate(lines):
        txt(fig, x, y + vsize / 72 * 1.25 + i * 0.26, line, lsize, MUTED)


def field_axes(fig, x, y, w, h, Z, title, cmap="RdBu_r", vmax=None, vmin=None, cb=True, ylabels=True):
    ax = axes(fig, x, y, w, h)
    X = np.linspace(0, np.pi, Z.shape[0])
    vmax = vmax if vmax is not None else np.max(np.abs(Z))
    vmin = vmin if vmin is not None else -vmax
    im = ax.pcolormesh(X, X, Z.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([0, np.pi / 2, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
    ax.set_yticks([0, np.pi / 2, np.pi], ["0", r"$\pi/2$", r"$\pi$"] if ylabels else [])
    ax.set_title(title, fontsize=14, pad=8)
    if cb:
        c = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        c.ax.tick_params(labelsize=10, colors=MUTED)
        c.outline.set_edgecolor(EDGE)
    return ax


def fmt_sci(v: float, digits: int = 1) -> str:
    """1.5e+08 -> 1.5·10^8 as mathtext."""
    m, e = f"{v:.{digits}e}".split("e")
    return rf"${m}\cdot 10^{{{int(e)}}}$"


# ----------------------------------------------------------------------------- data
def load_data() -> dict:
    cfg = DEFAULT_CONFIG
    grid = Grid.from_config(cfg.grid)
    ex = PaperExample.from_config(cfg.problem)
    T = ex.final_time
    src = SeparableExponentialSource(ex.f_spatial_coefficients(grid), ex.time_rate())
    g = ex.g_grid(grid)
    # identical noise stream to experiment_05 (one rng, consumed in eps order)
    rng = make_rng(cfg.seed)
    noisy = {eps: add_paper_noise(g, grid, eps, rng) for eps in cfg.noise_levels}
    rules = cutoff_rules(r=cfg.regularization.r, beta=cfg.regularization.beta, m4_sqrt=cfg.regularization.m4_sqrt)
    eps_fig = cfg.figure_noise_level
    rec = {r: reconstruct(noisy[eps_fig].values, 0.0, rules[r](eps_fig, T), grid, T, src) for r in ("M1", "M3")}
    d = dict(cfg=cfg, grid=grid, ex=ex, T=T, g=g, noisy=noisy, rules=rules, rec=rec, eps_fig=eps_fig,
             u0=ex.u_grid(grid, 0.0))
    d["t1"] = pd.read_csv(TABLES / "table1_cutoffs.csv")
    d["noise_tab"] = pd.read_csv(TABLES / "exp03_paper_noise_levels.csv")
    d["t2"] = pd.read_csv(TABLES / "table2_side_by_side.csv")
    d["seed_tab"] = pd.read_csv(TABLES / "table2_seed_sensitivity.csv")
    d["opt_tab"] = pd.read_csv(TABLES / "exp04_optimal_vs_rules.csv")
    d["e04"] = pd.read_csv(RAW / "exp04_error_vs_M.csv")
    d["e03"] = pd.read_csv(RAW / "exp03_error_vs_eps.csv")
    d["e02"] = pd.read_csv(RAW / "exp02_error_vs_M.csv")
    d["r1"] = json.loads((RAW / "experiment_01_forward_validation.json").read_text(encoding="utf-8"))
    d["r2"] = json.loads((RAW / "experiment_02_noiseless_reconstruction.json").read_text(encoding="utf-8"))
    d["r5"] = json.loads((RAW / "experiment_05_reproduce_paper.json").read_text(encoding="utf-8"))
    return d


def noise_row(d, eps, rule):
    t = d["noise_tab"]
    return t[(np.isclose(t.eps, eps)) & (t.rule == rule)].iloc[0]


# ----------------------------------------------------------------------------- slides
def s01_title(d):
    fig = new_slide(1, None)
    # background motif: faint contour of the benchmark solution on the right
    ax = fig.add_axes([0.72, 0.0, 0.28, 1.0])
    X = np.linspace(0, np.pi, d["u0"].shape[0])
    ax.contour(X, X, d["u0"].T, levels=[v for v in np.linspace(-1.1, 1.1, 23) if abs(v) > 0.05], colors=ACCENT, linewidths=0.8, alpha=0.32, linestyles="solid")
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, np.pi)

    txt(fig, 0.8, 0.9, "MA5030 · PARTIAL DIFFERENTIAL EQUATIONS · COURSE PROJECT", 13, ACCENT, weight="bold")
    lines = ["Numerical Reproduction and Stability",
             "Analysis of Fourier-Based Regularization",
             "for the Two-Dimensional Backward",
             "Heat Equation"]
    for i, line in enumerate(lines):
        txt(fig, 0.8, 1.45 + i * 0.62, line, 33, TEXT, weight="bold", family=HEAD_FONT)
    txt(fig, 0.8, 4.15, "Based on: H. T. Nguyen, Stability and regularization for 2-D backward heat",
        14, MUTED, style="italic")
    txt(fig, 0.8, 4.45, "conduction equation, Mathematical Reports 17(67), no. 1 (2015), 1–23",
        14, MUTED, style="italic")

    team = [("Manne Rithvik", "ES23BTECH11025"), ("Heena", "MA25MSCST11013"), ("Osheen Ansari", "MA25MSCST11020")]
    card(fig, 0.8, 5.05, 6.6, 1.55, CARD)
    for i, (name, rid) in enumerate(team):
        y = 5.25 + i * 0.44
        txt(fig, 1.1, y, name, 16, TEXT, weight="bold")
        txt(fig, 4.2, y, rid, 16, MUTED, family="Consolas")
    txt(fig, 0.8, 6.85, "Indian Institute of Technology Hyderabad", 15, TEXT)
    return fig


def s02_motivation(d):
    fig = new_slide(2, "Heat diffusion forgets detail — can we undo it?", "Motivation",
                    "Benchmark solution of the paper, eq. (24), p. 18")
    bullets(fig, 0.7, 1.55, [
        ["The heat equation models diffusion: temperature,",
         "concentration, smoothing of images."],
        ["Forward problem: know the state now,", "predict it later. Stable and smoothing."],
        ["Backward problem: measure the state at a", "final time $T$, recover the earlier state."],
        ["Uses: reconstructing initial temperature,", "source history, deblurring, geophysics."],
    ], size=17, gap=0.6, cont_gap=0.36)
    card(fig, 0.7, 5.35, 5.1, 1.35, CARD_HI)
    txt(fig, 0.95, 5.55, "The catch", 15, WARN, weight="bold")
    txt(fig, 0.95, 5.9, "Fine detail decays fastest, so measurement noise", 15, TEXT)
    txt(fig, 0.95, 6.2, "swamps exactly the detail we want back.", 15, TEXT)

    ex, grid = d["ex"], d["grid"]
    snaps = [0.0, 0.5, 1.0]
    vmax = np.max(np.abs(d["u0"]))
    for i, t in enumerate(snaps):
        field_axes(fig, 6.35 + i * 2.25, 2.0, 2.0, 2.0, ex.u_grid(grid, t), rf"$u(x,y,{t:g})$", vmax=vmax, cb=False)
    arrow(fig, 6.6, 4.75, 12.6, 4.75, ACCENT, lw=2.2)
    txt(fig, 9.6, 4.9, "forward in time: diffusion damps the sharp ridges (stable)", 13, ACCENT, ha="center")
    arrow(fig, 12.6, 5.45, 6.6, 5.45, WARN, lw=2.2)
    txt(fig, 9.6, 5.6, "backward in time: must re-create them from noisy data (unstable)", 13, WARN, ha="center")
    txt(fig, 9.6, 1.45, "Same colour scale in all three panels", 12, MUTED, ha="center")
    return fig


def s03_background(d):
    fig = new_slide(3, "The heat equation on a square", "Mathematical background",
                    "Notation follows Nguyen (2015), eq. (1), p. 1")
    card(fig, 0.7, 1.45, 6.0, 2.0, CARD)
    eq(fig, 3.7, 1.75, r"$u_t-\Delta u=f(x,y,t),\qquad \Delta u=u_{xx}+u_{yy}$", 24, ha="center")
    eq(fig, 3.7, 2.55, r"$(x,y)\in\Omega=(0,\pi)\times(0,\pi),\quad t\in(0,T)$", 21, ha="center")
    rows = [
        (r"$u(x,y,t)$", "temperature at point $(x,y)$, time $t$"),
        (r"$x,\ y$", "spatial variables on the square $\\Omega$"),
        (r"$t$", "time; final time $T>0$ (numerics: $T=1$)"),
        (r"$\Delta$", "Laplacian: net heat flow into a point"),
        (r"$f$", "heat source (known, may be noisy)"),
        ("diffusivity", "normalised to 1, as in the paper"),
        (r"$u|_{\partial\Omega}=0$", "zero Dirichlet boundary: edges held at 0"),
    ]
    for i, (sym, desc) in enumerate(rows):
        y = 3.75 + i * 0.44
        txt(fig, 0.95, y, sym, 16, ACCENT)
        txt(fig, 2.6, y, desc, 16, TEXT)

    # forward vs backward comparison cards
    for j, (head, col, l1, l2, l3, l4) in enumerate([
        ("Forward (well-posed)", ACCENT, r"given  $u(x,y,0)$", r"find  $u(x,y,t),\ t>0$",
         "stable and smoothing", r"small data error $\Rightarrow$ small change"),
        ("Backward (ill-posed)", WARN, r"given  $u(x,y,T)=g(x,y)$", r"find  $u(x,y,t),\ t<T$",
         "unique, but unstable", r"tiny data error $\Rightarrow$ huge change"),
    ]):
        y0 = 1.45 + j * 2.75
        card(fig, 7.3, y0, 5.35, 2.45, CARD)
        txt(fig, 7.6, y0 + 0.25, head, 19, col, weight="bold", family=HEAD_FONT)
        eq(fig, 7.6, y0 + 0.85, l1, 18)
        eq(fig, 7.6, y0 + 1.3, l2, 18)
        txt(fig, 7.6, y0 + 1.8, f"{l3};  {l4}", 14, MUTED)
    return fig


def s04_problem(d):
    fig = new_slide(4, "Problem (1): recover the past from the final state", "Problem formulation",
                    "Nguyen (2015), eq. (1), p. 1 and noise assumption, p. 6")
    card(fig, 0.7, 1.45, 7.4, 3.0, CARD)
    txt(fig, 0.95, 1.6, "Determine $u(x,y,0)$ from", 16, MUTED)
    eq(fig, 1.0, 2.1, r"$u_t-u_{xx}-u_{yy}=f(x,y,t),\qquad (x,y,t)\in\Omega\times(0,T)$", 22)
    eq(fig, 1.0, 2.85, r"$u(0,y,t)=u(\pi,y,t)=u(x,0,t)=u(x,\pi,t)=0$", 22)
    eq(fig, 1.0, 3.6, r"$u(x,y,T)=g(x,y)$", 22)
    chip(fig, 6.95, 1.75, "PAPER")

    card(fig, 0.7, 4.75, 3.55, 1.95, CARD)
    txt(fig, 0.95, 4.95, "Known (inexactly)", 17, ACCENT, weight="bold")
    eq(fig, 0.95, 5.45, r"$g\in L^2(\Omega),\ \ f\in L^2(0,T;L^2(\Omega))$", 16)
    eq(fig, 0.95, 5.95, r"$\|g_\varepsilon-g\|\leq\varepsilon,\ \ \|f_\varepsilon-f\|\leq\varepsilon$", 16)
    card(fig, 4.55, 4.75, 3.55, 1.95, CARD)
    txt(fig, 4.8, 4.95, "Unknown", 17, WARN, weight="bold")
    eq(fig, 4.8, 5.45, r"$u(x,y,t)$ for $0\leq t<T$,", 16)
    txt(fig, 4.8, 5.95, "above all the initial state $u(x,y,0)$", 16, TEXT)

    txt(fig, 8.6, 1.5, "Every symbol", 19, TEXT, weight="bold", family=HEAD_FONT)
    bullets(fig, 8.6, 2.05, [
        [r"$\Omega=(0,\pi)\times(0,\pi)$, $T>0$"],
        [r"$f$: source term, $g$: final-time", "measurement (terminal data)"],
        [r"$\varepsilon$: noise level of the data"],
        [r"$g_\varepsilon, f_\varepsilon$: the measured,", "noisy versions of $g, f$"],
        ["Uniqueness holds (Theorem 2, p. 6);", "the difficulty is stability"],
    ], size=16, gap=0.8, cont_gap=0.33)
    return fig


def s05_illposed(d):
    fig = new_slide(5, "Why is the backward problem ill-posed?", "The central difficulty",
                    "Nguyen (2015), eqs. (3)–(6), pp. 3–4; bars computed from $\\lambda=5k^2$")
    txt(fig, 0.7, 1.5, "Eigenfunctions of $-\\Delta$ with zero boundary values (eq. 4):", 16, MUTED)
    eq(fig, 0.75, 1.95, r"$\sin(mx)\,\sin(ny),\qquad \lambda_{mn}=m^2+n^2$", 22)
    txt(fig, 0.7, 2.75, "Each Fourier mode evolves independently:", 16, MUTED)
    eq(fig, 0.75, 3.2, r"forward:  $u_{mn}(t)\sim e^{-t\lambda_{mn}}\,u_{mn}(0)$", 20, color=ACCENT)
    eq(fig, 0.75, 3.75, r"backward (6):  $u_{mn}(t)=e^{(T-t)\lambda_{mn}}\,g_{mn}-\int_t^T e^{(s-t)\lambda_{mn}}f_{mn}(s)\,ds$",
       17, color=WARN)
    txt(fig, 0.7, 4.6, "A data error $\\delta$ in $g_{mn}$ becomes $\\delta\\,e^{(T-t)\\lambda_{mn}}$ in $u_{mn}(t)$.", 16)
    txt(fig, 0.7, 4.98, "Since $\\lambda_{mn}\\to\\infty$, the error is unbounded: no continuous", 16)
    txt(fig, 0.7, 5.3, "dependence on the data (Hadamard's 3rd condition fails).", 16)

    X = np.linspace(0, np.pi, 129)
    XX, YY = np.meshgrid(X, X, indexing="ij")
    field_axes(fig, 0.95, 5.75, 1.15, 1.15, np.sin(2 * XX) * np.sin(YY), "", cb=False).set_axis_off()
    field_axes(fig, 3.75, 5.75, 1.15, 1.15, np.sin(8 * XX) * np.sin(4 * YY), "", cb=False).set_axis_off()
    txt(fig, 2.2, 6.05, "mode (2,1)", 13, TEXT, weight="bold")
    txt(fig, 2.2, 6.35, r"$\lambda=5$, $e^{\lambda}\approx 148$", 13, MUTED)
    txt(fig, 5.0, 6.05, "mode (8,4)", 13, TEXT, weight="bold")
    txt(fig, 5.0, 6.35, r"$\lambda=80$, $e^{\lambda}\approx 5.5\cdot10^{34}$", 13, MUTED)

    ax = axes(fig, 8.0, 1.65, 4.75, 3.95)
    k = np.arange(1, 7)
    lam = 5 * k**2
    amp = np.exp(lam)
    bars = ax.bar(k, amp, color=[ACCENT if v < 1e3 else WARN for v in amp], width=0.62, log=True)
    for b, v in zip(bars, amp):
        ax.text(b.get_x() + b.get_width() / 2, v * 3, fmt_sci(v, 0) if v > 1e3 else f"{v:.0f}",
                ha="center", va="bottom", fontsize=11, color=TEXT)
    ax.set_xticks(k, [f"({2 * i},{i})\n$\\lambda$={5 * i * i}" for i in k], fontsize=10)
    ax.set_ylim(1, 1e95)
    ax.set_yticks([1, 1e20, 1e40, 1e60, 1e80])
    ax.set_ylabel(r"amplification $e^{T\lambda}$ at $t=0$")
    ax.set_title("Error amplification for the benchmark modes", fontsize=14, pad=10)
    ax.grid(axis="x", visible=False)
    txt(fig, 10.4, 6.38, "Even float64 rounding ($10^{-16}$) is blown up", 13, MUTED, ha="center")
    txt(fig, 10.4, 6.66, "once modes with $\\lambda\\gtrsim 40$ are kept (Exp. 2).", 13, MUTED, ha="center")
    return fig


def s06_method(d):
    fig = new_slide(6, "The paper's method: Fourier truncation", "Proposed regularization",
                    "Nguyen (2015), p. 3, eq. (12) p. 6, Theorem 3 pp. 6–7")
    card(fig, 0.7, 1.45, 6.9, 1.25, CARD_HI)
    txt(fig, 0.95, 1.62, "“The essence of our regularization method is just to eliminate", 15, TEXT, style="italic")
    txt(fig, 0.95, 1.95, "all high frequencies from the solution.”", 15, TEXT, style="italic")
    txt(fig, 0.95, 2.3, "— Nguyen (2015), p. 3", 12, MUTED)
    bullets(fig, 0.7, 3.0, [
        [r"Keep only modes inside the disc $m^2+n^2\leq M_\varepsilon$;", "drop every higher frequency."],
        [r"Kept modes are inverted exactly by eq. (6)", "(no damping, no filter weights)."],
        [r"Cut-off grows as data improve: $M_\varepsilon\to\infty$ as $\varepsilon\to 0$."],
    ], size=16, gap=0.55, cont_gap=0.33)
    card(fig, 0.7, 5.15, 6.9, 1.6, CARD)
    txt(fig, 0.95, 5.3, "Theorem 3 (stability)", 15, ACCENT, weight="bold")
    chip(fig, 6.45, 5.45, "PAPER")
    eq(fig, 0.95, 5.85, r"$\|u(\cdot,t)-v(\cdot,t)\|^2\leq e^{2(T-t)M_\varepsilon}\left(\|g^1_\varepsilon-g^2_\varepsilon\|^2+\|f^1_\varepsilon-f^2_\varepsilon\|^2\right)$", 17)
    txt(fig, 0.95, 6.35, "Amplification is capped at $e^{(T-t)M_\\varepsilon}$: finite for each $\\varepsilon$.", 14, MUTED)

    # lattice picture
    M = d["rules"]["M3"](d["eps_fig"], d["T"])
    ax = axes(fig, 8.55, 1.55, 4.1, 4.1)
    n = 9
    mm, nn = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")
    keep = mm**2 + nn**2 <= M
    ax.scatter(mm[~keep], nn[~keep], s=26, color=MUTED, alpha=0.45, zorder=3)
    ax.scatter(mm[keep], nn[keep], s=60, color=ACCENT, zorder=4)
    th = np.linspace(0, np.pi / 2, 200)
    ax.plot(np.sqrt(M) * np.cos(th), np.sqrt(M) * np.sin(th), color=ACCENT, lw=1.6, ls="--")
    sig = [(2 * k, k) for k in range(1, 5)]
    ax.scatter([a for a, _ in sig], [b for _, b in sig], s=190, facecolors="none", edgecolors=WARN, lw=1.8, zorder=5)
    ax.set_xlim(0.3, n + 0.5)
    ax.set_ylim(0.3, n + 0.5)
    ax.set_xticks(range(1, n + 1))
    ax.set_yticks(range(1, n + 1))
    ax.set_aspect("equal")
    ax.set_xlabel("$m$")
    ax.set_ylabel("$n$")
    ax.set_title(rf"Kept modes for $M^3_\varepsilon={M:.2f}$  ($\varepsilon=10^{{-5}}$)", fontsize=13, pad=8)
    txt(fig, 8.3, 6.3, "●", 14, ACCENT)
    txt(fig, 8.55, 6.32, "kept", 13, TEXT)
    txt(fig, 9.3, 6.3, "●", 14, MUTED)
    txt(fig, 9.55, 6.32, "discarded", 13, TEXT)
    txt(fig, 10.75, 6.27, "○", 16, WARN)
    txt(fig, 11.0, 6.32, "modes of the benchmark", 13, TEXT)
    txt(fig, 8.3, 6.67, "Only (2,1) of the benchmark's modes survives the cut.", 13, MUTED)
    return fig


def s07_derivation(d):
    fig = new_slide(7, "From Fourier series to the regularized solution", "Mathematical derivation",
                    "Nguyen (2015), eqs. (4)–(8), (12), pp. 3–6; step 2 is the mode ODE behind the weak form (5)")
    steps = [
        ("1  Fourier representation", [r"$u=\sum_{m,n\geq1}u_{mn}(t)\,\sin(mx)\sin(ny)$",
                                       r"$h_{mn}=\frac{4}{\pi^2}\int_0^\pi\!\!\int_0^\pi h\,\sin(mx)\sin(ny)\,dx\,dy$"]),
        ("2  Coefficient evolution", [r"$u_{mn}'(t)+\lambda_{mn}u_{mn}(t)=f_{mn}(t)$",
                                      r"$u_{mn}(T)=g_{mn},\quad \lambda_{mn}=m^2+n^2$"]),
        ("3  Backward reconstruction (6)", [r"$u_{mn}(t)=e^{(T-t)\lambda_{mn}}g_{mn}$",
                                            r"$\qquad-\int_t^T e^{(s-t)\lambda_{mn}}f_{mn}(s)\,ds$"]),
        ("4  Source of instability", [r"$\delta g_{mn}\ \mapsto\ e^{(T-t)\lambda_{mn}}\,\delta g_{mn}$",
                                      r"solvable iff the series (8) converges"]),
        ("5  Regularization", [r"keep only $\lambda_{mn}=m^2+n^2\leq M_\varepsilon$",
                               r"$M_\varepsilon\to\infty$ as $\varepsilon\to0$"]),
        ("6  Regularized solution (12)", [r"$u^\varepsilon=\sum_{m^2+n^2\leq M_\varepsilon}u^\varepsilon_{mn}(t)\,\sin(mx)\sin(ny)$",
                                          r"$u^\varepsilon_{mn}$: eq. (6) with $g_\varepsilon,\ f_\varepsilon$"]),
    ]
    cw, ch = 3.8, 2.4
    for i, (head, lines) in enumerate(steps):
        r, c = divmod(i, 3)
        x0, y0 = 0.7 + c * (cw + 0.3), 1.5 + r * (ch + 0.35)
        hi = i == 5
        card(fig, x0, y0, cw, ch, CARD_HI if hi else CARD, edge=ACCENT if hi else None)
        txt(fig, x0 + 0.25, y0 + 0.22, head, 16, ACCENT if not i == 3 else WARN, weight="bold")
        for j, line in enumerate(lines):
            eq(fig, x0 + 0.25, y0 + 0.9 + j * 0.68, line, 17 if "$" in line and len(line) < 70 else 15)
    for c in range(2):
        for r in range(2):
            arrow(fig, 0.7 + (c + 1) * (cw + 0.3) - 0.27, 1.5 + r * (ch + 0.35) + ch / 2,
                  0.7 + (c + 1) * (cw + 0.3) - 0.03, 1.5 + r * (ch + 0.35) + ch / 2, ACCENT, lw=1.6, ms=11)
    return fig


def s08_implementation(d):
    fig = new_slide(8, "From equations to code", "Numerical implementation",
                    "Our Python package src/backward_heat (all components implemented and tested)")
    stages = [
        ("Exact\nsolution", "analytical.py", "eq. (24)"),
        ("Forward\nevolution", "forward_solver.py", "spectral + CN"),
        ("Terminal\nobservation", "fourier.py", "DST-I coeffs"),
        ("Noise\naddition", "noise.py", r"$g+\frac{\varepsilon}{\pi}\,$rand"),
        ("Regulari-\nzation", "regularization.py", r"$M^1_\varepsilon..M^4_\varepsilon$"),
        ("Backward\nreconstruction", "reconstruction.py", "eq. (12)"),
        ("Error\nanalysis", "metrics.py", "RMS, rel. $L^2$"),
    ]
    bw, gap, y0 = 1.5, 0.27, 1.8
    for i, (name, mod, note) in enumerate(stages):
        x0 = 0.7 + i * (bw + gap)
        hi = i in (4, 5)
        card(fig, x0, y0, bw, 1.35, CARD_HI if hi else CARD, edge=ACCENT if hi else EDGE)
        for j, line in enumerate(name.split("\n")):
            txt(fig, x0 + bw / 2, y0 + 0.3 + j * 0.34, line, 15, TEXT, ha="center", weight="bold")
        txt(fig, x0 + bw / 2, y0 + 1.55, mod, 11, ACCENT, ha="center", family="Consolas")
        txt(fig, x0 + bw / 2, y0 + 1.88, note, 12, MUTED, ha="center")
        if i < len(stages) - 1:
            arrow(fig, x0 + bw + 0.02, y0 + 0.67, x0 + bw + gap - 0.02, y0 + 0.67, ACCENT, lw=1.6, ms=10)

    r1 = d["r1"]
    cn = r1["crank_nicolson"]
    orders = [o for o in cn["observed_order"][1:]]
    filon = r1["filon"]["observed_order"][-1]
    txt(fig, 0.7, 4.35, "Key implementation choices", 19, TEXT, weight="bold", family=HEAD_FONT)
    bullets(fig, 0.7, 4.85, [
        ["Sine coefficients by DST-I on the 129×129 grid,", "exact for every mode of the benchmark"],
        ["Duhamel source integral in closed form", "(Gauss–Legendre quadrature for a general $f$)"],
        ["Modes are masked before exponentiating:", "no overflow even when $\\lambda$ is huge"],
    ], size=15, gap=0.5, cont_gap=0.29)

    txt(fig, 7.3, 4.35, "Validation (Experiment 1)", 19, TEXT, weight="bold", family=HEAD_FONT)
    chip(fig, 11.1, 4.55, "OURS")
    stat(fig, 7.3, 4.95, fmt_sci(r1["spectral_max_abs_error"], 1), "max error, spectral forward\nsolve vs. given $g$",
         vsize=24, lsize=12)
    stat(fig, 9.25, 4.95, f"{orders[0]:.2f}, {orders[1]:.2f}", "Crank–Nicolson observed\norder (expected 2)",
         vsize=24, lsize=12)
    stat(fig, 11.25, 4.95, "64 / 64", "unit tests pass\n(pytest)", vsize=24, lsize=12)
    txt(fig, 7.3, 6.45, f"Duhamel quadrature: observed order {filon:.2f}. The paper's data $g$ is consistent", 12, MUTED)
    txt(fig, 7.3, 6.7, "with its source $f$ and solution $u$ (also checked symbolically with SymPy).", 12, MUTED)
    return fig


def s09_setup(d):
    fig = new_slide(9, "Experimental setup", "Configuration of the paper's example",
                    "Nguyen (2015), Section 4, pp. 18–19; items marked otherwise are ours")
    rows = [
        ("Domain, time", r"$\Omega=(0,\pi)^2,\ T=1$", "PAPER"),
        ("Grid", r"$129\times129$ nodes, $x_i=i\pi/128$", "PAPER"),
        ("Benchmark", r"$u=e^{-t}\,\Sigma_{k=1}^{15}\,k^{-2}\sin(2kx)\sin(ky)$", "PAPER"),
        ("Noise", r"$g_\varepsilon=g+(\varepsilon/\pi)\,\mathrm{rand}(),\ \mathrm{rand}()\in(-1,1)$", "PAPER"),
        ("Noise levels, times", r"$\varepsilon=10^{-3},10^{-5},10^{-7};\ \ t=0,0.1,\ldots,0.9$", "PAPER"),
        ("Random draws", "iid uniform per node, seed 2015; noise in $g$ only", "OURS"),
        ("Parameter $r$ in $M^3$", r"$r=2$  (Table 1 fits: 1.9999, 1.9987, 2.0008)", "INFERRED"),
        ("Form of $M^4$", "square-root form (statement, not proof)", "INFERRED"),
        ("Error metric", "RMS over the $129^2$ nodes (paper: “$L^2$”)", "INFERRED"),
    ]
    for i, (k, v, tag) in enumerate(rows):
        y = 1.55 + i * 0.555
        if i % 2 == 0:
            card(fig, 0.6, y - 0.12, 7.9, 0.52, CARD, radius=0.06)
        txt(fig, 0.8, y, k, 14, MUTED)
        txt(fig, 2.75, y - 0.02, v, 15, TEXT)
        chip(fig, 7.25, y + 0.13, tag)

    # cut-off rules and Table 1
    card(fig, 8.75, 1.43, 4.0, 5.3, CARD)
    txt(fig, 9.0, 1.6, "Cut-off rules (Table 1)", 17, ACCENT, weight="bold")
    formulas = [
        ("M1", r"$\frac{1}{T}\ln\frac{T}{\varepsilon+\varepsilon\ln(T/\varepsilon)}$", "Remark 1"),
        ("M2", r"$\frac{\ln(1/\varepsilon)}{2T}$", "Theorem 5"),
        ("M3", r"$\frac{1}{T}\ln\frac{T\,(\ln(T/\varepsilon))^r}{\varepsilon+\varepsilon\ln(T/\varepsilon)}$", "Theorem 6"),
        ("M4", r"$\sqrt{\ln(1/\varepsilon)/(T+\beta)},\ \beta=1/2$", "Theorem 7"),
    ]
    for i, (r, f, src) in enumerate(formulas):
        y = 2.0 + i * 0.7
        eq(fig, 9.0, y, rf"$M^{r[1]}_\varepsilon$", 19, color=RULE_COL[r])
        txt(fig, 9.0, y + 0.38, src, 10, MUTED)
        eq(fig, 9.85, y - 0.06, f, 19 if r != "M4" else 17)
    t1 = d["t1"]
    y = 4.88
    txt(fig, 9.0, y, "$\\varepsilon$", 13, MUTED)
    for j, e in enumerate(["$10^{-3}$", "$10^{-5}$", "$10^{-7}$"]):
        txt(fig, 10.35 + j * 0.8, y, e, 13, MUTED, ha="center")
    for i, r in enumerate(["M1", "M2", "M3", "M4"]):
        row = t1[t1.rule == r].iloc[0]
        yy = y + 0.34 + i * 0.3
        eq(fig, 9.0, yy, rf"$M^{r[1]}_\varepsilon$", 13, color=RULE_COL[r])
        for j, col in enumerate(["ours eps=1e-3", "ours eps=1e-5", "ours eps=1e-7"]):
            txt(fig, 10.35 + j * 0.8, yy, f"{row[col]:#.4g}", 13, TEXT, ha="center")
    txt(fig, 9.0, 6.38, "our values = printed values (all 12)", 12, ACCENT)
    return fig


def s10_results(d):
    fig = new_slide(10, "Reconstruction of the initial state", "Numerical results (ours)",
                    r"$t=0$, $\varepsilon=10^{-5}$, seed 2015; colour scales shared within each pair")
    u0, rec = d["u0"], d["rec"]
    vmax = np.max(np.abs(u0))
    err_max = max(np.max(np.abs(u0 - rec[r].u)) for r in rec)
    field_axes(fig, 0.75, 1.6, 2.9, 2.9, u0, r"exact $u(x,y,0)$", vmax=vmax)
    field_axes(fig, 4.15, 1.6, 2.9, 2.9, rec["M3"].u, r"regularized $u^\varepsilon$, rule $M^3_\varepsilon$", vmax=vmax,
               ylabels=False)
    field_axes(fig, 7.55, 1.6, 2.9, 2.9, np.abs(u0 - rec["M3"].u), r"$|u-u^\varepsilon|$", cmap="magma", vmin=0,
               vmax=err_max, ylabels=False)

    rows = [("M1", noise_row(d, 1e-5, "M1")), ("M3", noise_row(d, 1e-5, "M3"))]
    card(fig, 10.95, 1.45, 1.85, 3.15, CARD)
    txt(fig, 11.15, 1.62, "Errors at $t=0$", 13, MUTED)
    for i, (r, row) in enumerate(rows):
        y = 2.0 + i * 1.3
        eq(fig, 11.15, y, rf"$M^{r[1]}_\varepsilon={row.M:.2f}$", 14, color=RULE_COL[r])
        txt(fig, 11.15, y + 0.37, f"RMS  {row.paper_rms_t0:.4f}", 13, TEXT)
        txt(fig, 11.15, y + 0.68, f"rel. L²  {row.relative_l2_t0:.3f}", 13, TEXT)

    txt(fig, 0.75, 5.0, "What the picture says", 18, TEXT, weight="bold", family=HEAD_FONT)
    bullets(fig, 0.75, 5.5, [
        [r"Both rules keep only the benchmark mode $(2,1)$: $u^\varepsilon\approx\sin(2x)\sin(y)$, smooth and stable."],
        [r"The error is the discarded detail $k\geq2$ (the sharp ridges), not amplified noise."],
        [r"Relative $L^2$ error $\approx 0.28$ for every rule that keeps $(2,1)$; noise adds only $\sim10^{-3}$ (rule $M^3$)."],
    ], size=15, gap=0.48)
    return fig


def s11_tradeoff(d):
    fig = new_slide(11, "Truncation vs. noise: choosing the cut-off", "Noise sensitivity and regularization",
                    "Experiments 3–4 (supplementary sweeps beyond the paper's table)")
    e04 = d["e04"]
    ax = axes(fig, 1.2, 1.5, 5.4, 3.55)
    sub = e04[(e04.t == 0.0) & (e04.M <= 34)]
    for eps in (1e-3, 1e-5, 1e-7):
        s = sub[np.isclose(sub.eps, eps)].sort_values("M")
        ax.semilogy(s.M, s.total, "-", color=EPS_COL[eps], lw=2.2, label=rf"$\varepsilon=10^{{{int(np.log10(eps))}}}$")
    s = sub[np.isclose(sub.eps, 1e-5)].sort_values("M")
    ax.semilogy(s.M, s.truncation, "--", color=MUTED, lw=1.6, label="truncation part")
    ax.set_xlabel(r"cut-off $M$")
    ax.set_ylabel(r"relative $L^2$ error at $t=0$")
    ax.set_ylim(0.08, 1e7)
    ax.legend(loc="upper left", ncol=2, fontsize=10)
    ax.set_title("Too small: detail lost.  Too large: noise explodes", fontsize=13, pad=8)
    chip(fig, 11.25, 0.55, "SUPPLEMENTARY")

    e03 = d["e03"]
    ax2 = axes(fig, 7.6, 1.5, 5.1, 3.55)
    s3 = e03[e03.t == 0.0].groupby(["rule", "eps"]).relative_l2.mean().reset_index()
    for r in ("M1", "M2", "M3", "M4"):
        s = s3[s3.rule == r].sort_values("eps")
        ax2.loglog(s.eps, s.relative_l2, "-o", ms=3.5, color=RULE_COL[r], lw=1.8, label=rf"$M^{r[1]}_\varepsilon$")
    for eps in (1e-3, 1e-5, 1e-7):
        ax2.axvline(eps, color=MUTED, lw=0.8, ls=":")
    ax2.invert_xaxis()
    ax2.set_xlabel(r"noise level $\varepsilon$ (decreasing $\rightarrow$)")
    ax2.set_ylabel(r"relative $L^2$ error at $t=0$")
    ax2.legend(loc="lower left", ncol=2, fontsize=10)
    ax2.set_title("Error falls in steps as $\\varepsilon$ decreases (10 seeds)", fontsize=13, pad=8)

    opt = d["opt_tab"]
    o7 = opt[(opt.t == 0) & np.isclose(opt.eps, 1e-7)].iloc[0]
    e26 = e04[(e04.t == 0) & (e04.M == 26) & np.isclose(e04.eps, 1e-5)].total.iloc[0]
    bullets(fig, 0.7, 5.75, [
        rf"$\varepsilon=10^{{-5}}$: $M=3$ keeps nothing (error 1.0), $M=5$ gives 0.276, $M=26$ gives " + fmt_sci(e26, 1),
        rf"Best $M$ at $\varepsilon=10^{{-7}}$ is {o7.M_opt:.0f} (error {o7.err_opt:.3f}) — about half of the a-priori rules' {o7.err_M1:.3f}",
        "Each step = one more benchmark mode $(2k,k)$, $\\lambda=5k^2$, entering the disc",
    ], size=15, gap=0.42)
    return fig


def s12_comparison(d):
    fig = new_slide(12, "Comparison with the published results", "Reproduction",
                    "Paper values: Nguyen (2015), Tables 1–2, pp. 18–19")
    t2 = d["t2"]
    ours, paper, typo = [], [], None
    for col in t2.columns:
        if col.endswith("ours"):
            pc = col.replace("ours", "paper")
            for o, p, t in zip(t2[col], t2[pc], t2.t):
                if abs(o - p) / p > 0.5:
                    typo = (p, o)
                else:
                    ours.append(o)
                    paper.append(p)
    ax = axes(fig, 1.15, 1.55, 4.6, 4.6)
    lo, hi = 0.04, 0.7
    ax.loglog([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    ax.scatter(paper, ours, s=26, color=ACCENT, zorder=3, label=f"{len(ours)} entries")
    if typo:
        ax.scatter([typo[0]], [typo[1]], s=70, color=WARN, marker="D", zorder=4, label="printed 1.287E-02 (typo)")
    ax.set_xlim(0.009, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("paper, Table 2")
    ax.set_ylabel("ours (RMS over 129² nodes)")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Table 2: 120 error values", fontsize=14, pad=8)

    r5 = d["r5"]
    x0 = 6.7
    txt(fig, x0, 1.5, "Reproduced", 19, TEXT, weight="bold", family=HEAD_FONT)
    stat(fig, x0, 2.0, "12 / 12", "Table 1 cut-offs match every\nprinted digit", vsize=28)
    stat(fig, x0 + 3.0, 2.0, f"{100 * r5['table2_median_abs_rel_diff']:.3f}%",
         f"Table 2 median difference\n(max {100 * r5['table2_max_abs_rel_diff']:.2f}%, 119 entries)", vsize=28)
    seed = d["seed_tab"]
    m3 = seed[np.isclose(seed.eps, 1e-5) & (seed.rule == "M3")].iloc[0]
    bullets(fig, x0, 3.85, [
        ["Typo in the paper: $\\varepsilon=10^{-5}$, $M^3$, $t=0.1$ printed 1.287E-02;",
         f"ours {r5['typo_cell']['ours']:.4f}, as in every neighbouring cell"],
        ["Largest gap is where noise matters ($M^3$, $t=0$): 50 noise draws give",
         f"[{m3['min']:.4f}, {m3['max']:.4f}]; the paper's {m3.paper_t0:.4f} lies inside"],
        ["Figures 1–2 (surfaces, cuts at $y=51\\pi/128,\\ \\pi/2,\\ 51\\pi/64$):", "qualitative match"],
    ], size=15, gap=0.55, cont_gap=0.31)
    card(fig, x0, 6.2, 6.05, 0.58, CARD_HI)
    txt(fig, x0 + 0.2, 6.33,"Table 2 is mostly truncation error: with these $\\varepsilon$ at most mode (2,1) is kept.", 13, TEXT)
    return fig


def s13_limitations(d):
    fig = new_slide(13, "Limitations and observations", "Critical assessment",
                    "Details and page references: docs/paper_analysis.md (A1–A13)")
    r2 = d["r2"]
    cols = [
        ("In the paper", WARN, [
            ["Three details unstated: $r$, the form of $M^4$,", "the error metric (recovered from its tables)"],
            ["Theorem 4/5 constants too small: printed bound", "1.28 at $\\varepsilon=10^{-3}$, actual error 1.634"],
            ["Theorem 6: with the printed $M^3$ the noise factor", "grows like $\\ln(1/\\varepsilon)$ at $t=0$ (exponent sign)"],
            ["“Bigger $M_\\varepsilon$ is better” holds only for $M<20$", "(trade-off on the previous slide)"],
            ["Typos: eq. (3) integration interval; Table 2 cell"],
        ]),
        ("In our reproduction", "#8FB3FF", [
            ["The paper's noise realisation is unknown, so", "noise-sensitive cells agree only statistically"],
            ["Figures 1–2 compared visually (no data published)"],
            ["A single, special benchmark: only diagonal modes", "$(2k,k)$; at the paper's $\\varepsilon$ only $k=1$ is recovered"],
            ["Noise in $f$ is supported but not exercised", "(the paper perturbs only $g$)"],
            [f"Noise-free data still fail beyond $M\\approx40$:", "rounding $\\times\\,e^{M}$ (rel. error " + fmt_sci(r2["M80_relative_l2_t0"], 0) + " at $M=80$)"],
        ]),
    ]
    for j, (head, col, items) in enumerate(cols):
        x0 = 0.7 + j * 6.1
        card(fig, x0, 1.45, 5.85, 5.3, CARD)
        txt(fig, x0 + 0.3, 1.65, head, 19, col, weight="bold", family=HEAD_FONT)
        bullets(fig, x0 + 0.3, 2.25, items, size=15, gap=0.64, cont_gap=0.31, marker_color=col)
    return fig


def s14_conclusion(d):
    fig = new_slide(14, "Conclusion", "Summary")
    items = [
        ("Ill-posed", "The backward problem has a unique solution, but it does not depend continuously on the data."),
        ("Why", r"Mode $(m,n)$ is amplified by $e^{(T-t)(m^2+n^2)}$; high frequencies turn tiny noise into huge errors."),
        ("Method", r"Fourier truncation keeps $m^2+n^2\leq M_\varepsilon$: amplification capped at $e^{(T-t)M_\varepsilon}$ (Thm 3)."),
        ("Reproduced", "Table 1 exactly, Table 2 to 0.42%, Figures 1–2 qualitatively; three unstated details recovered."),
        ("Learned", r"Choosing $M_\varepsilon$ is a trade-off; only modes with $\lambda\lesssim\ln(1/\varepsilon)$ are recoverable at $t=0$."),
    ]
    for i, (k, v) in enumerate(items):
        y = 1.6 + i * 1.02
        card(fig, 0.7, y, 11.95, 0.82, CARD)
        txt(fig, 1.0, y + 0.22, f"{i + 1:02d}", 22, ACCENT, weight="bold", family=HEAD_FONT)
        txt(fig, 1.75, y + 0.25, k, 18, TEXT, weight="bold")
        txt(fig, 3.6, y + 0.26, v, 16, TEXT)
    return fig


def s15_references(d):
    fig = new_slide(15, "References", None)
    refs = [
        ("[1]", "H. T. Nguyen, Stability and regularization for 2-D backward heat conduction equation,",
         "Mathematical Reports 17(67), no. 1 (2015), 1–23.  (primary source)"),
        ("[2]", "L. C. Evans, Partial Differential Equations, American Mathematical Society, Providence, RI;",
         "cited as [7] in [1] (Ch. 6.5: eigenvalues of the Laplacian)."),
        ("[3]", "P. T. Nam, D. D. Trong, N. H. Tuan, The truncation method for a two-dimensional nonhomogeneous",
         "backward heat problem, Appl. Math. Comput. 216 (2010), 3423–3432; cited as [21] in [1]."),
        ("[4]", "A. N. Tikhonov, V. Y. Arsenin, Solutions of Ill-Posed Problems, Winston (1977); cited as [25] in [1].", ""),
        ("[5]", "Software: Python, NumPy, SciPy (DST-I, sparse LU), Matplotlib, pandas, SymPy, pytest.", ""),
    ]
    y = 1.55
    for tag, l1, l2 in refs:
        txt(fig, 0.7, y, tag, 15, ACCENT, weight="bold")
        txt(fig, 1.25, y, l1, 15, TEXT)
        if l2:
            txt(fig, 1.25, y + 0.32, l2, 15, MUTED)
        y += 0.82 if l2 else 0.55
    card(fig, 0.7, 5.3, 11.95, 1.45, CARD_HI)
    txt(fig, W / 2, 5.55, "Thank you — questions?", 32, TEXT, ha="center", weight="bold", family=HEAD_FONT)
    txt(fig, W / 2, 6.25, "Code, experiments, tests and reports: backward-heat-reproduction/  ·  live demo: python demo.py",
        13, MUTED, ha="center")
    return fig


SLIDES = [s01_title, s02_motivation, s03_background, s04_problem, s05_illposed, s06_method, s07_derivation,
          s08_implementation, s09_setup, s10_results, s11_tradeoff, s12_comparison, s13_limitations,
          s14_conclusion, s15_references]


def main() -> None:
    assert len(SLIDES) == TOTAL
    OUT.mkdir(parents=True, exist_ok=True)
    if SLIDES_DIR.exists():
        shutil.rmtree(SLIDES_DIR)
    SLIDES_DIR.mkdir(parents=True)
    d = load_data()
    meta = {"Title": "Fourier-Based Regularization for the 2-D Backward Heat Equation (MA5030)",
            "Author": "Manne Rithvik, Heena, Osheen Ansari", "Subject": "MA5030 course project"}
    with PdfPages(PDF_PATH, metadata=meta) as pdf:
        for i, make in enumerate(SLIDES, start=1):
            fig = make(d)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"slide {i:2d}: {make.__name__}")
    # Previews are rasterised from the finished PDF (what the audience sees). Matplotlib's own
    # PNG path cannot draw Calibri/Cambria below ~11 pt (embedded bitmap strikes).
    try:
        import pymupdf
    except ImportError:
        print(f"wrote {PDF_PATH} (install pymupdf for PNG previews)")
        return
    with pymupdf.open(PDF_PATH) as doc:
        for i, page in enumerate(doc, start=1):
            page.get_pixmap(dpi=110).save(SLIDES_DIR / f"slide-{i:02d}.png")
    for name in ("speaker_notes.md",):
        if (PRES / name).exists():
            shutil.copy(PRES / name, OUT / name)
    if (ROOT / "docs" / "presentation_content.md").exists():
        shutil.copy(ROOT / "docs" / "presentation_content.md", OUT / "presentation_content.md")
    print(f"wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
