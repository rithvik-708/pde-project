"""Experiment 5 -- reproduction of the published results (Section 4 of the paper).

* Table 1 : cut-off values M^i_eps, i = 1..4, eps = 1e-3, 1e-5, 1e-7.
* Table 2 : errors at t = 0, 0.1, ..., 0.9 (paper metric = RMS over the 129^2 nodes).
* Figure 1: 3-D surfaces of u(., ., 0) and u^eps(., ., 0) for eps = 1e-5, M1 and M3.
* Figure 2: section cuts at y = 51pi/128, pi/2, 51pi/64 (t = 0, eps = 1e-5).

Settings follow the paper exactly: T = 1, 128 x 128 intervals, g_eps = g + (eps/pi) rand(),
beta = 1/2. The paper leaves r (Theorem 6), the RNG and the metric unstated. We use
r = 2 and the RMS metric (both inferred from the paper's own numbers) and a fixed seed.
The seed sensitivity of Table 2 is quantified with 50 additional noise realisations.
"""

from __future__ import annotations

import _common as C  # noqa: I001

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backward_heat import add_paper_noise, make_rng, reconstruct
from backward_heat import visualization as viz
from backward_heat.config import PAPER_TABLE1, PAPER_TABLE2
from backward_heat.metrics import l2_error, paper_error
from backward_heat.regularization import cutoff_rules

NAME = "experiment_05_reproduce_paper"
RULES = ("M1", "M2", "M3", "M4")
N_EXTRA_SEEDS = 50
TYPO = (1e-5, "M3", 0.1)  # printed 1.287E-02 (paper_analysis.md, A6)


def eps_label(eps: float) -> str:
    return f"1e-{round(-np.log10(eps))}"


def run() -> dict:
    S = C.setup(NAME)
    cfg, grid, ex, T = S.cfg, S.grid, S.example, S.T
    rules = cutoff_rules(cfg.regularization.r, cfg.regularization.beta, cfg.regularization.m4_sqrt)
    eps_list, times = cfg.noise_levels, cfg.times
    g_clean = ex.g_grid(grid)
    exact = {t: ex.u_grid(grid, t) for t in times}

    # ------------------------------------------------------------------ Table 1
    cutoffs = {(rule, eps): rules[rule](eps, T) for rule in RULES for eps in eps_list}
    t1_rows = []
    for rule in RULES:
        row = {"rule": rule}
        for eps in eps_list:
            row[f"ours eps={eps_label(eps)}"] = round(cutoffs[(rule, eps)], 4)
            row[f"paper eps={eps_label(eps)}"] = PAPER_TABLE1[rule][eps]
        t1_rows.append(row)
    table1 = pd.DataFrame(t1_rows)
    C.save_table(table1, "table1_cutoffs", "%.4g")
    max_dev = max(abs(cutoffs[k] - PAPER_TABLE1[k[0]][k[1]]) / PAPER_TABLE1[k[0]][k[1]] for k in cutoffs)
    S.log.info("Table 1 reproduced; max relative deviation from printed values %.2e", max_dev)

    # ------------------------------------------------------------------ Table 2
    rng = make_rng(cfg.seed)
    noisy = {eps: add_paper_noise(g_clean, grid, eps, rng) for eps in eps_list}
    for eps, nd in noisy.items():
        S.log.info("eps=%g: actual ||g_eps-g||_L2 = %.3e (bound eps), max|noise| = %.3e (bound eps/pi = %.3e)",
                   eps, nd.l2_norm, nd.max_abs, eps / np.pi)

    records = []
    for eps in eps_list:
        for rule in RULES:
            M = cutoffs[(rule, eps)]
            for t in times:
                res = reconstruct(noisy[eps].values, t, M, grid, T, S.source)
                ours = paper_error(exact[t], res.u)
                pub = PAPER_TABLE2[(eps, rule)][round(t, 1)]
                records.append({
                    "eps": eps, "rule": rule, "M": M, "modes_kept": res.n_modes, "t": round(t, 1),
                    "ours": ours, "paper": pub, "rel_diff": (ours - pub) / pub,
                    "true_L2": l2_error(exact[t], res.u, grid),
                })
    long = pd.DataFrame(records)
    C.save_table(long, "table2_long")

    # Seed sensitivity: same computation for N_EXTRA_SEEDS further realisations (t = 0 suffices,
    # the t-dependence is the deterministic factor e^{-t} plus amplified noise).
    seed_stats = []
    for eps in eps_list:
        draws = {rule: [] for rule in RULES}
        for s in range(N_EXTRA_SEEDS):
            g_eps = add_paper_noise(g_clean, grid, eps, make_rng(10_000 + s)).values
            for rule in RULES:
                draws[rule].append(paper_error(exact[0.0], reconstruct(g_eps, 0.0, cutoffs[(rule, eps)], grid, T, S.source).u))
        for rule in RULES:
            d = np.array(draws[rule])
            seed_stats.append({"eps": eps, "rule": rule, "mean": d.mean(), "std": d.std(ddof=1),
                               "min": d.min(), "max": d.max(), "paper_t0": PAPER_TABLE2[(eps, rule)][0.0]})
    seeds = pd.DataFrame(seed_stats)
    C.save_table(seeds, "table2_seed_sensitivity")

    # Side-by-side wide table in the paper's layout
    wide = pd.DataFrame({"t": [round(t, 1) for t in times]})
    for eps in eps_list:
        for rule in RULES:
            sub = long[(long.eps == eps) & (long.rule == rule)].set_index("t")
            wide[f"{eps_label(eps)} {rule} ours"] = sub["ours"].values
            wide[f"{eps_label(eps)} {rule} paper"] = sub["paper"].values
    C.save_table(wide, "table2_side_by_side", "%.3e")

    comparable = long[~((long.eps == TYPO[0]) & (long.rule == TYPO[1]) & (long.t == TYPO[2]))]
    S.log.info("Table 2: max |rel diff| over %d entries (typo excluded) = %.3e; median = %.3e",
               len(comparable), comparable.rel_diff.abs().max(), comparable.rel_diff.abs().median())
    typo_row = long[(long.eps == TYPO[0]) & (long.rule == TYPO[1]) & (long.t == TYPO[2])].iloc[0]
    S.log.info("typo cell (eps=1e-5, M3, t=0.1): ours %.4e vs printed %.4e", typo_row.ours, typo_row.paper)

    # ------------------------------------------------------------------ Table 2 figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), sharey=True, constrained_layout=True)
    for ax, eps in zip(axes, eps_list):
        sub = long[long.eps == eps]
        series = {r: sub[sub.rule == r].ours.values for r in RULES}
        published = {r: sub[sub.rule == r].paper.values for r in RULES}
        viz.error_vs_time([round(t, 1) for t in times], series, viz.RULE_COLORS,
                          rf"$\varepsilon = 10^{{{round(np.log10(eps))}}}$", published, ax=ax)
        ax.set_ylim(0.009, 0.8)
        kept = {r: int(sub[sub.rule == r].modes_kept.iloc[0]) for r in RULES}
        ax.text(0.98, 0.97, "modes kept: " + ", ".join(f"{r}={k}" for r, k in kept.items()),
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=viz.MUTED)
        if eps == TYPO[0]:
            ax.annotate("printed 1.287E-02\n(typo, see A6)", xy=(TYPO[2], 0.01287), xytext=(0.3, 0.02),
                        fontsize=8.5, color=viz.RULE_COLORS["M3"],
                        arrowprops=dict(arrowstyle="->", color=viz.RULE_COLORS["M3"], lw=1))
    for ax in axes[1:]:
        ax.set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls="none", marker="x", color=viz.INK, mew=1.6))
    labels.append("published value (Table 2)")
    fig.legend(handles, labels, loc="outside lower center", ncol=5)
    fig.suptitle("Table 2 reproduced: error of the regularised solution vs. time")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.95))
    fig.text(0.5, 0.945, "lines: this code (seed 2015)  ·  crosses: values printed in the paper  ·  metric: RMS over the 129×129 nodes", ha="center", va="top", color=viz.MUTED, fontsize=10)
    viz.save_figure(fig, C.FIGURES_DIR / "table2_reproduction")

    # ------------------------------------------------------------------ Figures 1 and 2
    eps_fig = cfg.figure_noise_level
    rec = {r: reconstruct(noisy[eps_fig].values, 0.0, cutoffs[(r, eps_fig)], grid, T, S.source) for r in ("M1", "M3")}
    fig = viz.surface_panels(
        grid.x, [exact[0.0], rec["M1"].u, rec["M3"].u],
        [r"(a) exact $u(x,y,0)$",
         rf"(b) $u^\varepsilon(x,y,0)$, $M^1_\varepsilon={cutoffs[('M1', eps_fig)]:.3f}$",
         rf"(c) $u^\varepsilon(x,y,0)$, $M^3_\varepsilon={cutoffs[('M3', eps_fig)]:.2f}$"],
        "Figure 1 (reproduced): exact and regularised solutions at t = 0",
        rf"$\varepsilon=10^{{-5}}$ · modes kept: {rec['M1'].n_modes} ($M^1$), {rec['M3'].n_modes} ($M^3$)",
    )
    viz.save_figure(fig, C.FIGURES_DIR / "figure1_surfaces")

    idx = cfg.section_cut_indices
    labels = [r"$y = \frac{51}{128}\pi$", r"$y = \frac{1}{2}\pi$", r"$y = \frac{51}{64}\pi$"]
    fig = viz.section_cuts(
        grid.x, [exact[0.0][:, j] for j in idx],
        {r: [rec[r].u[:, j] for j in idx] for r in ("M1", "M3")}, labels,
        "Figure 2 (reproduced): section cuts at t = 0", r"$\varepsilon = 10^{-5}$, grid lines $j = 51, 64, 102$",
    )
    viz.save_figure(fig, C.FIGURES_DIR / "figure2_section_cuts")

    result = {
        "table1": {f"{r}|{e}": v for (r, e), v in cutoffs.items()},
        "table1_max_rel_dev": max_dev,
        "table2_max_abs_rel_diff": float(comparable.rel_diff.abs().max()),
        "table2_median_abs_rel_diff": float(comparable.rel_diff.abs().median()),
        "typo_cell": {"ours": float(typo_row.ours), "printed": float(typo_row.paper)},
        "noise_actual_l2": {str(e): nd.l2_norm for e, nd in noisy.items()},
        "modes_kept": {f"{r}|{e}": int(np.sum(grid.eigenvalues() <= v)) for (r, e), v in cutoffs.items()},
    }
    C.save_json(result, NAME)
    S.log.info("done")
    return result


if __name__ == "__main__":
    run()
