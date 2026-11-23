"""Experiment 3 -- noise sensitivity.

Part A (paper settings, eps = 1e-3, 1e-5, 1e-7): noisy data g_eps, reconstructions at
t = 0 with each cut-off rule, and the error maps. The *numbers* for these settings are
the reproduction of Table 2 in Experiment 5; this part visualises them.

Part B (SUPPLEMENTARY, not in the paper): error at t = 0 and t = 0.5 over a wide noise
range eps = 1e-1 ... 1e-14, averaged over 10 noise realisations. It shows the
convergence u^eps -> u as eps -> 0 claimed by Theorems 5-7. The Theorem 5 bound is
overlaid for M2.

Part C: Fourier picture of the instability. The plot shows the coefficients of the data,
the noise, and the noise after multiplication by e^{T lambda}, against lambda.
"""

from __future__ import annotations

import _common as C  # noqa: I001

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backward_heat import add_paper_noise, make_rng, reconstruct, sine_coefficients
from backward_heat import visualization as viz
from backward_heat.fourier import h10_norm_from_coeffs
from backward_heat.metrics import l2_norm_grid, paper_error, relative_l2_error
from backward_heat.regularization import cutoff_rules, theorem5_corrected_l2_bound, theorem5_l2_bound

NAME = "experiment_03_noise_sensitivity"
RULES = ("M1", "M2", "M3", "M4")
EPS_SWEEP = np.logspace(-1, -14, 27)
N_SEEDS = 10


def run() -> dict:
    S = C.setup(NAME)
    cfg, grid, ex, T = S.cfg, S.grid, S.example, S.T
    rules = cutoff_rules(cfg.regularization.r, cfg.regularization.beta, cfg.regularization.m4_sqrt)
    g = ex.g_grid(grid)
    exact0 = ex.u_grid(grid, 0.0)
    rng = make_rng(cfg.seed)

    # ------------------------------------------------------------- Part A
    rows, labels = [], []
    part_a = []
    for eps in cfg.noise_levels:
        nd = add_paper_noise(g, grid, eps, rng)
        M = rules["M1"](eps, T)
        res = reconstruct(nd.values, 0.0, M, grid, T, S.source)
        rows.append([nd.values, nd.perturbation, res.u, np.abs(exact0 - res.u)])
        labels.append(rf"$\varepsilon=10^{{{round(np.log10(eps))}}}$")
        for rule in RULES:
            r = reconstruct(nd.values, 0.0, rules[rule](eps, T), grid, T, S.source)
            part_a.append({"eps": eps, "rule": rule, "M": r.cutoff, "modes_kept": r.n_modes,
                           "noise_L2": nd.l2_norm, "paper_rms_t0": paper_error(exact0, r.u),
                           "relative_l2_t0": relative_l2_error(exact0, r.u, grid), "amplification": r.amplification})
    C.save_table(pd.DataFrame(part_a), "exp03_paper_noise_levels")
    fig = viz.field_grid(
        grid.x, rows, labels,
        [r"noisy data $g_\varepsilon$", r"noise $g_\varepsilon-g$", r"$u^\varepsilon(\cdot,\cdot,0)$, $M^1_\varepsilon$",
         r"$|u-u^\varepsilon|$ at $t=0$"],
        ["field", "noise", "field", "error"],
        "Noisy final data and regularised reconstructions (paper noise levels)",
        r"$g_\varepsilon=g+\frac{\varepsilon}{\pi}\,\mathrm{rand}()$, rand $\sim U(-1,1)$ i.i.d. per node; "
        r"cut-off $M^1_\varepsilon$ (Remark 1)",
    )
    viz.save_figure(fig, C.FIGURES_DIR / "exp03_noise_levels_comparison")

    # ------------------------------------------------------------- Part B
    records = []
    exact = {t: ex.u_grid(grid, t) for t in (0.0, 0.5)}
    for s in range(N_SEEDS):
        r_s = make_rng(1000 + s)
        for eps in EPS_SWEEP:
            g_eps = add_paper_noise(g, grid, eps, r_s).values
            for rule in RULES:
                M = rules[rule](eps, T)
                for t in exact:
                    if (T - t) * M > 600:
                        continue
                    u_eps = reconstruct(g_eps, t, M, grid, T, S.source).u
                    records.append({"seed": s, "eps": eps, "rule": rule, "t": t, "M": M,
                                    "relative_l2": relative_l2_error(exact[t], u_eps, grid)})
    sweep = pd.DataFrame(records)
    sweep.to_csv(C.RAW_DIR / "exp03_error_vs_eps.csv", index=False)
    agg = sweep.groupby(["rule", "t", "eps"]).relative_l2.agg(["mean", "std"]).reset_index()
    C.save_table(agg[agg.eps.isin(EPS_SWEEP[::4])], "exp03_error_vs_eps_excerpt")

    h10 = h10_norm_from_coeffs(ex.u_coefficients(grid, 0.0), grid.eigenvalues())
    l2_u0 = l2_norm_grid(exact[0.0], grid)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True, constrained_layout=True)
    for ax, t in zip(axes, exact):
        for rule in RULES:
            sub = agg[(agg.rule == rule) & (agg.t == t)]
            ax.plot(sub.eps, sub["mean"], color=viz.RULE_COLORS[rule], label=viz.RULE_LABELS[rule],
                    **{k: v for k, v in viz.RULE_STYLES[rule].items() if k != "lw"})
        if t == 0.0:
            bound = [theorem5_l2_bound(e, T, h10) / l2_u0 for e in EPS_SWEEP]
            ax.plot(EPS_SWEEP, bound, color=viz.RULE_COLORS["M2"], lw=1.2, alpha=0.6, ls="-.",
                    label=r"Thm 5 bound as printed (violated, A12)")
            corrected = [theorem5_corrected_l2_bound(e, T, h10) / l2_u0 for e in EPS_SWEEP]
            ax.plot(EPS_SWEEP, corrected, color=viz.RULE_COLORS["M2"], lw=1.2, alpha=0.9,
                    label=r"Thm 5 bound, Parseval-corrected")
        for e in cfg.noise_levels:
            ax.axvline(e, color=viz.GRID, lw=6, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"noise level $\varepsilon$")
        ax.set_title(rf"target time $t={t}$")
    axes[0].set_ylabel(r"relative $L^2$ error (mean of 10 noise draws)")
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Convergence of the regularised solution as the noise vanishes")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.95))
    fig.text(0.5, 0.945, "staircase: one more active mode (2k, k), λ = 5k², enters the disc m²+n² ≤ M_ε  ·  "
             "shaded: the paper's noise levels", ha="center", va="top", color=viz.MUTED, fontsize=10)
    viz.save_figure(fig, C.FIGURES_DIR / "exp03_error_vs_noise_level")

    # ------------------------------------------------------------- Part C
    eps = cfg.figure_noise_level
    nd = add_paper_noise(g, grid, eps, make_rng(cfg.seed))
    lam = grid.eigenvalues()
    noise_c = sine_coefficients(nd.perturbation, grid)
    sel = lam <= 200
    amplified = np.where(sel, np.exp(T * np.minimum(lam, 200)) * noise_c, 0)
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    viz.coefficient_spectrum(
        np.where(sel, lam, np.nan),
        {r"exact data $|g_{mn}|$": np.where(sel, ex.g_coefficients(grid), 0),
         r"noise $|(g_\varepsilon-g)_{mn}|$": np.where(sel, noise_c, 0),
         r"amplified noise $e^{T\lambda}|(g_\varepsilon-g)_{mn}|$": amplified,
         r"needed $|u_{mn}(0)|$": np.where(sel, ex.u_coefficients(grid, 0.0), 0)},
        ["#111827", "#94a3b8", "#e11d48", "#16a34a"],
        {r: rules[r](eps, T) for r in RULES},
        rf"Why truncation is needed ($\varepsilon=10^{{-5}}$, $t=0$)", ax=ax,
    )
    ax.axhline(1.0, color=viz.MUTED, lw=0.8, ls=":")
    viz.save_figure(fig, C.FIGURES_DIR / "exp03_fourier_coefficients")

    result = {"part_a": part_a, "h10_norm_u0": h10}
    C.save_json(result, NAME)
    S.log.info("done")
    return result


if __name__ == "__main__":
    run()
