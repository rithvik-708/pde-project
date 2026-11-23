"""Experiment 4 -- influence of the regularisation parameter M (SUPPLEMENTARY).

The paper fixes M by a-priori rules (Table 1) and remarks that "the bigger the value
M_eps, the better". This experiment sweeps M continuously. It splits the error into

    u^eps - u = [Gamma_M u_clean-data-recon - u]    (truncation / bias part, eps = 0)
              + [reconstruction of the pure noise]  (propagated noise, ~ e^{(T-t)M} eps)

so that the stability-versus-accuracy trade-off is visible. The paper's statement holds only
for M < 20. Beyond that, the propagated noise dominates.

Since the error only changes when M crosses an eigenvalue m^2 + n^2, the sweep evaluates
M exactly at the distinct eigenvalues, which captures the whole piecewise-constant curve.
"""

from __future__ import annotations

import _common as C  # noqa: I001

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backward_heat import add_paper_noise, make_rng, reconstruct
from backward_heat import visualization as viz
from backward_heat.metrics import l2_norm_grid, relative_l2_error
from backward_heat.regularization import cutoff_rules

NAME = "experiment_04_regularization_parameter"
RULES = ("M1", "M2", "M3", "M4")
M_MAX = 60.0
TIMES = (0.0, 0.5)


def run() -> dict:
    S = C.setup(NAME)
    cfg, grid, ex, T = S.cfg, S.grid, S.example, S.T
    rules = cutoff_rules(cfg.regularization.r, cfg.regularization.beta, cfg.regularization.m4_sqrt)
    g = ex.g_grid(grid)
    lam_sorted = np.unique(grid.eigenvalues())
    Ms = np.r_[1.0, lam_sorted[lam_sorted <= M_MAX]]
    rng = make_rng(cfg.seed)
    noisy = {eps: add_paper_noise(g, grid, eps, rng) for eps in cfg.noise_levels}

    rows = []
    for t in TIMES:
        exact = ex.u_grid(grid, t)
        norm_u = l2_norm_grid(exact, grid)
        for M in Ms:
            bias = reconstruct(g, t, float(M), grid, T, S.source).u - exact
            for eps, nd in noisy.items():
                noise_part = reconstruct(nd.perturbation, t, float(M), grid, T, None).u  # linear in data
                total = reconstruct(nd.values, t, float(M), grid, T, S.source).u - exact
                rows.append({"t": t, "M": float(M), "eps": eps,
                             "total": l2_norm_grid(total, grid) / norm_u,
                             "truncation": l2_norm_grid(bias, grid) / norm_u,
                             "noise": l2_norm_grid(noise_part, grid) / norm_u})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(C.RAW_DIR / "exp04_error_vs_M.csv", index=False)

    # Optimal M versus the a-priori rules
    opt_rows = []
    for (t, eps), sub in sweep.groupby(["t", "eps"]):
        i = sub.total.idxmin()
        row = {"t": t, "eps": eps, "M_opt": sub.loc[i, "M"], "err_opt": sub.loc[i, "total"]}
        for rule in RULES:
            M = rules[rule](eps, T)
            u = reconstruct(noisy[eps].values, t, M, grid, T, S.source).u
            row[f"M_{rule}"] = M
            row[f"err_{rule}"] = relative_l2_error(ex.u_grid(grid, t), u, grid)
        opt_rows.append(row)
    optimum = pd.DataFrame(opt_rows)
    C.save_table(optimum, "exp04_optimal_vs_rules")

    # ---------------------------------------------------------------- figure: trade-off
    fig, axes = plt.subplots(len(TIMES), len(cfg.noise_levels), figsize=(15, 8.2), sharex=True, sharey=True,
                             constrained_layout=True)
    for r, t in enumerate(TIMES):
        for c, eps in enumerate(cfg.noise_levels):
            ax = axes[r, c]
            sub = sweep[(sweep.t == t) & (sweep.eps == eps)]
            ax.step(sub.M, sub.truncation, where="post", color="#64748b", lw=1.6, ls="--", label="noise-free error (truncation, + rounding for large M)")
            ax.step(sub.M, sub.noise, where="post", color="#e11d48", lw=1.6, ls="--", label="propagated noise")
            ax.step(sub.M, sub.total, where="post", color=viz.INK, lw=2.4, label="total error")
            for rule in RULES:
                ax.axvline(rules[rule](eps, T), color=viz.RULE_COLORS[rule], lw=1.3, alpha=0.9)
            ax.set_yscale("log")
            ax.set_ylim(1e-9, 1e4)
            ax.set_xlim(0, M_MAX)
            ax.set_title(rf"$\varepsilon=10^{{{round(np.log10(eps))}}}$, $t={t}$", fontsize=11)
            if r == len(TIMES) - 1:
                ax.set_xlabel(r"cut-off $M$")
            if c == 0:
                ax.set_ylabel(r"relative $L^2$ error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=viz.RULE_COLORS[r]) for r in RULES]
    labels += [viz.RULE_LABELS[r] for r in RULES]
    fig.legend(handles, labels, loc="outside lower center", ncol=8, fontsize=9)
    fig.suptitle("Stability vs. accuracy: choosing the cut-off M")
    viz.save_figure(fig, C.FIGURES_DIR / "exp04_tradeoff")

    # ---------------------------------------------------------------- figure: three regimes
    eps = cfg.figure_noise_level
    exact0 = ex.u_grid(grid, 0.0)
    sub = sweep[(sweep.t == 0.0) & (sweep.eps == eps)]
    m_opt = float(sub.loc[sub.total.idxmin(), "M"])
    cases = [("too much regularisation", 3.0), ("appropriate (optimal)", m_opt), ("too little regularisation", 26.0)]
    recs = [reconstruct(noisy[eps].values, 0.0, M, grid, T, S.source) for _, M in cases]
    fig = viz.field_grid(
        grid.x,
        [[exact0, r.u, np.abs(exact0 - r.u)] for r in recs],
        [f"{name}\nM = {M:g} ({r.n_modes} modes)" for (name, M), r in zip(cases, recs)],
        [r"exact $u(x,y,0)$", r"reconstruction $u^\varepsilon(x,y,0)$", r"$|u-u^\varepsilon|$"],
        ["field", "field", "error"],
        r"Three regimes of the cut-off ($\varepsilon=10^{-5}$, $t=0$)",
        "small M: stable but over-smoothed  ·  large M: noise amplified by e^{(T−t)M}  ·  each panel has its own colour scale",
    )
    viz.save_figure(fig, C.FIGURES_DIR / "exp04_three_regimes")

    result = {"optimum": optimum.to_dict("records"), "regimes": {name: {"M": M, "modes": r.n_modes,
              "relative_l2": relative_l2_error(exact0, r.u, grid)} for (name, M), r in zip(cases, recs)}}
    C.save_json(result, NAME)
    S.log.info("regimes: %s", result["regimes"])
    S.log.info("done")
    return result


if __name__ == "__main__":
    run()
