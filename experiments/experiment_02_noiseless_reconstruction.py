"""Experiment 2 -- backward reconstruction from noiseless final data (SUPPLEMENTARY).

The paper reports only noisy experiments. Here eps = 0, so the data are exact up to
float64 rounding (~1e-16). This isolates two effects:

* the **truncation error**, which falls in steps whenever the disc m^2 + n^2 <= M
  takes in another active mode (2k, k), at lambda = 5, 20, 45, ...;
* **rounding errors amplified by e^{(T-t) lambda}**, which ruin the reconstruction
  once e^{(T-t) M} * 1e-16 ~ 1, even though the data contain *no* measurement noise.
  This is ill-posedness in its purest form.
"""

from __future__ import annotations

import _common as C  # noqa: I001

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backward_heat import reconstruct
from backward_heat import visualization as viz
from backward_heat.metrics import all_metrics, paper_error, relative_l2_error

NAME = "experiment_02_noiseless_reconstruction"
TIMES = (0.0, 0.5, 0.9)
M_MAX = 400.0


def run() -> dict:
    S = C.setup(NAME)
    grid, ex, T = S.grid, S.example, S.T
    g = ex.g_grid(grid)
    lam_sorted = np.unique(grid.eigenvalues())

    sweep_rows = []
    best = {}
    for t in TIMES:
        exact = ex.u_grid(grid, t)
        m_cap = min(M_MAX, 300.0 / max(T - t, 1e-12))  # keep e^{(T-t)M} far from overflow
        Ms = lam_sorted[lam_sorted <= m_cap]
        errs = []
        for M in Ms:
            res = reconstruct(g, t, float(M), grid, T, S.source)
            errs.append(relative_l2_error(exact, res.u, grid))
            sweep_rows.append({"t": t, "M": float(M), "modes_kept": res.n_modes, "relative_l2": errs[-1],
                               "paper_rms": paper_error(exact, res.u)})
        errs = np.array(errs)
        i = int(np.argmin(errs))
        best[t] = (float(Ms[i]), float(errs[i]))
        S.log.info("t=%.1f: best M=%g (rel L2 %.3e); worst rel L2 in sweep %.3e", t, Ms[i], errs[i], errs.max())
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(C.RAW_DIR / "exp02_error_vs_M.csv", index=False)

    # summary table at representative cut-offs
    summary_rows = []
    for t in TIMES:
        exact = ex.u_grid(grid, t)
        for M in (4.9, 5.0, 20.0, best[t][0], 60.0):
            if (T - t) * M > 300:
                continue
            res = reconstruct(g, t, M, grid, T, S.source)
            summary_rows.append({"t": t, "M": M, "modes_kept": res.n_modes, **all_metrics(exact, res.u, grid)})
    summary = pd.DataFrame(summary_rows).drop_duplicates(subset=["t", "M"])
    C.save_table(summary, "exp02_noiseless_summary")

    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    colors = viz.NOISE_COLORS
    for color, t in zip(colors, TIMES):
        sub = sweep[sweep.t == t]
        ax.step(sub.M, sub.relative_l2, where="post", color=color, label=rf"$t={t}$")
        ax.plot(*best[t], "o", color=color, ms=7, mec="white", mew=1.5)
    for k in (1, 2, 3, 4):
        ax.axvline(5 * k * k, color=viz.GRID, lw=1.5, zorder=0)
        ax.text(5 * k * k, 1.01, f"k={k}", transform=ax.get_xaxis_transform(), ha="center", fontsize=8.5, color=viz.MUTED)
    ax.axhline(1.0, color=viz.MUTED, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(5e-3, 1e10)  # errors reach 1e111 for t = 0; clipped for readability
    ax.set_xlabel(r"cut-off $M$ (keep $m^2+n^2\leq M$)")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title("Noise-free data: truncation error vs. amplified rounding error")
    ax.legend(title="target time", loc="upper left")
    viz.save_figure(fig, C.FIGURES_DIR / "exp02_error_vs_cutoff_noiseless")

    exact0 = ex.u_grid(grid, 0.0)
    good = reconstruct(g, 0.0, best[0.0][0], grid, T, S.source)
    bad = reconstruct(g, 0.0, 80.0, grid, T, S.source)
    fig = viz.field_grid(
        grid.x,
        [[exact0, good.u, np.abs(exact0 - good.u)], [exact0, bad.u, np.abs(exact0 - bad.u)]],
        [rf"$M={best[0.0][0]:g}$", r"$M=80$"],
        [r"exact $u(x,y,0)$", r"reconstruction $u^\varepsilon(x,y,0)$", r"$|u-u^\varepsilon|$"],
        ["field", "field", "error"],
        "Noise-free backward reconstruction at t = 0",
        rf"top: best cut-off ({good.n_modes} modes, rel. $L^2$ error {best[0.0][1]:.3f}) · bottom: $M=80$, "
        r"float64 rounding $\times\,e^{80}$ destroys the solution",
    )
    viz.save_figure(fig, C.FIGURES_DIR / "exp02_noiseless_reconstruction")

    result = {"best_cutoff": {str(t): {"M": m, "relative_l2": e} for t, (m, e) in best.items()},
              "M80_relative_l2_t0": relative_l2_error(exact0, bad.u, grid)}
    C.save_json(result, NAME)
    S.log.info("done")
    return result


if __name__ == "__main__":
    run()
