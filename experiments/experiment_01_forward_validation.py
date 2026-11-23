"""Experiment 1 -- forward solution validation.

The final data g of the backward problem must be consistent with the forward PDE.
Three checks:

1. Exact spectral evolution (eq. (11)) of u(., ., 0) to T = 1 versus the analytical g.
2. The general Filon source quadrature (``GridSource``) converges at O(h_t^2) to the
   closed-form Duhamel integral.
3. An independent Crank-Nicolson finite-difference solver converges at O(h^2) to g.

Checks 2-3 are supplementary verifications; the paper itself does not report a forward solve.
"""

from __future__ import annotations

import _common as C  # noqa: I001  (must come first: sets up sys.path)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backward_heat import Grid, GridSource, SeparableExponentialSource
from backward_heat import visualization as viz
from backward_heat.forward_solver import crank_nicolson, spectral_forward
from backward_heat.metrics import all_metrics, max_absolute_error, paper_error

NAME = "experiment_01_forward_validation"


def run() -> dict:
    S = C.setup(NAME)
    grid, ex, T = S.grid, S.example, S.T
    x = grid.x

    # 1) spectral forward solve -------------------------------------------------------
    with C.Timer(S.log, "spectral forward solve"):
        gT_spec = spectral_forward(ex.u_grid(grid, 0.0), grid, 0.0, T, S.source)
    g_exact = ex.g_grid(grid)
    spec_metrics = all_metrics(g_exact, gT_spec, grid)
    S.log.info("spectral forward: max|err| = %.3e, rel L2 = %.3e", spec_metrics["max_abs"], spec_metrics["relative_l2"])

    # 2) Filon quadrature convergence ---------------------------------------------------
    lam = grid.eigenvalues()
    exact_int = S.source.forward_integral(lam, 0.0, T)
    filon_rows = []
    for n in (25, 50, 100, 200, 400):
        approx = GridSource(lambda s: ex.f_grid(grid, s), grid, n_panels=n).forward_integral(lam, 0.0, T)
        filon_rows.append({"panels": n, "dt": T / n, "max_coeff_error": float(np.max(np.abs(approx - exact_int)))})
    filon = pd.DataFrame(filon_rows)
    filon["observed_order"] = np.r_[np.nan, np.log2(filon.max_coeff_error.values[:-1] / filon.max_coeff_error.values[1:])]
    S.log.info("Filon orders: %s", np.round(filon.observed_order.values[1:], 3))

    # 3) Crank-Nicolson convergence -----------------------------------------------------
    cn_rows, cn_fields = [], {}
    with C.Timer(S.log, "Crank-Nicolson study"):
        for n in (32, 64, 128):
            g_n = Grid(n)
            uT = crank_nicolson(ex.u_grid(g_n, 0.0), g_n, T, n_steps=2 * n, f_on_grid=lambda s, gg=g_n: ex.f_grid(gg, s))
            m = all_metrics(ex.g_grid(g_n), uT, g_n)
            cn_rows.append({"N": n, "h": g_n.h, "dt": T / (2 * n), "paper_rms": m["paper_rms"],
                            "relative_l2": m["relative_l2"], "max_abs": m["max_abs"]})
            cn_fields[n] = uT
    cn = pd.DataFrame(cn_rows)
    cn["observed_order"] = np.r_[np.nan, np.log2(cn.relative_l2.values[:-1] / cn.relative_l2.values[1:])]
    S.log.info("Crank-Nicolson rel-L2 errors: %s", cn.relative_l2.values)

    summary = pd.DataFrame([
        {"solver": "spectral (eq. 11, exact Duhamel)", "N": grid.n, **{k: spec_metrics[k] for k in ("paper_rms", "relative_l2", "max_abs")}},
        {"solver": "Crank-Nicolson FD (dt = h/(2pi))", "N": 128, **{k: cn_rows[-1][k] for k in ("paper_rms", "relative_l2", "max_abs")}},
    ])
    C.save_table(summary, "exp01_forward_summary")
    C.save_table(filon, "exp01_filon_convergence")
    C.save_table(cn, "exp01_crank_nicolson_convergence")

    # ---------------------------------------------------------------- figures
    snaps = [0.0, 0.25, 0.5, 1.0]
    fig = viz.field_panels(
        x, [ex.u_grid(grid, t) for t in snaps], [rf"$u(x,y,{t:g})$" for t in snaps],
        "Forward heat evolution of the benchmark solution",
        r"$u_t-\Delta u=f$ on $(0,\pi)^2$, zero Dirichlet data; the final state $u(\cdot,\cdot,1)=g$ is the input of the backward problem",
    )
    viz.save_figure(fig, C.FIGURES_DIR / "exp01_forward_evolution")

    fig = viz.field_panels(
        x, [g_exact, cn_fields[128], np.abs(g_exact - cn_fields[128])],
        [r"analytical $g=u(\cdot,\cdot,1)$", r"Crank–Nicolson, $N=128$", r"$|g-u_h(\cdot,\cdot,1)|$"],
        "Forward solver validation at t = T = 1", "independent finite-difference solve started from the exact u(·,·,0)",
        error_last=True,
    )
    viz.save_figure(fig, C.FIGURES_DIR / "exp01_forward_validation")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.loglog(cn.h, cn.relative_l2, "o-", color=viz.RULE_COLORS["M1"], label="Crank–Nicolson")
    ax.loglog(cn.h, cn.relative_l2.iloc[0] * (cn.h / cn.h.iloc[0]) ** 2, "--", color=viz.MUTED, label=r"slope 2")
    ax.set_xlabel(r"mesh width $h=\pi/N$")
    ax.set_ylabel(r"relative $L^2$ error at $t=1$")
    ax.set_title("FD solver: second-order convergence")
    ax.legend()
    ax = axes[1]
    ax.loglog(filon.dt, filon.max_coeff_error, "o-", color=viz.RULE_COLORS["M3"], label="Filon (linear) quadrature")
    ax.loglog(filon.dt, filon.max_coeff_error.iloc[0] * (filon.dt / filon.dt.iloc[0]) ** 2, "--", color=viz.MUTED, label="slope 2")
    ax.set_xlabel(r"time panel width $\Delta s$")
    ax.set_ylabel("max coefficient error")
    ax.set_title(r"Duhamel integral: $O(\Delta s^2)$ uniformly in $\lambda$")
    ax.legend()
    for a in axes:
        a.xaxis.set_minor_formatter(plt.NullFormatter())
    fig.suptitle("Convergence of the forward machinery")
    viz.save_figure(fig, C.FIGURES_DIR / "exp01_convergence")

    result = {
        "spectral": spec_metrics,
        "filon": filon.to_dict("list"),
        "crank_nicolson": cn.to_dict("list"),
        "spectral_max_abs_error": max_absolute_error(g_exact, gT_spec),
        "spectral_paper_rms": paper_error(g_exact, gT_spec),
    }
    C.save_json(result, NAME)
    S.log.info("done")
    return result


if __name__ == "__main__":
    run()
