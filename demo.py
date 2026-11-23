"""Interactive live demo of the Fourier-truncation regularisation (for presentations).

    python demo.py                 # interactive window with sliders
    python demo.py --snapshot      # write results/figures/demo_snapshot.png and exit

Controls
--------
* noise level  log10(eps)        -- size of the measurement error in g_eps = g + (eps/pi) rand()
* target time  t                 -- how far back in time to reconstruct (T = 1)
* cut-off rule M1 / M2 / M3 / M4 -- the paper's a-priori choices, or "manual" to use the M slider
* manual M                       -- keep the modes with m^2 + n^2 <= M (try M > 25 at t = 0!)
* "new noise" button             -- draw a fresh noise realisation

Everything shown is computed live with the same library code as the experiments.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--snapshot", action="store_true", help="render one frame to PNG instead of opening a window")
ARGS = parser.parse_args()
if not ARGS.snapshot:
    os.environ["BACKWARD_HEAT_INTERACTIVE"] = "1"

import matplotlib  # noqa: E402

if not ARGS.snapshot:
    try:
        matplotlib.use("TkAgg")
    except Exception:  # pragma: no cover - fall back to whatever is available
        pass

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.widgets import Button, RadioButtons, Slider  # noqa: E402

from backward_heat import Grid, PaperExample, SeparableExponentialSource, add_paper_noise, make_rng, reconstruct  # noqa: E402
from backward_heat import visualization as viz  # noqa: E402
from backward_heat.metrics import paper_error, relative_l2_error  # noqa: E402
from backward_heat.regularization import MAX_EXPONENT, cutoff_rules  # noqa: E402

GRID = Grid(128)
EX = PaperExample()
SRC = SeparableExponentialSource(EX.f_spatial_coefficients(GRID), EX.time_rate())
RULES = cutoff_rules()
G_CLEAN = EX.g_grid(GRID)
X = GRID.x


class Demo:
    def __init__(self) -> None:
        self.seed = 2015
        self.noise_unit = make_rng(self.seed).uniform(-1, 1, size=G_CLEAN.shape)
        self.fig = plt.figure(figsize=(15.5, 8.4))
        self.fig.suptitle("Backward heat equation on (0, π)² — Fourier truncation regularisation (Nguyen 2015)")
        gs = self.fig.add_gridspec(2, 4, left=0.04, right=0.985, top=0.9, bottom=0.36, wspace=0.28, hspace=0.05,
                                   height_ratios=[1, 0.07])
        titles = [r"noisy final data $g_\varepsilon = u(\cdot,\cdot,1)+$noise", r"exact $u(\cdot,\cdot,t)$",
                  r"regularised $u^\varepsilon(\cdot,\cdot,t)$", r"error $|u-u^\varepsilon|$"]
        self.axes, self.images, self.cbars = [], [], []
        for k, title in enumerate(titles):
            ax = self.fig.add_subplot(gs[0, k])
            ax.grid(False)
            ax.set_xticks([0, np.pi], ["0", "π"])
            ax.set_yticks([0, np.pi], ["0", "π"])
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=11)
            cmap = viz.ERROR_CMAP if k == 3 else viz.FIELD_CMAP
            im = ax.pcolormesh(X, X, np.zeros((129, 129)), cmap=cmap, shading="gouraud")
            cax = self.fig.add_subplot(gs[1, k])
            self.cbars.append(self.fig.colorbar(im, cax=cax, orientation="horizontal"))
            self.axes.append(ax)
            self.images.append(im)

        # profile + spectrum strip
        self.ax_cut = self.fig.add_axes([0.37, 0.06, 0.3, 0.2])
        self.ax_cut.set_title(r"section $y=\pi/2$", fontsize=10)
        (self.line_exact,) = self.ax_cut.plot(X, np.zeros_like(X), color=viz.EXACT_COLOR, lw=1.5, label="exact")
        (self.line_rec,) = self.ax_cut.plot(X, np.zeros_like(X), color=viz.RULE_COLORS["M3"], lw=2.2, label=r"$u^\varepsilon$")
        self.ax_cut.legend(loc="upper right", fontsize=8)
        self.ax_cut.set_xticks([0, np.pi / 2, np.pi], ["0", "π/2", "π"])
        self.info = self.fig.text(0.70, 0.27, "", va="top", family="DejaVu Sans Mono", fontsize=9.5, color=viz.INK)

        # widgets
        self.s_eps = Slider(self.fig.add_axes([0.08, 0.22, 0.22, 0.03]), r"log$_{10}\varepsilon$", -12, -1, valinit=-5, valstep=0.25)
        self.s_t = Slider(self.fig.add_axes([0.08, 0.17, 0.22, 0.03]), "t", 0.0, 0.95, valinit=0.0, valstep=0.05)
        self.s_M = Slider(self.fig.add_axes([0.08, 0.12, 0.22, 0.03]), "manual M", 1, 80, valinit=20, valstep=1)
        self.radio = RadioButtons(self.fig.add_axes([0.08, 0.005, 0.1, 0.105]), ["M1", "M2", "M3", "M4", "manual"], active=2)
        self.button = Button(self.fig.add_axes([0.2, 0.04, 0.1, 0.045]), "new noise")
        for w in (self.s_eps, self.s_t, self.s_M):
            w.on_changed(lambda _v: self.update())
        self.radio.on_clicked(lambda _l: self.update())
        self.button.on_clicked(self.reseed)
        self.update()

    def reseed(self, _event: object = None) -> None:
        self.seed += 1
        self.noise_unit = make_rng(self.seed).uniform(-1, 1, size=G_CLEAN.shape)
        self.update()

    def update(self) -> None:
        eps = 10.0 ** self.s_eps.val
        t = float(self.s_t.val)
        rule = self.radio.value_selected
        M = float(self.s_M.val) if rule == "manual" else RULES[rule](eps, 1.0)
        M = min(M, 0.95 * MAX_EXPONENT / max(1.0 - t, 1e-9))
        g_eps = G_CLEAN + eps / np.pi * self.noise_unit
        exact = EX.u_grid(GRID, t)
        res = reconstruct(g_eps, t, M, GRID, 1.0, SRC)
        err = np.abs(exact - res.u)

        lim_u = float(np.max(np.abs(exact)))
        fields = [g_eps, exact, res.u, err]
        for k, (im, F) in enumerate(zip(self.images, fields)):
            im.set_array(F.T.ravel())
            if k == 3:
                im.set_norm(mcolors.Normalize(0, max(float(F.max()), 1e-12)))
            else:
                lim = max(float(np.max(np.abs(F))), lim_u) if k == 2 else float(np.max(np.abs(F)))
                im.set_norm(mcolors.TwoSlopeNorm(0.0, -lim, lim))
            self.cbars[k].update_normal(im)
        self.line_exact.set_ydata(exact[:, 64])
        self.line_rec.set_ydata(res.u[:, 64])
        lo = min(exact[:, 64].min(), res.u[:, 64].min())
        hi = max(exact[:, 64].max(), res.u[:, 64].max())
        pad = 0.1 * (hi - lo + 1e-12)
        self.ax_cut.set_ylim(lo - pad, hi + pad)

        rel = relative_l2_error(exact, res.u, GRID)
        status = "OK" if rel < 0.5 else ("POOR" if rel < 2 else "UNSTABLE: noise amplified")
        self.info.set_text(
            f"noise level      ε = {eps:.2e}\n"
            f"target time      t = {t:.2f}   (T = 1)\n"
            f"cut-off rule       {rule}\n"
            f"cut-off          M = {M:.3f}\n"
            f"modes kept         {res.n_modes}  (m²+n² ≤ M)\n"
            f"amplification      e^((T-t)M) = {res.amplification:.2e}\n"
            f"paper metric (RMS) {paper_error(exact, res.u):.4e}\n"
            f"relative L² error  {rel:.4e}\n"
            f"status             {status}\n"
            f"noise seed         {self.seed}"
        )
        self.fig.canvas.draw_idle()


def main() -> None:
    demo = Demo()
    if ARGS.snapshot:
        out = ROOT / "results" / "figures" / "demo_snapshot.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        demo.fig.savefig(out, dpi=150)
        print(f"wrote {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
