"""End-to-end reconstruction pipeline: noisy final data -> regularised u^eps(., ., t).

    terminal data g_eps on grid
        --(DST-I, paper normalisation 4/pi^2)-->      (g_eps)_mn
        --(eq. (12): truncate + propagate back)-->    u^eps_mn(t)
        --(inverse DST)-->                            u^eps(x_i, y_j, t)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .fourier import Grid, sine_coefficients, synthesize, truncation_mask
from .regularization import regularized_coefficients, stability_factor
from .source import SourceTerm

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ReconstructionResult:
    """Regularised solution and diagnostics."""

    u: FloatArray  # u^eps on the (N+1)^2 grid
    coefficients: FloatArray  # u^eps_mn(t)
    data_coefficients: FloatArray  # (g_eps)_mn
    t: float
    cutoff: float
    n_modes: int  # number of (m, n) with m^2 + n^2 <= M
    kept_modes: list[tuple[int, int]]
    amplification: float  # e^{(T-t) M} (Theorem 3)


def reconstruct(
    terminal_data: FloatArray,
    t: float,
    cutoff: float,
    grid: Grid,
    final_time: float = 1.0,
    source: SourceTerm | None = None,
) -> ReconstructionResult:
    """Regularised backward reconstruction of u(., ., t) from g_eps = u(., ., T) + noise.

    Parameters
    ----------
    terminal_data : grid samples of g_eps, shape (N+1, N+1).
    t             : target time, 0 <= t <= T.
    cutoff        : regularisation parameter M_eps (keep m^2 + n^2 <= M_eps).
    grid          : spatial grid.
    final_time    : T.
    source        : source term f (``None`` for the homogeneous equation).
    """
    terminal_data = np.asarray(terminal_data, dtype=float)
    if not np.all(np.isfinite(terminal_data)):
        raise ValueError("terminal data contain NaN or inf")
    eigenvalues = grid.eigenvalues()
    g_coeffs = sine_coefficients(terminal_data, grid)
    coeffs = regularized_coefficients(g_coeffs, eigenvalues, t, final_time, cutoff, source)
    mask = truncation_mask(eigenvalues, cutoff)
    kept = [(int(m) + 1, int(n) + 1) for m, n in zip(*np.nonzero(mask))]
    u = synthesize(coeffs, grid)
    if not np.all(np.isfinite(u)):  # pragma: no cover - guarded by OverflowError upstream
        raise FloatingPointError("reconstruction produced non-finite values")
    return ReconstructionResult(
        u=u,
        coefficients=coeffs,
        data_coefficients=g_coeffs,
        t=t,
        cutoff=cutoff,
        n_modes=len(kept),
        kept_modes=kept,
        amplification=stability_factor(t, final_time, cutoff),
    )
