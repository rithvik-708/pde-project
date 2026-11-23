"""Forward (well-posed) heat evolution used to generate and validate final-time data.

Primary solver (consistent with the paper, eq. (11)): exact spectral evolution

    u_mn(t) = e^{-lambda (t - t0)} u_mn(t0) + int_{t0}^{t} e^{-lambda (t-s)} f_mn(s) ds.

There is no time-stepping error; the only error comes from the source quadrature,
which is exact for the paper's separable source.

Independent cross-check (supplementary): a second-order Crank-Nicolson finite-
difference scheme with the 5-point Laplacian. It is unconditionally stable,
O(h^2 + dt^2), and uses a sparse LU factorisation computed once per run, then O(N^2) work per step.
Agreement between two unrelated discretisations is strong evidence that the
data g used by the backward problem are right.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import splu

from .fourier import Grid, sine_coefficients, synthesize
from .source import SourceTerm, ZeroSource

FloatArray = NDArray[np.float64]


def spectral_forward_coefficients(
    u0_coeffs: FloatArray,
    eigenvalues: FloatArray,
    t0: float,
    t: float,
    source: SourceTerm | None = None,
) -> FloatArray:
    """Exact modal evolution from t0 to t >= t0 (eq. (11))."""
    if t < t0:
        raise ValueError("the forward solver only runs forward in time (t >= t0)")
    source = source or ZeroSource()
    return np.exp(-eigenvalues * (t - t0)) * u0_coeffs + source.forward_integral(eigenvalues, t0, t)


def spectral_forward(
    u0_grid: FloatArray,
    grid: Grid,
    t0: float,
    t: float,
    source: SourceTerm | None = None,
) -> FloatArray:
    """Grid-to-grid forward solve: DST -> exact modal evolution -> inverse DST."""
    coeffs = sine_coefficients(u0_grid, grid)
    return synthesize(spectral_forward_coefficients(coeffs, grid.eigenvalues(), t0, t, source), grid)


def laplacian_matrix(grid: Grid) -> sp.csc_matrix:
    """5-point Dirichlet Laplacian on the (N-1)^2 interior nodes (ij ordering)."""
    n_int = grid.n - 1
    main = -2.0 * np.ones(n_int)
    off = np.ones(n_int - 1)
    d2 = sp.diags([off, main, off], [-1, 0, 1]) / grid.h**2
    eye = sp.identity(n_int)
    return (sp.kron(d2, eye) + sp.kron(eye, d2)).tocsc()


def crank_nicolson(
    u0_grid: FloatArray,
    grid: Grid,
    final_time: float,
    n_steps: int,
    f_on_grid: Callable[[float], FloatArray] | None = None,
) -> FloatArray:
    """Crank-Nicolson finite differences for u_t = Delta u + f, zero Dirichlet BCs."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if final_time <= 0:
        raise ValueError("final_time must be positive")
    dt = final_time / n_steps
    A = laplacian_matrix(grid)
    eye = sp.identity(A.shape[0], format="csc")
    lhs = splu((eye - 0.5 * dt * A).tocsc())
    rhs_op = (eye + 0.5 * dt * A).tocsr()

    def interior_source(s: float) -> FloatArray:
        if f_on_grid is None:
            return np.zeros(A.shape[0])
        return f_on_grid(s)[1:-1, 1:-1].ravel()

    u = np.asarray(u0_grid, dtype=float)[1:-1, 1:-1].ravel().copy()
    f_old = interior_source(0.0)
    for step in range(1, n_steps + 1):
        f_new = interior_source(step * dt)
        u = lhs.solve(rhs_op @ u + 0.5 * dt * (f_old + f_new))
        f_old = f_new
    out = np.zeros((grid.n + 1, grid.n + 1))
    out[1:-1, 1:-1] = u.reshape(grid.n - 1, grid.n - 1)
    return out
