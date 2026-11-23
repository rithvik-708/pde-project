"""Exact benchmark of Section 4, problem (24), of Nguyen (2015).

    u(x, y, t) = e^{-t}  sum_{k=1}^{K} (1/k^2)       sin(2kx) sin(ky)       (exact solution)
    f(x, y, t) = e^{-t}  sum_{k=1}^{K} (5 - 1/k^2)   sin(2kx) sin(ky)       (source)
    g(x, y)    = e^{-1}  sum_{k=1}^{K} (1/k^2)       sin(2kx) sin(ky)       (final data, T = 1)

with K = 15. The active Fourier modes are (m, n) = (2k, k), with eigenvalue 5k^2.

The class also provides the exact sine coefficients and the closed-form Duhamel
integrals needed by the regularised formula (12). Because f is separable,
f_mn(s) = e^{-s} F_mn, so

    int_t^T e^{(s-t) lam} e^{-s} ds = e^{-t} * expm1((lam-1)(T-t)) / (lam-1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import ProblemConfig
from .fourier import Grid

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PaperExample:
    """Benchmark problem (24); ``T`` and ``K`` default to the paper's values."""

    final_time: float = 1.0
    n_terms: int = 15

    @classmethod
    def from_config(cls, cfg: ProblemConfig) -> "PaperExample":
        return cls(final_time=cfg.final_time, n_terms=cfg.n_terms)

    def __post_init__(self) -> None:
        if self.final_time <= 0:
            raise ValueError("T must be positive")
        if self.n_terms < 1:
            raise ValueError("K must be >= 1")

    # ------------------------------------------------------------ mode data
    @property
    def k(self) -> NDArray[np.int64]:
        return np.arange(1, self.n_terms + 1)

    @property
    def mode_m(self) -> NDArray[np.int64]:
        return 2 * self.k

    @property
    def mode_n(self) -> NDArray[np.int64]:
        return self.k

    @property
    def mode_eigenvalues(self) -> FloatArray:
        """lambda = (2k)^2 + k^2 = 5k^2."""
        return 5.0 * self.k.astype(float) ** 2

    @property
    def amplitudes(self) -> FloatArray:
        """Spatial amplitudes 1/k^2 of u (without the e^{-t} factor)."""
        return 1.0 / self.k.astype(float) ** 2

    @property
    def source_amplitudes(self) -> FloatArray:
        """Spatial amplitudes 5 - 1/k^2 of f."""
        return 5.0 - 1.0 / self.k.astype(float) ** 2

    # ------------------------------------------------------------ pointwise functions
    def _series(self, amps: FloatArray, x: FloatArray, y: FloatArray, m_factor: FloatArray | None = None) -> FloatArray:
        x = np.asarray(x, dtype=float)[..., None]
        y = np.asarray(y, dtype=float)[..., None]
        terms = amps * np.sin(self.mode_m * x) * np.sin(self.mode_n * y)
        if m_factor is not None:
            terms = terms * m_factor
        return terms.sum(axis=-1)

    def u(self, x: FloatArray, y: FloatArray, t: float | FloatArray) -> FloatArray:
        """Exact solution u(x, y, t) (vectorised in x, y; t scalar or broadcastable)."""
        return np.exp(-np.asarray(t, dtype=float)) * self._series(self.amplitudes, x, y)

    def u_t(self, x: FloatArray, y: FloatArray, t: float | FloatArray) -> FloatArray:
        """du/dt = -u."""
        return -self.u(x, y, t)

    def u_xx(self, x: FloatArray, y: FloatArray, t: float | FloatArray) -> FloatArray:
        """d^2u/dx^2 = -sum (2k)^2 (...)."""
        return np.exp(-np.asarray(t, dtype=float)) * self._series(
            self.amplitudes, x, y, m_factor=-(self.mode_m.astype(float) ** 2)
        )

    def u_yy(self, x: FloatArray, y: FloatArray, t: float | FloatArray) -> FloatArray:
        """d^2u/dy^2 = -sum k^2 (...)."""
        return np.exp(-np.asarray(t, dtype=float)) * self._series(
            self.amplitudes, x, y, m_factor=-(self.mode_n.astype(float) ** 2)
        )

    def laplacian(self, x: FloatArray, y: FloatArray, t: float | FloatArray) -> FloatArray:
        return self.u_xx(x, y, t) + self.u_yy(x, y, t)

    def f(self, x: FloatArray, y: FloatArray, t: float | FloatArray) -> FloatArray:
        """Source term f(x, y, t) of (24)."""
        return np.exp(-np.asarray(t, dtype=float)) * self._series(self.source_amplitudes, x, y)

    def g(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Final data g(x, y) = u(x, y, T)."""
        return np.exp(-self.final_time) * self._series(self.amplitudes, x, y)

    def pde_residual(self, x: FloatArray, y: FloatArray, t: float) -> FloatArray:
        """u_t - (u_xx + u_yy) - f; identically zero for the exact solution."""
        return self.u_t(x, y, t) - self.laplacian(x, y, t) - self.f(x, y, t)

    # ------------------------------------------------------------ grid helpers
    def u_grid(self, grid: Grid, t: float) -> FloatArray:
        X, Y = grid.mesh()
        return self.u(X, Y, t)

    def g_grid(self, grid: Grid) -> FloatArray:
        X, Y = grid.mesh()
        return self.g(X, Y)

    def f_grid(self, grid: Grid, t: float) -> FloatArray:
        X, Y = grid.mesh()
        return self.f(X, Y, t)

    # ------------------------------------------------------------ exact sine coefficients
    def _scatter(self, grid: Grid, values: FloatArray) -> FloatArray:
        if 2 * self.n_terms >= grid.n:
            raise ValueError("grid too coarse to represent mode (2K, K)")
        C = np.zeros((grid.n - 1, grid.n - 1))
        C[self.mode_m - 1, self.mode_n - 1] = values
        return C

    def u_coefficients(self, grid: Grid, t: float) -> FloatArray:
        """u_mn(t): non-zero only at (2k, k), equal to e^{-t}/k^2."""
        return self._scatter(grid, np.exp(-t) * self.amplitudes)

    def g_coefficients(self, grid: Grid) -> FloatArray:
        return self.u_coefficients(grid, self.final_time)

    def f_spatial_coefficients(self, grid: Grid) -> FloatArray:
        """F_mn with f_mn(s) = e^{-s} F_mn."""
        return self._scatter(grid, self.source_amplitudes)

    # ------------------------------------------------------------ closed-form Duhamel terms
    @staticmethod
    def time_rate() -> float:
        """f_mn(s) = e^{-rate * s} F_mn with rate = 1."""
        return 1.0
