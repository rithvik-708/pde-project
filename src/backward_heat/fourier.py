"""Double sine-series machinery on Omega = (0, pi)^2 with Dirichlet conditions.

Conventions (paper, p. 4):

    h_mn = (4/pi^2) * int_0^pi int_0^pi h(x, y) sin(mx) sin(ny) dx dy,
    h(x, y) = sum_{m,n>=1} h_mn sin(mx) sin(ny),
    lambda_mn = m^2 + n^2         (eigenvalues of -Laplacian),
    ||h||^2_{L2}  = (pi^2/4) * sum h_mn^2                 (Parseval),
    ||h||^2_{H10} = (pi^2/4) * sum (m^2 + n^2) h_mn^2.

Array conventions: grid values ``V[i, j] = h(x_i, y_j)`` with ``indexing="ij"``,
shape ``(N+1, N+1)`` including the boundary nodes. Coefficient arrays ``C[m-1, n-1]``
have shape ``(N-1, N-1)`` (modes m, n = 1..N-1 are the ones a grid with N
intervals can represent).

Discrete transform. On the uniform grid x_i = i*pi/N the trapezoidal rule for the
coefficient integral is

    h_mn ~ (4/pi^2) (pi/N)^2 sum_{i,j=1}^{N-1} V[i,j] sin(m i pi/N) sin(n j pi/N)
         = DST-I(V_interior)[m-1, n-1] / N^2,

because SciPy's (unnormalised) DST-I already contains a factor 2 per axis. The rule is
*exact* for every trigonometric polynomial with m, n < N (discrete orthogonality).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dstn

from .config import GridConfig

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Grid:
    """Uniform tensor grid on [0, L]^2 with ``n`` intervals in each direction."""

    n: int
    length: float = float(np.pi)

    @classmethod
    def from_config(cls, cfg: GridConfig) -> "Grid":
        return cls(n=cfg.n_intervals, length=cfg.length)

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("Grid needs at least 2 intervals")
        if not np.isclose(self.length, np.pi):
            # The sine basis sin(mx) is tied to the side length pi used by the paper.
            raise ValueError("This implementation follows the paper: the domain must be (0, pi)^2")

    @property
    def h(self) -> float:
        """Mesh width pi / N."""
        return self.length / self.n

    @property
    def x(self) -> FloatArray:
        """Nodes x_i = i*pi/N, i = 0..N (identical for y)."""
        return np.linspace(0.0, self.length, self.n + 1)

    def mesh(self) -> tuple[FloatArray, FloatArray]:
        """(X, Y) with ``indexing='ij'`` so that ``X[i, j] = x_i``."""
        return np.meshgrid(self.x, self.x, indexing="ij")

    @property
    def modes(self) -> NDArray[np.int64]:
        """Representable mode numbers 1..N-1."""
        return np.arange(1, self.n, dtype=np.int64)

    def mode_mesh(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """(Mx, Ny) integer mode numbers with ``Mx[m-1, n-1] = m``."""
        return np.meshgrid(self.modes, self.modes, indexing="ij")

    def eigenvalues(self) -> FloatArray:
        """lambda_mn = m^2 + n^2, shape (N-1, N-1)  [paper, p. 3]."""
        m, n = self.mode_mesh()
        return (m.astype(float) ** 2 + n.astype(float) ** 2)


# --------------------------------------------------------------------------- transforms
def sine_coefficients(values: FloatArray, grid: Grid) -> FloatArray:
    """Coefficients h_mn (paper normalisation 4/pi^2) of grid samples.

    Parameters
    ----------
    values:
        Array of shape ``(..., N+1, N+1)``; the last two axes are (x, y). Boundary
        rows/columns are ignored (sin vanishes there), so data that violate the
        Dirichlet condition, e.g. noisy boundary values, are handled consistently.
    """
    values = np.asarray(values, dtype=float)
    if values.shape[-2:] != (grid.n + 1, grid.n + 1):
        raise ValueError(f"expected trailing shape {(grid.n + 1, grid.n + 1)}, got {values.shape}")
    interior = values[..., 1:-1, 1:-1]
    return dstn(interior, type=1, axes=(-2, -1)) / grid.n**2


def synthesize(coeffs: FloatArray, grid: Grid) -> FloatArray:
    """Inverse transform: grid values of sum c_mn sin(mx) sin(ny), boundary = 0.

    Uses DST-I(c)/4 = sum_{m,n} c_mn sin(m x_i) sin(n y_j) at interior nodes.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.shape[-2:] != (grid.n - 1, grid.n - 1):
        raise ValueError(f"expected trailing shape {(grid.n - 1, grid.n - 1)}, got {coeffs.shape}")
    out = np.zeros(coeffs.shape[:-2] + (grid.n + 1, grid.n + 1))
    out[..., 1:-1, 1:-1] = dstn(coeffs, type=1, axes=(-2, -1)) / 4.0
    return out


def evaluate_series(coeffs: FloatArray, x: FloatArray, y: FloatArray) -> FloatArray:
    """Evaluate sum c_mn sin(mx) sin(ny) at arbitrary (broadcastable) points.

    Only non-zero coefficients are visited, so truncated series are cheap.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = np.broadcast_arrays(x, y)
    result = np.zeros(x.shape)
    for mi, ni in zip(*np.nonzero(coeffs)):
        result += coeffs[mi, ni] * np.sin((mi + 1) * x) * np.sin((ni + 1) * y)
    return result


# --------------------------------------------------------------------------- truncation
def truncation_mask(eigenvalues: FloatArray, cutoff: float) -> NDArray[np.bool_]:
    """Indicator of the disc m^2 + n^2 <= M (paper eq. (12))."""
    if not np.isfinite(cutoff):
        raise ValueError("cut-off M must be finite")
    return eigenvalues <= cutoff


def truncate(coeffs: FloatArray, eigenvalues: FloatArray, cutoff: float) -> FloatArray:
    """Projection Gamma_M of Theorem 4: zero all modes with m^2 + n^2 > M."""
    return np.where(truncation_mask(eigenvalues, cutoff), coeffs, 0.0)


# --------------------------------------------------------------------------- norms via Parseval
def l2_norm_from_coeffs(coeffs: FloatArray) -> float:
    """||h||_{L2(Omega)} = (pi/2) sqrt(sum h_mn^2)."""
    return float(np.pi / 2 * np.sqrt(np.sum(np.square(coeffs))))


def h10_norm_from_coeffs(coeffs: FloatArray, eigenvalues: FloatArray) -> float:
    """||h||_{H^1_0(Omega)} = (pi/2) sqrt(sum (m^2+n^2) h_mn^2)."""
    return float(np.pi / 2 * np.sqrt(np.sum(eigenvalues * np.square(coeffs))))
