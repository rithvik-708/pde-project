"""Error metrics.

Two kinds of norm are used:

* **Paper metric** ``paper_error`` -- root-mean-square over all (N+1)^2 grid nodes.
  This is the metric that reproduces Table 2 exactly [inferred, see paper_analysis.md A3]:
      E = sqrt( (1/(N+1)^2) sum_{i,j=0}^{N} |u_eps - u|^2 ).
* **Continuous L2(Omega)** norm, approximated by the 2-D trapezoidal rule (spectrally
  accurate for the smooth periodic-type integrands considered here).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .fourier import Grid

FloatArray = NDArray[np.float64]


def _check(a: FloatArray, b: FloatArray) -> tuple[FloatArray, FloatArray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return a, b


def absolute_error(exact: FloatArray, approx: FloatArray) -> FloatArray:
    """Pointwise |u_exact - u_approx|."""
    exact, approx = _check(exact, approx)
    return np.abs(exact - approx)


def trapezoid_weights(grid: Grid) -> FloatArray:
    """2-D trapezoidal weights on the (N+1) x (N+1) grid."""
    w = np.full(grid.n + 1, grid.h)
    w[[0, -1]] = grid.h / 2
    return np.outer(w, w)


def l2_norm_grid(values: FloatArray, grid: Grid) -> float:
    """||v||_{L2(Omega)} approximated by the trapezoidal rule."""
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.sum(trapezoid_weights(grid) * values**2)))


def l2_error(exact: FloatArray, approx: FloatArray, grid: Grid) -> float:
    """||u_exact - u_approx||_{L2(Omega)}."""
    exact, approx = _check(exact, approx)
    return l2_norm_grid(exact - approx, grid)


def relative_l2_error(exact: FloatArray, approx: FloatArray, grid: Grid) -> float:
    """||u_exact - u_approx||_{L2} / ||u_exact||_{L2}."""
    denom = l2_norm_grid(exact, grid)
    if denom == 0:
        raise ZeroDivisionError("exact solution has zero norm")
    return l2_error(exact, approx, grid) / denom


def mean_squared_error(exact: FloatArray, approx: FloatArray) -> float:
    """Mean over all grid nodes of |u_exact - u_approx|^2."""
    exact, approx = _check(exact, approx)
    return float(np.mean((exact - approx) ** 2))


def root_mean_squared_error(exact: FloatArray, approx: FloatArray) -> float:
    """sqrt(MSE)."""
    return float(np.sqrt(mean_squared_error(exact, approx)))


def paper_error(exact: FloatArray, approx: FloatArray) -> float:
    """Metric of Table 2: RMS over all (N+1)^2 nodes, boundary included (= RMSE)."""
    return root_mean_squared_error(exact, approx)


def max_absolute_error(exact: FloatArray, approx: FloatArray) -> float:
    """max |u_exact - u_approx|."""
    return float(np.max(absolute_error(exact, approx)))


def relative_reconstruction_error(exact: FloatArray, approx: FloatArray) -> float:
    """Discrete relative error ||e||_2 / ||u||_2 (Euclidean norms of the node vectors)."""
    exact, approx = _check(exact, approx)
    denom = np.linalg.norm(exact)
    if denom == 0:
        raise ZeroDivisionError("exact solution has zero norm")
    return float(np.linalg.norm(exact - approx) / denom)


def all_metrics(exact: FloatArray, approx: FloatArray, grid: Grid) -> dict[str, float]:
    """Every metric at once (for tables)."""
    return {
        "paper_rms": paper_error(exact, approx),
        "l2": l2_error(exact, approx, grid),
        "relative_l2": relative_l2_error(exact, approx, grid),
        "mse": mean_squared_error(exact, approx),
        "rmse": root_mean_squared_error(exact, approx),
        "max_abs": max_absolute_error(exact, approx),
        "relative_discrete": relative_reconstruction_error(exact, approx),
    }
