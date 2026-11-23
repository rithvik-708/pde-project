"""Measurement-noise model of the paper (Section 4, p. 18).

    g_eps(x, y) = g(x, y) + (eps / pi) * rand(),     rand() in (-1, 1).

Since |g_eps - g| <= eps/pi pointwise and |Omega| = pi^2, ||g_eps - g||_{L2} <= eps.

Implementation choices (not specified in the paper):
* ``rand()`` is an independent U(-1, 1) draw at every grid node (boundary included).
* A NumPy ``Generator`` with a fixed seed makes every run reproducible.
* Only the final data g are perturbed; the source f is taken exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .fourier import Grid
from .metrics import l2_norm_grid

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NoisyData:
    """Noisy observation together with diagnostics of the perturbation."""

    values: FloatArray
    perturbation: FloatArray
    noise_level: float  # the nominal eps
    l2_norm: float  # actual ||g_eps - g||_{L2}, trapezoidal rule
    max_abs: float  # actual max |g_eps - g|  (<= eps/pi)


def make_rng(seed: int | None) -> np.random.Generator:
    """Seeded generator (``None`` gives fresh entropy)."""
    return np.random.default_rng(seed)


def uniform_noise(shape: tuple[int, ...], noise_level: float, rng: np.random.Generator) -> FloatArray:
    """(eps/pi) * U(-1, 1) samples of the given shape."""
    if noise_level < 0:
        raise ValueError("noise level eps must be non-negative")
    return (noise_level / np.pi) * rng.uniform(-1.0, 1.0, size=shape)


def add_paper_noise(clean: FloatArray, grid: Grid, noise_level: float, rng: np.random.Generator) -> NoisyData:
    """Return g_eps = g + (eps/pi) rand() on the grid, plus the actual noise magnitude."""
    clean = np.asarray(clean, dtype=float)
    if noise_level == 0.0:
        perturbation = np.zeros_like(clean)
    else:
        perturbation = uniform_noise(clean.shape, noise_level, rng)
    return NoisyData(
        values=clean + perturbation,
        perturbation=perturbation,
        noise_level=noise_level,
        l2_norm=l2_norm_grid(perturbation, grid),
        max_abs=float(np.max(np.abs(perturbation))),
    )
