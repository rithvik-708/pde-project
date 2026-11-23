"""Source-term (Duhamel) integrals appearing in eqs. (6), (11) and (12) of the paper.

Backward (regularised formula (12)):

    B_mn(t) = int_t^T e^{(s-t) lam_mn} f_mn(s) ds      -- only for kept modes lam <= M

Forward (solution of the direct problem, eq. (11)):

    F_mn(t0, t) = int_{t0}^{t} e^{-(t-s) lam_mn} f_mn(s) ds

Two implementations are provided:

* :class:`SeparableExponentialSource` -- closed form for f_mn(s) = e^{-a s} F_mn
  (this is the paper's test problem with a = 1). Exact to rounding.
* :class:`GridSource` -- general source given only through grid samples
  f(x_i, y_j, s). The coefficients f_mn(s) are computed with the DST at uniformly
  spaced times. The time integral uses a Filon-type rule: f_mn is interpolated
  linearly on each panel, and the product with the exponential is integrated *exactly*.
  The rule is second-order accurate *uniformly in lambda*, which plain Gauss
  quadrature is not for very stiff modes. It is an implementation choice (the paper does not discuss
  quadrature). It is cross-validated against the closed form in the tests and in Experiment 1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .fourier import Grid, sine_coefficients

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


# --------------------------------------------------------------------------- phi functions
def phi1(z: FloatArray) -> FloatArray:
    """phi1(z) = int_0^1 e^{z tau} d tau = (e^z - 1)/z, phi1(0) = 1 (stable via expm1)."""
    z = np.asarray(z, dtype=float)
    out = np.ones_like(z)
    nz = z != 0
    out[nz] = np.expm1(z[nz]) / z[nz]
    return out


def psi(z: FloatArray) -> FloatArray:
    """psi(z) = int_0^1 tau e^{z tau} d tau = (e^z (z-1) + 1)/z^2, psi(0) = 1/2.

    For |z| < 0.5 the power series sum z^n / (n! (n+2)) is used to avoid cancellation.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    small = np.abs(z) < 0.5
    zs = z[small]
    series = np.zeros_like(zs)
    term = np.ones_like(zs)
    for n in range(16):
        if n > 0:
            term = term * zs / n
        series += term / (n + 2)
    out[small] = series
    zl = z[~small]
    out[~small] = (np.exp(zl) * (zl - 1.0) + 1.0) / zl**2
    return out


# --------------------------------------------------------------------------- interface
class SourceTerm(ABC):
    """Abstract source f(x, y, t) seen through its sine coefficients."""

    @abstractmethod
    def backward_integral(self, eigenvalues: FloatArray, t: float, final_time: float, mask: BoolArray) -> FloatArray:
        """int_t^T e^{(s-t) lam} f_mn(s) ds on ``mask`` (zero elsewhere)."""

    @abstractmethod
    def forward_integral(self, eigenvalues: FloatArray, t0: float, t: float) -> FloatArray:
        """int_{t0}^{t} e^{-(t-s) lam} f_mn(s) ds for all modes."""


class ZeroSource(SourceTerm):
    """Homogeneous equation, f = 0."""

    def backward_integral(self, eigenvalues: FloatArray, t: float, final_time: float, mask: BoolArray) -> FloatArray:
        return np.zeros_like(eigenvalues)

    def forward_integral(self, eigenvalues: FloatArray, t0: float, t: float) -> FloatArray:
        return np.zeros_like(eigenvalues)


@dataclass
class SeparableExponentialSource(SourceTerm):
    """f_mn(s) = e^{-rate*s} F_mn (closed-form Duhamel integrals)."""

    spatial_coefficients: FloatArray
    rate: float = 1.0

    def backward_integral(self, eigenvalues: FloatArray, t: float, final_time: float, mask: BoolArray) -> FloatArray:
        # int_t^T e^{lam(s-t)} e^{-a s} ds = e^{-a t} (T-t) phi1((lam-a)(T-t))
        _check_interval(t, final_time)
        out = np.zeros_like(eigenvalues)
        dt = final_time - t
        lam = eigenvalues[mask]
        out[mask] = np.exp(-self.rate * t) * dt * phi1((lam - self.rate) * dt) * self.spatial_coefficients[mask]
        return out

    def forward_integral(self, eigenvalues: FloatArray, t0: float, t: float) -> FloatArray:
        # int_{t0}^t e^{-lam(t-s)} e^{-a s} ds = e^{-a t} (t-t0) phi1(-(lam-a)(t-t0))
        _check_interval(t0, t)
        dt = t - t0
        return np.exp(-self.rate * t) * dt * phi1(-(eigenvalues - self.rate) * dt) * self.spatial_coefficients


@dataclass
class GridSource(SourceTerm):
    """General source known only through grid samples ``f_on_grid(s) -> (N+1, N+1)``."""

    f_on_grid: Callable[[float], FloatArray]
    grid: Grid
    n_panels: int = 200  # forward (Filon) panels
    n_gauss: int = 48  # backward Gauss-Legendre nodes
    _cache: dict[tuple, FloatArray] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.n_panels < 1 or self.n_gauss < 1:
            raise ValueError("n_panels and n_gauss must be >= 1")

    def _filon_samples(self, a: float, b: float) -> FloatArray:
        """f_mn(s_j) at s_j = a + j (b-a)/n, shape (n+1, N-1, N-1)."""
        key = ("filon", a, b, self.n_panels)
        if key not in self._cache:
            times = np.linspace(a, b, self.n_panels + 1)
            values = np.stack([self.f_on_grid(float(s)) for s in times])
            self._cache[key] = sine_coefficients(values, self.grid)
        return self._cache[key]

    def _gauss_samples(self, a: float, b: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Gauss-Legendre nodes/weights on [a, b] and f_mn at the nodes."""
        key = ("gauss", a, b, self.n_gauss)
        xi, wi = np.polynomial.legendre.leggauss(self.n_gauss)
        nodes = a + (b - a) * (xi + 1.0) / 2.0
        if key not in self._cache:
            values = np.stack([self.f_on_grid(float(s)) for s in nodes])
            self._cache[key] = sine_coefficients(values, self.grid)
        return nodes, (b - a) / 2.0 * wi, self._cache[key]

    def backward_integral(self, eigenvalues: FloatArray, t: float, final_time: float, mask: BoolArray) -> FloatArray:
        # Gauss-Legendre (spectral accuracy). Any *relative* quadrature error is multiplied
        # by e^{(T-t) lambda} and must cancel against e^{(T-t) lambda} g_mn, so a low-order
        # rule (e.g. Filon-linear, O(h^2)) would act like a large data error -- exactly the
        # instability the paper analyses (Theorem 3 also covers perturbations of f).
        _check_interval(t, final_time)
        out = np.zeros_like(eigenvalues)
        if final_time == t or not mask.any():
            return out
        nodes, weights, fq = self._gauss_samples(t, final_time)
        lam = eigenvalues[mask]
        kernel = np.exp(np.outer(nodes - t, lam))  # (q, K), bounded by e^{(T-t) M}
        out[mask] = np.sum(weights[:, None] * kernel * fq[:, mask], axis=0)
        return out

    def forward_integral(self, eigenvalues: FloatArray, t0: float, t: float) -> FloatArray:
        _check_interval(t0, t)
        if t == t0:
            return np.zeros_like(eigenvalues)
        h = (t - t0) / self.n_panels
        fj = self._filon_samples(t0, t)  # (n+1, N-1, N-1)
        z = -eigenvalues * h
        p1, p2 = phi1(z), psi(z)
        # Right-endpoint form, decaying exponentials only (no overflow for stiff modes):
        # panel j: e^{-lam (t - s_{j+1})} * h * [f_{j+1} phi1(-lam h) + (f_j - f_{j+1}) psi(-lam h)]
        total = np.zeros_like(eigenvalues)
        for j in range(self.n_panels):
            decay = np.exp(-eigenvalues * (t - (t0 + (j + 1) * h)))
            total += h * decay * (fj[j + 1] * p1 + (fj[j] - fj[j + 1]) * p2)
        return total


def _check_interval(a: float, b: float) -> None:
    if b < a:
        raise ValueError(f"invalid time interval [{a}, {b}]")
