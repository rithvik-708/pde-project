"""Fourier-truncation regularisation of Nguyen (2015), Section 3.

Regularised solution, eq. (12):

    u^eps(x, y, t) = sum_{m,n>=1, m^2+n^2 <= M_eps} u^eps_mn(t) sin(mx) sin(ny),
    u^eps_mn(t)   = e^{(T-t)(m^2+n^2)} (g_eps)_mn - int_t^T e^{(s-t)(m^2+n^2)} (f_eps)_mn(s) ds.

The cut-off M_eps is chosen by one of the four rules used in Table 1:

    M1 (Remark 1)  : (1/T) ln( T / (eps + eps ln(T/eps)) )
    M2 (Theorem 5) : ln(1/eps) / (2T)
    M3 (Theorem 6) : (1/T) ln( T (ln(T/eps))^r / (eps + eps ln(T/eps)) )      (r = 2 inferred)
    M4 (Theorem 7) : sqrt( ln(1/eps) / (T + beta) )                            (beta = 1/2)
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .fourier import truncation_mask
from .source import SourceTerm, ZeroSource

FloatArray = NDArray[np.float64]

#: Largest exponent for which exp() is finite in float64 (ln(1.797e308) ~ 709.78).
MAX_EXPONENT: float = float(np.log(np.finfo(float).max))


# --------------------------------------------------------------------------- cut-off rules
def _check_eps_T(eps: float, final_time: float) -> None:
    if not (0 < eps < 1):
        raise ValueError("noise level eps must lie in (0, 1)")
    if final_time <= 0:
        raise ValueError("T must be positive")
    if eps >= final_time:
        raise ValueError("the rules require eps < T")


def cutoff_remark1(eps: float, final_time: float) -> float:
    """M1 = (1/T) ln( T / (eps + eps ln(T/eps)) )   [Remark 1, p. 7]."""
    _check_eps_T(eps, final_time)
    return math.log(final_time / (eps + eps * math.log(final_time / eps))) / final_time


def cutoff_theorem5(eps: float, final_time: float) -> float:
    """M2 = ln(1/eps) / (2T)   [Theorem 5, p. 11]."""
    _check_eps_T(eps, final_time)
    return math.log(1.0 / eps) / (2.0 * final_time)


def cutoff_theorem6(eps: float, final_time: float, r: float = 2.0) -> float:
    """M3 = (1/T) ln( T (ln(T/eps))^r / (eps + eps ln(T/eps)) ), r > 0   [Theorem 6, p. 14]."""
    _check_eps_T(eps, final_time)
    if r <= 0:
        raise ValueError("r must be positive")
    log_ratio = math.log(final_time / eps)
    return math.log(final_time * log_ratio**r / (eps + eps * log_ratio)) / final_time


def cutoff_theorem7(eps: float, final_time: float, beta: float = 0.5, use_sqrt: bool = True) -> float:
    """M4 = sqrt( ln(1/eps) / (T + beta) )   [Theorem 7, p. 16].

    The statement of Theorem 7 contains the square root (matches Table 1); its proof uses
    ln(1/eps)/(T+beta) without it. ``use_sqrt=False`` gives the proof's version.
    """
    _check_eps_T(eps, final_time)
    if beta <= 0:
        raise ValueError("beta must be positive")
    value = math.log(1.0 / eps) / (final_time + beta)
    return math.sqrt(value) if use_sqrt else value


def cutoff_rules(r: float = 2.0, beta: float = 0.5, m4_sqrt: bool = True) -> dict[str, Callable[[float, float], float]]:
    """The four rules of Table 1 as ``{name: M(eps, T)}``."""
    return {
        "M1": cutoff_remark1,
        "M2": cutoff_theorem5,
        "M3": lambda eps, T: cutoff_theorem6(eps, T, r=r),
        "M4": lambda eps, T: cutoff_theorem7(eps, T, beta=beta, use_sqrt=m4_sqrt),
    }


RULE_DESCRIPTIONS: dict[str, str] = {
    "M1": "Remark 1",
    "M2": "Theorem 5",
    "M3": "Theorem 6 (r=2)",
    "M4": "Theorem 7 (β=1/2)",
}


# --------------------------------------------------------------------------- the method
def stability_factor(t: float, final_time: float, cutoff: float) -> float:
    """Amplification bound e^{(T-t) M} of Theorem 3."""
    return math.exp((final_time - t) * cutoff)


def regularized_coefficients(
    g_coeffs: FloatArray,
    eigenvalues: FloatArray,
    t: float,
    final_time: float,
    cutoff: float,
    source: SourceTerm | None = None,
) -> FloatArray:
    """Coefficients u^eps_mn(t) of eq. (12); zero outside the disc m^2 + n^2 <= M.

    The exponential is evaluated **only** on the kept modes, so high modes never overflow.
    Raises ``OverflowError`` instead of silently returning inf/NaN if the requested cut-off
    is so large that e^{(T-t) M} is not representable.
    """
    if not (0.0 <= t <= final_time):
        raise ValueError(f"t must lie in [0, T]; got t={t}, T={final_time}")
    if g_coeffs.shape != eigenvalues.shape:
        raise ValueError("g_coeffs and eigenvalues must have the same shape")
    source = source or ZeroSource()

    # Step 1 -- choose the modes: m^2 + n^2 <= M_eps            (summation range in (12))
    mask = truncation_mask(eigenvalues, cutoff)
    if not mask.any():
        return np.zeros_like(g_coeffs)
    largest_exponent = (final_time - t) * float(eigenvalues[mask].max())
    if largest_exponent > MAX_EXPONENT:
        raise OverflowError(
            f"e^{{(T-t)*lambda}} = e^{largest_exponent:.1f} overflows float64; reduce the cut-off M"
        )

    coeffs = np.zeros_like(g_coeffs)
    # Step 2 -- backward propagation of the data: e^{(T-t) lambda} (g_eps)_mn
    coeffs[mask] = np.exp((final_time - t) * eigenvalues[mask]) * g_coeffs[mask]
    # Step 3 -- subtract the Duhamel term int_t^T e^{(s-t) lambda} (f_eps)_mn(s) ds
    coeffs -= source.backward_integral(eigenvalues, t, final_time, mask)
    return coeffs


# --------------------------------------------------------------------------- a-priori bounds
def theorem5_l2_bound(eps: float, final_time: float, h10_norm_u0: float) -> float:
    """Right-hand side of the first L2 estimate in Theorem 5 (t = 0)."""
    return (4 * math.sqrt(2) / (math.pi * final_time)) * eps**0.25 + (
        4 * math.sqrt(2) * final_time / math.pi**2
    ) * h10_norm_u0 / math.sqrt(math.log(1 / eps))


def theorem5_corrected_l2_bound(eps: float, final_time: float, h10_norm_u0: float) -> float:
    """Parseval-consistent version of the Theorem 5 estimate (t = 0, f exact).

    noise part      ||u^eps - Gamma_M u(0)|| <= e^{T M} ||g_eps - g|| = sqrt(eps)    (Theorem 3, M = M2)
    truncation part ||Gamma_M u(0) - u(0)||  <= ||u(0)||_{H^1_0} / sqrt(M)
                                             = sqrt(2T) ||u(0)||_{H^1_0} / sqrt(ln(1/eps)).
    The printed constants (4 sqrt2 / (pi T), 4 sqrt2 T / pi^2) are too small (paper_analysis.md A12).
    """
    return math.sqrt(eps) + math.sqrt(2 * final_time) * h10_norm_u0 / math.sqrt(math.log(1 / eps))


def theorem7_bound(eps: float, t: float, final_time: float, beta: float, a2: float) -> float:
    """Right-hand side of (19): eps^{beta/(T+beta)} (A2 + 2 eps^{t/(T+beta)})."""
    return eps ** (beta / (final_time + beta)) * (a2 + 2 * eps ** (t / (final_time + beta)))
