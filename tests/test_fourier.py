"""Sine basis, normalisation (4/pi^2), DST-I transforms, truncation and Parseval."""

import numpy as np
import pytest
from scipy import integrate

from backward_heat.fourier import (
    Grid, evaluate_series, h10_norm_from_coeffs, l2_norm_from_coeffs, sine_coefficients,
    synthesize, truncate, truncation_mask,
)


def test_eigenvalues(grid):
    lam = grid.eigenvalues()
    assert lam.shape == (127, 127)
    assert lam[0, 0] == 2 and lam[1, 0] == 5 and lam[3, 1] == 20  # (1,1), (2,1), (4,2)


def test_single_mode_has_unit_coefficient(grid):
    X, Y = grid.mesh()
    C = sine_coefficients(np.sin(3 * X) * np.sin(7 * Y), grid)
    expected = np.zeros_like(C)
    expected[2, 6] = 1.0
    np.testing.assert_allclose(C, expected, atol=1e-13)


def test_normalisation_matches_quadrature():
    """(4/pi^2) int int h sin(mx) sin(ny) for a non-polynomial h, vs scipy dblquad."""
    grid = Grid(256)
    h = lambda x, y: x * (np.pi - x) * y**2 * (np.pi - y)  # noqa: E731
    X, Y = grid.mesh()
    C = sine_coefficients(h(X, Y), grid)
    for m, n in [(1, 1), (2, 3), (5, 2)]:
        ref = 4 / np.pi**2 * integrate.dblquad(
            lambda yy, xx: h(xx, yy) * np.sin(m * xx) * np.sin(n * yy), 0, np.pi, 0, np.pi
        )[0]
        assert abs(C[m - 1, n - 1] - ref) < 1e-4 * max(1, abs(ref))


def test_round_trip(grid, rng):
    C = rng.normal(size=(127, 127)) / (1 + grid.eigenvalues())
    np.testing.assert_allclose(sine_coefficients(synthesize(C, grid), grid), C, atol=1e-13)


def test_reconstruction_of_benchmark(example, grid):
    U = example.u_grid(grid, 0.2)
    C = sine_coefficients(U, grid)
    np.testing.assert_allclose(C, example.u_coefficients(grid, 0.2), atol=1e-14)
    np.testing.assert_allclose(synthesize(C, grid), U, atol=1e-13)


def test_evaluate_series_off_grid(example, grid):
    C = example.u_coefficients(grid, 0.0)
    x = np.array([0.1, 1.234, 2.9])
    y = np.array([0.7, 0.3, 3.0])
    np.testing.assert_allclose(evaluate_series(C, x, y), example.u(x, y, 0.0), atol=1e-13)


def test_parseval_norms(example, grid):
    C = example.u_coefficients(grid, 0.0)
    k = np.arange(1, 16)
    assert np.isclose(l2_norm_from_coeffs(C), np.pi / 2 * np.sqrt(np.sum(k**-4.0)))
    assert np.isclose(h10_norm_from_coeffs(C, grid.eigenvalues()), np.pi / 2 * np.sqrt(np.sum(5 * k**2 * k**-4.0)))


def test_truncation_projection_is_idempotent_and_shrinks(example, grid):
    lam = grid.eigenvalues()
    C = example.u_coefficients(grid, 0.0)
    for M in (4.9, 5.0, 20.0, 100.0):
        P = truncate(C, lam, M)
        np.testing.assert_array_equal(truncate(P, lam, M), P)
        assert l2_norm_from_coeffs(P) <= l2_norm_from_coeffs(C) + 1e-15
    assert truncation_mask(lam, 4.99).sum() == 1  # only (1,1): lambda = 2
    assert truncation_mask(lam, 5.0).sum() == 3  # + (1,2), (2,1)
    assert truncation_mask(lam, 8.0).sum() == 4  # + (2,2)


def test_theorem4_projection_bounds(example, grid):
    """Theorem 4 (ii)-(iii), with the constants that follow from Parseval.

    The paper prints ||Gamma_M h - h|| <= 4/(pi^2 sqrt M) ||h||_H10. That constant is too small
    (see the next test and paper_analysis.md A12). Parseval gives
    ||Gamma_M h - h|| <= ||h||_H10 / sqrt(M) and <= ||Delta h|| / M; both are verified here.
    """
    lam = grid.eigenvalues()
    C = example.u_coefficients(grid, 0.0)
    h10 = h10_norm_from_coeffs(C, lam)
    lap = l2_norm_from_coeffs(lam * C)
    for M in (3.0, 10.0, 50.0, 300.0):
        err = l2_norm_from_coeffs(C - truncate(C, lam, M))
        assert err <= h10 / np.sqrt(M) + 1e-15
        assert err <= lap / M + 1e-15
    # convergence Gamma_M h -> h (Theorem 4 (i))
    assert l2_norm_from_coeffs(C - truncate(C, lam, 1125.0)) == 0.0


def test_paper_constant_in_theorem4_is_too_small(example, grid):
    """Documents discrepancy A12: the printed constant 4/(pi^2 sqrt M) is violated at M = 3."""
    lam = grid.eigenvalues()
    C = example.u_coefficients(grid, 0.0)
    err = l2_norm_from_coeffs(C - truncate(C, lam, 3.0))
    assert err > 4 / (np.pi**2 * np.sqrt(3.0)) * h10_norm_from_coeffs(C, lam)
