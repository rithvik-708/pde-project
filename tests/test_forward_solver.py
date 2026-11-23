"""Forward evolution: spectral (exact) solver and Crank-Nicolson cross-check."""

import numpy as np
import pytest
from scipy.integrate import quad

from backward_heat import GridSource, SeparableExponentialSource
from backward_heat.forward_solver import crank_nicolson, spectral_forward, spectral_forward_coefficients
from backward_heat.fourier import Grid
from backward_heat.metrics import paper_error
from backward_heat.source import phi1, psi


def test_spectral_forward_reproduces_final_data(example, grid, exact_source):
    """u(., ., 0) --(eq. 11)--> u(., ., T) must equal g to rounding."""
    gT = spectral_forward(example.u_grid(grid, 0.0), grid, 0.0, 1.0, exact_source)
    assert np.max(np.abs(gT - example.g_grid(grid))) < 1e-13


def test_spectral_forward_semigroup(example, grid, exact_source):
    lam = grid.eigenvalues()
    c0 = example.u_coefficients(grid, 0.0)
    direct = spectral_forward_coefficients(c0, lam, 0.0, 0.8, exact_source)
    two_step = spectral_forward_coefficients(
        spectral_forward_coefficients(c0, lam, 0.0, 0.3, exact_source), lam, 0.3, 0.8, exact_source
    )
    np.testing.assert_allclose(direct, two_step, atol=1e-15)
    np.testing.assert_allclose(direct, example.u_coefficients(grid, 0.8), atol=1e-15)


def test_homogeneous_decay_rate(grid):
    X, Y = grid.mesh()
    u0 = np.sin(2 * X) * np.sin(3 * Y)
    u1 = spectral_forward(u0, grid, 0.0, 0.1, None)
    np.testing.assert_allclose(u1, np.exp(-13 * 0.1) * u0, atol=1e-14)


def test_grid_source_forward_is_second_order(example):
    """Filon-linear quadrature of the Duhamel integral converges at O(h^2), uniformly in lambda."""
    grid = Grid(64)
    lam = grid.eigenvalues()
    exact = SeparableExponentialSource(example.f_spatial_coefficients(grid)).forward_integral(lam, 0.0, 1.0)
    errs = []
    for n in (25, 50, 100):
        approx = GridSource(lambda s: example.f_grid(grid, s), grid, n_panels=n).forward_integral(lam, 0.0, 1.0)
        errs.append(np.max(np.abs(approx - exact)))
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(rates > 1.9), rates


def test_crank_nicolson_converges_to_analytical(example):
    """FD solution at T = 1 approaches g at second order in h (dt ~ h)."""
    errs = []
    for n in (32, 64):
        grid = Grid(n)
        uT = crank_nicolson(
            example.u_grid(grid, 0.0), grid, 1.0, n_steps=2 * n,
            f_on_grid=lambda s, g=grid: example.f_grid(g, s),
        )
        errs.append(paper_error(example.g_grid(grid), uT))
    assert errs[1] < errs[0] / 3.0, errs


def test_phi_functions():
    z = np.array([-50.0, -1.0, -1e-9, 0.0, 1e-9, 0.3, 2.0, 30.0])
    for zi, p1, p2 in zip(z, phi1(z), psi(z)):
        assert np.isclose(p1, quad(lambda s: np.exp(zi * s), 0, 1)[0], rtol=1e-10)
        assert np.isclose(p2, quad(lambda s: s * np.exp(zi * s), 0, 1)[0], rtol=1e-10)


def test_forward_rejects_backward_time(example, grid):
    with pytest.raises(ValueError):
        spectral_forward(example.u_grid(grid, 0.0), grid, 1.0, 0.5)
