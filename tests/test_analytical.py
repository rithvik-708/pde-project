"""The benchmark of Section 4 satisfies problem (24) exactly."""

import numpy as np
import pytest
import sympy as sp

from backward_heat import PaperExample


def test_symbolic_pde_boundary_and_final_condition():
    """SymPy: u_t - (u_xx + u_yy) = f, u = 0 on the boundary, u(.,.,1) = g."""
    x, y, t = sp.symbols("x y t", real=True)
    K = 15
    u = sp.exp(-t) * sum(sp.sin(2 * k * x) * sp.sin(k * y) / k**2 for k in range(1, K + 1))
    f = sp.exp(-t) * sum((5 - sp.Rational(1, k**2)) * sp.sin(2 * k * x) * sp.sin(k * y) for k in range(1, K + 1))
    g = sp.exp(-1) * sum(sp.sin(2 * k * x) * sp.sin(k * y) / k**2 for k in range(1, K + 1))
    assert sp.simplify(sp.diff(u, t) - sp.diff(u, x, 2) - sp.diff(u, y, 2) - f) == 0
    assert sp.simplify(u.subs(t, 1) - g) == 0
    for bc in (u.subs(x, 0), u.subs(x, sp.pi), u.subs(y, 0), u.subs(y, sp.pi)):
        assert sp.simplify(bc) == 0


def test_numeric_pde_residual(example, grid):
    X, Y = grid.mesh()
    for t in (0.0, 0.37, 1.0):
        assert np.max(np.abs(example.pde_residual(X, Y, t))) < 1e-11


def test_derivatives_against_finite_differences(example):
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0.2, 2.9, 20), rng.uniform(0.2, 2.9, 20)
    t, h = 0.3, 1e-4
    fd_t = (example.u(x, y, t + h) - example.u(x, y, t - h)) / (2 * h)
    fd_xx = (example.u(x + h, y, t) - 2 * example.u(x, y, t) + example.u(x - h, y, t)) / h**2
    fd_yy = (example.u(x, y + h, t) - 2 * example.u(x, y, t) + example.u(x, y - h, t)) / h**2
    np.testing.assert_allclose(example.u_t(x, y, t), fd_t, atol=1e-7)
    np.testing.assert_allclose(example.u_xx(x, y, t), fd_xx, atol=5e-4)
    np.testing.assert_allclose(example.u_yy(x, y, t), fd_yy, atol=5e-4)


def test_dirichlet_boundary(example, grid):
    U = example.u_grid(grid, 0.0)
    for edge in (U[0], U[-1], U[:, 0], U[:, -1]):
        assert np.max(np.abs(edge)) < 1e-13


def test_final_condition(example, grid):
    np.testing.assert_allclose(example.u_grid(grid, 1.0), example.g_grid(grid), atol=1e-15)


def test_modes_and_eigenvalues(example):
    np.testing.assert_array_equal(example.mode_m, 2 * np.arange(1, 16))
    np.testing.assert_allclose(example.mode_eigenvalues, 5 * np.arange(1, 16) ** 2)


def test_vectorised_shapes(example):
    x = np.linspace(0, np.pi, 7)
    assert example.u(x[:, None], x[None, :], 0.5).shape == (7, 7)


@pytest.mark.parametrize("bad", [dict(final_time=0.0), dict(n_terms=0)])
def test_invalid_parameters(bad):
    with pytest.raises(ValueError):
        PaperExample(**bad)
