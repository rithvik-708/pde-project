"""Error metrics and the noise model."""

import numpy as np
import pytest

from backward_heat import add_paper_noise, make_rng
from backward_heat.metrics import (
    absolute_error, all_metrics, l2_error, l2_norm_grid, max_absolute_error, mean_squared_error,
    paper_error, relative_l2_error, relative_reconstruction_error, root_mean_squared_error,
)


def test_trapezoid_l2_norm_is_exact_for_sine_modes(grid):
    X, Y = grid.mesh()
    # ||sin(mx) sin(ny)||_{L2(Omega)} = pi/2
    assert np.isclose(l2_norm_grid(np.sin(4 * X) * np.sin(9 * Y), grid), np.pi / 2, rtol=1e-12)


def test_paper_metric_reproduces_published_baseline(example, grid):
    """u^eps = 0 gives 0.5161 at t = 0 (Table 2) -- the inferred metric."""
    assert round(paper_error(example.u_grid(grid, 0.0), np.zeros((129, 129))), 4) == 0.5161


def test_continuous_l2_differs_from_paper_metric(example, grid):
    """The true L2 norm of u(0) is 1.634, not 0.5161 (paper_analysis.md A3)."""
    k = np.arange(1, 16)
    assert l2_norm_grid(example.u_grid(grid, 0.0), grid) == pytest.approx(np.pi / 2 * np.sqrt(np.sum(k**-4.0)))


def test_metric_definitions():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[1.0, 1.0], [3.0, 6.0]])
    assert mean_squared_error(a, b) == pytest.approx(5 / 4)
    assert root_mean_squared_error(a, b) == pytest.approx(np.sqrt(5 / 4))
    assert max_absolute_error(a, b) == 2.0
    np.testing.assert_array_equal(absolute_error(a, b), [[0, 1], [0, 2]])
    assert relative_reconstruction_error(a, b) == pytest.approx(np.sqrt(5) / np.sqrt(30))


def test_relative_l2_and_zero_error(example, grid):
    U = example.u_grid(grid, 0.0)
    assert l2_error(U, U, grid) == 0.0
    assert relative_l2_error(U, 0.9 * U, grid) == pytest.approx(0.1)
    assert all(v == 0 for v in all_metrics(U, U, grid).values())


def test_shape_mismatch_and_zero_division(grid):
    with pytest.raises(ValueError):
        paper_error(np.zeros((3, 3)), np.zeros((4, 4)))
    with pytest.raises(ZeroDivisionError):
        relative_l2_error(np.zeros((129, 129)), np.ones((129, 129)), grid)


# ------------------------------------------------------------------ noise model
def test_noise_bound_and_reproducibility(example, grid):
    g = example.g_grid(grid)
    for eps in (1e-1, 1e-3, 1e-7):
        n1 = add_paper_noise(g, grid, eps, make_rng(5))
        n2 = add_paper_noise(g, grid, eps, make_rng(5))
        np.testing.assert_array_equal(n1.values, n2.values)
        assert n1.max_abs <= eps / np.pi
        assert n1.l2_norm <= eps  # ||g_eps - g||_{L2} <= eps (paper, p. 18)
        # iid U(-1, 1) has RMS 1/sqrt(3), so ||.||_{L2} ~ (eps/pi) * pi / sqrt(3)
        assert n1.l2_norm == pytest.approx(eps / np.sqrt(3), rel=0.03)


def test_zero_noise_is_exact(example, grid):
    g = example.g_grid(grid)
    nd = add_paper_noise(g, grid, 0.0, make_rng(0))
    np.testing.assert_array_equal(nd.values, g)
    assert nd.l2_norm == 0.0


def test_negative_noise_rejected(example, grid):
    with pytest.raises(ValueError):
        add_paper_noise(example.g_grid(grid), grid, -1.0, make_rng(0))
