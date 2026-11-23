"""Fourier-truncation regularisation (eq. 12), cut-off rules (Table 1) and Table 2."""

import math

import numpy as np
import pytest

from backward_heat import GridSource, add_paper_noise, make_rng, reconstruct
from backward_heat.config import PAPER_TABLE1, PAPER_TABLE2
from backward_heat.metrics import l2_norm_grid, paper_error
from backward_heat.fourier import h10_norm_from_coeffs
from backward_heat.regularization import (
    cutoff_rules, cutoff_theorem5, cutoff_theorem7, stability_factor, theorem5_corrected_l2_bound,
    theorem5_l2_bound,
)


# --------------------------------------------------------------------- Table 1
@pytest.mark.parametrize("rule", ["M1", "M2", "M3", "M4"])
@pytest.mark.parametrize("eps", [1e-3, 1e-5, 1e-7])
def test_table1_regression(rule, eps):
    """All 12 published cut-offs, to the printed precision (4 significant digits)."""
    value = cutoff_rules()[rule](eps, 1.0)
    published = PAPER_TABLE1[rule][eps]
    half_ulp = 0.5 * 10 ** (math.floor(math.log10(published)) - 3)
    assert abs(value - published) <= half_ulp + 1e-12


def test_m4_proof_variant_does_not_match_table1():
    """A2: without the square root, M4(1e-3) = 4.605, not the published 2.146."""
    assert abs(cutoff_theorem7(1e-3, 1.0, use_sqrt=False) - 2.146) > 1


def test_cutoffs_grow_as_noise_vanishes():
    for rule in cutoff_rules().values():
        vals = [rule(eps, 1.0) for eps in (1e-2, 1e-4, 1e-8, 1e-12)]
        assert np.all(np.diff(vals) > 0)


@pytest.mark.parametrize("eps", [0.0, 1.0, -1e-3])
def test_invalid_eps(eps):
    with pytest.raises(ValueError):
        cutoff_rules()["M1"](eps, 1.0)


# --------------------------------------------------------------------- the method
def test_noiseless_exact_recovery_of_kept_modes(example, grid, exact_source):
    """With exact data and M = 20 the kept active modes (k = 1, 2) are recovered."""
    res = reconstruct(example.g_grid(grid), 0.0, 20.0, grid, 1.0, exact_source)
    c_exact = example.u_coefficients(grid, 0.0)
    assert abs(res.coefficients[1, 0] - c_exact[1, 0]) < 1e-12
    # lambda = 20: rounding of g_mn (~1e-17) and of the Duhamel term is amplified by e^{20} ~ 5e8,
    # so ~1e-8 is the best float64 can do -- ill-posedness even without noise.
    assert abs(res.coefficients[3, 1] - c_exact[3, 1]) < 1e-7
    # inactive kept modes stay ~0 (only rounding times e^{lambda})
    inactive = np.ones_like(c_exact, dtype=bool)
    inactive[[1, 3], [0, 1]] = False
    assert np.max(np.abs(res.coefficients[inactive & (grid.eigenvalues() <= 20)])) < 1e-6


def test_truncation_zeroes_high_modes(example, grid, exact_source):
    res = reconstruct(example.g_grid(grid), 0.3, 13.87, grid, 1.0, exact_source)
    assert np.all(res.coefficients[grid.eigenvalues() > 13.87] == 0)
    assert res.n_modes == int(np.sum(grid.eigenvalues() <= 13.87))


def test_reconstruction_at_final_time_returns_data(example, grid, exact_source):
    res = reconstruct(example.g_grid(grid), 1.0, 2000.0, grid, 1.0, exact_source)
    np.testing.assert_allclose(res.u, example.g_grid(grid), atol=1e-13)


def test_no_silent_overflow(example, grid, exact_source):
    with pytest.raises(OverflowError):
        reconstruct(example.g_grid(grid), 0.0, 5000.0, grid, 1.0, exact_source)


def test_result_is_finite(example, grid, exact_source):
    noisy = add_paper_noise(example.g_grid(grid), grid, 1e-3, make_rng(1))
    for M in (0.5, 4.84, 50.0, 300.0):
        assert np.all(np.isfinite(reconstruct(noisy.values, 0.0, M, grid, 1.0, exact_source).u))


def test_theorem3_stability_estimate(example, grid, exact_source):
    """||u1 - u2|| <= e^{(T-t)M} ||g1 - g2||   (Theorem 3 with f1 = f2)."""
    rng = make_rng(7)
    g = example.g_grid(grid)
    for M, t in [(5.0, 0.0), (13.87, 0.0), (40.0, 0.5)]:
        g1 = add_paper_noise(g, grid, 1e-3, rng).values
        g2 = add_paper_noise(g, grid, 1e-3, rng).values
        u1 = reconstruct(g1, t, M, grid, 1.0, exact_source).u
        u2 = reconstruct(g2, t, M, grid, 1.0, exact_source).u
        assert l2_norm_grid(u1 - u2, grid) <= stability_factor(t, 1.0, M) * l2_norm_grid(g1 - g2, grid)


def test_noise_amplification_matches_exponential(grid):
    """A single-mode perturbation delta*sin(3x)sin(2y) is amplified by exactly e^{(T-t)*13}."""
    X, Y = grid.mesh()
    delta = 1e-6
    res = reconstruct(delta * np.sin(3 * X) * np.sin(2 * Y), 0.0, 13.0, grid, 1.0, None)
    assert np.isclose(res.coefficients[2, 1], delta * np.exp(13.0), rtol=1e-12)


def test_general_grid_source_matches_closed_form(example, grid, exact_source):
    gsrc = GridSource(lambda s: example.f_grid(grid, s), grid)
    for t, M in [(0.0, 18.84), (0.4, 30.0)]:
        a = reconstruct(example.g_grid(grid), t, M, grid, 1.0, exact_source).u
        b = reconstruct(example.g_grid(grid), t, M, grid, 1.0, gsrc).u
        assert np.max(np.abs(a - b)) < 1e-8


def test_invalid_time(example, grid):
    with pytest.raises(ValueError):
        reconstruct(example.g_grid(grid), 1.5, 5.0, grid, 1.0)


# --------------------------------------------------------------------- Table 2
def test_table2_noiseless_closed_form(example, grid, exact_source):
    """Without noise, the paper metric equals (64/129) e^{-t} sqrt(sum_{k not kept} k^-4)."""
    k = np.arange(1, 16)
    for M, kept in [(4.84, 0), (8.705, 1)]:
        for t in (0.0, 0.5, 0.9):
            res = reconstruct(example.g_grid(grid), t, M, grid, 1.0, exact_source)
            predicted = 64 / 129 * np.exp(-t) * np.sqrt(np.sum(k[kept:] ** -4.0))
            assert abs(paper_error(example.u_grid(grid, t), res.u) - predicted) < 1e-12


def test_table2_with_paper_noise(example, grid, exact_source):
    """Every Table 2 entry (A6 typo corrected) is matched to within 1 % using the paper's noise."""
    rules = cutoff_rules()
    rng = make_rng(2015)
    for eps in (1e-3, 1e-5, 1e-7):
        g_eps = add_paper_noise(example.g_grid(grid), grid, eps, rng).values
        for rule, fn in rules.items():
            M = fn(eps, 1.0)
            for t in (0.0, 0.1, 0.5, 0.9):
                published = PAPER_TABLE2[(eps, rule)][t]
                if (eps, rule, t) == (1e-5, "M3", 0.1):
                    published *= 10  # A6: printed 1.287E-02 is a typo for 1.287E-01
                u_eps = reconstruct(g_eps, t, M, grid, 1.0, exact_source).u
                err = paper_error(example.u_grid(grid, t), u_eps)
                assert abs(err - published) / published < 0.01, (eps, rule, t, err, published)


def test_theorem5_bounds(example, grid, exact_source):
    """Printed Theorem 5 bound is violated (A12); the Parseval-corrected bound holds."""
    from backward_heat.metrics import l2_error
    h10 = h10_norm_from_coeffs(example.u_coefficients(grid, 0.0), grid.eigenvalues())
    exact = example.u_grid(grid, 0.0)
    violated = False
    for k, eps in enumerate((1e-2, 1e-3, 1e-4, 1e-6, 1e-9, 1e-12)):
        g_eps = add_paper_noise(example.g_grid(grid), grid, eps, make_rng(k)).values
        u_eps = reconstruct(g_eps, 0.0, cutoff_theorem5(eps, 1.0), grid, 1.0, exact_source).u
        err = l2_error(exact, u_eps, grid)
        assert err <= theorem5_corrected_l2_bound(eps, 1.0, h10)
        violated |= err > theorem5_l2_bound(eps, 1.0, h10)
    assert violated  # e.g. eps = 1e-3: error 1.634 > printed bound 1.283
