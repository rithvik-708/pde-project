# Implementation Notes

## Module map ↔ paper

| Module | Responsibility | Paper reference |
|---|---|---|
| `config.py` | all parameters, published Table 1/2 values, result paths | Section 4 |
| `fourier.py` | grid, eigenvalues `m²+n²`, DST-I with the `4/π²` normalisation, truncation `Γ_M`, Parseval norms | p. 3–4, Thm 4 |
| `analytical.py` | exact `u, u_t, u_xx, u_yy, f, g` and exact sine coefficients of problem (24) | eq. (24) |
| `source.py` | Duhamel integrals `∫ e^{±λ(·)} f_mn`: closed form plus general grid-based quadrature | eqs. (6), (11), (12) |
| `forward_solver.py` | exact spectral forward evolution; Crank–Nicolson cross-check | eq. (11) |
| `noise.py` | `g_ε = g + (ε/π)·rand()`, seeded, reports the actual `‖g_ε−g‖` | p. 18 |
| `regularization.py` | cut-off rules M¹–M⁴, regularised coefficients, stability factor, a-priori bounds | eq. (12), Rem. 1, Thms 3, 5–7 |
| `reconstruction.py` | pipeline `g_ε → (g_ε)_mn → u^ε_mn(t) → u^ε` with diagnostics | eq. (12) |
| `metrics.py` | paper metric (RMS over nodes), trapezoidal `L²`, relative, MSE, RMSE, max | Table 2 |
| `visualization.py` | plotting only (no mathematics) | Figs 1–2 |

## How eq. (12) maps to code (`regularization.regularized_coefficients`)

```python
mask = eigenvalues <= cutoff                                   # summation range m²+n² ≤ M_ε
coeffs[mask] = np.exp((T - t) * eigenvalues[mask]) * g_coeffs[mask]   # e^{(T−t)λ}(g_ε)_mn
coeffs -= source.backward_integral(eigenvalues, t, T, mask)    # − ∫_t^T e^{(s−t)λ}(f_ε)_mn ds
```

* The exponential is evaluated **only on kept modes**. For high modes `e^{λ}` would overflow (λ up to 32 258).
* If even the kept modes overflow (`(T−t)·max λ > 709.78`), an `OverflowError` is raised. The code never returns inf/NaN silently.

## Implementation choices (not specified by the paper)

| Choice | Reason |
|---|---|
| Coefficients by DST-I (trapezoidal rule) on the paper's grid | Exact for all representable modes; `O(N² log N)` |
| `rand()` = iid `U(−1,1)` per node, `default_rng(2015)` | Paper gives only the range; seed fixed for reproducibility |
| Only `g` is noisy, `f` is exact | The paper's example perturbs only `g` |
| Closed-form Duhamel integral for the experiments | `f` is given analytically; exact to rounding |
| General `GridSource` (for arbitrary `f`): Gauss–Legendre (48 nodes) backward, Filon-linear forward | See below |
| Metric = RMS over 129² nodes | Inferred: it reproduces Table 2 exactly; the true L² is reported in addition |
| `r = 2` (Theorem 6), square root in M⁴ | Inferred from Table 1 |

### Why the backward source quadrature must be high-order

`u^ε_mn = e^{(T−t)λ} g_mn − I_mn` is a difference of two numbers of size `~e^{(T−t)λ}`, and the result is `O(1)`.
A *relative* error `δ` in the quadrature `I_mn` therefore becomes an absolute error `δ·e^{(T−t)λ}`.
This is the same mechanism as data noise; Theorem 3 also covers perturbations of `f`.
A first version used a second-order Filon rule (`δ ~ 10⁻⁵`) and produced errors of ~370 at `M = 30`.
Switching to Gauss–Legendre, whose accuracy is spectral, reduced the discrepancy from the closed form to `< 10⁻⁸` (tested).
For the *forward* integral the kernel `e^{−λ(t−s)}` damps errors. There the Filon rule is used because it is
second-order accurate *uniformly in λ*, which plain Gauss quadrature is not for λ ≈ 10⁴. Experiment 1 confirms order 2.0.

## Numerical facts worth knowing

* **Rounding alone is amplified.** With noise-free data, `M = 20` at `t = 0` recovers the `k = 2` coefficient only to `~10⁻⁸` (= `2⁻⁵³·e²⁰`).
  At `M = 80` the error is `~10¹⁷` (Experiment 2).
* **Complexity.** One reconstruction costs two DSTs plus `O(#kept modes)` work: about 1 ms on a laptop for 129² nodes.
  The Crank–Nicolson check costs one sparse LU factorisation plus `O(N²)` per step.
* **Stability of the forward CN solver.** CN is unconditionally stable. We use `Δt = T/(2N)`, giving `O(h²)` overall (observed 2.15 and 2.03).

## Reproducibility

* All parameters: `src/backward_heat/config.py`. Each run saves its configuration to `results/raw/<experiment>_config.json`.
* Logs: `results/logs/<experiment>.log`.
* Seeds: Table 2 uses seed 2015. The seed study uses seeds 10000–10049, and Experiment 3 uses 1000–1009.
* Encoding pitfall seen during development: on Windows, `Path.write_text` defaults to cp1252. All project I/O passes `encoding="utf-8"`.
