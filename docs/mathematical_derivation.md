# Mathematical Derivation

Notation follows the paper. `Ω = (0, π)²`, `T > 0`, and `λ_mn = m² + n²`.

## 1. Forward problem and its spectral solution

The forward (well-posed) problem `u_t − Δu = f`, `u|_∂Ω = 0`, `u(·,·,0) = v` is solved by expanding in the
Dirichlet eigenfunctions of `−Δ`:

```
−Δ [sin(mx) sin(ny)] = (m² + n²) sin(mx) sin(ny),      m, n ≥ 1.
```

Write `u(x, y, t) = Σ u_mn(t) sin(mx) sin(ny)`. Testing the PDE with `sin(mx) sin(ny)` (the weak form (5))
decouples it into scalar ODEs:

```
u'_mn(t) + λ_mn u_mn(t) = f_mn(t).
```

**Normalisation.** Since `∫_0^π sin²(mx) dx = π/2`, we have `‖sin(mx)sin(ny)‖²_{L²} = π²/4`, so

```
h_mn = (4/π²) ∫∫_Ω h(x, y) sin(mx) sin(ny) dx dy,       ‖h‖²_{L²} = (π²/4) Σ h_mn².
```

**Forward (Duhamel) solution**, eq. (11):

```
u_mn(t) = e^{−λ_mn t} v_mn + ∫_0^t e^{−λ_mn (t−s)} f_mn(s) ds.
```

Every mode is *damped*. A perturbation of the initial data shrinks by `e^{−λ t}`, so the forward problem is stable.

## 2. Backward problem, eq. (6)

Solve the same ODE backwards from `u_mn(T) = g_mn`:

```
u_mn(t) = e^{(T−t) λ_mn} g_mn − ∫_t^T e^{(s−t) λ_mn} f_mn(s) ds,       0 ≤ t ≤ T.
```

*Check.* Differentiate: `u'_mn = −λ e^{(T−t)λ} g + f_mn(t) + λ ∫_t^T e^{(s−t)λ} f ds = −λ u_mn + f_mn` ✔, and `u_mn(T) = g_mn` ✔.

**Note on eq. (3).** The paper's introduction integrates over `(0, T)`. With that range the final condition fails, so (3) contains a typo; (6), (7) and (12) are correct.

## 3. Ill-posedness

Let `g^ε = g + δ` with `δ = η sin(m₀x) sin(n₀y)`. Then `‖δ‖_{L²} = (π/2)|η|`, and the solution changes by

```
‖u^ε(t) − u(t)‖_{L²} = (π/2) |η| e^{(T−t)(m₀² + n₀²)}.
```

Choose `η = 2ε/π` and `m₀² + n₀² → ∞`. The data error stays `ε` while the solution error tends to `∞`.
The solution does not depend continuously on `g`. In addition, a solution exists only if
`Σ e^{2Tλ} |g_mn − …|² < ∞` (Theorem 1). Generic noisy data violate this condition, so a solution need not exist at all.

The test in `tests/test_regularization.py::test_noise_amplification_matches_exponential` checks that the amplification factor
for a single mode is exactly `e^{(T−t)λ}`, to 12 digits.

## 4. Fourier truncation, eq. (12)

Keep only the modes in the disc `λ_mn ≤ M_ε`:

```
u^ε_mn(t) = 1{λ_mn ≤ M_ε} · [ e^{(T−t) λ_mn} (g_ε)_mn − ∫_t^T e^{(s−t) λ_mn} (f_ε)_mn(s) ds ].
```

**Stability (Theorem 3).** For two data sets the coefficient differences satisfy
`|Δu_mn| ≤ e^{(T−t)M_ε} ( |Δg_mn| + ∫ |Δf_mn| )`. Parseval and Cauchy–Schwarz then give

```
‖u¹(t) − u²(t)‖² ≤ e^{2(T−t)M_ε} ( ‖g¹ − g²‖² + ‖f¹ − f²‖² ).
```

The regularised problem is well-posed for every fixed `M_ε`, with stability constant `e^{(T−t)M_ε}`.

**Error splitting (proof of Theorems 5–7).** For exact data `(f, g)` and noisy data `(f_ε, g_ε)`:

```
u^ε − u = (u^ε − Γ_{M_ε} u)  +  (Γ_{M_ε} u − u)
           ─────────────────     ─────────────────
           propagated noise      truncation (bias)
           ≤ C e^{(T−t)M_ε} ε     → 0 as M_ε → ∞, rate set by the smoothness of u
```

The cut-off rules balance the two terms. `M_ε` must grow as ε → 0, so that the bias vanishes, but slowly
enough that `e^{(T−t)M_ε} ε → 0`. For example, `M²_ε = ln(1/ε)/(2T)` gives `e^{T M} ε = √ε` at `t = 0`.
Experiment 4 computes this decomposition explicitly (`results/figures/exp04_tradeoff.png`).

### Cut-off rules

| Rule | Formula | Resulting noise factor `e^{TM}ε` at `t = 0` |
|---|---|---|
| M¹ (Remark 1) | `(1/T) ln(T/(ε + ε ln(T/ε)))` | `T/(1 + ln(T/ε))` → 0 logarithmically |
| M² (Thm 5) | `ln(1/ε)/(2T)` | `√ε` |
| M³ (Thm 6) | `(1/T) ln(T (ln(T/ε))^r/(ε + ε ln(T/ε)))` | `T (ln(T/ε))^r/(1 + ln(T/ε))`, which **grows** for r > 1 |
| M⁴ (Thm 7) | `sqrt(ln(1/ε)/(T+β))` | `→ 0` very fast. The bias decays only like `e^{−βM}` |

Note that for `r = 2` the rule M³ does not drive the propagated noise to zero at `t = 0`. In the experiments this
shows up as the small increase of the M³ errors (1.430E-01 instead of 1.423E-01), both in the paper and here.

## 5. Benchmark (24)

`u = e^{−t} Σ_{k=1}^{15} k^{−2} sin(2kx) sin(ky)`. Its only modes are `(2k, k)`, with `λ = 5k²`, and

```
u_t − Δu = e^{−t} Σ (−1 + 5k²) k^{−2} sin(2kx) sin(ky) = e^{−t} Σ (5 − k^{−2}) sin(2kx) sin(ky) = f.   ✔
```

The source is separable, `f_mn(s) = e^{−s} F_mn`, so the Duhamel term has a closed form:

```
∫_t^T e^{(s−t)λ} e^{−s} ds = e^{−t} (T−t) φ₁((λ−1)(T−t)),       φ₁(z) = (e^z − 1)/z.
```

`φ₁` is evaluated with `expm1` so that it stays accurate for small `z`.

### Closed form of the Table 2 errors

If the truncation keeps the active modes `k ≤ k₀` and the noise is negligible, then

```
u − u^ε = e^{−t} Σ_{k > k₀} k^{−2} sin(2kx) sin(ky).
```

On the 129×129 grid the RMS of `sin(2kx)sin(ky)` (with `2k, k < 128`) is `(128/129)·(1/2)`. Distinct modes are discretely orthogonal, so

```
E(t) = (64/129) e^{−t} ( Σ_{k > k₀} k^{−4} )^{1/2}  =  0.51612 e^{−t}   (k₀ = 0)
                                                    =  0.14227 e^{−t}   (k₀ = 1).
```

These are exactly the two families of values printed in Table 2. This is how the paper's metric was identified
(`tests/test_regularization.py::test_table2_noiseless_closed_form`).

## 6. Discrete transform

On `x_i = iπ/N` the trapezoidal rule for `h_mn` reduces to the type-I discrete sine transform:

```
h_mn ≈ (4/π²)(π/N)² Σ_{i,j=1}^{N−1} h(x_i, y_j) sin(miπ/N) sin(njπ/N) = DST-I(h)_{m−1,n−1} / N²
```

SciPy's DST-I carries a factor 2 per axis. Discrete orthogonality makes this formula exact for all
`m, n ≤ N − 1`; the benchmark uses at most `m = 30`. The inverse transform is `DST-I(c)/4`. The cost of both is `O(N² log N)`.
