# Paper Analysis

> Milestone 1 deliverable. Everything marked **[paper]** is taken directly from the PDF
> (`paper.pdf`, page numbers refer to the journal pagination 1–23).
> Everything marked **[inferred]** was recovered by reverse-engineering the paper's own
> published numbers, and everything marked **[choice]** is an implementation decision
> that the paper does not specify.

---

## 1. Bibliographic information

| Field | Value |
|---|---|
| Title | *Stability and Regularization for 2-D Backward Heat Conduction Equation* |
| Author | Huy Tuan Nguyen (Ton Duc Thang University, Ho Chi Minh City) |
| Journal | *Mathematical Reports*, **17(67)**, no. 1 |
| Year / pages | 2015, pp. 1–23 (received 1 Aug 2012) |
| Communicated by | Marius Tucsnak |
| AMS class. | 35K05, 35K99, 47J06, 47H10 |
| Method | **Fourier truncation** regularization |

---

## 2. Problem formulation [paper, p. 1, eq. (1)]

Let `T > 0`, `Ω = (0, π) × (0, π)`. Determine `u(x, y, 0)` from

```
u_t − u_xx − u_yy = f(x, y, t),          (x, y, t) ∈ Ω × (0, T)
u(0, y, t) = u(π, y, t) = u(x, 0, t) = u(x, π, t) = 0      (homogeneous Dirichlet)
u(x, y, T) = g(x, y)                       (final / terminal condition)
```

* `f ∈ L²(0, T; L²(Ω))` and `g ∈ L²(Ω)` are **data that may be given inexactly**.
* Time direction: the *physical* equation is the **forward** heat equation (diffusion
  coefficient +1). We are given the state at the **final** time `T` and must go **backwards**
  to `t < T` (in particular `t = 0`). This is the *backward heat problem* (BHP) / final value problem.
* Noisy data satisfy `‖g_ε − g‖_{L²(Ω)} ≤ ε`, `‖f_ε − f‖_{L²(0,T;L²(Ω))} ≤ ε` (p. 6).

### Eigen-structure [paper, p. 3, eq. (4)]

`−Δ` with zero Dirichlet conditions on the square has eigenfunctions
`sin(mx) sin(ny)`, `m, n ≥ 1`, with eigenvalues

```
λ_mn = m² + n².
```

The orthonormal basis is `{(2/π) sin(mx) sin(ny)}` (p. 4).

### Sine coefficients (normalisation!) [paper, p. 4, below eq. (6)]

```
h_mn = (4/π²) ∫_0^π ∫_0^π h(x, y) sin(mx) sin(ny) dx dy,
h(x, y) = Σ_{m,n≥1} h_mn sin(mx) sin(ny).
```

Parseval with this normalisation (p. 8): `Σ |h_mn|² = (4/π²) ‖h‖²_{L²(Ω)}`, i.e.
`‖h‖²_{L²} = (π²/4) Σ |h_mn|²`, and `‖h‖²_{H¹_0} = (π²/4) Σ (m²+n²) |h_mn|²`.

---

## 3. Exact solution formula and ill-posedness

### Mode-wise solution [paper, eq. (6)–(7), p. 4]

```
u_mn(t) = e^{(T−t)(m²+n²)} g_mn − ∫_t^T e^{(s−t)(m²+n²)} f_mn(s) ds,     m, n ≥ 1.        (6)
```

*What it represents:* the Fourier coefficient of `u(·,·,t)` obtained by solving the scalar ODE
`u'_mn + λ_mn u_mn = f_mn` backwards from `u_mn(T) = g_mn`.

### Existence (Theorem 1, p. 4, eq. (8))

A solution exists **iff** `Σ ( e^{T λ_mn} g_mn − ∫_0^T e^{s λ_mn} f_mn(s) ds )² < ∞`.
Uniqueness (Theorem 2, p. 6) holds in `C([0,T];H¹_0) ∩ C¹((0,T);H²)`.

### Why the problem is ill-posed (p. 3)

The factor `e^{(T−t)(m²+n²)}` grows without bound as `m² + n² → ∞`. A perturbation of size
`δ` in the coefficient `g_mn` produces an error of size `δ e^{(T−t)λ_mn}` in `u_mn(t)`.
For `T − t = 1` and `λ = 50` this is a factor `5·10²¹`. A data error that is arbitrarily small
in `L²` can therefore cause an arbitrarily large error in the reconstruction. The solution does not
depend continuously on the data (Hadamard's third condition fails).

---

## 4. Regularization method: Fourier truncation [paper, eq. (12), p. 6]

Keep only the modes in the disc `m² + n² ≤ M_ε`:

```
u^ε(x, y, t) = Σ_{m,n ≥ 1, m²+n² ≤ M_ε} u^ε_mn(t) sin(mx) sin(ny),                         (12)
u^ε_mn(t)    = e^{(T−t)(m²+n²)} (g_ε)_mn − ∫_t^T e^{(s−t)(m²+n²)} (f_ε)_mn(s) ds,
(g_ε)_mn     = (4/π²) ⟨g_ε, sin(mx) sin(ny)⟩,   (f_ε)_mn = (4/π²) ⟨f_ε(·,·,s), sin(mx) sin(ny)⟩.
```

The cutoff satisfies `M_ε → ∞` as `ε → 0`.

**Stability (Theorem 3, p. 6):**
`‖u(·,·,t) − v(·,·,t)‖² ≤ e^{2(T−t)M_ε} ( ‖g¹ − g²‖² + ‖f¹ − f²‖² )`. The amplification is at most `e^{(T−t)M_ε}`.

### Parameter choice rules (the four cutoffs used in the numerics)

| Label | Source | Formula |
|---|---|---|
| `M¹_ε` | Remark 1, p. 7 | `(1/T) ln( T / (ε + ε ln(T/ε)) )` |
| `M²_ε` | Theorem 5, p. 11 | `ln(ε⁻¹) / (2T)` |
| `M³_ε` | Theorem 6, p. 14 | `(1/T) ln( T (ln(T/ε))^r / (ε + ε ln(T/ε)) )`, `r > 0` |
| `M⁴_ε` | Theorem 7, p. 16 | `sqrt( ln(1/ε) / (T + β) )`, `β > 0` |

### Convergence results

* **Theorem 4 (p. 8):** projection error of `Γ_M` (truncation operator): `‖Γ_M h − h‖ → 0`;
  `≤ (4/(π²√M)) ‖h‖_{H¹_0}` for `h ∈ H¹_0`; `≤ (4/(π²M)) ‖Δh‖` if `Δh ∈ L²`.
* **Theorem 5 (p. 11):** with `M_ε = ln(ε⁻¹)/(2T)`:
  `‖u^ε(0) − u(0)‖_{L²} ≤ (4√2/(πT)) ε^{1/4} + (4√2 T/π²) ‖u(0)‖_{H¹_0} / √ln(ε⁻¹)`,
  which is a logarithmic rate at `t = 0`. There is also an `H¹_0` estimate, and a `1/ln(ε⁻¹)` rate if `Δu(0) ∈ L²`.
* **Theorem 6 (p. 14):** under `(π²/4) Σ λ^k u²_mn < A₁²`, a mixed Hölder–logarithmic rate.
* **Theorem 7 (p. 16):** under the Gevrey-type condition `(π²/4) Σ e^{2βλ} u²_mn(t) < A₂²` (18):
  `‖u^ε(t) − u(t)‖ ≤ ε^{β/(T+β)} ( A₂ + 2 ε^{t/(T+β)} )`, which is Hölder at every `t ∈ [0, T]`, including `t = 0`.

---

## 5. Numerical experiment specification [paper, Section 4, pp. 18–21]

### Test problem (24)

`T = 1`, `Ω = (0, π)²`:

```
g(x, y)    = Σ_{k=1}^{15} (e^{−1}/k²) sin(2kx) sin(ky)
f(x, y, t) = e^{−t} Σ_{k=1}^{15} (5 − 1/k²) sin(2kx) sin(ky)
exact:  u(x, y, t) = e^{−t} Σ_{k=1}^{15} (1/k²) sin(2kx) sin(ky)
```

Check: `u_t − Δu = e^{−t} Σ (1/k²)(−1 + (4k² + k²)) sin sin = e^{−t} Σ (5 − 1/k²) sin sin = f`. ✔
(Verified symbolically in `tests/test_analytical.py`.)

**Important structural fact [inferred].** The only active modes are `(m, n) = (2k, k)`, with
eigenvalues `λ = 5k² ∈ {5, 20, 45, 80, …, 1125}`. The truncation keeps mode `k` iff `5k² ≤ M_ε`.

### Noise model [paper, p. 18]

```
g_ε(x, y) = g(x, y) + (ε/π) · rand(),     rand() ∈ (−1, 1).
```

Since `|g_ε − g| ≤ ε/π` pointwise, `‖g_ε − g‖_{L²(Ω)} ≤ (ε/π)·π = ε`. ✔

* **[choice]** `rand()` is read as an **independent uniform U(−1, 1) draw at every grid node**,
  with a fixed seed (the paper gives no seed or distribution details beyond the range).
* **[choice]** Only `g` is perturbed. The paper writes "For the measured noised data g_ε"
  and gives no noisy `f`, so `f_ε = f` (exact).

### Discretisation [paper, p. 18]

`(I, J) = (128, 128)`, `x_i = iπ/I`, `y_j = jπ/J`, `i, j = 0, …, 128`, i.e. 129 × 129 nodes.

* **[choice]** `(g_ε)_mn` is computed from the grid samples by the trapezoidal rule. On this grid the
  rule is the type-I discrete sine transform (DST-I) over the 127 × 127 interior nodes (the boundary
  values of `sin(mx)sin(ny)` vanish). It is **exact** for trigonometric polynomials with `m, n < 128`,
  so for every mode of the test problem.
* **[choice]** The source integral `∫_t^T e^{(s−t)λ} f_mn(s) ds` is evaluated in **closed form**
  for this separable `f`. A general path (Filon-type exponential quadrature over grid samples of
  `f`) is also implemented and cross-validated.

### Parameters [paper, p. 18]

`ε ∈ {10⁻³, 10⁻⁵, 10⁻⁷}`, `t ∈ {0, 0.1, …, 0.9}`, `β = 1/2` (for `M⁴`).

### Error metric [inferred]

Table 2 is described as "error estimations … in `L²((0,π)²)`" (p. 18). The continuous `L²` norm
does **not** match the table. `‖u(·,·,0)‖_{L²} = (π/2)(Σ k⁻⁴)^{1/2} ≈ 1.634`, while the table shows 0.5161
in the columns where no mode is kept. The table **is** reproduced exactly by the root-mean-square over
all `129²` grid nodes, boundary included:

```
E = sqrt( (1/129²) Σ_{i,j=0}^{128} |u^ε(x_i, y_j, t) − u(x_i, y_j, t)|² ).
```

When no mode is kept: `E = (128/129)(1/π)·(π/2)(Σ_{k=1}^{15} k⁻⁴)^{1/2} = (64/129)(Σ k⁻⁴)^{1/2} = 0.51612` ✔.
When only `k = 1` is kept: `(64/129)(Σ_{k=2}^{15} k⁻⁴)^{1/2} = 0.14227` ✔.
This metric is used for the reproduction. The true `L²` error is reported **in addition**.

### Tables and figures to reproduce

| Item | Content | Status target |
|---|---|---|
| **Table 1** (p. 18) | `M^i_ε`, i = 1..4, for ε = 10⁻³, 10⁻⁵, 10⁻⁷ | exact numbers |
| **Table 2** (p. 19) | error at t = 0, 0.1, …, 0.9 for 4 rules × 3 noise levels | exact numbers (up to noise realisation) |
| **Figure 1** (p. 20) | 3-D surfaces at t = 0: exact `u`; `u^ε` for ε = 10⁻⁵ with `M¹`, `M³` | qualitative |
| **Figure 2** (p. 21) | section cuts at y = 51π/128, π/2, 51π/64 (t = 0, ε = 10⁻⁵, `M¹`, `M³`) | qualitative |

Published Table 1:

| | ε = 10⁻³ | ε = 10⁻⁵ | ε = 10⁻⁷ |
|---|---|---|---|
| M¹ | 4.840 | 8.986 | 13.28 |
| M² | 3.454 | 5.756 | 8.059 |
| M³ | 8.705 | 13.87 | 18.84 |
| M⁴ | 2.146 | 2.770 | 3.278 |

---

## 6. Ambiguities, inconsistencies and missing information

| # | Issue | Resolution |
|---|---|---|
| A1 | **`r` in Theorem 6 is never given.** | **[inferred] r = 2.** Solving Table 1 for `r` gives 1.9999, 1.9987 and 2.0008 at the three ε values. |
| A2 | **`M⁴`: the statement of Theorem 7 has a square root** (`M_ε = sqrt(ln(1/ε)/(T+β))`), **but the proof uses `M_ε = ln(1/ε)/(T+β)`** and writes `e^{−βM²_ε}`. | Table 1 (2.146, 2.770, 3.278) matches the **square-root** version. The proof version would give 4.605, 7.675, 10.75. The square root is used for reproduction. Note that the Hölder rate (19) is derived for the non-root version. |
| A3 | Error metric "in L²" does not match the continuous norm. | **[inferred]** RMS over the 129² nodes (see §5). |
| A4 | Eq. (3) integrates the source over `(0, T)`, while (6), (7) and (12) integrate over `(t, T)`. | Typo in (3). `(t, T)` is the correct Duhamel form (it reproduces `u` exactly) and is used. |
| A5 | Eq. (5) has `⟨f, ΔW⟩` on the right; it should be `⟨f, W⟩`. | Typo. It does not affect the numerics. |
| A6 | Table 2, ε = 10⁻⁵, `M³`, t = 0.1 shows **1.287E-02**. Every neighbour shows 1.287E-01, and the value must equal `0.1423·e^{−0.1} = 0.1287`. | Typo (missing digit in the exponent). Reported, not reproduced. |
| A7 | Fig. 1/2 do not state which noise realisation was used. | Same fixed seed as Table 2. |
| A8 | Remark 1 claims `U₁(ε) < U₂(ε)` "for all t"; `U₁` does not depend on `t`. | Cosmetic; not used numerically. |
| A9 | Theorem 7 requires the constant `A₂`. For the test problem `(π²/4) Σ e^{2β·5k²} k⁻⁴ e^{−2t}` is finite but astronomically large (~e^{1125}), so the bound is vacuous in practice. | Discussed in the report; the bound is not plotted at that scale. |
| A10 | The paper says "the bigger the value `M_ε`, the better". | True **only** in the range of Table 1 (all `M < 20`, so at most mode `k = 1` is kept and the error is pure truncation error). For larger `M` the noise is amplified by `e^{(T−t)M}` and the error explodes. Shown in supplementary Experiment 4. |
| A11 | Seed/RNG and noise distribution beyond "rand() ∈ (−1,1)". | Uniform iid, `numpy.random.default_rng(seed=2015)`. |
| A12 | **Theorem 4's constants are too small.** The proof writes `Σ(m²+n²)|h_mn|² ≤ (64/π⁶)‖h‖²_{H¹_0}`, but Parseval gives `= (4/π²)‖h‖²_{H¹_0}`. The printed bound `‖Γ_M h − h‖ ≤ 4/(π²√M)‖h‖_{H¹_0}` therefore fails, e.g. for the benchmark at M = 3 (1.634 > 1.033). | Verified numerically (`tests/test_fourier.py`). The correct bounds `‖Γ_M h − h‖ ≤ ‖h‖_{H¹_0}/√M` and `≤ ‖Δh‖/M` are tested instead. The constants in Theorem 5 inherit this error: its printed L² bound (1.28 at ε = 10⁻³) is below the actual error (1.634). A Parseval-corrected bound `√ε + √(2T)‖u(0)‖_{H¹₀}/√ln(1/ε)` is implemented and tested. The method itself and the numerics are unaffected. |
| A13 | **Theorem 6, sign of an exponent.** With the printed `M³_ε`, `e^{(T−t)M}ε = ε^{t/T} (T (ln(T/ε))^r/(1+ln(T/ε)))^{1−t/T}`. The factor `(ln T/ε)` therefore enters with exponent **+r(1−t/T)**, while the printed estimate has `(ln T/ε)^{2r(t/T−1)}` (a negative exponent). For `r = 2` and `t = 0` the propagated-noise factor *grows* like `ln(1/ε)`, so the stated convergence at `t = 0` does not follow from the printed rule. | Reported; we did not try to correct the theorem. Table 1 is reproduced with the rule *as printed* (r = 2). Numerically the effect is small at the paper's ε: it explains why the M³ errors at ε = 10⁻⁵, 10⁻⁷ are slightly *larger* (1.424E-01 vs 1.423E-01), both in the paper and here. |

### What Table 2 actually measures [inferred, then verified numerically]

All `M` values in Table 1 are `< 20`. So:

* `M < 5` (M¹ and M² at ε = 10⁻³; M⁴ always) → **no mode is kept** → `u^ε ≡ 0` → `E = 0.5161·e^{−t}`.
* `5 ≤ M < 20` (every other entry) → only mode `k = 1` is kept → `E ≈ 0.1423·e^{−t}`.

The noise contribution is tiny. At `ε = 10⁻³` the amplified noise in mode (2,1) is about `e^{5}·2ε/(√3·π·128) ≈ 10⁻⁴`, which changes only the 4th significant digit (e.g. 1.424E-01). So Table 2 is in essence a table of **truncation errors**. Reproducing it confirms the method, the normalisation, the cutoff rules and the metric. It is not a stress test of the noise amplification, which Experiments 3–4 add (labelled supplementary).
