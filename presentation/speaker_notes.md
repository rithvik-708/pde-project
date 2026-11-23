# Speaker Notes — MA5030 Presentation

*Fourier-based regularization for the 2-D backward heat equation*

| | |
|---|---|
| **Total time** | About 13–15 minutes (roughly 1 minute per slide) |
| **Suggested split** | Rithvik: slides 1–5 · Heena: slides 6–9 · Osheen: slides 10–15 (adjust as you like) |
| **Ending** | Optionally run the live demo (`python demo.py`) after slide 15 |

---

## Slide 1: Title (≈30 s)

**Say:**
- "Our project reproduces Huy Tuan Nguyen's 2015 paper on stabilising the two-dimensional backward heat equation by Fourier truncation."
- "We implemented the method from scratch in Python. We reproduced every table and figure in the paper's numerical section and checked its theorems numerically."

**Transition:** "First, why is this problem interesting at all?"

---

## Slide 2: Motivation (≈1 min)

**Say:**
- The heat equation describes diffusion: temperature, concentrations, even the blurring of an image.
- The three heatmaps are the paper's own benchmark solution at t = 0, 0.5 and 1, on the same colour scale. Going forward in time, the sharp ridges fade out.
- The backward question is: given only the final picture, recover the first one.
- Applications include reconstructing an initial temperature field, the history of a pollutant source, and deblurring.

**Key idea for the audience:** detail that diffusion has wiped out is exactly the detail that noise destroys when we try to recover it.

**Transition:** "Let's fix the mathematical setting."

---

## Slide 3: Mathematical background (≈1 min)

**Say:**
- The equation is u_t − Δu = f, with Δu = u_xx + u_yy, on the square (0,π)².
- The diffusivity is 1, as in the paper.
- The boundary is held at zero (Dirichlet condition).
- **Forward problem:** start from u at time 0. It is well-posed in Hadamard's sense: a solution exists, it is unique, and it depends continuously on the data.
- **Backward problem:** start from u at the final time T. The solution is unique, but stability fails.

**Explain the equation:** u_t is the rate of change of temperature. Δu measures how much hotter the neighbourhood is than the point itself, so heat flows in or out accordingly. f is an external heat source.

**Possible question:** "Why the domain (0,π)²?" Because then the eigenfunctions are exactly sin(mx) sin(ny), which makes the analysis and the numerics clean.

---

## Slide 4: Problem formulation (≈1 min)

**Say:**
- This is the paper's system (1), quoted exactly.
- We know g, the measurement at the final time T, and the source f. Both may be noisy, with errors at most ε in the L² norm (p. 6).
- We want u at earlier times, especially u(x,y,0).
- Uniqueness holds (Theorem 2), so the difficulty is purely about stability.

**Explain the symbols:**
- g_ε and f_ε are the measured, noisy data.
- ε is the noise level.
- L²(0,T;L²(Ω)) means square-integrable in time, with values that are square-integrable in space.

**Transition:** "So where exactly does stability break?"

---

## Slide 5: Why the problem is ill-posed (≈1.5 min, the key slide)

**Say:**
- −Δ with zero boundary values has eigenfunctions sin(mx) sin(ny), with eigenvalues λ = m² + n².
- Each Fourier mode decouples into a scalar ODE.
- Forward in time, mode (m,n) is damped by e^{−tλ}.
- Backward, the paper's eq. (6) multiplies the data coefficient by e^{(T−t)λ}.
- So an error δ in g_mn becomes δ·e^{(T−t)λ}. Because λ grows without bound, no constant can bound the output error in terms of the input error: there is no continuous dependence on the data.

**Walk through the bar chart:**
- It shows the amplification at t = 0 for the benchmark's own modes (2k, k), where λ = 5k².
- Mode (2,1) is amplified only about 148 times.
- Mode (8,4) is amplified about 5·10³⁴ times.
- Our Experiment 2 showed that even machine rounding error (about 10⁻¹⁶) destroys the reconstruction once modes with λ above about 40 are kept, even with no measurement noise at all.

**Possible question:** "Isn't uniqueness enough?" No. Hadamard's three conditions are existence, uniqueness and continuous dependence; the third one fails here. Also, a solution exists only if the data satisfy the decay condition (8) of Theorem 1, and real noisy data never do.

---

## Slide 6: The paper's method, Fourier truncation (≈1 min)

**Say:**
- The quote is from p. 3. The method throws away every mode outside the disc m² + n² ≤ M_ε.
- Modes inside the disc are inverted exactly by eq. (6). There are no filter weights; it is a sharp cut-off.
- M_ε grows as the noise shrinks, so in the limit we recover the full solution.
- Theorem 3 says the regularized problem is stable: the amplification is at most e^{(T−t)M_ε}, which is finite for each ε.

**Explain the plot:**
- Lattice points are the pairs (m,n).
- Teal points are the modes kept by rule M³ at ε = 10⁻⁵ (M = 13.87).
- Orange circles are the modes that actually appear in the benchmark.
- Only (2,1) is inside the disc, which explains the results later on.

**Possible question:** "How does this differ from Tikhonov regularization?" Tikhonov damps every mode smoothly with a weight. Truncation uses a 0/1 weight: keep the mode or discard it. Both trade bias (truncation error) for variance (amplified noise).

---

## Slide 7: Mathematical derivation (≈1.5 min)

Read the cards left to right, top to bottom.

1. **Fourier representation.**
   - Expand u in the sine basis.
   - The coefficient h_mn uses the factor 4/π², because ∫₀^π sin² = π/2 in each variable.
2. **Coefficient evolution.**
   - Substituting into the PDE gives the ODE u_mn' + λ u_mn = f_mn for each mode.
   - The condition at the final time is u_mn(T) = g_mn.
3. **Backward reconstruction.**
   - Solve the ODE backwards with an integrating factor (Duhamel's formula). This gives eq. (6).
   - Eq. (3) in the paper integrates over (0,T). That is a typo; (6) and (12) use (t,T), which we verified.
4. **Source of instability.** The factor e^{(T−t)λ}. Theorem 1: a solution exists only if the series (8) converges.
5. **Regularization.** Keep λ ≤ M_ε.
6. **Regularized solution.** Eq. (12) is the result: eq. (6) applied to the noisy data, summed only over the disc.

---

## Slide 8: Numerical implementation (≈1 min)

**Say:**
- Each box in the pipeline is one module of our Python package.
- Each module is tested separately (64 tests, all passing).
- The sine coefficients come from the type-I discrete sine transform. On the paper's 129 × 129 grid this equals the trapezoidal rule, and it is exact for every mode of the benchmark.
- The source integral is computed in closed form. We also have a general quadrature path and cross-checked it against the closed form.
- We apply the mask before exponentiating, so e^{(T−t)λ} is never evaluated for a discarded λ, and nothing overflows.

**Validation:**
- Evolving the exact initial state forward reproduces the paper's g to 4.7·10⁻¹⁶.
- An independent Crank–Nicolson finite-difference solver converges at second order: observed orders 2.15 and 2.03.
- So the paper's data are consistent with the PDE.

---

## Slide 9: Experimental setup (≈1 min)

**Say:**
- The green "PAPER" tags mark everything the paper specifies: the domain, T = 1, the 128 × 128 grid, the benchmark (24), the noise formula, the three ε values and the ten times.
- The blue "OURS" tag marks our choices: iid uniform noise per grid node with seed 2015, and noise in g only.
- The orange "INFERRED" tags mark three things the paper never states, which we recovered from its own tables:
  - **r = 2:** solving Table 1 for r gives 1.9999, 1.9987 and 2.0008.
  - **M⁴ uses the square root:** the statement of Theorem 7 has it, the proof drops it, and only the square-root version reproduces Table 1.
  - **The error metric is an RMS over grid nodes,** not the true L² norm. It reproduces 0.5161 and 0.1423 exactly; the continuous L² norm would be about 3.2 times larger.
- On the right are the four cut-off rules; all 12 values match Table 1.

---

## Slide 10: Numerical results (≈1 min)

**Say:**
- This is the case the paper plots: t = 0, ε = 10⁻⁵.
- Left: the exact initial state, with sharp diagonal ridges, because the coefficients decay only like 1/k².
- Middle: our reconstruction with rule M³. It is a single smooth mode, sin(2x) sin(y).
- Right: the absolute error. It sits exactly on the ridges, i.e. on the discarded detail.
- The numbers on the right:
  - M¹ and M³ both keep only mode (2,1);
  - relative L² error ≈ 0.28;
  - in the paper's metric, 0.1423 and 0.1430.
- The tiny difference for M³ is amplified noise (it keeps more of the grid modes, with amplification up to about 10⁶).

**Key message:** the reconstruction is stable but blurry. That is the price of regularization.

---

## Slide 11: Truncation vs. noise (≈1.5 min)

These are supplementary experiments, beyond the paper's table.

**Left plot** (error against the cut-off M, at t = 0):
- For M < 5 nothing is kept, so the error is 100%.
- At M = 5 the first mode comes in (error 0.276).
- At M = 20 the second mode comes in.
- Increasing M further only adds noise amplified by e^M, so the error explodes. With ε = 10⁻⁵, M = 26 already gives an error of 6.8·10³.
- The dashed line is the truncation error alone: it keeps going down.
- This is the classic bias–variance trade-off.

**Right plot** (error against ε, for the four rules):
- The error falls in steps. Each step is one more benchmark mode (λ = 5k²) entering the disc.
- M⁴ never keeps a single mode in this range of ε.

**Takeaway:** the a-priori rules are safe but conservative. At ε = 10⁻⁷ the best M (20) halves the error compared with the rules.

**Possible question:** "How would you choose M without knowing the exact solution?" With an a-posteriori rule such as Morozov's discrepancy principle, or with an L-curve or cross-validation. The paper uses only a-priori rules, which depend on ε and T alone.

---

## Slide 12: Comparison with the paper (≈1 min)

**Say:**
- Each dot is one Table 2 entry: the paper's value against ours. They sit on the diagonal.
- Table 1: all 12 values match to every printed digit.
- Table 2:
  - median relative difference 0.005%, maximum 0.42%;
  - the largest gaps are the cells where noise matters (M³ at t = 0).
- The paper's random noise realisation is unknown. Over 50 noise draws, our range for that cell contains the paper's 0.1424.
- The orange diamond is a typo in the paper. It prints 1.287E-02 at ε = 10⁻⁵, M³, t = 0.1. Every neighbouring cell, and the formula, give 0.1288.

**Honest caveat:** with these ε values at most one benchmark mode is ever kept. So Table 2 mostly measures truncation error, not noise handling.

---

## Slide 13: Limitations and observations (≈1 min)

**Say:**
- **Paper side** (all documented with page numbers in `docs/paper_analysis.md`):
  - three details are unstated;
  - the constants in Theorem 4 are too small. Parseval gives 4/π², not 64/π⁶. As a result, the printed bound of Theorem 5 (1.28 at ε = 10⁻³) is below the actual error (1.634);
  - with the printed M³, the noise factor in Theorem 6 grows like ln(1/ε) at t = 0, so the stated convergence does not follow (an exponent sign).
  - None of this breaks the method itself.
- **Our side:**
  - the paper's noise realisation is unknown;
  - the figures can only be compared visually;
  - the benchmark is special (diagonal modes only);
  - noise in f is not tested;
  - floating-point rounding limits how many modes can ever be used.

**Possible question:** "Did you fix Theorem 5?" We derived and tested a Parseval-correct bound, √ε + √(2T)‖u(0)‖_{H¹₀}/√ln(1/ε). It holds at every ε we tested. We did not attempt to re-prove Theorem 6.

---

## Slide 14: Conclusion (≈45 s)

Read the five lines.

**Closing sentence:** "Truncation turns an impossible problem into a stable one. But the exponential amplification limits how much of the initial state we can ever recover: at t = 0, roughly only modes with λ up to ln(1/ε)."

---

## Slide 15: References and Q&A

- Point to the primary paper and to the references it cites ([7], [21], [25] in its own numbering).
- If time allows, run `python demo.py`: move the noise slider and the M slider and watch the reconstruction blow up.

---

# Viva Questions and Answers

**1. What makes the backward heat equation ill-posed?**
- The solution does not depend continuously on the data, so Hadamard's third condition fails.
- In Fourier space, u_mn(t) = e^{(T−t)λ_mn} g_mn − …, and λ_mn = m² + n² → ∞. A perturbation δ in g_mn is therefore multiplied by an unbounded factor: arbitrarily small data errors in L² can produce arbitrarily large errors in the solution.
- In addition, a solution exists only for data whose coefficients decay fast enough (Theorem 1, condition (8)).

**2. What is the difference between the forward and the backward heat problem?**
- **Forward:** given u(·,0), find u(·,t) for t > 0. Modes are multiplied by e^{−tλ} ≤ 1, so the problem is well-posed and smoothing.
- **Backward:** given u(·,T), find u(·,t) for t < T. Modes are multiplied by e^{(T−t)λ} ≥ 1, unboundedly. The solution is unique when it exists, but it is unstable.

**3. Why do high-frequency Fourier modes become unstable?**
- The eigenvalue λ = m² + n² is large for high frequencies.
- Diffusion damps them like e^{−Tλ}, so their information in g is buried below the noise level. Inverting multiplies whatever is in that coefficient, which is mostly noise, by e^{Tλ}.
- For example, mode (8,4) has λ = 80 and is amplified about 5·10³⁴ times.

**4. What is regularization?**
- It replaces an ill-posed problem by a nearby well-posed one that depends on a parameter (here M_ε), such that:
  - for fixed ε the new problem is stable;
  - as ε → 0, with the parameter chosen appropriately, its solution converges to the true one.
- Examples: Tikhonov, quasi-reversibility, quasi-boundary value methods, and truncation (this paper).

**5. Why is Fourier truncation useful here?**
- The instability lives entirely in the high modes, and on the square the eigenfunctions are known explicitly (sines).
- Dropping modes with m² + n² > M_ε caps the amplification at e^{(T−t)M_ε} (Theorem 3).
- The modes that are kept are inverted exactly, so there is no extra damping bias.
- It is cheap: one sine transform plus one exponential factor per mode.

**6. What is the role of the regularization parameter M_ε?**
- It sets the trade-off:
  - a small M discards signal (truncation error, i.e. bias);
  - a large M amplifies noise by e^{(T−t)M} (variance).
- The paper's rules are a-priori choices that depend only on ε and T:
  - M¹ and M³ are of the form (1/T)·ln(…/ε);
  - M² is ln(1/ε)/(2T);
  - M⁴ is the much smaller sqrt(ln(1/ε)/(T+β)).
- In each case M_ε → ∞ as ε → 0. With M¹ and M², the propagated noise e^{TM}·ε also goes to 0, so the error converges.
- Caveat (erratum A13): with M³ as printed, the noise factor at t = 0 grows like ln(1/ε), so convergence at t = 0 does not follow from Theorem 6 as stated.
- Our Experiment 4 shows the trade-off: at ε = 10⁻⁵, t = 0, M = 3 gives error 1.0, M = 5 gives 0.276, and M = 26 gives 6.8·10³.

**7. How is noise introduced?**
- As in the paper: g_ε = g + (ε/π)·rand(), with rand() ∈ (−1, 1).
- We draw rand() independently and uniformly at each of the 129² grid nodes, with a fixed seed (2015).
- Since |g_ε − g| ≤ ε/π and |Ω| = π², we get ‖g_ε − g‖_{L²} ≤ ε, as the theory requires. The actual value is about 0.58ε.
- The source f is not perturbed, because the paper does not perturb it.

**8. How is the reconstruction error calculated?**
- Table 2 says "L²", but its numbers match the root-mean-square difference over all 129 × 129 grid nodes, E = sqrt(mean |u^ε − u|²).
- With no mode kept, this gives exactly 0.5161; with mode (2,1) kept, 0.1423.
- We also report the true L² norm (about 3.2 times larger) and the relative L² error ‖u^ε − u‖/‖u‖.

**9. How was the numerical method validated?**
- The spectral forward solve of the exact u(·,0) reproduces the paper's g to 4.7·10⁻¹⁶.
- An independent Crank–Nicolson solver converges at second order (observed 2.15 and 2.03).
- The quadrature for the source integral converges at order 2.00.
- A SymPy check confirms that the benchmark satisfies the PDE.
- Theorem 3's stability bound is verified numerically.
- Table 1 is reproduced exactly, and Table 2 to within 0.42%.
- The pytest suite has 64 tests, all passing.

**10. What are the limitations of the implementation?**
- The paper's noise realisation is unknown, so noise-sensitive cells agree only statistically.
- Figures 1–2 can only be compared qualitatively.
- The square domain with explicit sine modes does not generalise directly to arbitrary domains, where the eigenpairs would have to be computed numerically.
- The benchmark contains only diagonal modes (2k, k), and at the paper's ε only one of them is recovered.
- Noise in f is supported but not exercised.
- float64 rounding limits the usable cut-off to roughly M ≲ 40 at t = 0.

**11. What did you find wrong or missing in the paper?**
- **Unstated details:** the value of r (we inferred 2), the exact form of M⁴ (statement vs. proof), and the error metric.
- **Theorem 4/5 constants:** they are too small (64/π⁶ instead of 4/π² from Parseval). The printed Theorem 5 bound is violated at ε = 10⁻³: 1.28 against an actual error of 1.634.
- **Theorem 6:** an exponent sign problem.
- **Typos:** eq. (3) and one Table 2 cell.
- **The claim "the bigger M_ε, the better":** it holds only for M < 20.

**12. Why does Table 2 barely change with ε?**
- All of the paper's cut-offs are below 20.
- The benchmark's modes have λ = 5, 20, 45, …, so at most mode (2,1) is ever kept.
- The error is therefore almost entirely truncation error: (64/129)·e^{−t}·sqrt(Σ_{k≥2} k⁻⁴) = 0.1423·e^{−t}. Noise only changes the fourth significant digit.

**13. What does Theorem 7 add?**
- It applies under a Gevrey-type smoothness assumption on the solution, (π²/4) Σ e^{2βλ} u_mn² < A₂².
- It states a Hölder rate ε^{β/(T+β)} that holds at every t, including t = 0.
- Caveat (A2): the statement uses the cut-off with a square root, but the proof (and therefore the rate) uses ln(1/ε)/(T+β) without the root. Table 1 matches the square-root version, so that is what the numerics use.
- For this benchmark, however, A₂ is astronomically large, so the bound is not informative. And M⁴ is so small that it keeps no mode at all.
