# Experiment 2 results — the α sweep: the IPS exponent as a diversity knob

Numbers below are from the run logged in
`results/exp2_ips_alpha_sweep/data/exp2_summary.json` and the CSVs next to it
(regenerate with `python3 experiments/exp2_ips_alpha_sweep.py`, about 2.5 min
on a laptop CPU, all seeds fixed). Every "theory" value is derived in
`derivation.md`, in the sections *Linearization at the stationary point*,
*α = 0: algebraic, not exponential, convergence* and *Finite group size*.

Setting: generalized scaling `r̃ = r / p^α`, idealized flow
`ż_i = r_i p_i^{1-α} - p_i Σ_k r_k p_k^{1-α}` (Eq. * of `derivation.md`),
predicted stationary law `p* ∝ r^{1/α}`. α=0 is Experiment 1's collapse
dynamics and α=1 is the paper's IPS.

## Figure 1 — the diversity knob (idealized gradient flow, RK4)

**(a) K=2, r=(4,1)**, uniform start, RK4 with h=0.005, T=200:

| α | 0 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| p₁(T=200), RK4 | 0.99916 | 0.941176 | 0.800000 | 0.666667 | 0.585786 |
| p₁* = 4^{1/α}/(4^{1/α}+1) | 1 (limit) | 16/17 = 0.941176 | 0.8 | 2/3 | 0.585786 |

This reproduces the proposal's 94/6, 80/20, 67/33 split. α=0 still collapses:
p₁ is 0.99916 at T=200 and still creeping upward (only algebraically, see Fig. 2c).

**(b) Stationary p₁* vs α** for r ∈ {(4,1), (2,1), (1.25,1)}: 27 RK4 runs
over α ∈ [0.3, 5] land on the analytical curve with

> **max ℓ₁ error = 7.5 × 10⁻¹⁵** (machine precision)

**(c) K=5, r=(5,4,3,2,1)**, full stationary distributions
(max ℓ₁ error 1.0 × 10⁻¹⁵). Normalized entropy `H(p*)/ln K`:

| α | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|
| H/ln K | 0.794 | 0.926 | 0.978 | 0.994 |

α=1 gives exactly `r/Σr` = (1/3, 4/15, 1/5, 2/15, 1/15).

**Takeaway:** α is a continuous, monotone diversity dial. α→0 concentrates on
argmax r, α=1 is reward-proportional, and α>1 flattens toward uniform. The
reward *ratio* sets how quickly the dial moves (panel b).

## Figure 2 — convergence rates (least-squares fits, Methodology step 5)

Linearizing at `p*` gives `J* = -α ‖r‖_{1/α} (diag p* − p* p*ᵀ)`, so every
α>0 converges exponentially at rate `λ_min = α ‖r‖_{1/α} μ_min(Cov p*)`.

**(a) r=(2,1):** `‖p(t) − p*‖₁` is a straight line on a semilog plot. The
rates span two orders of magnitude (λ = 0.358, 1.33, 5.66, 45.6 for
α = 0.5, 1, 2, 4), but in rescaled time `λt` all four curves fall on one line
down to the ~10⁻¹⁵ round-off floor.

**(b) Fitted vs linearized rate.** Least-squares fit of `ln‖p − p*‖₁` against t
inside the window [10⁻¹², 10⁻⁶] (past the nonlinear transient, above
round-off), 34 runs:

- **K=2** (three reward ratios × 9 values of α ∈ [0.3, 4]):
  **relative error ≤ 1.7 × 10⁻⁵**, R² ≥ 0.9999999999.
- **K=5**, r=(5,4,3,2,1), α ∈ [0.5, 4]: relative error ≤ 6.3 × 10⁻⁶ up to
  α=1.41, then 6.2 × 10⁻⁵ (α=2), 3.2 × 10⁻⁴ (α=2.83), 8.9 × 10⁻⁴ (α=4).
  This is not a failure of the theory. At large α the two slowest rates
  nearly coincide (λ₂/λ₁ = 1.18 at α=4, against 1.97 at α=1), so the second
  mode has not died out inside the fit window and biases the slope slightly
  (min R² = 0.99999991).

**(c) α → 0 is a singular limit** (r=(2,1), log–log plot of the minority
probability p₂(t)):

- α=0: least-squares tail slope **−1.007** (R² = 0.99999) over t ∈ [100, 2×10⁴],
  matching the exact asymptote `p₂ ≈ 1/(2Δt)`. Collapse is algebraic, with no
  exponential rate at all.
- α>0: p₂ plateaus at p₂* with rate 1.33 (α=1), 0.358 (α=0.5), 0.0237 (α=0.2),
  3.9 × 10⁻⁴ (α=0.1). The small-α formula `λ ≈ 2α r₁ (r₂/r₁)^{1/α}` gives
  3.906 × 10⁻⁴ against the exact 3.899 × 10⁻⁴ at α=0.1: the rate vanishes
  faster than any power of α. On any finite training horizon a small α
  looks like collapse (α=0.1 only settles at t ~ 10⁴).
- Bonus consistency check with Experiment 1: the exact α=0 solution
  `u + sinh u = Δt + const` predicts Exp. 1's collapse-time constant as
  53.89/Δ, against the fitted 53.94 · Δ^(−1.00).

## Figure 3 — stiffness and explicit-integrator stability

**(a) Spectrum of J\* vs α** (r=(5,4,3,2,1)). The closed form matches the
numerical eigenvalues of the analytical Jacobian at z\* to **5 × 10⁻¹³**
(relative). The two ends of the dial are hard for different reasons:

| α | λ_min | λ_max | λ_max/λ_min |
|---|---|---|---|
| 0.1 | 5.8 × 10⁻⁸ | 0.089 | 1.5 × 10⁶ |
| 1 | 1.15 | 4.53 | 3.9 |
| 5 | 7.2 × 10³ | 9.3 × 10³ | 1.3 |

- **Small α** is classically stiff: a huge spread of time scales, so a long
  horizon is needed for the slow mode.
- **Large α** has every rate large (`‖r‖_{1/α}` grows like K^α; exact tie:
  `λ = α K^{α−1}`), so stability, not accuracy, dictates the step size.
  Unlike α=0, the right-hand side is also no longer bounded for α>1.

**(b) Critical step size by bisection.** For each α ∈ [0.5, 4] a vectorized
bisection (30 geometric halvings) on the *actual nonlinear* iteration finds
the largest h for which a 10⁻⁴ perturbation of z\* shrinks. It agrees with
linear stability theory:

> Euler `h_crit = 2/λ_max`: **max relative error 2.6 × 10⁻⁴**
> RK4 `h_crit = 2.7852935634/λ_max`: **max relative error 1.3 × 10⁻⁴**

The RK4 constant is itself found by bisection on x³ − 4x² + 12x − 24 = 0.
Across the sweep h_crit falls from 1.48 to 1.3 × 10⁻³ (Euler) and from 2.06
to 1.8 × 10⁻³ (RK4). **Experiment 1's production step h=0.05 becomes
unstable above α = 2.06 (Euler) and α = 2.23 (RK4)** for these rewards.

**(c) α=2, r=(4,1), λ\*=8**, start p₁=0.6: Euler with h=0.2 (hλ=1.6)
converges with a damped zig-zag, Euler with h=0.3 (hλ=2.4) is thrown into a
0 ↔ 1 oscillation, and RK4 with the same h=0.3 converges. This is the Euler
instability that Experiment 1 looked for and could not find at α=0.

One caveat, found while building this panel: **linear stability is local.**
From the uniform policy the local rate is 10, not λ\*=8, so RK4 with h=0.3
(h·10 = 3 > 2.785) diverges during the transient. h=0.26 converges from
there. In practice, the learning rate for α>1 has to be set for the stiffest
point along the path, not just at the equilibrium.

## Figure 4 — Monte-Carlo agent (finite group G; each update is a noisy Euler step)

**(a) K=5, exact tie**, symmetric start, G=16, h=0.5, 6000 steps, 100 seeds
per α (the same noise-driven setting as Exp. 1, Fig. 2b):

| α | 0 | 0.25 | 0.5 | 1 |
|---|---|---|---|---|
| seeds collapsed (max p > 0.95) | **64%** | 0% | 0% | 0% |
| final median H/ln K | 0.105 | 0.984 | 0.994 | 0.9999 |

Sampling noise alone collapses α=0 (Exp. 1), but any α>0 adds a restoring
force toward p\*, so the same noise leaves the policy diverse. (α>1 is left
out here only because h=0.5 would violate Fig. 3's stability limit.)

**(b, c) K=2, r=(4,1): sweep over (α, G)**, 12 values of α ∈ [0.3, 2] ×
G ∈ {2, …, 128}, h=0.05, 30 000 steps, policy time-averaged over the second
half, 16 seeds each. The finite-G mean-field prediction (the exact binomial
expectation of the sampled update, stationary point found by bisection)
tracks the simulation:

> max |MC − mean field| = **0.013** (worst at G=4 near the boundary),
> ≤ 0.0048 for G ≥ 8, ≤ 0.0008 for G ≥ 32

The new result is that **a finite group caps the inverse-probability weight
at min(G, 1/ε)^α**, because a rare outcome is seen at most once, at
frequency 1/G. So IPS protects the weaker outcome only when

```
α > α_c = ln(r₁/r₂) / ln min(G, 1/ε)
```

| G | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| α_c (r=(4,1), ε=10⁻³) | 2 | 1 | 0.667 | 0.5 | 0.4 | 0.333 | 0.286 |

The phase diagram (panel c) switches from "minority outcome extinct" to
"kept" exactly along this curve. At the paper's own α=1 (grid point 1.003):

| G | 4 | 16 | 64 | ideal |
|---|---|---|---|---|
| long-run p₁ (MC) | **0.9965** | 0.8041 | 0.7993 | 0.7993 |

**With G=4, IPS-GRPO at α=1 fails to keep an outcome worth 1/4 of the best
one.** Well above α_c the finite-G bias is small and changes sign: G=4
under-corrects at α=2 (0.742 vs 0.667), while G=16 slightly over-corrects
(0.652, i.e. 104% of the ideal minority mass).

Side remark: when ε < 1/G the clip can never activate (p̂ is 0 or ≥ 1/G), so
results cannot depend on ε there. That is consistent with the identical
entries in the paper's Table 4 (G=4 for all ε; G=8 for ε = 0.01 and 0.1).

## Change to shared code made for this experiment

`src/dynamics.py::rhs_sampled` used to give an outcome absent from the group
(p̂=0) the weight `r·ε^{1−α}` instead of 0: the code shortened
`p̂·max(p̂,ε)^{−α}` to `max(p̂,ε)^{1−α}`, which is only valid for p̂ ≥ ε.
At α=2 that pushed unseen outcomes with ±1000·r, so a Monte-Carlo α-sweep
was impossible without the fix. At α=0 the effect was ~10⁻³.

**Experiment 1 re-run with the fix** (on a copy; the committed Exp. 1 figures
were not overwritten):

- Figs. 1 and 3 and Fig. 2a (50% split): identical.
- Fig. 2b K=5 winner frequencies: 18.5/16.0/21.5/23.0/21.0%, all still inside
  the 95% band [14.5, 25.5]%.
- Fig. 2c collapse-time exponent: 1.066 (was 1.07). Collapsed within budget
  at G=32/64: 100%/92.5% (was 97.5%/87.5%). Medians 177/383/654/2051/3066
  (were 191/418/793/2480/3138). Collapse is slightly faster because, without
  the spurious floor, the minority outcome can actually go extinct.

Every Experiment 1 conclusion stands. New shared modules: `src/rootfinding.py`
(scalar and vectorized bisection), `src/finite_group.py` (finite-G mean
field), `integrate_at` in `src/integrators.py`, and `jacobian_alpha` /
`stationary_jacobian` / `linear_rates` / `reward_norm` in `src/dynamics.py`.

## Scope notes

- The sampled update is the group-REINFORCE estimator used in Exp. 1, without
  GRPO's per-group advantage standardization or PPO clipping.
- The finite-G threshold is exact for K=2. For K>2 the ceiling
  `min(G, 1/ε)^α` still bounds any outcome's boost, but the boundary also
  depends on how the dominant outcomes share mass.
