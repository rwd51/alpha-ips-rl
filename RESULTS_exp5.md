# Experiment 5 results — estimator bias, variance, and what they cost downstream

Numbers below are from the run logged in
`results/exp5_bias_variance/data/exp5_summary.json` and the CSVs next to it
(regenerate with `python experiments/exp5_bias_variance.py`, 597 s on an 8-core
laptop CPU, all seeds fixed — rerunning reproduces every CSV byte for byte). The theory is in `derivation_exp5.md`; the unit
tests are `tests/test_estimators.py` (55 tests).

This is the project pitch's Section 4.4, second half: *with N samples, how far is
`p̂` from the true `p`, and how do ε-clipping, Laplace smoothing and moving
averages trade bias against variance?* The base paper uses only clipping (its
Eq. 9, ablated in its Table 4), so the comparison is ours.

**Setting.** A group of `G` rollouts gives counts `n_i ~ Binomial(G, p_i)`. The
α-IPS update needs the weight `w_i = p_i^{-α}`, which the algorithm never knows.
A *weight rule* `ω` supplies a substitute:

| rule | `ω(n)` | knob |
|---|---|---|
| clipped (the paper) | `max(n/G, ε)^{-α}` | ε |
| add-λ (Laplace λ=1, Jeffreys λ=½) | `((n+λ)/(G+Kλ))^{-α}` | λ |
| split-half Richardson | `2f(n/G) − ½[f(n_A/M)+f(n_B/M)]`, `M=G/2` | guard |
| α-matched offset (new, Fig. 4) | `max((n−c)/(G−c), ε)^{-α}`, `c=(1−α)/2` | — |
| EMA over steps | `max(p̄, ε)^{-α}` | β |

Every bias and variance below is an **exact binomial sum**, not a Monte-Carlo
estimate. Monte Carlo is used only to validate those sums and the training
dynamics. Default α = 1 (the paper's IPS) and ε = 10⁻³ unless stated.

---

## Figure 1 — how far `p̂` is from `p`, and what that does to the weight

**(a) Exact error budget, G=16, α=1, ε=10⁻³.** Relative bias, standard deviation
and RMSE of `max(p̂,ε)^{-α}` against `p`, with the delta-method (truncation)
asymptotes `b ≈ α(α+1)(1−p)/(2Gp)` and `s ≈ α√((1−p)/(Gp))`.

Monte-Carlo validation, 400 000 groups per point drawn with the same
inverse-transform sampler Experiments 1–3 use, at 10 values of `p`:

> worst z-score against the exact sum: **2.20** (mean), **2.46** (variance)

The variance check needs its own exact standard error `√((μ₄−σ⁴)/N)`, because
the clipped weight is heavy-tailed: **up to 99.5% of its variance is carried by
the single empty-group draw** (`p̂=0`, probability `(1−p)^G`). At p=0.59, G=16
that draw has probability 3.4×10⁻⁷, so 400k samples contain 0.14 of them on
average and the Monte-Carlo standard deviation is 51% low — exactly as its own
predicted error says it should be. **This is why the exact sums are necessary
rather than merely convenient**: a sampling-based study of this estimator's
variance would need ≫10⁷ groups per point.

The bias changes sign at **p = 1.016×10⁻³ ≈ ε**: below that, the clip floors the
estimate at `ε^{-α} < p^{-α}` and under-weights; above it, convexity of `x^{-α}`
makes the bias positive.

**(b) The expansion parameter is `Gp`, not `G`.** Rescaled bias
`b·Gp/(1−p)` for G ∈ {8,…,256} collapses onto the single constant
`α(α+1)/2 = 1`. Bisection on the 10%-accuracy level gives the validity boundary:

| G | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| `Gp` above which the delta method is within 10% | — | 13.2 | 16.9 | 19.6 | 21.2 | — |

So roughly **`Gp ≳ 20` expected hits** are needed before the textbook asymptotic
is usable — a concrete reading of the paper's own observation that rare outcomes
get unstable weights. The two dashes are genuine, not failures of the search,
and the script records why: at **G=8** the asymptote is never within 10% at any
`p` (the `(Gp)⁻²` term is still 19% at the largest `Gp` the group allows), and
at **G=256** it is already within 10% at `Gp = 1`, because there `p ≈ 4ε` and
the clip's negative bias partly cancels the positive convexity bias.

Below the threshold the single empty-group term (5.3) takes over:

| | p=0.2, G=16 | p=0.1, G=16 | p=0.3, G=16 |
|---|---|---|---|
| exact relative bias | 5.87 | 18.38 | 1.206 |
| exact / delta method alone | 23.5× | 32.7× | 8.3× |
| exact / (delta + empty group) | 1.003 | 0.972 | 1.059 |

**(c) The total-error curve, and an exact optimum.** `MSE(ε)` falls (the heavy
upper tail is truncated) and then rises (systematic under-weighting) — the same
shape as the course's truncation-plus-round-off total error. Golden-section
search in `log ε` finds the minimum, and it has a closed form:

> **ε\* = p, exactly** (derivation_exp5.md, Eq. 5.4)

| p | 0.05 | 0.2 | 0.5 |
|---|---|---|---|
| ε\* (golden section) | 0.0500000001 | 0.2000000005 | 0.5000000026 |
| relative MSE at ε\* | 0.0914 | 0.0440 | 0.0167 |
| relative MSE at ε=10⁻⁴ | 1.1×10⁵ | 1.1×10⁵ | 3.8×10² |

Over a 5×9 grid of `G ∈ {8,…,128}` and `p ∈ [0.01, 0.5]`, a least-squares fit of
`log ε\*` against `log p` gives slope **1.0000000** with **R² = 1.0**, and
`ε\*/p ∈ [1−7×10⁻⁹, 1+7×10⁻⁹]` — the residual being the
`√ε_machine ≈ 1.5×10⁻⁸` floor of any minimiser that only sees function values
near a smooth minimum. The same search reproduces the independent closed form
`λ\* = Kp(1−p)/(1−Kp)²` for the add-λ estimator of `p̂` itself to 3.6×10⁻⁸,
which is how we know the optimiser is trustworthy on objectives with no closed
form.

**Why this matters.** ε is one global constant but ε\* is per-outcome. For the
`r=(4,1)` problem, tuning ε to the majority outcome (`p₁\*=0.8`) caps the weight
dynamic range at `1/ε = 1.25 < 4` and **guarantees** the minority dies at any
group size; tuning it to the minority (`ε=0.2`) gives `5 > 4` and it survives.
There is no single winner, which is an honest explanation of why the paper's own
ε ablation (its Table 4) does not have one either.

---

## Figure 2 — the three devices on the bias–variance plane

**(a) Trade-off curves, G=16, p=0.1, α=1.** Each knob traces a curve in
(|relative bias|, relative sd); the lower-left envelope is what one wants.

| rule | abs. rel. bias | rel. sd | rel. RMSE |
|---|---|---|---|
| clip ε=10⁻³ (the paper) | 18.4 | 38.5 | 42.6 |
| clip ε=0.1 | 0.164 | 0.204 | 0.262 |
| clip ε=0.2 | 0.508 | 0.033 | 0.509 |
| add-λ=1 | 0.118 | 0.474 | 0.488 |
| add-λ=5 | 0.593 | 0.071 | 0.598 |
| guarded Richardson | 18.4 | 38.5 | 42.6 |
| naive Richardson | 5.66 | 54.3 | 54.6 |
| **EMA β=0.02** | **0.0059** | **0.0765** | **0.0767** |

The EMA curve extends furthest toward the origin, and it is the only device that
gets there **without spending more rollouts**: it reaches a relative RMSE of
0.077 where the paper's setting has 42.6, a factor of 555, purely by remembering
previous groups. The `Binomial(G_eff)` proxy for the EMA is accurate to **2.0%
for β ≤ 0.2** but off by up to 78% at β ≈ 0.5–0.7, where `p̄` is an average of
too few groups to look binomial; the plotted EMA curve is therefore simulated,
not proxied.

Guarded Richardson is indistinguishable from plain clipping here — their
relative RMSEs agree to six significant figures (42.6184 vs 42.6186) — because
at `Gp = 1.6` the guard rejects the extrapolation in essentially every draw. It
is a well-sampled-regime correction, and panel (b) is where it lives.

**(b) Richardson extrapolation in 1/G raises the bias order from 1 to 2.**
Least-squares fits of `log|rel. bias|` against `log G`:

| | p = 0.3 | p = 0.1 |
|---|---|---|
| plain clip | **1.03** (R² 0.99985) | **1.03** (R² 0.99989) |
| guarded Richardson | **2.10** (R² 0.99967) | **2.14** (R² 0.99975) |

At G=2048, p=0.3 the bias drops from 1.14×10⁻³ to 6.4×10⁻⁶ — a factor of 179 —
while the relative standard deviation is **unchanged to within 0.4–3.8%**
(sd ratio 0.962 to 0.998). This is the theoretical prediction that the
correction is variance-free to leading order, because `2p̂ = p̂_A + p̂_B` makes
the first-order term of the combination identical to the plain estimator's.

**(c) Why the correction needs a guard.** For a singleton (`n=1`, so one half is
empty) the naive rule returns

| G | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| `ω_R(1)` | −486 | −472 | −444 | −388 | −276 |

a **negative weight** — a sign-flipped reward on exactly the rare outcomes IPS
exists to protect. The probability of drawing a negative weight peaks at
**50.0%** around `Gp ≈ 1.3`, at every group size. Worse, `ω_R(1)` is precisely
the `w_G(0)` of Eq. (5.8), so by the survival criterion the naive rule makes the
minority outcome's extinction *unconditional*: no `α` and no `G` can rescue it
(Figure 3b).

Requiring both halves non-empty is not enough either: for α=1 the raw rule is
negative whenever `8 n_A n_B < (n_A+n_B)²`, i.e. whenever one half has more than
`3+2√2 ≈ 5.83` times the other, which still happens with probability up to
**2.9%**. Adding the positivity check of the classical Richardson/Romberg
convergence test (Eq. 5.11) drives it to **exactly 0** at every G tested, while
keeping `0 < ω ≤ 2·(plain weight)` and leaving the endpoints `ω(1), ω(G)`
untouched. The guard's rejection probability decays exponentially in `Gp`, which
is why it does not spoil the order-2 bias:

| `Gp` | 2 | 5 | 19 | 26 | 77 |
|---|---|---|---|---|---|
| fallback probability | 0.676 | 0.116 | 7.8×10⁻⁴ | 3.4×10⁻⁴ | 7.6×10⁻¹² |

---

## Figure 3 — downstream: which trades the policy notices

**(a) The effective weight.** Whatever the rule, the mean-field dynamics see
only `w_G(p) = E[ω(1+B)]`, `B ~ Binomial(G−1, p)` (the weight given to a group
already known to contain one copy of this outcome; Eq. 5.5, verified against the
direct `φ_G(p)/p` sum to 5.1×10⁻¹⁵). Its two endpoints decide everything:

> **survival ⟺ `ω(1)/ω(G) > r₁/r₂`** (Eq. 5.8)

At G=16, α=1 the singleton-to-full-group dynamic range is

| rule | clip ε=10⁻³ | add-λ=1 | add-λ=½ | guarded Richardson | naive Richardson |
|---|---|---|---|---|---|
| `ω(1)/ω(G)` | **16.0** | 8.5 | 11.0 | **16.0** | **−472** |

This generalises Experiments 2–4's `min(G, 1/ε)^α` to an arbitrary estimator,
and it is reproduced to 3×10⁻¹⁶ by the closed forms `min(G,1/ε)^α` (clip) and
`((G+λ)/(1+λ))^α` (add-λ, independent of K).

*Aside on K.* add-λ's normaliser `G+Kλ` multiplies every weight by the same
constant, which cancels from `r_i w_G(p_i) = S`. The stationary distribution is
therefore independent of K — checked to 2.6×10⁻¹⁵ across K ∈ {2, 5, 17} — so
panel (a) (drawn with K=2) and panel (c) (trained with K=5) are comparable.

**Corollary, measured:** add-λ smoothing **always** shrinks the range, so
Laplace smoothing buys its variance reduction in the same currency clipping
does. The first group size at which the minority of `r=(4,1)` survives at α=1:

| rule | clip | add-λ=½ | add-λ=1 | guarded Richardson | naive |
|---|---|---|---|---|---|
| smallest surviving G | **5** | 6 | **8** | 5 (6 on the even-G grid) | never |

*(The split-half rule needs an even G, so the sweep only evaluates it at even
group sizes; its criterion is identical to the clip's, so its true threshold is
also G=5.)*

**(b) Extinction exponents by bisection.** `α_c` is the exponent below which the
minority dies:

| G | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| clip (= Experiment 2's `ln4/lnG`) | 1.0000 | 0.6667 | 0.5000 | 0.4000 | 0.3333 |
| guarded Richardson | 1.0000 | 0.6667 | 0.5000 | 0.4000 | 0.3333 |
| add-λ=½ | 1.2619 | 0.7992 | 0.5781 | 0.4507 | 0.3686 |
| add-λ=1 | 1.5129 | 0.9217 | 0.6478 | 0.4945 | 0.3982 |

- bisection against the closed form: max difference **7.0×10⁻¹¹**;
- `α_c(clip)` vs `α_c(guarded Richardson)`: **0.0 exactly** — the guarded
  correction is free of any diversity cost;
- the clip row reproduces Experiment 2's table exactly, here derived through a
  completely different route (the general dynamic-range theorem rather than the
  `min(G,1/ε)^α` ceiling argument).

**(c) Monte-Carlo training, K=5, r=(5,4,3,2,1), α=1**, h=0.01, 100 000 steps,
32 seeds, policy time-averaged over the second half. `ℓ₁` distance to the ideal
`p\* ∝ r^{1/α}`:

| G | clip (paper) | add-λ=1 | add-λ=½ | guarded Richardson |
|---|---|---|---|---|
| 8 | 0.2439 | 0.4294 | 0.3383 | **0.2172** |
| 16 | 0.1092 | 0.2199 | 0.1807 | **0.0703** |
| 32 | **0.0190** | 0.1106 | 0.0666 | 0.0266 |
| 64 | **0.0016** | 0.0419 | 0.0221 | 0.0294 |

> max `ℓ₁` |Monte Carlo − finite-G mean field| = **0.0060** (worst at G=8),
> and at most 7×10⁻⁵ at G=64

So the mean field of Experiments 2–3, extended to an arbitrary weight rule,
predicts the sampled training runs; the residual is the finite-h bias
Experiment 3 already identified. Three readings:

1. **Laplace smoothing is straightforwardly worse**, at every group size. It
   reduces variance, but the variance was not what set the fixed point.
2. **The guarded Richardson correction helps where the bias is large** (11%
   closer at G=8, 36% closer at G=16) …
3. … **but stops improving at large G** (0.0266 at G=32, 0.0294 at G=64) while
   plain clipping keeps converging (0.0016 at G=64). The better *estimator*
   trains *worse*. Figure 4 explains why.

---

## Figure 4 — the estimator objective is not the dynamics objective

The rule that minimises `E[(ω(n) − p^{-α})²]` is not the rule the learning
dynamics want, because they see the **size-biased** functional
`w_G(p) = E[ω(1+B)]`. The extra guaranteed hit shifts the argument up by about
`(1−p)/G`, which partly cancels the convexity bias. Writing the dynamics-level
distortion `D(p) = w_G(p)·p^α − 1`, the exact expansions are

```
clip, c = 0              D = α(α−1)(1−p)/(2Gp) + O(G⁻²)
guarded Richardson       D = −α(1−p)/(Gp)      + O(G⁻²)
offset c = (1−α)/2       D =                      O(G⁻²)
```

**(a) α = 1 is special, and it is exactly the exponent IPS uses.** The first
coefficient contains a factor `(α−1)`, so it vanishes at α = 1 — and in fact the
closed form is exact there. For ε ≤ 1/G,

> **`w_G(p) = (1 − (1−p)^G)/p`**, so **`D(p) = −(1−p)^G`** exactly

verified against the binomial sum to **4.0×10⁻¹¹** over `G ∈ {2,…,256}` and
`p ∈ [10⁻¹², 1]`, and the measured `D` matches `−(1−p)^G` to **1.2×10⁻¹⁰**
wherever the law is above the binomial sum's own round-off floor. The base
paper's weight rule is therefore accurate, at the level the dynamics care about,
to *minus the probability that the group misses the outcome entirely* —
exponentially small in `Gp`, not merely `O(1/G)`. Measured orders at p=0.1:

| rule | order of the distortion in 1/G |
|---|---|
| clip (paper) | exponential, `−(1−p)^G` |
| guarded Richardson | **1.07** |
| add-λ=1 | **1.01** |

So the estimator-optimal correction *introduces* an `O(1/G)` distortion where
the paper's rule had none. That is precisely the Figure 3(c) plateau.

**(b) The coefficient as a function of α.** Measured `lim Gp·D/(1−p)` at G=4096
against theory, for α from 0.25 to 3:

| rule | theory | max relative error |
|---|---|---|
| clip | `α(α−1)/2` | **0.24%** |
| guarded Richardson | `−α` | **1.5%** |
| α-matched offset | `0` | measured coefficient ≤ **0.0011** in absolute value |

The clip's curve crosses zero at α = 1 and nowhere else, so **the accident is
exactly at IPS and nowhere else on the diversity knob.** For α ≠ 1 the paper's
rule is only first-order accurate for the dynamics.

**(c) A correction that targets the right objective.** Setting the offset to
`c = (1−α)/2` in `ω(n) = max((n−c)/(G−c), ε)^{-α}` cancels the `O(1/G)` term
exactly, and reduces to the paper's own rule at α = 1. Fitted orders of `|D|`
over `G ∈ [64, 2048]`, p = 0.3:

| α | clip | α-matched offset |
|---|---|---|
| 0.5 | 1.01 | **2.03** |
| 2 | 1.03 | **2.04** |
| 3 | 1.05 | **2.05** |

**Downstream (K=5 mean field, `ℓ₁` to the ideal `p\*`), and the honest cost.**
The offset also changes the dynamic range to `((G−c)/(1−c))^α`, which is *larger*
than `G^α` for α < 1 and *smaller* for α > 1:

| α = 0.5 | G=8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| clip | 0.2524 | 0.1260 | 0.0573 | 0.0315 | 0.0100 |
| offset c = +0.25 | **0.2051** | **0.0824** | **0.0405** | **0.0141** | **0.0021** |
| dynamic range, clip → offset | 2.83 → 3.21 | 4.00 → 4.58 | 5.66 → 6.51 | 8.00 → 9.22 | 11.3 → 13.1 |

| α = 2 | G=8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| clip | **0.2277** | **0.0349** | 0.0179 | 0.0114 | 0.0049 |
| offset c = −0.5 | 0.3053 | 0.0922 | **0.0145** | **0.0010** | **0.00013** |
| dynamic range, clip → offset | 64 → 32 | 256 → 121 | 1024 → 469 | 4096 → 1849 | 16384 → 7339 |

- For **α < 1 the offset is a strict improvement**: 5× closer at G=128 *and* a
  larger dynamic range.
- For **α > 1 it is a trade**: cutting the dynamic range by a factor 2–2.2
  costs an outcome at G=8 (support 4 instead of 5) and makes the answer worse at G=8 and G=16, but
  once the group is large enough to keep every outcome it is 11× closer at G=64
  and 37× closer at G=128.

So the practical rule is: use the α-matched offset when `G` comfortably exceeds
the group size needed for full support (Experiment 4's criterion), and the
paper's plain clip below it.

---

## Figure 5 — moving averages, the device with a memory

`p̄_t = (1−β) p̄_{t−1} + β p̂_t`, the third device in the pitch, and the only one
the paper does not consider.

**(a) Free variance reduction, no steady-state bias.** For a static policy,

> `Var(p̄) / Var(p̂) = β/(2−β)`, i.e. `G_eff = G(2−β)/β`

- Monte Carlo (60 000 chains, G=32, p=0.3, 9 values of β): max relative error
  **1.2%**, against the estimator's own ≈0.6% statistical error;
- the measured steady-state bias of `p̄` never exceeds **4.8×10⁻⁴** (≈1.4
  standard errors of the mean), consistent with the exact `E[p̄] = p`.

β = 0.1 therefore buys `G_eff = 19 G` — the variance of a 19× larger group, for
no extra rollouts. Neither ε nor λ can claim that; they buy variance by biasing
the weight.

**(b) The price: lag.** When the policy is moving at `δ` per step, the EMA
reports it as it was `(1−β)/β` steps ago:

| β | 0.02 | 0.053 | 0.14 | 0.38 | 1.0 |
|---|---|---|---|---|---|
| lag, simulated | 0.02456 | 0.008898 | 0.003043 | 0.000988 | −0.000135 |
| lag, theory `(1−β)δ/β` | 0.02450 | 0.008902 | 0.003036 | 0.000830 | 0 |

> worst z-score against the exact formula over the sweep: **1.40**

Total error `((1−β)δ/β)² + β/(2−β)·p(1−p)/G` therefore has an interior optimum.
Golden-section search gives **β\* = 0.051572** against a 4001-point scan's
0.051559 (relative difference 2.5×10⁻⁴, which is the scan's own grid resolution), and the simulated
drift test matches the predicted total error to 1.3% across the sweep.

**(c) An EMA makes the learning dynamics second order.** The `h→0` mean field
with an EMA weight has *the same fixed point* `p\* ∝ r^{1/α}` (an EMA is
unbiased in steady state, unlike ε and λ), but the linearisation changes
character. With `κ = β/h`, `∂F/∂z = 0` at the fixed point and

> **`x'' + κ x' + κ λ_j x = 0`**, i.e. `s² + κs + κλ_j = 0`

where `λ_j` are **exactly Experiment 2's linear rates**
`λ_j = α‖r‖_{1/α} μ_j`. Experiment 2's first-order relaxation has become a
damped harmonic oscillator whose damping is the EMA and whose stiffness is the
reward landscape. For r=(5,4,3,2,1), α=1 (`λ_min = 1.1498`, `λ_max = 4.5285`):

- the roots of the quadratic are eigenvalues of the analytical 2K×2K Jacobian to
  **4.2×10⁻¹⁵**, and the Jacobian matches central differences to 3×10⁻⁹;
- the asymptotic decay rate matches the closed form at every κ tested, to
  **5.1×10⁻¹⁵**;
- the policy **rings** for `κ < 4λ`, i.e. `β < 4hλ` (the inset shows it clearly
  at κ=1 and not at all at κ=200);
- the rate is **maximised at `κ\* = 4λ_min = 4.599`, where it is `2λ_min`** —
  **exactly 2.000× Experiment 2's rate**, the familiar momentum effect, here in
  closed form;
- `κ → ∞` recovers `λ_min` (no EMA), as it must.

The eigenvalues `μ_j` feeding all of this are obtained with the course's own
**power method** (with Hotelling deflation) and **QR algorithm**, written out in
`src/eigen.py` and checked against `numpy.linalg.eigvalsh`: errors **0.0** and
**2.2×10⁻¹⁶** respectively.

---

## Numerical hygiene

- **Round-off.** `p^{-α}` overflows `float64` below `p ≈ 10^{-308/α}` and the
  absolute bias `E[Ŵ] − p^{-α}` is a difference of two nearly equal huge
  numbers. Every error in this experiment is formed as `E[Ŵ p^α] − 1`, a sum of
  `O(1)` terms, and every power is evaluated as `exp(−α ln x)`.
  The clearest instance is the α=1 closed form `w_G(p) = (1−(1−p)^G)/p`, where
  `1 − (1−p)^G` cancels catastrophically for small `p`. Measured against the
  binomial sum (which subtracts nothing):

  | p | 10⁻⁹ | 10⁻¹¹ | 10⁻¹² | 10⁻¹³ |
  |---|---|---|---|---|
  | relative error, direct `(1−(1−p)^G)/p` | 2.8×10⁻⁸ | 8.3×10⁻⁸ | 2.2×10⁻⁵ | 3.1×10⁻⁴ |
  | relative error, `−expm1(G·log1p(−p))/p` | 1.2×10⁻¹⁵ | 2.1×10⁻¹⁵ | 8.9×10⁻¹⁶ | 2.3×10⁻¹⁵ |

  The direct form loses accuracy in step with the number of leading digits that
  cancel; the guarded form keeps all of them. `src/estimators.py` switches to the
  guarded form below p = 10⁻⁶.
- **Golden-section search** is only valid on a unimodal bracket. Every call is
  cross-checked against a dense scan on the same interval, and the unimodality
  of the scan is asserted rather than assumed. Its attainable accuracy is the
  `√ε_machine ≈ 1.5×10⁻⁸` floor, confirmed against two independent closed forms.
- **The unshifted QR algorithm** converges at rate `λ_{i+1}/λ_i`, so it stalls
  on a near-degenerate spectrum: over 17 test matrices (the softmax covariances,
  12 random symmetric, 2 deliberately rank-deficient) it reached the 20 000
  iteration cap on exactly one, while still returning every eigenvalue correct
  to 2.7×10⁻¹⁵ — the diagonal converges long before the off-diagonal norm
  does. Worst error over all 17: 1.2×10⁻¹⁴. A shifted,
  deflating implementation would be needed for production use.
- **Gram–Schmidt on a singular matrix.** `diag(p\*) − p\* p\*ᵀ` is singular by
  construction, which makes a plain modified-Gram–Schmidt QR produce a zero
  column and silently destroy the similarity `RQ = QᵀAQ` that the QR algorithm
  relies on. `src/eigen.py` detects the rank deficiency and completes `Q` to an
  orthonormal basis; without that the algorithm returned the right eigenvalues
  with an off-diagonal norm that never fell below 0.07.
- **Deflation needs eigenvectors, not eigenvalues.** For a symmetric matrix the
  Rayleigh quotient is accurate to the *square* of the eigenvector error, so a
  power method stopped on the change in `λ` returns a vector that is only
  `√tol`-accurate — good enough to report an eigenvalue, not good enough to
  deflate with. `power_method` therefore stops on the residual `‖Ax − λx‖`.

## Files added for this experiment

No existing file was modified: Experiments 1–4, their `src/` modules, their
derivations and their results are exactly as committed, and
`tests/test_hypergrid.py` still passes. Everything here is new:

- `experiments/exp5_bias_variance.py` — the experiment (writes to
  `results/exp5_bias_variance/{figures,data}/`).
- `src/estimators.py` — `ClipRule`, `AddLambdaRule`, `RichardsonRule`,
  `OffsetRule`, `IdealRule`; exact relative moments and central moments;
  `effective_weight` (both the count and the split size-biasing forms);
  `effective_weight_alpha1` (the closed form); `dynamic_range`;
  `negative_weight_probability` / `fallback_probability`; `sampled_step`, which
  is `dynamics.rhs_sampled` with the weight rule made pluggable.
- `src/meanfield_rule.py` — Experiment 3's nested mean-field solver generalised
  to an arbitrary weight rule, plus `check_monotone` and `extinction_alpha_K2`.
- `src/optimize1d.py` — golden-section search (linear and log), a dense scan to
  audit it, and a unimodality test.
- `src/eigen.py` — power method, Hotelling deflation, modified Gram–Schmidt QR
  with a rank-deficiency guard, and the unshifted QR algorithm.
- `src/ema.py` — the EMA variance factor, effective group size, lag, and the
  coupled second-order mean-field dynamics.
- `tests/test_estimators.py` — 55 tests.
- `derivation_exp5.md`, `RESULTS_exp5.md` (this file).
- `main.ipynb` at the repository root — runs any experiment and displays its
  figures; this is what gets imported as a Kaggle notebook.

## Scope notes

- Everything is the outcome-selection bandit of Experiments 1–4: the sampled
  update is the group-REINFORCE estimator, without GRPO's per-group advantage
  standardisation or PPO clipping. The conclusions are about the weight rule,
  not about GRPO's other machinery.
- The `O(1/G)` coefficients in Figure 4(b) are *measured* at a single large
  `G = 4096` rather than extrapolated; the residual `O(1/G²)` term is what the
  0.24%–1.5% disagreement with theory is.
- The exact `α=1` identity `w_G(p) = (1−(1−p)^G)/p` assumes `ε ≤ 1/G`, so that
  the clip never fires on a non-empty count. Above that threshold it does not
  hold, and Experiment 4's clip-limited regime applies instead.
- The α-matched offset is derived from the leading term of the size-biased
  expansion. It is second order, not exact; it is not claimed to be optimal
  within any larger family, and for α > 1 it costs dynamic range.
- A per-outcome EMA presumes the outcome set is small and enumerable, which is
  true of this bandit and of Experiment 4's grids, but not of the base paper's
  LLM setting, where outcomes are not revisited often enough for a running
  average to mean anything. The second-order dynamics result (5.15) is a
  statement about this model, not a recommendation for IPS-GRPO at scale.
- The training runs use one reward vector, `r=(5,4,3,2,1)`, one learning rate,
  and α = 1. The finite-h bias is of the size Experiment 3 measured and is not
  separately re-studied here.
- Figures 5(a) and 5(b) simulate only one outcome's marginal count, so they draw
  it with `rng.binomial(G, p)` rather than through the policy-level
  inverse-transform sampler of Experiments 1–3. The two are the same
  distribution; every place a *policy* is sampled (Figure 1's validation and all
  training runs) goes through `dynamics.inverse_transform_sample_batch`.
