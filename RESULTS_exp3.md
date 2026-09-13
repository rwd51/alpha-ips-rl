# Experiment 3 results — root-finding for the stationary point

Numbers below are from the run logged in
`results/exp3_rootfinding/data/exp3_summary.json` and the CSVs next to it
(regenerate with `python3 experiments/exp3_rootfinding.py`, about 6 min on a
laptop CPU, all seeds fixed). The theory is in `derivation_exp3.md`, sections
*Root-finding formulations* and *Finite group size for K > 2*.

Experiment 2 reached `p* ∝ r^{1/α}` by integrating the ODE for many e-folding
times. Here we solve `ż = 0` directly (Methodology step 4) and check every
root against that analytical `p*`. The same condition is written in three
equivalent forms:

- **drift:** `F_i = ż_i`, the ODE right-hand side itself.
- **balance:** `F_i = r_i p_i^{-α} − r_K p_K^{-α}`, i.e. the drift divided by `p_i`.
- **log:** `F_i = ln(r_i/r_K) − α(z_i − z_K)`, which is linear in `z`.

Dividing by `p_i` removes the same probability multiplier the base paper
blames for collapse. It turns out to matter for Newton too.

## Figure 1 — K=2: bisection, secant and Newton (one unknown u = z₁ − z₂)

**(a) Convergence at α=1, r=(4,1)**, root `u* = ln 4`, start `u₀ = 0`. The
empirical order is the least-squares slope of `ln e_{n+1}` against `ln e_n`:

| method | iterations to \|Δu\| < 10⁻¹⁵ | empirical order |
|---|---|---|
| bisection on [−20, 20] | 56 | linear, `e_{n+1}/e_n ≈ 0.502` |
| secant (drift) | 9 | **1.61** (theory 1.618) |
| Newton (drift) | 5 | **2.00** |
| Newton (balance) | 6 | **2.00** |
| Newton (log) | 1 | exact in one step |

All of them land on the analytical root to machine precision.

**(b) Newton's basin on the drift**, over α ∈ [0.2, 4] × u₀ ∈ [−15, 15]:

| α | 0.5 | 1 | 2 |
|---|---|---|---|
| starts converging (drift form) | **14%** | **17%** | 100% |

- For α ≤ 1 the drift is bounded, and flat far from the root, so a Newton
  step from far away overshoots and never comes back. The basin is a strip
  a few logits wide around `u*`.
- From **α = 1.113** on, Newton on the drift converges from every start in
  the grid. At α = 1.076 it still converges from only 59%.
- Newton on the balance form converged from **every** start at every α (and
  the log form trivially does).

This is the mirror image of Experiment 1's Fig. 3 note. There, a bounded
right-hand side meant no explicit-Euler instability. Here the same
boundedness means Newton is only locally convergent. The unbounded α > 1
drift that made Experiment 2 stiff is the one Newton handles globally.

**(c) Far from the root Newton is only linear.** In the exponential tails a
Newton step moves `u` by a constant `1/k`, so the iteration count grows like
`k |u₀ − u*|`. Least-squares slopes of iterations against `|u₀ − u*|` (for
`|u₀ − u*| > 8`):

| case | fitted slope | theory `k` |
|---|---|---|
| balance, α=0.5 | 0.504 | 0.5 |
| balance, α=1 | 1.000 | 1 |
| balance, α=2 | 2.000 | 2 |
| drift, α=2 | 1.002 | α − 1 = 1 |

Bisection on [−40, 40] always takes 47 iterations, whatever the start. The
log form always takes one step, plus one more to confirm convergence.
Balance-form Newton from 30 logits away at α=2 (63 iterations) is slower
than bisection.

**Takeaway:** Newton's quadratic rate is a local statement. How the
stationarity equation is written decides the basin and the far-field cost.
The log form is best on both counts.

## Figure 2 — K=5, r=(5,4,3,2,1): multivariate Newton-Raphson

**Gauge.** Shifting every logit leaves `ż` unchanged, so the full 5×5 Jacobian
is singular. At `p*` (α=1) its smallest singular value is **7.1 × 10⁻¹⁷**.
Every run below fixes `z₅ = 0` and solves for 4 unknowns.

**(a) From the uniform policy at α=1**, all forms reach `‖y − y*‖∞ ~ 10⁻¹⁶`:

| form | iterations | empirical order |
|---|---|---|
| drift | 6 | 1.98 |
| balance | 6 | 1.94 |
| log | 2 (1 + check) | exact in one step |

At α=0.5 the drift form diverges from the same uniform start. After 4 steps
it sits at a simplex vertex, with p = (6×10⁻⁹, 0.99999999, 0, 3×10⁻⁹, 2×10⁻¹⁰).

**(b) Success from 200 random starts** (logits ~ N(0, 3²)), α ∈ [0.2, 4]:

| α | 0.2 | 0.47 | 0.89 | 1.11 | 4 |
|---|---|---|---|---|---|
| drift | 2.5% | 0.5% | 3% | 100% | 100% |
| drift + backtracking | 3% | 1.5% | 6% | 100% | 100% |
| balance | 100% | 100% | 100% | 100% | 100% |
| log | 100% | 100% | 100% | 100% | 100% |

**Backtracking does not rescue the drift form.** For α < 1 the drift also
vanishes on every face of the simplex (`r_i p_i^{1−α} → 0` as `p_i → 0`), so
each vertex is a zero "at infinity". A line search that only asks for
`‖F‖` to decrease is drawn toward those zeros.

- **Plain drift Newton, α < 1:** 97–99.5% of the starts end with some
  `p_i < e⁻³⁰`.
- **With backtracking:** 63–87% still end there.

The 100% success of the balance form costs more iterations at large α: the
median over converged starts is 7 at α=0.2 and 25 at α=4. That is the
far-field linear phase from Fig. 1c.

**(c) Condition number of the Newton system at the root:**

| α | 0.1 | 1 | 5 |
|---|---|---|---|
| drift, no gauge (5×5) | 6 × 10¹⁴ | 1.6 × 10¹⁶ | 2.4 × 10¹⁶ |
| drift, gauge `z₅=0` | 7.7 × 10⁶ | 20.3 | 6.5 |
| balance / log, gauge | **1** | **1** | **1** |
| `λ_max/λ_min` (Exp. 2) | 1.5 × 10⁶ | 4.08 | 1.30 |

- Without the gauge the system is numerically singular (cond ~ 1/ε_machine).
- With the gauge, the drift Jacobian gets worse as α → 0, following
  Experiment 2's stiffness ratio.
- At the root the balance Jacobian is exactly `−α‖r‖_{1/α} I` and the log
  Jacobian is `−α I`, so both have condition number 1.

## Figure 3 — finite group size G, K=5: a stationary point with no closed form

**(a) The effective weight.** The size-biased form
`w_G(p) = E[max((1+B)/G, ε)^{−α}]`, `B ~ Binomial(G−1, p)`, matches
Experiment 2's direct binomial sum `φ_G(p)/p` to
**2.3 × 10⁻¹⁵** (max relative difference over p ∈ [10⁻⁴, 1], G ∈ {2,4,16,64}).

- It is strictly decreasing, from the ceiling `G^α` at p=0 to 1 at p=1.
- It is well defined at p = 0, where `φ_G(p)/p` is 0/0.
- It has an exact derivative, which Newton needs.

**Nested root-finding.**

- **Inner:** for a given `S`, solve `w_G(p_i) = S/r_i` for each outcome.
- **Outer:** find `S` with `Σ p_i(S) = 1`.

Both levels are monotone with a guaranteed bracket. Over G ∈ {4, 16, 64} ×
α ∈ {0.5, 1, 2}:

| | outer iterations | total inner iterations | work (w_G + w_G′ evaluations) |
|---|---|---|---|
| bisection / bisection | 43–55 | 4 322–13 207 | 4 506–13 769 |
| safeguarded Newton / Newton | 6–16 | 89–727 | 225–1 703 |

- Both methods agree to **9.8 × 10⁻¹⁴** (max |Δp|).
- Newton does **8–20×** less work, and took 8–19× less wall time in this run.
- Two independent checks:
  - For K=2 the nested solver reproduces Experiment 2's
    `meanfield_stationary_K2`.
  - For G=2, `w₂(p) = 2^α(1−p) + p` is linear, and solving by hand gives
    p = (2/3, 1/3, 0, 0, 0) at α=1, as computed.

**Logit-space Newton cannot represent extinction.** Newton on the gauge-fixed
log-balance residual of the mean field:

- **G=16** (all 5 outcomes kept): converges in 7 iterations and matches the
  nested solver to 2.2 × 10⁻¹⁶.
- **G=4** (outcomes 4 and 5 extinct): diverges. Their logits must go to −∞.

The probability-space bracketing formulation is needed here, not just
convenient.

**(b) Stationary distribution vs G at α=1** (lines: nested solver; markers:
the Monte-Carlo agent, h=0.01, 100 000 steps, 16 seeds, averaged over the
second half):

| G | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 32 | 64 | ideal |
|---|---|---|---|---|---|---|---|---|---|---|
| outcomes kept | 2 | 3 | 3 | 4 | 4 | 4 | 5 | 5 | 5 | 5 |
| p₅ (r=1) | 0 | 0 | 0 | 0 | 0 | 0 | 0.0187 | 0.0572 | 0.0659 | 0.0667 |
| p₁ (r=5) | 0.667 | 0.554 | 0.495 | 0.438 | 0.403 | 0.375 | 0.358 | 0.337 | 0.334 | 0.333 |

- **MC against mean field:** max |MC − mean field| = **0.0049** (at G=2),
  and it falls to 1.3 × 10⁻⁵ at G=64.
- **The remaining gap is a learning-rate effect.** At G=4 it shrinks with h:

| h | 0.05 | 0.02 | 0.01 |
|---|---|---|---|
| max \|MC − mean field\|, G=4 | 0.0188 | 0.0064 | 0.0028 |

The mean field is the h → 0 limit of the sampled update. Experiment 2's
worst-case K=2 gap (0.013 at G=4, h=0.05) is of the same size as this
finite-h bias.

**(c) Extinction thresholds.** For each outcome j and group size G, bisection
on α finds the root of `h_j(α) = ln(r_j G^α) − ln S*(α)`:

| G = 16 | O2 (r=4) | O3 (r=3) | O4 (r=2) | O5 (r=1) |
|---|---|---|---|---|
| α_c, K=5 mean field | 0.0805 | 0.203 | 0.419 | **0.886** |
| two-outcome formula `ln(r₁/r_j)/ln G` | 0.0805 | 0.184 | 0.330 | 0.580 |

- **Runner-up:** its threshold equals Experiment 2's K=2 formula to within
  the bisection tolerance (max |diff| 5.0 × 10⁻¹⁰ over G ∈ [2, 256]). When
  it goes extinct the weaker outcomes are already gone and `p₁ = 1`, so
  `S* = r₁`.
- **Weaker outcomes:** their thresholds are strictly higher, because
  stronger outcomes still share mass and push `S*` above `r₁`. At G=16 the
  paper's α=1 keeps every outcome, but only just (O5 needs 0.886). The
  two-outcome formula would have claimed a comfortable margin (0.58).

**New for K > 2: some outcomes are lost at every α.** In the α → ∞ limit
outcome j survives iff `Σ_{i∈A, i≠j} (r_j/r_i)^{1/(G−1)} > |A| − 2`. For
r=(5,4,3,2,1) that gives the smallest group that keeps each outcome at
*some* α:

| outcome | O2 (r=4) | O3 (r=3) | O4 (r=2) | O5 (r=1) |
|---|---|---|---|---|
| minimum G | 2 | 2 | **3** | **6** |

- At G = 2 (O4) or G ≤ 5 (O5) no α keeps the outcome, and the bisection
  bracket up to α=40 never changes sign.
- The script checks that the α → ∞ criterion and the bisection agree for all
  52 (G, outcome) pairs.
- For K=2 the criterion always holds, which is why Experiment 2 always found
  a finite `α_c`.

In IPS-GRPO terms, the group size sets a hard floor on how many
near-optimal outcomes can be kept, and turning up the exponent cannot get
past it.

## Files added for this experiment

No existing file was modified. Experiments 1–2, `src/rootfinding.py`,
`src/finite_group.py`, `derivation.md` and `README.md` are exactly as
committed. Everything for Experiment 3 lives in new files:

- `experiments/exp3_rootfinding.py`: the experiment script (writes to
  `results/exp3_rootfinding/{figures,data}/`).
- `src/newton.py`: `newton_raphson`, `secant`, `newton_bracketed`
  (safeguarded Newton), `newton_system` (multivariate, optional Armijo
  backtracking), and `traced`. `traced` records bisection's midpoints by
  wrapping the function, so the existing `rootfinding.bisection` is reused
  unchanged.
- `src/stationarity.py`: the drift / balance / log residuals, for K=2 (with
  closed-form derivatives) and for general K (gauge-fixed, with Jacobians),
  plus `root_K2` and `gauge_root`.
- `src/finite_group_general.py`: the general-K mean field, built on
  `src/finite_group.py`:
  - `effective_weight` and `effective_weight_prime` (size-biased form).
  - `meanfield_stationary` (nested solver, any K).
  - `extinction_alpha` and `kept_at_large_alpha`.
  - `meanfield_log_residual` / `meanfield_log_jacobian`.
- `derivation_exp3.md`: the derivations, continuing `derivation.md`.
- `RESULTS_exp3.md`: this write-up.

## Scope notes

- The single sign change of `h_j(α)`, and hence that the α → ∞ criterion
  decides whether a finite threshold exists, was checked on an α grid for
  G ∈ {2,3,4,5,6,8,16,64}. It is not proven.
- The α → ∞ criterion assumes G < 1/ε (true for every G used, with ε = 10⁻³).
- The mean field is the h → 0 limit of the group-REINFORCE update of
  Experiments 1–2. It has no GRPO advantage standardization and no PPO
  clipping.
- All K=5 results use one reward vector, r=(5,4,3,2,1). The thresholds
  depend on the whole reward profile, not just on `r₁/r_j`.
