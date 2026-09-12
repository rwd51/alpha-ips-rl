# Experiment 1 results — mode collapse under the standard objective (α=0)

Numbers below are from the actual run logged in
`results/data/exp1_summary.json` (regenerate with
`python3 experiments/exp1_collapse.py`).

## Figure 1 — deterministic (idealized) gradient flow

**(a) Exact reward tie ⇒ flow is provably flat.**
Maximum drift in `p_1(t)` over T=300 (RK4), across four different initial
policies (p₁(0) ∈ {0.50, 0.65, 0.80, 0.95}) under `r₁=r₂=1`:

> **max drift = 1.11 × 10⁻¹⁶** (machine epsilon)

Confirms the flow is exactly flat, not just slow — the objective
`J(p)=Σp_i r_i` is literally constant on the simplex when rewards are tied,
so gradient ascent has nothing to climb. See `derivation.md` for why this
sharpens (rather than contradicts) the paper's informal claim.

**(b)** Trajectories for reward gaps Δ ∈ {0.05, 0.1, 0.25, 0.5, 1.0, 2.0}
all collapse onto the higher-reward outcome, with the transition visibly
slower and later for smaller Δ (semi-log time axis).

**(c) Collapse-time power law**, least-squares fit in log-log space over
Δ ∈ [0.02, 4] (12 points):

> **t_collapse ≈ 53.94 · Δ^(−1.00)**,  R² = 0.99999994

A remarkably clean inverse-linear law: collapse time is (to numerical
precision) exactly proportional to 1/Δ for this system.

## Figure 2 — realistic Monte-Carlo (stochastic) training

**(a)** K=2, exact tie, exactly symmetric initialization (`z0=0`), group
size G=16, 36 seeds: sampling noise alone produces full fan-out collapse.
Outcome 1 wins **50.0%** of seeds — consistent with pure chance (expected
50%), confirming no built-in bias in the simulator.

**(b)** K=5, exact tie, 200 seeds, group size G=16 — winner frequencies:

| Outcome | O1 | O2 | O3 | O4 | O5 | uniform (1/K) |
|---|---|---|---|---|---|---|
| Frequency | 18.5% | 22.5% | 19.0% | 19.0% | 21.0% | 20.0% |

All five lie inside the 95% confidence band around 1/K=20% under the null
hypothesis of a fair/unbiased collapse (±1.96·SE ≈ [14.5%, 25.5%] for
n=200) — the collapse target really is arbitrary.

**(c)** Time-to-collapse (median ± IQR) vs Monte-Carlo group size G:

| G | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| Collapsed within budget | 100% | 100% | 100% | 97.5% | 87.5% |
| Median collapse time | 190.5 | 418.3 | 793.0 | 2479.8 | 3138.0 |

Least-squares power-law fit: **t_collapse ∝ G^1.07** (R² = 0.974) — larger
groups (less sampling noise per update) delay collapse roughly linearly in
G, a genetic-drift-style scaling law directly relevant to a real GRPO
hyperparameter (group size).

## Figure 3 — integrator validation

**(a) Empirical order of accuracy** (fit only above the ~10⁻¹¹
round-off-contaminated region):

> **Euler order ≈ 1.00** (theory: 1)
> **RK4 order ≈ 3.99** (theory: 4)

Below step size h≈0.025 the RK4 error plateaus at ~10⁻¹⁴–10⁻¹⁵ (machine
round-off floor), no longer decreasing with h — the classic truncation-error
→ round-off-error crossover from the course's error-analysis unit.

**(b) Production step size validated.** At h=0.05 (the step size used
throughout Figs. 1–2), maximum absolute error in `p₁(t)` against a
high-precision reference (`scipy.solve_ivp`, DOP853, rtol=1e-12):

> RK4: **5.1 × 10⁻⁸**   Euler: **1.4 × 10⁻⁴**

Both are far below anything visible on the probability scale used in the
other figures — the qualitative conclusions in Figs. 1–2 are not
integrator artifacts.

## A note on what Figure 3 does *not* show, and why

We searched a wide grid of reward scales and step sizes looking for
classical Euler instability (oscillation/blow-up) to contrast against RK4.
We did not find one, for a principled reason: this ODE is a *bounded,
self-damping* gradient flow (softmax keeps `p ∈ (0,1)` and the right-hand
side vanishes as any `p_i → 0` or `1`) — it is monotone ascent on a
*linear* objective over the simplex, so no step size can make an explicit
step overshoot past an equilibrium the way it can for a stiff or
oscillatory system. We report this honestly (Fig. 3a/b) rather than
manufacture an instability the system does not have.
