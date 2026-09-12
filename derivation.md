# Derivation: the α-generalized learning dynamics

This is the ~10-line derivation the project pitch calls for (Appendix A.2 of
the base paper, with one substitution). It is verified numerically in
`src/dynamics.py` (see the sanity checks in that file's tests).

## Setup

K outcomes, fixed rewards `r_1,...,r_K`. Policy is softmax over logits
`z ∈ R^K`: `p_i = softmax(z)_i`. We scale the terminal reward by inverse
probability to a tunable power α:

```
r̃_α(o) = r(o) / p(o)^α
```

α=0 recovers the standard objective (no scaling); α=1 recovers the paper's
Inverse Probability Scaling (IPS).

## Objective and stop-gradient

Following the paper's own proof technique for IPS (Appendix A.2), define a
stop-gradient copy of the current probability, `p̄_i := stopgrad(p_i)`, so it
is treated as a constant when differentiating the objective at time t:

```
J_α(z) = E_{o~p(z)}[ r(o) / p̄(o)^α ] = Σ_k p_k(z) * (r_k / p̄_k^α)
```

## Gradient

Let `w_k := r_k / p̄_k^α` (constant w.r.t. z at this instant). Then
`J_α(z) = Σ_k p_k(z) w_k`, and using the softmax log-derivative identity
`∂p_k/∂z_i = p_k(1{i=k} - p_i)`:

```
∂J_α/∂z_i = Σ_k w_k p_k (1{i=k} - p_i)
          = p_i w_i - p_i Σ_k p_k w_k
```

Because `p̄_k(t) = stopgrad(p_k(t))` equals `p_k(t)` numerically at the
instant of evaluation, `w_i = r_i / p_i^α`, so:

```
ż_i = p_i * (r_i / p_i^α) - p_i * Σ_k p_k * (r_k / p_k^α)
    = r_i * p_i^{1-α} - p_i * Σ_k r_k * p_k^{1-α}
```

**This is equation (*) used throughout `src/dynamics.py::rhs_alpha`.**

## Sanity checks (both verified numerically to machine precision)

- **α=0:** `ż_i = r_i p_i - p_i Σ_k r_k p_k = p_i (r_i - r̄) = p_i a_i`
  — exactly the paper's Eq. (2), the standard collapse dynamics.
- **α=1:** `ż_i = r_i - p_i Σ_k r_k`
  — exactly the paper's Eq. (4), IPS.

## Stationary point

Setting `ż_i = 0` for all `i`: `r_i p_i^{-α} = Σ_k r_k p_k^{1-α} =: S_α`
(a constant, independent of `i`). Hence `p_i^α ∝ r_i`, i.e.

```
p*_i ∝ r_i^{1/α}
```

which is exactly Eq. (7) of the project pitch — the tunable "diversity
knob." As α→0 this collapses all mass onto argmax(r) (the standard
objective's true equilibrium when rewards are not exactly tied); at α=1 it
is IPS's reward-proportional law; α>1 flattens further towards uniform.

## An important subtlety this project makes precise (Experiment 1, Fig. 1a)

The paper's informal Section 3 argues collapse happens "even with exactly
equal rewards." Taken literally about the *deterministic, noiseless*
gradient flow, this is **not quite right**: if `r_1 = r_2 = ... = r_K`
exactly, then `a_i = r_i - r̄ ≡ 0` for every `i` and every `p`, so `ż ≡ 0`
identically — the objective `J(p) = Σ p_i r_i` is literally constant on the
whole simplex, and the idealized flow does not move at all (we verify this
to machine precision, drift ~1e-16, in Fig. 1a).

What the paper's claim is really describing is the **finite-sample,
stochastic** training process: real algorithms never observe the exact
population mean reward, only a finite-group empirical estimate, so the
*realized* advantage is never exactly zero even when the *true* rewards
are tied. That sampling noise is what breaks the tie, and the probability
multiplier `p_i` then amplifies it into full collapse — a Pólya-urn-style
rich-get-richer mechanism. This is exactly what Experiment 1's Figure 2
demonstrates: starting from an **exactly symmetric** initial policy
(`z0 = 0`, no artificial perturbation at all), Monte-Carlo training still
collapses, purely from sampling noise, onto an outcome chosen uniformly at
random (Fig. 2a-b), and larger Monte-Carlo group sizes `G` (less sampling
noise) measurably delay this collapse (Fig. 2c).

## Linearization at the stationary point (Experiment 2)

Perturb the logits around `z* = log p*`. With the softmax derivative
`C = diag(p) - p pᵀ`, a logit perturbation moves the probabilities by
`δp = C δz`. At `p*` every outcome shares the same value

```
r_i / p*_i^α = c = ‖r‖_{1/α} := (Σ_k r_k^{1/α})^α      (since p*_i = r_i^{1/α} / Σ_k r_k^{1/α})
```

and `S* = Σ_k r_k p*_k^{1-α} = c Σ_k p*_k = c`. Perturbing each piece of (*):

```
δ(r_i p_i^{1-α}) = (1-α) r_i p_i^{-α} δp_i = (1-α) c δp_i
δS              = Σ_k (1-α) c δp_k        = 0          (probabilities sum to 1)
δ(p_i S)        = S* δp_i + p*_i δS        = c δp_i
```

so `δż_i = (1-α) c δp_i - c δp_i = -α c δp_i`, i.e.

```
J* = -α ‖r‖_{1/α} (diag(p*) - p* p*ᵀ)
```

(`src/dynamics.py::stationary_jacobian`; it agrees with the general
analytical Jacobian `jacobian_alpha` evaluated at `z*` to ~1e-16.) `J*` is
symmetric negative semidefinite with one zero eigenvalue along `1` (shifting
every logit leaves `p` unchanged). The other K-1 eigenvalues are the
**linear rates** `λ_j = α ‖r‖_{1/α} μ_j`, where the `μ_j` are the nonzero
eigenvalues of `Cov(p*) = diag(p*) - p* p*ᵀ`.

Special cases:

- **α=1:** `J* = -R Cov(p*)`, consistent with the paper's `ż = R(p* - p)`.
- **K=2:** `μ = 2 p* q*`, so `λ = 2α ‖r‖_{1/α} p* q*` (r=(4,1), α=1: λ = 2·5·0.8·0.2 = 1.6).
- **Exact tie** (all `r_i = r̄`): `p*` is uniform, `‖r‖_{1/α} = r̄ K^α`, and every
  `μ_j = 1/K`, so `λ = α r̄ K^{α-1}` (K-1 fold).

What this predicts:

- `‖p(t) - p*‖ ~ exp(-λ_min t)` asymptotically.
- An explicit scheme is linearly stable only for `h λ_max < 2` (Euler) or
  `h λ_max < 2.7853` (RK4: the real-axis root of `R(-x) = 1` for
  `R(z) = 1 + z + z²/2 + z³/6 + z⁴/24`, i.e. of `x³ - 4x² + 12x - 24 = 0`).
- **Large α is stiff:** `‖r‖_{1/α}` grows like `K^α`, so `λ_max` grows
  exponentially in α. For α > 1 the right-hand side is also no longer bounded
  (`r p^{1-α}` blows up as `p → 0`), which is why Experiment 1 (α=0, bounded
  RHS) could not find an Euler instability but Experiment 2 can.
- **Small α is slow:** for K=2, `q* ≈ (r_2/r_1)^{1/α}`, so
  `λ ≈ 2α r_1 (r_2/r_1)^{1/α}`, which vanishes faster than any power of α.

## α = 0: algebraic, not exponential, convergence

For K=2 let `u = z_1 - z_2` and `Δ = r_1 - r_2 > 0`. With α=0, (*) gives
`u̇ = 2pqΔ = Δ / (1 + cosh u)`, which integrates exactly to

```
(u + sinh u) - (u_0 + sinh u_0) = Δ t .
```

For large u, `sinh u ≈ e^u / 2`, so `p_2 = 1/(1 + e^u) ≈ 1/(2Δt)`: the policy
reaches its boundary equilibrium only algebraically, like `t^{-1}`, with no
exponential rate. This makes α=0 a singular limit rather than just the end of
the sweep: `λ_min → 0` as α → 0, and at α=0 the exponential law is replaced by
a power law. The same formula reproduces Experiment 1's collapse-time
constant: from `u_0 = 0.1` to `p_1 = 0.99` (`u = ln 99`) it gives
`t = 53.89/Δ`, against the fitted `53.94 · Δ^{-1.00}`.
