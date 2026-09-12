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
