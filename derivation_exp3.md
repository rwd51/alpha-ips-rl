# Derivation: Experiment 3 — root-finding for the stationary point

Continues `derivation.md` (the α-generalized dynamics, the linearization at
`p*`, and the K = 2 finite-group theory). Equation (*) below is
`ż_i = r_i p_i^{1-α} - p_i Σ_k r_k p_k^{1-α}` from that file. Everything here is
checked numerically in `experiments/exp3_rootfinding.py` (see `RESULTS_exp3.md`).

## Root-finding formulations

At a stationary point every `ż_i = p_i (r_i p_i^{-α} - S)` vanishes. The same
root can be handed to a root finder in three forms (`src/stationarity.py`):

```
drift     F_i = ż_i                                   (the ODE right-hand side)
balance   F_i = r_i p_i^{-α} - r_K p_K^{-α}            (ż_i / p_i, differenced)
log       F_i = ln(r_i p_i^{-α}) - ln(r_K p_K^{-α})
          = ln(r_i / r_K) - α (z_i - z_K)             (linear in z)
```

**Gauge.** `ż` does not change when every logit shifts by the same amount, so
the K×K Jacobian satisfies `J 1 = 0` and is singular. Newton needs the gauge
`z_K = 0` and K-1 unknowns `y = (z_1, …, z_{K-1})`. Since `Σ_i ż_i = S - S = 0`,
dropping the last drift component loses no information.

**K = 2.** With `u = z_1 - z_2`, `p = σ(u)` and `q = 1 - p`, the drift is
`ż_1 = pq (r_1 p^{-α} - r_2 q^{-α}) = -ż_2`, so

```
g(u) = u̇ = 2pq (r_1 p^{-α} - r_2 q^{-α})
b(u)     = r_1 p^{-α} - r_2 q^{-α}          = g / (2pq),   strictly decreasing
l(u)     = ln(r_1/r_2) - α u                                 root u* = ln(r_1/r_2)/α
g'(u)    = 2 r_1 p^{1-α} q ((1-α) q - p) - 2 r_2 p q^{1-α} (q - (1-α) p)
b'(u)    = -α (r_1 p^{-α} q + r_2 q^{-α} p)
```

**Shape of the tails decides whether Newton converges.**

- `u → -∞` (so `p ≈ e^u → 0`): `g ≈ 2 r_1 e^{(1-α) u}` and `b ≈ r_1 e^{-α u}`. At `u → +∞` it is the mirror image.
- **α < 1:** `g → 0` at both ends. The drift is bounded and flat far from
  the root, so a Newton step from far away is huge and lands even further
  out. For K > 2 the same holds on every face of the simplex:
  `r_i p_i^{1-α} → 0` as `p_i → 0`, so every vertex is a zero of the
  drift "at infinity". A line search on `‖F‖` is pulled toward those
  zeros, so backtracking cannot globalize Newton on the drift.
- **α = 1:** `g → 2r_1` and `g → -2r_2`. The drift is bounded but its limits are nonzero, and it is still flat far away.
- **α > 1:** `g` grows like `e^{(α-1)|u|}`. Newton on `A e^{k|u|} - C` moves
  `u` by `(1 - C e^{-k|u|})/k → 1/k` per step. Newton converges from every
  start, but only linearly far out: about `k |u_0 - u*|` iterations, with
  `k = α - 1`.
- **Balance form:** monotone with exponential tails, so it is globally
  convergent with `k = α`: about `α |u_0 - u*|` iterations from far away.
- **Log form:** linear, so it is exact after one Newton step from anywhere.

Bisection needs no shape information: `⌈log₂(width/tol)⌉` iterations from
any bracket.

**Conditioning at the root.** With `v_i = r_i p_i^{-α}`, every `v_i = c`
(`= ‖r‖_{1/α}`) at `p*`, and `d ln p_i / d z_j = δ_ij - p_j`:

```
drift    J_red = -α c (diag(p') - p' p'ᵀ),   p' = (p*_1, …, p*_{K-1})
balance  J_ij  = -α v_i δ_ij + α p_j (v_i - v_K)  =  -α c I   at the root
log      J     = -α I
```

The balance and log Jacobians have condition number exactly 1 at the root.
The gauge-fixed drift Jacobian inherits the spread of `p*`, and its
condition number grows as α → 0 the way Experiment 2's stiffness ratio
`λ_max/λ_min` does. Good conditioning *at* the root does not guarantee a
large basin; the tails above decide that.

## Finite group size for K > 2

**Size-biased form of the weight.** `n Binom(n; G, p) / (Gp) = Binom(n-1; G-1, p)`, so

```
w_G(p) = φ_G(p) / p = E[ max((1+B)/G, ε)^{-α} ],     B ~ Binomial(G-1, p)
```

It follows that:

- `w_G` is well defined at `p = 0`, where `w_G(0) = min(G, 1/ε)^α` (the ceiling).
- `w_G(1) = 1`.
- `w_G` is **strictly decreasing** on [0, 1]: `(1+B)/G` is stochastically increasing in `p` and `x^{-α}` is decreasing.
- With `f(n) = max(n/G, ε)^{-α}` and `m = G-1`, the identity
  `d/dp E_{Bin(m,p)}[f(1+B)] = m E_{Bin(m-1,p)}[f(2+B) - f(1+B)]` gives the exact derivative Newton needs.

**Stationary conditions.** The mean drift is
`m_i = p_i (r_i w_G(p_i) - S)` with `S = Σ_k p_k r_k w_G(p_k)`. Every
outcome is either interior, with `r_i w_G(p_i) = S`, or extinct, with
`p_i = 0`. The extinct state resists invasion iff `r_i w_G(0) ≤ S`. Given
`S`, each probability is therefore

```
p_i(S) = 0                       if S / r_i ≥ w_G(0)
       = 1                       if S / r_i ≤ 1
       = w_G^{-1}(S / r_i)        otherwise (unique: w_G is monotone)
```

- `p_i(S)` is non-increasing in `S`, so `T(S) = Σ_i p_i(S) - 1` has exactly one root on `[r_max, r_max w_G(0)]`.
- `T(r_max) ≥ 0` and `T(r_max w_G(0)) = -1`.
- Both levels are monotone scalar problems with a guaranteed bracket, so bisection always works. Newton safeguarded by the bracket takes
  `dT/dS = Σ_interior 1 / (r_i w_G'(p_i))`.
- `p_i(S)` is increasing in `r_i`, so the support is always the top-reward outcomes.

**Extinction thresholds.** Outcome j is kept iff
`h_j(α) = ln(r_j min(G,1/ε)^α) - ln S*(α) > 0`.

- **Runner-up.** When outcome 2 goes extinct, all weaker outcomes are
  already extinct and `p_1 = 1`. Then `S* = r_1 w_G(1) = r_1`, and the
  threshold is exactly Experiment 2's two-outcome formula
  `α_c = ln(r_1/r_2) / ln min(G, 1/ε)`.
- **Weaker outcomes (j ≥ 3).** At outcome j's margin at least two stronger
  outcomes are interior, so `p_1 < 1` and `S* = r_1 w_G(p_1) > r_1`.
  Outcome j needs a strictly larger α than the pairwise formula says.

**α → ∞ with G < 1/ε.** The `B = 0` term dominates, so
`w_G(p) / G^α → (1-p)^{G-1}`. Interior outcomes satisfy
`r_i (1-p_i)^{G-1} = s`, and `j` is kept iff `s < r_j`. At the margin
`s = r_j` the support is `A = {i : r_i ≥ r_j}`, and `Σ_A p_i = 1` gives

```
outcome j is kept for large α   ⇔   Σ_{i∈A, i≠j} (r_j / r_i)^{1/(G-1)}  >  |A| - 2
```

- **K = 2:** the criterion always holds (the sum is positive and `|A| - 2 = 0`), matching the paper's picture.
- **K > 2:** it can fail, and then raising α never keeps outcome j. For
  r = (5,4,3,2,1) the reward-1 outcome needs `G ≥ 6` and the reward-2
  outcome needs `G ≥ 3`.
- **G = 2** can be checked by hand: `w_2(p) = 2^α (1-p) + p` is linear, and the condition reduces to
  `r_j Σ_A 1/r_i > |A| - 1`.

Experiment 3 finds `h_j(α)` has a single sign change on every scanned grid,
so this limit decides whether a finite threshold exists. That monotonicity
is checked numerically, not proven.
