# Derivation: Experiment 4 — multidimensional finite-group hypergrids

Experiment 4 extends the finite-group analysis of `derivation.md` and
`derivation_exp3.md`.  Its purpose is not to introduce another learning rule;
it tests the existing finite-group theory jointly over several parameters and
then asks how accurately a tensor grid can serve as a numerical surrogate.

All rewards used here are strictly positive, `alpha > 0`, `G >= 2`, and
`0 < epsilon <= 1`.  Rewards are normalized by their maximum before the
general-`K` root solve.  This cannot change a stationary probability, and it
prevents an absolute root-solver tolerance from depending on the reward unit.

## 1. Exact finite-group boundary for K = 2

For outcome probability `p`, a group of `G` independent samples gives
`N ~ Binomial(G,p)` and empirical frequency `p_hat=N/G`.  The expected sampled
weight and its size-biased version are

```text
phi_G(p) = E[p_hat max(p_hat, epsilon)^(-alpha)]
w_G(p)   = phi_G(p)/p
         = E[max((1+B)/G, epsilon)^(-alpha)],
           B ~ Binomial(G-1,p).
```

The size-biased form is well defined at the boundary:

```text
w_G(1) = 1,
w_G(0) = min(G, 1/epsilon)^alpha =: C_G.
```

Let `rho=r_1/r_2 >= 1` and let `p_1` be the majority probability.  An interior
stationary point satisfies

```text
rho w_G(p_1) = w_G(1-p_1).
```

The left minus right side is positive at `p_1=1/2`.  At the collapsed boundary
`p_1=1` it is `rho-C_G`.  Therefore an interior root exists precisely when

```text
C_G > rho
alpha log min(G,1/epsilon) > log rho.
```

Experiment 4 uses the signed distance

```text
m(alpha,G,epsilon,rho)
    = alpha log min(G,1/epsilon) - log rho.
```

`m>0` means the minority outcome survives; `m<=0` means the finite-group mean
field has the collapsed stationary solution.  This reduces the four-dimensional
classification boundary to the single threshold `m=0`.  It does **not** imply
that the positive minority mass is a function of `m` alone, because the full
shape of `w_G(p)` still depends on `G`, `alpha`, and `epsilon`.

### Group-limited and clip-limited regimes

The ceiling base is

```text
min(G,1/epsilon) = G              when epsilon <= 1/G,
                 = 1/epsilon      when epsilon > 1/G.
```

Thus reducing `epsilon` below `1/G` cannot change the finite-group update:
every nonzero empirical frequency is already at least `1/G`.  Conversely, when
`epsilon>1/G`, increasing `G` eventually stops increasing the maximum inverse
weight.  Figure 1b isolates this crossover.

## 2. General K: support and entropy

For any number of outcomes, a mean-field stationary point obeys the KKT-like
conditions

```text
r_i w_G(p_i) = S       when p_i > 0,
r_i w_G(0) <= S        when p_i = 0,
sum_i p_i = 1.
```

Given `S`, monotonicity of `w_G` makes every interior `p_i` a unique scalar
root.  An outer scalar solve chooses `S` so the probabilities sum to one.  The
Experiment 4 wrapper validates every result using both normalization and the
scale-free residual

```text
max(
  max_active |r_i w_G(p_i)-S|,
  max_extinct max(r_i w_G(0)-S,0)
) / |S|.
```

The two plotted diversity summaries are

```text
support size = number of p_i > 10^(-11),
normalized entropy = -sum_i p_i log(p_i) / log K.
```

For the minimum-group-size grid, rewards follow a geometric profile with a
specified endpoint spread `R=r_max/r_min`:

```text
r_i = exp[-i log(R)/(K-1)],     i=0,...,K-1.
```

Every integer `G` is tested starting at 2, so the reported first full-support
group is not an approximation from a sparse plotting grid.  “Full support” is
still numerical: it uses the declared `10^(-11)` probability tolerance.

## 3. Multilinear interpolation in log-parameter coordinates

The surrogate grid uses

```text
x = (log alpha, log rho, log epsilon)
```

at fixed `G=16`.  Log coordinates are appropriate because all three parameters
are positive and span orders of magnitude.  For a query inside one grid cell,
let `t_d in [0,1]` be its fractional position along dimension `d`.  The
multilinear interpolant is the weighted sum over the cell's `2^D` corners:

```text
f_hat(x) = sum_{c in {0,1}^D} f(x_c)
           product_d [t_d if c_d=1 else (1-t_d)].
```

It is exact at grid nodes and for every function that is affine in each
coordinate separately.  For a twice-differentiable response, uniform linear
interpolation normally has second-order error.  Here the stationary minority
mass is continuous but has derivative kinks at

```text
m = 0                 (extinction boundary),
epsilon = 1/G         (clip-activation boundary).
```

Cells crossing either surface cannot retain the smooth second-order behavior.
Experiment 4 therefore reports holdout RMSE both away from and near those
boundaries instead of quoting one convergence rate without qualification.

## Numerical safeguards specific to Experiment 4

- The batched `K=2` solver uses 64 deterministic bisection iterations.
- Forty stratified atlas cells are independently compared with the original
  scalar `meanfield_stationary_K2` implementation.
- General-`K` solves are reward-normalized, checked against the conditions
  above, and rerun with pure bisection if safeguarded Newton fails validation.
- Tensor interpolation is compared against SciPy in one through five
  dimensions by the Experiment 4 test suite.
- The full script raises rather than writing a successful summary if phase
  classification, stationarity, normalization, node exactness, or refinement
  monotonicity fails.
