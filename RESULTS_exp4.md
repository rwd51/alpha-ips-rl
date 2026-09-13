# Experiment 4 results — a finite-group diversity hypergrid

Experiment 4 turns the one- and two-parameter sweeps of Experiments 2–3 into
a multidimensional atlas.  It tests the exact `K=2` phase boundary, extends the
map to general `K`, and audits whether multilinear interpolation can safely
stand in for direct finite-group root solving.

The numbers below come from `results/exp4_hypergrid/data/exp4_summary.json`,
generated with:

```text
python experiments/exp4_hypergrid.py
```

The final verification run took 225.0 seconds on Python 3.11.3 with NumPy 1.26.4,
SciPy 1.14.0, and Matplotlib 3.8.2.  The derivations and numerical safeguards
are documented in `derivation_exp4.md`.

## Figure 1 — K=2 over (alpha, G, epsilon, reward ratio)

The atlas contains

```text
51 alpha values × 15 group sizes × 5 clipping values × 5 reward ratios
= 19,125 stationary points.
```

The minority outcome should survive exactly when

```text
m = alpha log min(G,1/epsilon) - log(r1/r2) > 0.
```

### Numerical checks

- Phase-boundary classification mismatches away from floating-point equality:
  **0 / 19,125**.
- Maximum scaled stationarity residual: **1.18 × 10^-15**.
- Maximum difference between the new batched solver and 40 independently
  selected calls to the original scalar solver: **8.32 × 10^-13**.

Figure 1a shows the canonical `r1/r2=4`, `epsilon=10^-3` slice.  The numerical
zero-to-positive transition follows `alpha_c=log(4)/log(G)` over the entire
plot.  At `alpha=1`, equality occurs at `G=4`, so `G<=4` is collapsed and the
minority first becomes positive at `G=5`.

For selected plotted group sizes:

| G | 2 | 3 | 4 | 6 | 8 | 16 | 64 | 256 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p2, epsilon=0.001 | 0 | 0 | 0 | 0.1149 | 0.1570 | 0.1950 | 0.2000 | 0.2000 |
| p2, epsilon=0.1 | 0 | 0 | 0 | 0.1149 | 0.1570 | 0.1855 | 0.1996 | 0.2000 |
| p2, epsilon=0.3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The last row is not a group-size failure: clipping caps the boost at
`1/epsilon=3.33`, below the reward ratio 4, so no group size can rescue the
minority at `alpha=1`.  Conversely, when `epsilon<1/G`, changing epsilon has no
effect because every nonzero empirical frequency is already at least `1/G`.

Figure 1c combines all four axes.  Every extinct point lies at `m<=0`; positive
minority mass begins at the common `m=0` boundary.  The spread above the
boundary is expected: the sign of `m` determines survival, while the full
stationary mass depends on the shape of the finite-binomial weight.

## Figure 2 — general K support and entropy

For `r=(5,4,3,2,1)` and `epsilon=10^-3`, 765 independently checked stationary
points cover 45 alpha values and 17 group sizes.

### Solver validation

- Maximum probability-normalization error: **5.98 × 10^-12**.
- Maximum scaled KKT/stationarity residual: **2.07 × 10^-13**.
- Newton solutions requiring bisection fallback: **0**.
- Support-size decreases while increasing alpha on the grid: **0**.
- Support-size decreases while increasing G on the grid: **0**.

At `alpha=1`, the stationary distributions illustrate the group-size effect:

| G | stationary p | outcomes kept | H / log 5 |
|---:|---|---:|---:|
| 2 | (0.6667, 0.3333, 0, 0, 0) | 2 | 0.3955 |
| 4 | (0.4953, 0.3464, 0.1583, 0, 0) | 3 | 0.6257 |
| 8 | (0.4028, 0.3109, 0.2074, 0.0789, 0) | 4 | 0.7805 |
| 16 | (0.3583, 0.2856, 0.2102, 0.1272, 0.0187) | 5 | 0.8637 |
| 64 | (0.3336, 0.2669, 0.2002, 0.1334, 0.0659) | 5 | 0.9250 |
| 256 | (0.3333, 0.2667, 0.2000, 0.1333, 0.0667) | 5 | 0.9256 |

The final row is the ideal reward-proportional distribution to displayed
precision.  Increasing alpha and increasing group size both expand the support
on this atlas, but entropy continues changing after all five outcomes have
entered the support.

### Exact integer search for the minimum full-support group

Figure 2c uses geometric reward profiles and checks every integer `G` from 2
upward.  Across `K in {2,3,4,5,6,8,10}` and reward spreads from 1.25 to 32,
the minimum group size ranges from **2 to 96**.

At the endpoints of the reward-spread grid:

| K | minimum G, spread=1.25 | minimum G, spread=32 |
|---:|---:|---:|
| 2 | 2 | 33 |
| 3 | 2 | 38 |
| 4 | 2 | 46 |
| 5 | 3 | 54 |
| 6 | 3 | 62 |
| 8 | 3 | 79 |
| 10 | 4 | 96 |

Thus `G` must grow with both the number of outcomes and reward imbalance.  A
pairwise best-versus-worst ratio alone does not capture the full-support
requirement when many stronger outcomes compete for probability mass.

## Figure 3 — interpolation audit

A `17 × 17 × 17` tensor grid was constructed in
`(log alpha, log reward ratio, log epsilon)` at `G=16`.  Direct root solves at
800 independently seeded off-grid points provide the holdout truth.

### Exactness and independent implementation checks

- Maximum interpolation error at all 4,913 grid nodes: **0**.
- Affine self-check error: **4.44 × 10^-16**.
- The test suite compared scalar and vector-valued interpolation against
  SciPy `RegularGridInterpolator` in one through five dimensions: **exact
  agreement to machine precision**.

### Uniform-refinement error

| points per axis | all-point RMSE | smooth-region RMSE | near-boundary RMSE | maximum error |
|---:|---:|---:|---:|---:|
| 5 | 2.767 × 10^-2 | 2.223 × 10^-2 | 4.032 × 10^-2 | 1.699 × 10^-1 |
| 9 | 1.229 × 10^-2 | 9.905 × 10^-3 | 1.786 × 10^-2 | 6.952 × 10^-2 |
| 17 | **5.215 × 10^-3** | **2.485 × 10^-3** | **9.683 × 10^-3** | 6.337 × 10^-2 |

Empirical orders from the three refinements are 1.20 overall, 1.58 in the
declared smooth region, and 1.03 near an extinction or clip-activation
boundary.  This is the expected qualitative behavior: multilinear
interpolation improves reliably, but cells crossing a derivative kink lose
the near-second-order behavior available to a smooth response.

### Practical conclusion

A uniform hypergrid is suitable for broad exploration, but it should not be
used to make precise survival decisions near `m=0` or `epsilon=1/G`.  Direct
root solving or locally refined/adaptive grids are needed there.  The phase
margin is therefore useful twice: as the analytical survival criterion and as
an error-warning signal for a numerical surrogate.

## Scope

- This experiment studies the finite-group mean field of the repository's
  categorical sampled update, not full GRPO with group-standardized advantages
  or PPO clipping.
- The general-`K` full-support criterion uses a numerical probability tolerance
  of `10^-11`, stated in both the code and summary.
- No Monte Carlo trajectories are added here: Experiments 2–3 already validate
  the finite-group mean field against sampled dynamics.  Experiment 4 focuses
  specifically on multidimensional coverage and interpolation error.
- Nothing in Experiment 4 changes the implementations or outputs of
  Experiments 1–3.
