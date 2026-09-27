# Experiment 4 results — a finite-group diversity hypergrid

Experiment 4 turns the one- and two-parameter sweeps of Experiments 2–3 into
a multidimensional atlas. It tests the exact $K=2$ phase boundary, extends the
map to general $K$, and audits whether multilinear interpolation can safely
stand in for direct finite-group root solving.

The numbers below come from `results/exp4_hypergrid/data/exp4_summary.json` and
the CSV tables beside it, generated with:

```text
python experiments/exp4_hypergrid.py
```

Like every experiment's data folder, these files are gitignored; running the
script regenerates them.  The final run took 427 seconds on Python 3.11.3 with
NumPy 1.26.4, SciPy 1.14.0, and Matplotlib 3.8.2.  The derivations and
numerical safeguards are documented in `derivation_exp4.md`.

## Figure 1 — $K=2$ over $(\alpha,G,\varepsilon,\text{reward ratio})$

The atlas contains

$$
\underbrace{51}_{\alpha\text{ values}}
\times
\underbrace{15}_{\text{group sizes}}
\times
\underbrace{5}_{\text{clipping values}}
\times
\underbrace{5}_{\text{reward ratios}}
=19{,}125\ \text{stationary points}.
$$

The minority outcome should survive exactly when

$$
m
=\alpha\log\min\{G,\frac{1}{\varepsilon}\}
-\log(\frac{r_1}{r_2})>0.
$$

### Numerical checks

The batched solver chooses between the collapsed and the interior branch from
$m$ itself, so comparing its output with $m$ cannot test the boundary.  Every
point was therefore also solved with the original scalar solver of
Experiment 2, which decides collapse from the sign of the balance function and
never uses $m$.

- Phase classification of the scalar solver against the sign of $m$:
  **$0$ mismatches at the $19{,}120$ points with $|m|>10^{-9}$**.  The other
  $5$ points lie exactly on $m=0$ ($\alpha=4$, $G=2$, $r_1/r_2=16$, every
  $\varepsilon$) and are collapsed, as the strict inequality requires.
- Maximum difference between the batched and scalar solvers over all
  $19{,}125$ points: **$9.09\times10^{-13}$**.
- Maximum relative balance residual, measured against the size of the two
  balanced terms: **$5.92\times10^{-15}$**.

Figure 1a shows the canonical $r_1/r_2=4$, $\varepsilon=10^{-3}$ slice. The
zero-to-positive transition follows $\alpha_c=\log(4)/\log(G)$ over the entire
plot. At $\alpha=1$, equality occurs at $G=4$, so $G\leq4$ is collapsed; an
integer search with the scalar solver finds the first surviving group at $G=5$,
with $p_2=0.0735$ ($G=5$ is not on the plotted grid).

The colour scale runs to $0.45$ because for $\alpha>1$ the minority holds more
than the $\alpha=1$ ideal of $0.2$.  At $\alpha=4$ it is not even monotone in
$G$:

| $G$ | $2$ | $4$ | $6$ | $8$ | $12$ | $16$ | $32$ | $64$ | $256$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $p_2$, $\alpha=4$ | $0.1600$ | $0.3661$ | $0.4142$ | $0.4334$ | $0.4447$ | $0.4413$ | $0.4241$ | $0.4186$ | $0.4152$ |

The $G\to\infty$ value is $1/(1+4^{1/4})=0.4142$.  The minority passes it near
$G=6$, peaks at $G=12$, and then decays back toward it.  At the peak the
finite-group weight exceeds the ideal $p^{-4}$ by a factor $3.42$ for the
minority but only $2.08$ for the majority: for large $\alpha$ the average of the
strongly convex $((1+B)/G)^{-\alpha}$ over-corrects, more so for the rarer
outcome (derivation §1).  Moderate groups therefore over-protect the minority
before large groups bring it back to the ideal.  At $\alpha=1$ the opposite
holds, and a finite group always under-protects it.

For selected group sizes at $\alpha=1$ (Figure 1b):

| $G$ | $2$ | $3$ | $4$ | $6$ | $8$ | $16$ | $64$ | $256$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $p_2$, $\varepsilon=0.001$ | $0$ | $0$ | $0$ | $0.1149$ | $0.1570$ | $0.1950$ | $0.2000$ | $0.2000$ |
| $p_2$, $\varepsilon=0.1$ | $0$ | $0$ | $0$ | $0.1149$ | $0.1570$ | $0.1855$ | $0.1996$ | $0.2000$ |
| $p_2$, $\varepsilon=0.3$ | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ |

The last row is not a group-size failure: clipping caps the boost at
$1/\varepsilon=3.33$, below the reward ratio $4$, so no group size can rescue
the minority at $\alpha=1$. Conversely, when $\varepsilon\leq1/G$, changing
$\varepsilon$ has no effect because every nonzero empirical frequency is
already at least $1/G$.  In Figure 1b the curves for $\varepsilon=10^{-3}$,
$10^{-2}$ and $0.03$ differ by at most $1.3\times10^{-5}$ and are drawn with
nested markers: the two larger values only start clipping at $G>100$ and
$G>33$, where the minority ($p_2\approx0.2$) is almost never seen few enough
times for the clip to act.

Figure 1c combines all four axes and draws all $19{,}125$ points.  Every
extinct point lies at $m\leq0$; positive minority mass begins at the common
$m=0$ boundary. The spread above the boundary is expected: the sign of $m$
determines survival, while the full stationary mass depends on the shape of the
finite-binomial weight.

## Figure 2 — general $K$ support and entropy

For $\mathbf r=(5,4,3,2,1)$ and $\varepsilon=10^{-3}$, $765$ independently
checked stationary points cover $45$ values of $\alpha$ and $17$ group sizes.

### Solver validation

Over the $765$ atlas points and the six $\alpha=1$ solves below:

- Maximum probability-normalization error: **$5.98\times10^{-12}$**.
- Maximum scaled KKT/stationarity residual: **$2.07\times10^{-13}$**.
- Newton solutions requiring bisection fallback: **$0$**.
- Support-size decreases while increasing $\alpha$ on the grid: **$0$**.
- Support-size decreases while increasing $G$ on the grid: **$0$**.

The minimum-$G$ search below adds $1{,}281$ solves, with maximum normalization
error $1.42\times10^{-12}$, maximum KKT residual $4.04\times10^{-15}$, and no
fallbacks.

$\alpha=1$ is not a node of the $\alpha$ grid, so the script solves it
separately.  The stationary distributions illustrate the group-size effect:

| $G$ | stationary $\mathbf p$ | outcomes kept | $H/\log 5$ |
|---:|---|---:|---:|
| $2$ | $(0.6667, 0.3333, 0, 0, 0)$ | $2$ | $0.3955$ |
| $4$ | $(0.4953, 0.3464, 0.1583, 0, 0)$ | $3$ | $0.6257$ |
| $8$ | $(0.4028, 0.3109, 0.2074, 0.0789, 0)$ | $4$ | $0.7805$ |
| $16$ | $(0.3583, 0.2856, 0.2102, 0.1272, 0.0187)$ | $5$ | $0.8637$ |
| $64$ | $(0.3336, 0.2669, 0.2002, 0.1334, 0.0659)$ | $5$ | $0.9250$ |
| $256$ | $(0.3333, 0.2667, 0.2000, 0.1333, 0.0667)$ | $5$ | $0.9256$ |

The final row is the ideal reward-proportional distribution to displayed
precision. Increasing $\alpha$ and increasing group size both expand the support
on this atlas, but entropy continues changing after all five outcomes have
entered the support.

### Exact integer search for the minimum full-support group

Figure 2c uses geometric reward profiles and checks every integer $G$ from $2$
upward. Across $K\in\{2,3,4,5,6,8,10\}$ and reward spreads from $1.25$ to $32$,
the minimum group size ranges from **$2$ to $96$**.

At the endpoints of the reward-spread grid:

| $K$ | minimum $G$, spread $=1.25$ | minimum $G$, spread $=32$ |
|---:|---:|---:|
| $2$ | $2$ | $33$ |
| $3$ | $2$ | $38$ |
| $4$ | $2$ | $46$ |
| $5$ | $3$ | $54$ |
| $6$ | $3$ | $62$ |
| $8$ | $3$ | $79$ |
| $10$ | $4$ | $96$ |

Thus, $G$ must grow with both the number of outcomes and reward imbalance. A
pairwise best-versus-worst ratio alone does not capture the full-support
requirement when many stronger outcomes compete for probability mass.

## Figure 3 — interpolation audit

A $17\times17\times17$ tensor grid was constructed in
$(\log\alpha,\log\rho,\log\varepsilon)$ at $G=16$; the $5$- and $9$-point grids
used for refinement are its subgrids.  Direct root solves at $800$
independently seeded off-grid points provide the holdout truth.

### Exactness and independent implementation checks

- Maximum interpolation error at all $4{,}913$ grid nodes: **$0$**.
- Affine self-check error: **$4.44\times10^{-16}$**.
- The test suite compares scalar- and vector-valued interpolation against SciPy
  `RegularGridInterpolator` in one through five dimensions: **agreement to
  $2\times10^{-15}$**.

### Where the kinks are

The minority mass has a derivative kink on the extinction surface $m=0$ and at
every clipping threshold $\varepsilon=k/G$, not only at $1/G$ (derivation §3).
In this grid's range there are six, $\varepsilon\in\{1/16,\ldots,6/16\}$.  For
$\alpha=1$ and $r_1/r_2=2$, where the minority survives at every $\varepsilon$,
the slope of $p_2$ with respect to $\log\varepsilon$ jumps by:

| $\varepsilon$ | $1/16$ | $2/16$ | $3/16$ | $4/16$ | $5/16$ | $6/16$ |
|---|---:|---:|---:|---:|---:|---:|
| $p_2$ | $0.3330$ | $0.3316$ | $0.3271$ | $0.3160$ | $0.2905$ | $0.2333$ |
| slope jump | $-0.0027$ | $-0.0109$ | $-0.0287$ | $-0.0582$ | $-0.0931$ | $-0.0868$ |

At $\varepsilon=0.09$, between two thresholds, the jump is $2\times10^{-9}$,
i.e. none.  Here the minority is seen about five times per group, so the higher
thresholds carry the larger jumps.  On the $17$-point grid every cell reaching
above $\varepsilon=1/G$ contains a threshold: all $151$ holdout points with
$\varepsilon>1/G$ whose cell is not entirely extinct sit in a cell with a clip
kink.

Each holdout point is attributed to the cell its interpolation uses.  At $17^3$
the $800$ points split into $364$ kink-free cells with a surviving minority,
$230$ cells where the minority is extinct throughout (interpolated exactly), and
$206$ kinked cells: $110$ with a clip threshold only, $51$ with the extinction
surface only, and $45$ with both.

### Uniform-refinement error

Two fixed populations are followed across refinements: the $213$ points whose
$5$-point cell is kink-free with a surviving minority, which stay in such cells
at $9$ and $17$ points, and the $206$ points whose $17$-point cell contains a
kink, so every coarser cell does too.  Holdout RMSE:

| points per axis | all $800$ points | kink-free cells ($213$) | kinked cells ($206$) | maximum error |
|---:|---:|---:|---:|---:|
| $5$ | $2.767\times10^{-2}$ | $5.182\times10^{-3}$ | $4.777\times10^{-2}$ | $1.699\times10^{-1}$ |
| $9$ | $1.229\times10^{-2}$ | $1.135\times10^{-3}$ | $2.369\times10^{-2}$ | $6.952\times10^{-2}$ |
| $17$ | **$5.215\times10^{-3}$** | **$3.137\times10^{-4}$** | **$1.024\times10^{-2}$** | $6.337\times10^{-2}$ |

Observed orders between successive refinements, $\log_2(e_N/e_{2N})$:

| population | $4\to8$ intervals | $8\to16$ intervals |
|---|---:|---:|
| all points | $1.17$ | $1.24$ |
| kink-free cells | $2.19$ | $1.86$ |
| kinked cells | $1.01$ | $1.21$ |

This is what the kink structure predicts: second order where the response is
smooth and about first order in cells that contain a kink.  The kinked cells
dominate the overall error.  At $17^3$ all $38$ errors above $0.01$ are in
kinked cells, and the RMSE there is $16$ times that of the kink-free cells
($1.024\times10^{-2}$ against $6.26\times10^{-4}$); at $9^3$, $112$ of the $118$
errors above $0.01$ are.

The maximum error barely improves from $9$ to $17$ points per axis ($0.0695$ to
$0.0634$).  At every level the worst point sits at $\varepsilon\approx0.27$–$0.32$
in a cell that contains both the extinction surface and a clip threshold.  How
much a kink costs a nearby point depends on where the kink falls inside its
cell, so the worst of $800$ points need not halve when the spacing does.

### Survival decisions from the surrogate

$277$ of the $800$ holdout points are exactly extinct.  The interpolant never
predicts extinction for a surviving point: the corner of its cell with the
largest margin also survives and carries positive weight.  The reverse error is
common near $m=0$:

| points per axis | $5$ | $9$ | $17$ |
|---:|---:|---:|---:|
| extinct points predicted $p_2>10^{-3}$ | $137$ | $73$ | $37$ |
| extinct points predicted $p_2>10^{-2}$ | $85$ | $36$ | $17$ |

### Practical conclusion

A uniform hypergrid is suitable for broad exploration, but not for precise
values near $m=0$ or anywhere above $\varepsilon=1/G$, where the clip thresholds
are about as dense as the grid.  Survival should be decided from the
closed-form margin, not from the surrogate.  Because the clip kinks sit at the
known values $k/G$, an $\varepsilon$ axis with nodes on those values would avoid
them; near $m=0$, direct root solving or locally refined grids are needed.  The
phase margin is therefore useful twice: as the analytical survival criterion
and as an error-warning signal for a numerical surrogate.

## Scope

- This experiment studies the finite-group mean field of the repository's
  categorical sampled update, not full GRPO with group-standardized advantages
  or PPO clipping.
- The general $K$ full-support criterion uses a numerical probability tolerance
  of $10^{-11}$, stated in both the code and summary.
- No Monte Carlo trajectories are added here: Experiments 2–3 already validate
  the finite-group mean field against sampled dynamics.  Experiment 4 focuses
  specifically on multidimensional coverage and interpolation error.
- Nothing in Experiment 4 changes the implementations or outputs of the other
  experiments.

## Revision notes

Revised on 2026-09-27 after an audit.  No stationary point, interpolant, or
minimum-$G$ value changed; the changes are in how the results are checked,
analysed, and shown.

- Figure 1's classification check compared the batched solver with the margin
  it uses internally, so it could not fail.  It now uses the margin-free scalar
  solver at every point (still $0$ mismatches).  The balance residual was
  divided by the weight ceiling, which reaches $7\times10^{10}$; it is now
  relative to the balanced terms.
- Figure 3 treated $\varepsilon=1/G$ as the only clip kink and used fixed-width
  bands around the kinks.  There is a kink at every $\varepsilon=k/G$: $124$ of
  the $608$ points the old analysis called smooth sat in kinked $9^3$ cells, and
  the old "smooth-region order 1.58" was a three-level least-squares fit across
  both regimes.  Errors are now attributed per cell and orders are pairwise.
- Figure 1a's colour scale stopped at $0.20$ and hid $37\%$ of the slice,
  including the $\alpha=4$ overshoot; Figure 1c's axis limits cut $12\%$ of the
  points.  Both now show all data.  Figure 1b's overlapping curves, Figure 2c's
  labels, and Figure 3b's legend were made legible.
- The $\alpha=1$ tables are now computed by the script and stored in the
  summary; the old `support_at_alpha1_G16` entry was evaluated at the nearest
  grid value, $\alpha=0.956$.
