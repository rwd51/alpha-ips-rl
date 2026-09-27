# Derivation: Experiment 4 — multidimensional finite-group hypergrids

Experiment 4 extends the finite-group analysis of `derivation.md` and
`derivation_exp3.md`.  Its purpose is not to introduce another learning rule;
it tests the existing finite-group theory jointly over several parameters and
then asks how accurately a tensor grid can serve as a numerical surrogate.

All rewards used here are strictly positive, with $r_i>0$, $\alpha>0$,
$G\geq 2$, and $0<\varepsilon\leq 1$. Rewards are normalized by their maximum
before the general $K$ root solve. This cannot change a stationary probability,
and it prevents an absolute root-solver tolerance from depending on the reward
unit.

## 1. Exact finite-group boundary for $K=2$

For outcome probability $p$, a group of $G$ independent samples gives
$N\sim\mathrm{Binomial}(G,p)$ and empirical frequency
$\widehat p=N/G$. The expected sampled weight and its size-biased version are

$$
\begin{aligned}
\phi_G(p)
  &= \mathbb{E}[
       \widehat p\,\max\{\widehat p,\varepsilon\}^{-\alpha}
     ], \\
w_G(p)
  &= \frac{\phi_G(p)}{p} \\
  &= \mathbb{E}[
       \max\{\frac{1+B}{G},\varepsilon\}^{-\alpha}
     ],
\qquad B\sim\mathrm{Binomial}(G-1,p).
\end{aligned}
$$

The size-biased form is well defined at the boundary:

$$
w_G(1)=1,
\qquad
w_G(0)=\min\{G,\frac{1}{\varepsilon}\}^{\alpha}
\equiv C_G.
$$

Let $\rho=r_1/r_2\geq 1$ and let $p_1$ be the majority probability. An interior
stationary point satisfies

$$
\rho\,w_G(p_1)=w_G(1-p_1).
$$

The left-minus-right side is positive at $p_1=\tfrac12$. At the collapsed
boundary $p_1=1$, it is $\rho-C_G$. Therefore, an interior root exists precisely
when

$$
C_G>\rho
\quad\Longleftrightarrow\quad
\alpha\log\min\{G,\frac{1}{\varepsilon}\}>\log\rho.
$$

Experiment 4 uses the signed distance

$$
m(\alpha,G,\varepsilon,\rho)
=\alpha\log\min\{G,\frac{1}{\varepsilon}\}-\log\rho.
$$

$m>0$ means the minority outcome survives; $m\leq0$ means the finite-group mean
field has the collapsed stationary solution.  This reduces the four-dimensional
classification boundary to the single threshold $m=0$. It does **not** imply
that the positive minority mass is a function of $m$ alone, because the full
shape of $w_G(p)$ still depends on $G$, $\alpha$, and $\varepsilon$.

Nor is a finite group a uniform under-correction. At $\alpha=1$ with
$\varepsilon\leq1/G$ the weight has the closed form

$$
w_G(p)=\frac{1-(1-p)^G}{p}<\frac{1}{p},
$$

so both outcomes are under-weighted, the rarer one more (its deficit $(1-p)^G$
is larger), and the minority always holds less mass than in the $G\to\infty$
limit.  For large $\alpha$, however, $((1+B)/G)^{-\alpha}$ is strongly convex
and its average over the binomial spread can exceed $p^{-\alpha}$, by more for
the rarer outcome.  At moderate $G$ the minority can then hold *more* mass than
in the limit; Figure 1a shows this at $\alpha=4$.

### Group-limited and clip-limited regimes

The ceiling base is

$$
\min\{G,\frac{1}{\varepsilon}\}=
\begin{cases}
G, & \varepsilon\leq \dfrac{1}{G}, \\
\dfrac{1}{\varepsilon}, & \varepsilon>\dfrac{1}{G}.
\end{cases}
$$

Thus, reducing $\varepsilon$ below $1/G$ cannot change the finite-group update:
every nonzero empirical frequency is already at least $1/G$. Conversely, when
$\varepsilon>1/G$, increasing $G$ eventually stops increasing the maximum inverse
weight.  Figure 1b isolates this crossover.

## 2. General $K$: support and entropy

For any number of outcomes, a mean-field stationary point obeys the KKT-like
conditions

$$
\begin{aligned}
r_i w_G(p_i)&=S && \text{when }p_i>0, \\
r_i w_G(0)&\leq S && \text{when }p_i=0, \\
\sum_i p_i&=1.
\end{aligned}
$$

Given $S$, monotonicity of $w_G$ makes every interior $p_i$ a unique scalar
root. An outer scalar solve chooses $S$ so the probabilities sum to one. The
Experiment 4 wrapper validates every result using both normalization and the
scale-free residual

$$
\frac{1}{|S|}
\max\{
  \max_{i:\,p_i>0}|r_iw_G(p_i)-S|,
  \max_{i:\,p_i=0}[r_iw_G(0)-S]_+
\},
\qquad [x]_+\equiv\max\{x,0\}.
$$

The two plotted diversity summaries are

$$
\begin{aligned}
\text{support size}
  &= \mathrm{card}\{i:p_i>10^{-11}\}, \\
\text{normalized entropy}
  &= -\frac{\sum_i p_i\log p_i}{\log K}.
\end{aligned}
$$

For the minimum-group-size grid, rewards follow a geometric profile with a
specified endpoint spread $R=r_{\max}/r_{\min}$:

$$
r_i=\exp(-\frac{i\log R}{K-1}),
\qquad i=0,\ldots,K-1.
$$

Every integer $G$ is tested starting at $2$, so the reported first full-support
group is not an approximation from a sparse plotting grid.  “Full support” is
still numerical: it uses the declared $10^{-11}$ probability tolerance.

## 3. Multilinear interpolation in log-parameter coordinates

The surrogate grid uses

$$
\mathbf{x}=(\log\alpha,\log\rho,\log\varepsilon)
$$

at fixed $G=16$. Log coordinates are appropriate because all three parameters
are positive and span orders of magnitude.  For a query inside one grid cell,
let $t_d\in[0,1]$ be its fractional position along dimension $d$. The
multilinear interpolant is the weighted sum over the cell's $2^D$ corners:

$$
\widehat f(\mathbf{x})
=\sum_{\mathbf{c}\in\{0,1\}^{D}}
  f(\mathbf{x}_{\mathbf{c}})
  \prod_{d=1}^{D}
  t_d^{c_d}(1-t_d)^{1-c_d}.
$$

It is exact at grid nodes and for every function that is affine in each
coordinate separately.  For a twice-differentiable response, uniform linear
interpolation has second-order error.  The stationary minority mass is
continuous, but it has two kinds of derivative kink.

**The extinction kink.** On the surface $m=0$ the minority mass leaves zero with
a nonzero slope.

**The clip kinks.** Write the size-biased weight term by term:

$$
w_G(p)=\sum_{n=0}^{G-1}\binom{G-1}{n}p^n(1-p)^{G-1-n}f_n(\varepsilon),
\qquad
f_n(\varepsilon)=\max\{\frac{1+n}{G},\varepsilon\}^{-\alpha}.
$$

Term $n$ is constant for $\varepsilon\leq(1+n)/G$ and equals
$\varepsilon^{-\alpha}$ above it, so each term switches on at its own threshold.
Across $\varepsilon=k/G$ the slope of $w_G$ with respect to $\log\varepsilon$
jumps by

$$
\Delta_k=-\alpha\,(\frac{k}{G})^{-\alpha}\,\Pr[B=k-1],
\qquad B\sim\mathrm{Binomial}(G-1,p),
\qquad k=1,\ldots,G-1.
$$

Every $k/G$ is a kink, not only the first one at $1/G$ where clipping switches
on.  The jump is weighted by $\Pr[B=k-1]$, which peaks near
$k-1\approx(G-1)p$, so for an outcome that is typically seen more than once per
group the thresholds $2/G, 3/G,\ldots$ can matter more than $1/G$.  The
stationary point inherits every kink of the balance function through the
implicit-function theorem.  In the audited range, $G=16$ and
$10^{-3}\leq\varepsilon\leq0.4$, there are six:
$\varepsilon\in\{1/16,2/16,\ldots,6/16\}$.  Their spacing in $\log\varepsilon$
shrinks from $\log 2=0.69$ to $\log(6/5)=0.18$, comparable to the $17$-point
grid spacing of $0.37$, and on that grid every cell reaching above $1/G$
contains at least one of them.

**Attributing error to cells.** A multilinear interpolant uses only the $2^D$
corners of the cell holding the query point, so what matters is whether a kink
passes through that cell.  The margin $m$ increases with $\alpha$, decreases
with $\rho$ and does not increase with $\varepsilon$, so over a box its extremes
sit at two opposite corners:

$$
m_{\max}=m(\alpha_{\mathrm{hi}},\rho_{\mathrm{lo}},\varepsilon_{\mathrm{lo}}),
\qquad
m_{\min}=m(\alpha_{\mathrm{lo}},\rho_{\mathrm{hi}},\varepsilon_{\mathrm{hi}}).
$$

The extinction surface crosses the cell exactly when $m_{\min}<0<m_{\max}$.  A
clip kink crosses it when some $k/G$ lies strictly inside the cell's
$\varepsilon$ range and the minority survives somewhere in the cell
($m_{\max}>0$); where $m\leq0$ throughout, the response is identically zero and
has no kink at all.

A fixed-width band around the kink surfaces is the wrong unit, because the
affected region is one cell wide and shrinks as the grid is refined.  The
$5$-, $9$- and $17$-point grids are nested, so a point whose $5$-point cell is
kink-free, with the minority alive in it, stays in such cells at every level,
and a kink inside a point's $17$-point cell lies inside all of its coarser
cells too.  These two fixed
populations give convergence orders that do not mix regimes.  Orders are
reported between successive refinements,

$$
\text{order}=\log_2\frac{e_N}{e_{2N}},
$$

for RMSE $e_N$ on a grid with $N$ intervals per axis, rather than as one
least-squares slope through three levels, which would hide pre-asymptotic
behavior.

## Numerical safeguards specific to Experiment 4

- The batched $K=2$ solver uses $64$ deterministic bisection iterations.  It
  chooses between the collapsed and the interior branch from the margin $m$, so
  its own output cannot test the boundary.
- Every one of the $19{,}125$ atlas points is therefore re-solved with the
  original scalar `meanfield_stationary_K2` of Experiment 2.  That solver
  decides collapse from the sign of the balance function next to $p_1=1$ and
  evaluates the weight as $\phi_G(p)/p$ from the direct binomial sum, so neither
  $m$ nor the batched solver enters the phase classification.
- The $K=2$ residual is measured relative to the size of the two balanced terms,
  $\rho\,w_G(p_1)+w_G(p_2)$, not relative to the weight ceiling.
- General $K$ solves are reward-normalized, checked against the conditions
  above, and rerun with pure bisection if safeguarded Newton fails validation.
- Tensor interpolation is compared against SciPy for scalar- and vector-valued
  data in one through five dimensions, and the jump formula $\Delta_k$ is
  checked numerically, by the Experiment 4 test suite.
- The full script raises rather than writing a successful summary if phase
  classification, stationarity, normalization, node exactness, nesting of the
  cell classification, or refinement monotonicity fails.
