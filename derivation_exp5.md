# Derivation: Experiment 5 — estimator error, bias–variance, and what it costs downstream

Continues `derivation.md` (the $\alpha$-generalized flow and the $K=2$ finite-group
theory), `derivation_exp3.md` (general-$K$ mean field) and `derivation_exp4.md`
(the four-parameter atlas). Everything here is checked numerically in
`experiments/exp5_bias_variance.py` and `tests/test_estimators.py`; the measured
numbers are in `RESULTS_exp5.md`.

This is the project pitch's Section 4.4 ("Approximations and Error Analysis"),
second half: *with $N$ samples, how far is $\hat p$ from the true $p$, and how do
$\varepsilon$-clipping, Laplace smoothing and moving averages trade bias against
variance?* The paper only uses clipping (its Eq. 9 and Table 4 ablation), so the
comparison is ours.

The organising idea is the course's own total-error picture. In numerical
differentiation the step $h$ trades **truncation error** (which grows with $h$)
against **round-off error** (which grows as $h\to0$), and the total error has an
interior minimum. Here the group size $G$ and the clip $\varepsilon$ play exactly
that role: the estimator's **bias** is the truncation term and its **variance**
is the noise term. The new part is that the two are not symmetric downstream:
bias moves the fixed point the training converges to, variance does not.

---

## 0. Notation and the estimand

At a training step the policy is $p\in\Delta^{K-1}$. A group of $G$ independent
rollouts gives counts $n\sim\mathrm{Multinomial}(G,p)$, so marginally
$n_i\sim\mathrm{Binomial}(G,p_i)$ and $\hat p_i=n_i/G$. The $\alpha$-IPS update
needs the **inverse-probability weight**

$$
w_i \;=\; p_i^{-\alpha},
$$

which the algorithm never knows: it only has the group. Write $\widehat W$ for
whatever the algorithm uses in its place.

**Definition (count rule).** A *count rule* is a map
$\omega:\{0,1,\dots,G\}\to\mathbb{R}$ applied as $\widehat W_i=\omega(n_i)$. The
three devices in the pitch are count rules (or, for Richardson, a rule on a
*split* of the group):

| name | rule $\omega(n)$ | knob |
|---|---|---|
| clipped (the paper, Eq. 9) | $\max\\{n/G,\ \varepsilon\\}^{-\alpha}$ | $\varepsilon$ |
| add-$\lambda$ (Laplace $\lambda=1$, Jeffreys $\lambda=\tfrac12$) | $\left(\dfrac{n+\lambda}{G+K\lambda}\right)^{-\alpha}$ | $\lambda$ |
| Richardson (split-half), §4 | $2f(n/G)-\tfrac12\left[f(n_A/M)+f(n_B/M)\right]$, $M=G/2$ | guard $m_0$ |
| $\alpha$-matched offset, §5.1 (new) | $\max\\{\tfrac{n-c}{G-c},\ \varepsilon\\}^{-\alpha}$, $c=\tfrac{1-\alpha}{2}$ | — |
| EMA over steps, §6 | $\max\\{\bar p_i,\varepsilon\\}^{-\alpha}$, $\bar p$ exponentially averaged | $\beta$ |

with $f(x)=\max\\{x,\varepsilon\\}^{-\alpha}$ throughout.

Because $\omega$ takes finitely many values and $n_i$ is binomial, **every**
moment below is an exact finite sum — no Monte Carlo is needed to obtain the
ground truth. Monte Carlo is used only to *validate* those sums (and the
simulator that Experiments 1–3 already rely on).

$$
\mathbb{E}[\omega(n)] = \sum_{n=0}^{G}\binom{G}{n}p^n(1-p)^{G-n}\,\omega(n),
\qquad
\mathrm{Var} = \mathbb{E}[\omega^2]-\mathbb{E}[\omega]^2 .
$$

We report **relative** error throughout,

$$
b(p)=\frac{\mathbb{E}[\widehat W]}{p^{-\alpha}}-1,
\qquad
s(p)=\frac{\sqrt{\mathrm{Var}(\widehat W)}}{p^{-\alpha}},
\qquad
\mathrm{RMSE}=\sqrt{b^2+s^2},
$$

because $p^{-\alpha}$ spans many decades across outcomes and an absolute error is
meaningless when compared across $p$.

---

## 1. Truncation term: the delta method, and its expansion parameter

$\hat p$ is unbiased, $\mathbb{E}\hat p=p$, with
$\mathrm{Var}(\hat p)=p(1-p)/G$, $\mathbb{E}(\hat p-p)^3=p(1-p)(1-2p)/G^2$ and
$\mathbb{E}(\hat p-p)^4=3p^2(1-p)^2/G^2+O(G^{-3})$. Taylor-expanding a smooth $f$
about $p$ — this is literally the course's truncation-error expansion, with
$\hat p-p$ in place of the step — gives

$$
\mathbb{E}[f(\hat p)]
= f(p)+\tfrac12 f''(p)\frac{p(1-p)}{G}
+\tfrac16 f'''(p)\frac{p(1-p)(1-2p)}{G^{2}}
+\tfrac18 f''''(p)\frac{p^{2}(1-p)^{2}}{G^{2}}
+O(G^{-3}).
\tag{5.1}
$$

For $f(x)=x^{-\alpha}$ we have $f'=-\alpha x^{-\alpha-1}$ and
$f''=\alpha(\alpha+1)x^{-\alpha-2}$, so

$$
\boxed{\;b(p)\;=\;\frac{\alpha(\alpha+1)(1-p)}{2\,Gp}+O\!\left((Gp)^{-2}\right),
\qquad
s(p)\;=\;\alpha\sqrt{\frac{1-p}{Gp}}+O\!\left((Gp)^{-1}\right).}
\tag{5.2}
$$

Three consequences, all tested in Figure 1:

1. **The expansion parameter is $Gp$, not $G$.** The *expected number of hits*
   $Gp$ is what controls the error. Doubling the group helps a rare outcome
   exactly as much as doubling its probability.
2. **Variance dominates the MSE** ($s^2=O((Gp)^{-1})$ against $b^2=O((Gp)^{-2})$),
   but **bias dominates the learning dynamics**, because the mean field (§5)
   depends only on $\mathbb{E}[\cdot]$ while the zero-mean part averages out over
   steps. That asymmetry is the reason a bias correction (§4) is worth paying
   for even though it barely moves the MSE. §5.1 sharpens this: the bias the
   dynamics feel is not (5.2) but the *size-biased* distortion (5.16), and the
   two are not the same — correcting one can make the other worse.
3. **The expansion fails for $Gp\lesssim1$.** $f$ is not smooth on the realized
   range: $\hat p=0$ occurs with probability $(1-p)^G$ and $f(0)=\infty$. This is
   exactly why the paper clips. With clipping, the $n=0$ term contributes

$$
(1-p)^{G}\left(\varepsilon^{-\alpha}-p^{-\alpha}\right)
\quad\text{to the bias, and}\quad
(1-p)^{G}\left(\varepsilon^{-\alpha}-p^{-\alpha}\right)^{2}
\ \text{to the second moment,}
\tag{5.3}
$$

which, at $\varepsilon\ll p$, is astronomically larger than (5.2) until
$(1-p)^{G}\varepsilon^{-\alpha}$ falls below it. The crossover in $G$ (or in $p$)
is found by bisection in the experiment.

---

## 2. The clip as a truncation knob: the total-error curve

Raising $\varepsilon$ does two opposite things:

* it **removes the upper tail** (the rare $\hat p\approx0$ draws that produce an
  enormous weight), cutting both the $(5.3)$ bias and the variance;
* it **truncates the estimator from above** once $\varepsilon\gtrsim p$, so the
  weight is systematically too small: $\omega(n)\le\varepsilon^{-\alpha}$ always,
  and $\varepsilon^{-\alpha}<p^{-\alpha}$ once $\varepsilon>p$.

So $\mathrm{MSE}(\varepsilon)$ falls, then rises: an interior optimum, the same
shape as the course's truncation-plus-round-off total error. $\mathrm{MSE}$ is
continuous and piecewise smooth in $\varepsilon$ (kinks at $\varepsilon=n/G$), so
the minimiser is located with **golden-section search**. It turns out to have a
closed form.

### Theorem (the MSE-optimal clip is exactly $\varepsilon^\star=p$)

Only the draws with $n/G<\varepsilon$ depend on $\varepsilon$, and each of them
contributes $(\varepsilon^{-\alpha}-p^{-\alpha})^2$, so

$$
\mathrm{MSE}(\varepsilon)
=\underbrace{\Pr[n<G\varepsilon]}_{=:\Pi(\varepsilon)>0}
\left(\varepsilon^{-\alpha}-p^{-\alpha}\right)^{2}
+\sum_{n\ge G\varepsilon}\Pr[n]\left(\left(\tfrac nG\right)^{-\alpha}-p^{-\alpha}\right)^{2},
$$

and on the interior of each interval $\varepsilon\in(n/G,(n+1)/G)$, where
$\Pi$ is constant,

$$
\frac{\mathrm d\,\mathrm{MSE}}{\mathrm d\varepsilon}
=-2\alpha\,\Pi(\varepsilon)\,\varepsilon^{-\alpha-1}
\left(\varepsilon^{-\alpha}-p^{-\alpha}\right).
$$

Since $\Pi(\varepsilon)\ge\Pr[n=0]>0$ and $\alpha,\varepsilon>0$, the sign of the
derivative is the sign of $p^{-\alpha}-\varepsilon^{-\alpha}$: **negative for
every $\varepsilon<p$ and positive for every $\varepsilon>p$**, on every piece.
$\mathrm{MSE}$ is continuous, so it is strictly decreasing on $(0,p)$ and
strictly increasing on $(p,1)$:

$$
\boxed{\;\varepsilon^{\star}=p\;}
\tag{5.4}
$$

exactly, for every $\alpha>0$, every $G$, and every $p$ — and it is the unique
global minimiser, which also justifies the golden-section search. The
interpretation is simple: the best a clip can do is to make the floor equal the
quantity being estimated, $\varepsilon^{-\alpha}=p^{-\alpha}$, so that the
truncated draws carry no error at all. Measured: golden-section search returns
$\varepsilon^\star/p=1\pm7\times10^{-9}$ over $\alpha\in\{0.5,1,2,3\}$,
$G\in\{4,8,\dots,128\}$ and $p\in[0.007,0.77]$ — the residual being the
$\sqrt{\epsilon_{\text{mach}}}\approx1.5\times10^{-8}$ floor that any
derivative-free minimiser has near a smooth quadratic minimum.

**Corollary (one clip cannot serve two outcomes).** $\varepsilon$ is a single
global constant but $\varepsilon^\star$ is per-outcome, and by Eq. (5.8) below a
clip of $\varepsilon$ caps the dynamic range at $\min\\{G,1/\varepsilon\\}^\alpha$.
Tuning $\varepsilon$ to the majority outcome of the $r=(4,1)$ problem
($p_1^\star=0.8$, so $\varepsilon=0.8$) caps the range at $1.25<4$ and
**guarantees** the minority dies, at any group size; tuning it to the minority
($\varepsilon=0.2$) gives $5>4$ and it survives. This is the honest explanation
of why the paper's own $\varepsilon$ ablation (its Table 4) has no single winner:
the objective that $\varepsilon$ optimises pointwise is not the objective the
learning dynamics care about.

### A second closed form, as an optimiser sanity anchor

For the *probability* $\hat p$ itself
(not the weight) the add-$\lambda$ estimator
$\tilde p=(n+\lambda)/(G+K\lambda)$ has

$$
\mathrm{bias}=\frac{\lambda(1-Kp)}{G+K\lambda},
\qquad
\mathrm{Var}=\frac{Gp(1-p)}{(G+K\lambda)^{2}},
$$

so, with $D=G+K\lambda$, $\mathrm{MSE}(\lambda)=[\lambda^2(1-Kp)^2+Gp(1-p)]/D^2$,
and $\mathrm{d\,MSE}/\mathrm{d}\lambda=0$ reduces to
$\lambda(1-Kp)^{2}G=KGp(1-p)$, i.e.

$$
\boxed{\;\lambda^{\star}=\frac{K\,p(1-p)}{(1-Kp)^{2}}\;}
\tag{5.4b}
$$

independent of $G$. Golden-section search must reproduce (5.4b); it does, to
$4\times10^{-8}$ (checked in the test suite for several $(G,K,p)$), which is the
$\sqrt{\epsilon_{\text{mach}}}$ accuracy limit of any minimiser that only sees
function values near a smooth quadratic minimum — itself a course error-analysis
statement. This is the verification that the optimiser is trustworthy before it
is pointed at quantities with no closed form.

---

## 3. Size-biasing: the only thing about an estimator that the dynamics see

Experiments 2–3 showed that the finite-$G$ mean drift is
$m_i=p_i\left(r_i w_G(p_i)-S\right)$ with $S=\sum_k p_k r_k w_G(p_k)$, where
$w_G=\phi_G(p)/p$ and $\phi_G(p)=\mathbb{E}[\hat p\,\widehat W]$. For a general
rule the same computation goes through, and the size-biasing identity
$n\binom{G}{n}p^n(1-p)^{G-n}=Gp\binom{G-1}{n-1}p^{n-1}(1-p)^{G-n}$ gives

$$
\boxed{\;
w_G(p)\;=\;\frac{\mathbb{E}\left[\hat p\,\omega(n)\right]}{p}
\;=\;\mathbb{E}\left[\omega(1+B)\right],\qquad B\sim\mathrm{Binomial}(G-1,p).}
\tag{5.5}
$$

In words: **the effective weight is the weight the rule gives to a group in which
one designated sample is already known to be this outcome.** (5.5) is well
defined at $p=0$, where the $0/0$ of $\phi_G/p$ is resolved, and immediately
gives the two endpoints

$$
w_G(0)=\omega(1),
\qquad
w_G(1)=\omega(G).
\tag{5.6}
$$

For a *split* rule $\omega(n_A,n_B)$ the same argument, applied to whichever half
the designated sample falls in, gives the mixture

$$
w_G(p)=\tfrac12\,\mathbb{E}\!\left[\omega(1+B_A,\,B_B')\right]
      +\tfrac12\,\mathbb{E}\!\left[\omega(B_A',\,1+B_B)\right],
\tag{5.7}
$$

$B_\bullet\sim\mathrm{Binomial}(M-1,p)$, $B'_\bullet\sim\mathrm{Binomial}(M,p)$,
independent, $M=G/2$. Both forms are verified against the direct
$\phi_G(p)/p$ sum to $\sim10^{-16}$.

### Theorem (dynamic range decides survival)

For $K=2$ with $\rho=r_1/r_2\ge1$, the interior stationary condition is
$r_1w_G(p_1)=r_2w_G(1-p_1)$. The left-minus-right side is positive at
$p_1=\tfrac12$ and equals $r_1w_G(1)-r_2w_G(0)$ at $p_1=1$, so an interior root
(a surviving minority) exists **iff**

$$
\boxed{\;
\frac{w_G(0)}{w_G(1)}=\frac{\omega(1)}{\omega(G)}>\rho .}
\tag{5.8}
$$

The whole four-parameter atlas of Experiment 4 is the special case
$\omega=\max\\{n/G,\varepsilon\\}^{-\alpha}$, for which
$\omega(1)/\omega(G)=\min\\{G,1/\varepsilon\\}^{\alpha}$ — recovering
$\alpha_c=\ln\rho/\ln\min\\{G,1/\varepsilon\\}$ exactly. (5.8) generalises that to
any estimator: *only the ratio of the weight given to a singleton and the weight
given to a full group matters.*

### Corollary (Laplace smoothing always costs diversity)

For add-$\lambda$, the normalising constant $G+K\lambda$ cancels in the ratio:

$$
\frac{\omega(1)}{\omega(G)}=\left(\frac{G+\lambda}{1+\lambda}\right)^{\alpha}
\;<\;G^{\alpha}\quad\text{for every }\lambda>0,\ G>1,
$$

with no dependence on $K$. Hence

$$
\alpha_c^{\text{add-}\lambda}
=\frac{\ln\rho}{\ln\frac{G+\lambda}{1+\lambda}}
\;>\;
\frac{\ln\rho}{\ln G}
=\alpha_c^{\text{clip}}\quad(\varepsilon\le 1/G).
\tag{5.9}
$$

Laplace smoothing ($\lambda=1$) shrinks the usable dynamic range from $G$ to
$(G+1)/2$ — roughly halving the reward ratio IPS can protect. It buys variance
the same way $\varepsilon$ does, and it pays in the same currency. Verified to
machine precision for $G\in\\{8,16,32\\}$, $\lambda\in\\{\tfrac12,1,2\\}$,
$K\in\\{2,5\\}$.

---

## 4. Richardson extrapolation in $1/G$

(5.1) is an asymptotic expansion in the single small parameter $1/G$ whose
coefficients $c_1(p)=\tfrac12f''(p)p(1-p)$, $c_2(p),\dots$ do not depend on $G$:

$$
E_G:=\mathbb{E}_G[f(\hat p)]=f(p)+\frac{c_1}{G}+\frac{c_2}{G^{2}}+\cdots
$$

This is precisely the structure Richardson's extrapolation is built for. With
step ratio 2,

$$
2E_G-E_{G/2}
=\left(2f+\frac{2c_1}{G}+\frac{2c_2}{G^2}\right)-\left(f+\frac{2c_1}{G}+\frac{4c_2}{G^2}\right)
=f(p)-\frac{2c_2}{G^{2}}+O(G^{-3}),
$$

so the $O(1/G)$ bias is annihilated. The *estimator* that realises this splits
the group into two halves $A,B$ of size $M=G/2$:

$$
\boxed{\;
\omega_{\mathrm R}(n_A,n_B)
=2f\!\left(\frac{n_A+n_B}{G}\right)-\frac12\left[f\!\left(\frac{n_A}{M}\right)+f\!\left(\frac{n_B}{M}\right)\right].}
\tag{5.10}
$$

Each half is $\mathrm{Binomial}(M,p)$, so $\mathbb{E}[\omega_{\mathrm R}]=2E_G-E_{M}$
exactly, i.e. bias $O(G^{-2})$.

**The variance is free, to leading order.** Because $2\hat p=\hat p_A+\hat p_B$,
the first-order term of (5.10) is

$$
f'(p)\left[2(\hat p-p)-\tfrac12\big((\hat p_A-p)+(\hat p_B-p)\big)\right]
=f'(p)\,(\hat p-p),
$$

identical to the plain estimator's. So Richardson raises the bias order from 1 to
2 while leaving the leading variance unchanged; only $O(G^{-1})$ relative
corrections to the standard deviation remain. (Measured: variance ratio
$0.993$–$0.997$ at $G\ge512$, $p=0.3$.)

**Why the naive version must not be used.** The expansion assumes $f$ is smooth
over the realized range, which fails whenever a half is empty:
$f(0)=\varepsilon^{-\alpha}$. For a singleton ($n=1$, so one half has $0$),

$$
\omega_{\mathrm R}(1,0)=2G^{\alpha}-\tfrac12\left(\tfrac{G}{2}\right)^{\alpha}-\tfrac12\varepsilon^{-\alpha},
$$

which is **negative** for any realistic $\varepsilon$ ($\alpha=1$,
$\varepsilon=10^{-3}$: negative for all $G<286$). A negative weight is a
sign-flipped reward: the update actively pushes a rare outcome *down*. By (5.6)
and (5.8), $w_G(0)=\omega_{\mathrm R}(1,0)<0<\rho\,\omega_{\mathrm R}(G/2,G/2)$,
so naive Richardson guarantees collapse. This is the clean example of the
course's warning that extrapolation is only valid where the assumed asymptotic
error form actually holds.

**Guarded Richardson.** Apply the correction only where the expansion is valid.
Requiring both halves to be informative is *not enough*: even with
$\min\\{n_A,n_B\\}\ge1$, a lopsided split still gives a negative value (for
$\alpha=1$, exactly when $8n_An_B<(n_A+n_B)^2$, i.e. when one half has more than
$3+2\sqrt2\approx5.83$ times the other), which happens with probability up to
about $3\%$ at small $Gp$. The guard therefore also rejects an extrapolation
that disagrees this violently with the unextrapolated value — the standard
Richardson/Romberg convergence check:

$$
\omega_{\mathrm{RG}}(n_A,n_B)=
\begin{cases}
\omega_{\mathrm R}(n_A,n_B), & \min\\{n_A,n_B\\}\ge m_0\ \text{ and }\ \omega_{\mathrm R}>0,\\[2pt]
f\!\left(\dfrac{n_A+n_B}{G}\right), & \text{otherwise,}
\end{cases}
\qquad m_0\ge1 .
\tag{5.11}
$$

Because $\omega_{\mathrm R}\le 2f((n_A+n_B)/G)$ always (a positive quantity is
subtracted from twice the plain estimate), the guarded rule obeys the a-priori
bound

$$
0<\omega_{\mathrm{RG}}\le 2\,f\!\left(\frac{n_A+n_B}{G}\right),
$$

so it can never flip the sign of a reward and can never exceed twice the paper's
own weight. A singleton always falls back, so

$$
\omega_{\mathrm{RG}}(1)=\min\\{G,1/\varepsilon\\}^{\alpha},
\qquad
\omega_{\mathrm{RG}}(G)=1,
$$

and by (5.8) the guarded rule has **exactly the same extinction threshold as the
paper's clipped rule**, while removing the $O(1/G)$ bias wherever the group
actually resolves the outcome. The rejection probability is exponentially small
in $Gp$, so it does not disturb the $O(G^{-2})$ order in the well-sampled
regime.

Measured at $\alpha=1$, $\varepsilon=10^{-3}$: fitted bias orders $1.03$ (plain)
against $2.10$ (guarded) at $p=0.3$, and $1.03$ against $2.14$ at $p=0.1$;
relative standard deviations agree to better than $0.4\%$; and
$\Pr[\omega<0]=0$ exactly, against up to $3\%$ for the $\min\ge1$ guard alone and
$100\%$ for the naive rule at small $p$.

---

## 5. Downstream: the mean field for an arbitrary rule

With $w_G$ from (5.5)/(5.7), Experiment 3's nested solver applies verbatim:
interior outcomes satisfy $r_iw_G(p_i)=S$, extinct ones satisfy
$r_iw_G(0)\le S$, and $\sum_ip_i=1$ fixes $S$. The solver needs $w_G$ to be
strictly decreasing; that is immediate for clip and add-$\lambda$ (both are
decreasing in $n$ and $1+B$ is stochastically increasing in $p$), and it is
*checked numerically* on a dense grid for the guarded Richardson rule, which is
not monotone in $(n_A,n_B)$ termwise. The metric reported is the $\ell_1$
distance between the finite-$G$ mean-field policy and the ideal
$p^\star\propto r^{1/\alpha}$ — the same $\ell_1$ the base paper uses in its
Table 1.

---

## 5.1 The dynamics objective is not the estimator objective

Sections 1–4 minimised $\mathbb{E}[(\omega(n)-p^{-\alpha})^2]$ with
$n\sim\mathrm{Binomial}(G,p)$. That is the wrong target. By (5.5) the dynamics
see $w_G(p)=\mathbb{E}[\omega(1+B)]$ with $B\sim\mathrm{Binomial}(G-1,p)$: the
weight of a group already known to contain one copy of this outcome. Define the
**dynamics-level distortion**

$$
D(p)\;:=\;w_G(p)\,p^{\alpha}-1 ,
$$

which is zero exactly when the finite-$G$ mean field has the ideal stationary
law $p^\star\propto r^{1/\alpha}$.

### Expansion

Write $X=(1+B)/G$. Then

$$
\mathbb{E}[X]=p+\frac{1-p}{G},
\qquad
\mathrm{Var}(X)=\frac{(G-1)p(1-p)}{G^{2}} .
$$

The extra guaranteed hit biases the argument *upward* by $(1-p)/G$, while
convexity of $x^{-\alpha}$ biases the value upward through the variance. Applying
(5.1) about $\mathbb{E}[X]$,

$$
\mathbb{E}[X^{-\alpha}]
=\mathbb{E}[X]^{-\alpha}\left(1+\frac{\alpha(\alpha+1)}{2}\frac{\mathrm{Var}(X)}{\mathbb{E}[X]^{2}}\right)+O(G^{-2})
=p^{-\alpha}\left(1-\frac{\alpha(1-p)}{Gp}+\frac{\alpha(\alpha+1)(1-p)}{2Gp}\right)+O(G^{-2}),
$$

so the two effects partly cancel and

$$
\boxed{\;D_{\text{clip}}(p)=\frac{\alpha(\alpha-1)(1-p)}{2\,Gp}+O(G^{-2}).}
\tag{5.16}
$$

The factor $(\alpha-1)$ is the whole story: **the leading dynamics-level
distortion of the base paper's own weight rule vanishes at $\alpha=1$**, exactly
the exponent IPS uses, and nowhere else on the diversity knob.

### At $\alpha=1$ it is not merely second order — it is exact

For $\varepsilon\le1/G$ the clip never fires on a non-empty count, and with
$\binom{G-1}{b}/(1+b)=\binom{G}{b+1}/G$,

$$
\mathbb{E}\left[\frac{1}{1+B}\right]
=\sum_{b=0}^{G-1}\binom{G-1}{b}\frac{p^{b}q^{G-1-b}}{1+b}
=\frac{1}{Gp}\sum_{b=0}^{G-1}\binom{G}{b+1}p^{b+1}q^{G-(b+1)}
=\frac{1-q^{G}}{Gp},
\qquad q=1-p .
$$

Hence $w_G(p)=G\,\mathbb{E}[(1+B)^{-1}]$ is available in closed form:

$$
\boxed{\;w_G(p)=\frac{1-(1-p)^{G}}{p},
\qquad
D(p)=-(1-p)^{G}.}
\tag{5.17}
$$

So at $\alpha=1$ the paper's rule is right to within *minus the probability that
the group misses the outcome entirely* — exponentially small in $Gp$, not
$O(1/G)$. Two corollaries fall out immediately and agree with Experiments 2–4:
$w_G(0)=G$, $w_G(1)=1$, so the dynamic range is exactly $G$; and the whole
$\alpha=1$ finite-$G$ mean field becomes a closed-form root problem with no
binomial sums (checked against the Experiment 2 and 3 solvers to $10^{-12}$).

### Why a better estimator can train worse

The guarded Richardson rule of §4 is built so that
$\mathbb{E}_{n\sim\mathrm{Bin}(G,p)}[\omega_{\mathrm{RG}}(n)]=p^{-\alpha}+O(G^{-2})$;
that is, it removes the term $+\alpha(\alpha+1)(1-p)/(2Gp)$. Evaluated instead on
the size-biased count, its distortion is (5.16) minus that same term:

$$
\boxed{\;D_{\text{Richardson}}(p)=-\frac{\alpha(1-p)}{Gp}+O(G^{-2}).}
\tag{5.18}
$$

Which is *not* zero at $\alpha=1$. **Correcting the estimator bias destroys the
cancellation that made the plain rule dynamics-exact.** This is the precise
reason Experiment 5's Figure 3(c) shows the guarded Richardson rule beating the
clip at $G=8,16$ (where the raw bias dominates) and losing to it at $G=32,64$
(where only the $O(1/G)$ distortion is left). Measured coefficients at
$G=4096$, $p=0.3$: (5.16) to $0.24\%$ and (5.18) to $1.5\%$.

### A correction that does target the dynamics

Take the one-parameter offset family

$$
\omega_c(n)=\max\\{\tfrac{n-c}{G-c},\ \varepsilon\\}^{-\alpha}.
$$

With $X_c=(1+B-c)/(G-c)$ we get, using $(G-1)p+1-c=p(G-c)+(1-c)(1-p)$,

$$
\mathbb{E}[X_c]=p+\frac{(1-c)(1-p)}{G-c},
\qquad
\mathrm{Var}(X_c)=\frac{(G-1)p(1-p)}{(G-c)^{2}},
$$

so the leading distortion is

$$
D_c(p)=\left[-\alpha(1-c)+\frac{\alpha(\alpha+1)}{2}\right]\frac{1-p}{Gp}+O(G^{-2}),
$$

which vanishes when $1-c=(\alpha+1)/2$:

$$
\boxed{\;c^{\star}=\frac{1-\alpha}{2}.}
\tag{5.19}
$$

At $\alpha=1$ this is $c^\star=0$: the base paper's rule, consistent with (5.17).
For $\alpha\neq1$ it is a different rule, and the measured order of $|D|$ rises
from $1.0$ to $2.0$ (Experiment 5, Figure 4c).

**The price, and when it is worth paying.** By (5.6) the offset rule's dynamic
range is

$$
\frac{\omega_{c}(1)}{\omega_{c}(G)}=\left(\frac{G-c}{1-c}\right)^{\alpha},
$$

and $(G-c)/(1-c)>G\iff c(G-1)>0$. So the offset **widens** the range for
$\alpha<1$ ($c^\star>0$) and **narrows** it for $\alpha>1$ ($c^\star<0$):

* $\alpha<1$: a strict improvement — better dynamics accuracy *and* better
  minority protection.
* $\alpha>1$: a trade. At $\alpha=2$ the range falls from $G^2$ to
  $((G+\tfrac12)/\tfrac32)^2$, roughly a factor $2.25$, which can cost an outcome
  at small $G$. Above the group size that keeps every outcome (Experiment 4's
  full-support criterion) the second-order accuracy wins by an order of
  magnitude; below it, the plain clip is safer.

---

## 6. Moving averages: an extra state variable

The third device in the pitch is a moving average across training steps,

$$
\bar p_t=(1-\beta)\bar p_{t-1}+\beta\,\hat p_t,
\qquad
\widehat W_t=\max\\{\bar p_t,\varepsilon\\}^{-\alpha}.
$$

### 6.1 Static policy: pure variance reduction

If $p$ is fixed, the $\hat p_t$ are i.i.d. with variance $\sigma^2=p(1-p)/G$ and
$\bar p_t=\beta\sum_{j\ge0}(1-\beta)^{j}\hat p_{t-j}$, so $\mathbb{E}\bar p=p$ and

$$
\mathrm{Var}(\bar p_\infty)=\beta^{2}\sigma^{2}\sum_{j\ge0}(1-\beta)^{2j}
=\frac{\beta}{2-\beta}\,\sigma^{2}
\;\Longrightarrow\;
\boxed{\,G_{\mathrm{eff}}=G\,\frac{2-\beta}{\beta}\,}.
\tag{5.12}
$$

An EMA with $\beta=0.1$ is worth a $19\times$ larger group *for free* — and by
(5.2) it divides the weight bias by the same factor. It costs no extra rollouts,
which neither $\varepsilon$ nor $\lambda$ can claim.

### 6.2 Moving policy: lag bias

Let $p$ drift linearly, $p_t=p_{t-1}+\delta$. With $e_t=p_t-\mathbb{E}\bar p_t$,
the recursion gives $e_t=(1-\beta)(\delta+e_{t-1})$, hence

$$
e_\infty=\frac{1-\beta}{\beta}\,\delta .
\tag{5.13}
$$

The EMA reports the policy as it was $(1-\beta)/\beta$ steps ago. Combining
(5.12) and (5.13), the per-step error of $\bar p$ is

$$
\mathrm{MSE}(\beta)\approx\left(\frac{1-\beta}{\beta}\right)^{2}\delta^{2}
+\frac{\beta}{2-\beta}\frac{p(1-p)}{G},
\tag{5.14}
$$

decreasing then increasing: another interior optimum, located by golden-section
search and compared against a direct simulation of the transient.

### 6.3 The EMA makes the learning dynamics second order

Take the $h\to0$ mean field of the update with the EMA weight. Writing
$\kappa=\beta/h$ (the EMA's relaxation rate in continuous time),

$$
\dot z_i=p_i\left(r_i\bar p_i^{-\alpha}-S\right),\quad
S=\sum_k p_k r_k\bar p_k^{-\alpha},
\qquad
\dot{\bar p}=\kappa\,(p-\bar p).
$$

Its fixed point is unchanged ($\bar p=p$ and $r_ip_i^{-\alpha}=S$, i.e.
$p^\star\propto r^{1/\alpha}$): **an EMA is unbiased in steady state**, unlike
$\varepsilon$ and $\lambda$. Linearise at $(z^\star,p^\star)$ with $x=\delta z$,
$y=\delta\bar p$, $C=\mathrm{diag}(p^\star)-p^\star p^{\star\top}$ and
$c=\lVert r\rVert_{1/\alpha}$ (so every $r_ip_i^{\star-\alpha}=c$):

* $\partial F_i/\partial z_j=0$, because the only $z$-dependence sits in a factor
  multiplying the advantage, which vanishes at $p^\star$;
* $\partial F_i/\partial\bar p_j=-\alpha c\,(\delta_{ij}-p^\star_i)$, i.e.
  $M=-\alpha c\,(I-p^\star\mathbf{1}^{\top})$;
* $\mathbf 1^{\top}C=0$, hence $MC=-\alpha c\,C$.

So $x'=My$ and $y'=\kappa(Cx-y)$, giving $x''=\kappa MCx-\kappa x'$:

$$
\boxed{\;x''+\kappa\,x'+\kappa\,\alpha c\,C\,x=0
\quad\Longleftrightarrow\quad
s^{2}+\kappa s+\kappa\lambda_j=0\ \text{ per mode},}
\tag{5.15}
$$

where $\lambda_j=\alpha\lVert r\rVert_{1/\alpha}\mu_j$ are *exactly* Experiment 2's
linear rates ($\mu_j$ the nonzero eigenvalues of $C$). Experiment 2's first-order
relaxation $\dot x=-\lambda x$ has become a **damped harmonic oscillator** whose
damping is set by the EMA and whose stiffness is set by the reward landscape:

* $s=\tfrac12\left(-\kappa\pm\sqrt{\kappa^{2}-4\kappa\lambda_j}\right)$;
* **oscillatory iff $\kappa<4\lambda_j$**, i.e. $\beta<4h\lambda_j$ — too long a
  memory makes training ring;
* decay rate
  $\mathrm{Re}(-s)=\kappa/2$ when $\kappa\le4\lambda$, and
  $\tfrac{\kappa}{2}\left(1-\sqrt{1-4\lambda/\kappa}\right)$ when $\kappa>4\lambda$;
* $\kappa\to\infty$ recovers $-\lambda$ (no EMA, Experiment 2);
* the rate is **maximised at $\kappa^{\star}=4\lambda$, where it equals
  $2\lambda$** — a correctly tuned EMA converges *twice as fast* as no EMA, the
  familiar momentum effect, here in closed form.

Verified against the numerically differentiated $2K\times2K$ Jacobian
($\lVert\partial F/\partial z\rVert\le6\times10^{-10}$, eigenvalues matching
(5.15) to $10^{-9}$ at large $\kappa$, slowest-rate formula matching at every
$\kappa$ tested).

The eigenvalues $\mu_j$ needed in (5.15) come from the symmetric PSD matrix $C$,
so they are obtained with the course's own **power method** (largest $\mu$) and
**QR algorithm** (full spectrum), cross-checked against `numpy.linalg.eigvalsh`.

---

## 7. Round-off, and why every error is reported relative

$p^{-\alpha}$ overflows `float64` for $p\lesssim10^{-308/\alpha}$, and the
absolute bias $\mathbb{E}[\widehat W]-p^{-\alpha}$ is a difference of two nearly
equal huge numbers — catastrophic cancellation, the course's round-off case. Two
safeguards are used throughout Experiment 5:

1. all binomial weights are formed from `scipy.stats.binom.pmf` and all powers
   from $\exp(-\alpha\ln x)$, so the arithmetic stays in the exponent;
2. every error is divided by $p^{-\alpha}$ *before* it is subtracted where
   possible, i.e. $b(p)=\mathbb{E}[\widehat W p^{\alpha}]-1$ is formed as a sum of
   $O(1)$ terms rather than as a difference of $O(p^{-\alpha})$ terms.

The experiment reports the measured difference between the naive and the guarded
formulation as a round-off audit, alongside the Monte-Carlo validation of the
exact binomial sums.

---
## Summary of what Experiment 5 adds

| claim | where |
|---|---|
| The expansion parameter of the estimator error is $Gp$; relative bias $\to\alpha(\alpha+1)(1-p)/(2Gp)$, relative sd $\to\alpha\sqrt{(1-p)/(Gp)}$, and the expansion needs $Gp\gtrsim20$ | (5.2), Fig. 1a–b |
| The MSE-optimal clip is **exactly** $\varepsilon^\star=p$, for every $\alpha$, $G$ and $p$ — so no single clip can suit a majority and a minority outcome at once | (5.4), Fig. 1c |
| Only $\omega(1)/\omega(G)$ — the singleton-to-full-group dynamic range — decides minority survival, for *any* weight rule | (5.8), Fig. 3a–b |
| Add-$\lambda$ smoothing strictly shrinks that range to $(G+\lambda)/(1+\lambda)$, independent of $K$, so Laplace smoothing buys variance in the same currency clipping does | (5.9), Fig. 3b |
| Richardson extrapolation in $1/G$ raises the estimator's bias order $1\to2$ at no leading-order variance cost | §4, Fig. 2b |
| Naive Richardson gives rare outcomes *negative* weights and guarantees collapse; a positivity-checked guard removes them exactly, bounds the rule by $2f$, and keeps the paper's extinction threshold unchanged | (5.11), Fig. 2b–c, 3b |
| **The dynamics see the size-biased functional, not the estimator.** The distortion is $\alpha(\alpha-1)(1-p)/(2Gp)$ for the paper's clip and $-\alpha(1-p)/(Gp)$ for the bias-corrected rule | (5.16), (5.18), Fig. 4a–b |
| At $\alpha=1$ — exactly the exponent IPS uses — the paper's rule is *exact* up to the miss probability: $w_G(p)=(1-(1-p)^G)/p$, so $D=-(1-p)^G$. The better estimator therefore trains worse | (5.17), Fig. 4a, 3c |
| Matching the offset to the exponent, $c^\star=(1-\alpha)/2$, restores second-order dynamics accuracy for $\alpha\neq1$; it widens the dynamic range for $\alpha<1$ and narrows it for $\alpha>1$ | (5.19), Fig. 4c |
| An EMA is the only device that is unbiased in steady state; it gives $G_{\mathrm{eff}}=G(2-\beta)/\beta$ free of extra rollouts, at the price of a lag $(1-\beta)\delta/\beta$ | (5.12), (5.13), Fig. 5a–b |
| With an EMA the learning dynamics become a damped oscillator $s^2+\kappa s+\kappa\lambda=0$; ringing for $\beta<4h\lambda$, and rate $2\lambda$ at $\kappa=4\lambda$ | (5.15), Fig. 5c |
