"""
Weight rules: how a finite group of G samples is turned into an estimate of the
inverse-probability weight p^-alpha, and the exact error statistics of each
choice (Experiment 5, `derivation_exp5.md`).

The alpha-IPS update needs w_i = p_i^-alpha, which the algorithm never knows --
it only has a group of G rollouts, i.e. counts n_i ~ Binomial(G, p_i). A
**count rule** is any map omega: {0,...,G} -> R applied as W_hat_i = omega(n_i).
The devices compared here are

  ClipRule       omega(n) = max(n/G, eps)^-alpha            (the paper, Eq. 9)
  AddLambdaRule  omega(n) = ((n+lam)/(G+K lam))^-alpha       (Laplace / Jeffreys)
  RichardsonRule split-half Richardson extrapolation in 1/G, optionally guarded
  IdealRule      omega = p^-alpha (not implementable; the reference)

Because omega takes finitely many values and n is binomial, every moment below
is an *exact* finite sum -- no Monte Carlo is needed for the ground truth
(Monte Carlo only validates it). Three quantities matter:

  moments(p, G)          -> E[W_hat], Var(W_hat)                  (error analysis)
  relative_moments(p, G) -> the same divided by p^-alpha           (round-off safe)
  effective_weight(p, G) -> w_G(p) = E[omega(1 + B)], B ~ Bin(G-1, p)

The last one is the only thing the learning dynamics ever see: Experiment 3's
mean drift is m_i = p_i (r_i w_G(p_i) - S). Its two endpoints,

    w_G(0) = omega(1)        (weight given to an outcome seen exactly once)
    w_G(1) = omega(G)        (weight given to an outcome that fills the group)

decide minority survival for K=2 through the dynamic-range criterion
omega(1)/omega(G) > r_1/r_2 (derivation_exp5.md, Eq. 5.8), which generalizes
Experiments 2-4's min(G, 1/eps)^alpha to an arbitrary estimator.

`sampled_step` is `dynamics.rhs_sampled` with the weight rule made pluggable; it
reproduces that function to round-off when given a ClipRule -- the two differ
only in how the power is evaluated (checked in tests/test_estimators.py).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom

from .dynamics import softmax, inverse_transform_sample_batch, empirical_counts


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _pow_neg_alpha(x, alpha):
    """x^-alpha evaluated as exp(-alpha ln x): keeps the arithmetic in the
    exponent, where the dynamic range of p^-alpha lives."""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-alpha * np.log(x))


def _as_p_array(p):
    p = np.atleast_1d(np.asarray(p, dtype=np.float64))
    if p.ndim != 1:
        raise ValueError("p must be a scalar or a one-dimensional array")
    if np.any(p < 0.0) or np.any(p > 1.0) or not np.all(np.isfinite(p)):
        raise ValueError("p must lie in [0, 1] and be finite")
    return p


def _check_group(G, even=False):
    if not isinstance(G, (int, np.integer)) or G < 2:
        raise ValueError("group size G must be an integer >= 2")
    if even and G % 2 != 0:
        raise ValueError("split-half rules need an even group size G")
    return int(G)


# --------------------------------------------------------------------------
# base class
# --------------------------------------------------------------------------

class WeightRule:
    """Base class. Subclasses implement `values`, `moments_relative`,
    `effective_weight` and `weights_from_samples`."""

    name = "rule"
    label = "rule"
    alpha = 1.0

    # -- exact statistics ---------------------------------------------------

    def moments_relative(self, p, G):
        """(E[W_hat]/p^-alpha, Var(W_hat)/p^-2alpha) as exact sums, computed in
        units of the ideal weight so that no huge number is ever subtracted
        from another (derivation_exp5.md, section 7)."""
        raise NotImplementedError

    def moments(self, p, G):
        """(E[W_hat], Var(W_hat)) in absolute units."""
        p = _as_p_array(p)
        m1r, varr = self.moments_relative(p, G)
        ideal = _pow_neg_alpha(p, self.alpha)
        return m1r * ideal, varr * ideal ** 2

    def error_table(self, p, G):
        """Relative bias, relative standard deviation and relative RMSE."""
        p = _as_p_array(p)
        m1r, varr = self.moments_relative(p, G)
        bias = m1r - 1.0
        sd = np.sqrt(np.maximum(varr, 0.0))
        return {"rel_bias": bias, "rel_sd": sd,
                "rel_rmse": np.sqrt(bias ** 2 + varr),
                "rel_mse": bias ** 2 + varr}

    def relative_central_moment(self, p, G, order):
        """E[(W_hat/p^-alpha - E[W_hat]/p^-alpha)^order], exactly. `order` = 4
        gives the standard error sqrt((mu4 - var^2)/N) of a Monte-Carlo
        variance estimate, which is what makes a heavy-tailed estimator's
        variance hard to measure by sampling."""
        raise NotImplementedError

    def effective_weight(self, p, G):
        """w_G(p) = phi_G(p)/p, the only functional of the rule the mean-field
        dynamics depend on (derivation_exp5.md, Eq. 5.5 / 5.7)."""
        raise NotImplementedError

    def dynamic_range(self, G):
        """omega(1)/omega(G): the singleton-to-full-group ratio that decides
        minority survival (Eq. 5.8)."""
        w = self.effective_weight(np.array([0.0, 1.0]), G)
        return float(w[0] / w[1])

    def critical_alpha_K2(self, G, reward_ratio):
        """The exponent below which the K=2 minority outcome is driven extinct
        under this rule. Only meaningful for rules whose dynamic range scales
        as (something)^alpha; the experiment solves the general case by
        bisection instead."""
        return float(np.log(reward_ratio) / np.log(self.dynamic_range(G)))

    # -- Monte-Carlo side ---------------------------------------------------

    def weights_from_samples(self, idx, K):
        """(S, G) sampled outcome indices -> (S, K) weight estimates."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# count rules: omega depends on the total count n only
# --------------------------------------------------------------------------

class CountRule(WeightRule):
    """A rule of the form W_hat_i = omega(n_i)."""

    def values(self, G):
        """omega(0), ..., omega(G)."""
        raise NotImplementedError

    def values_relative(self, p, G):
        """omega(n) * p^alpha, shape (len(p), G+1)."""
        p = _as_p_array(p)
        return self.values(G)[None, :] * (p ** self.alpha)[:, None]

    def moments_relative(self, p, G):
        p = _as_p_array(p)
        G = _check_group(G)
        n = np.arange(G + 1)
        P = binom.pmf(n[None, :], G, p[:, None])
        V = self.values_relative(p, G)
        m1 = np.sum(P * V, axis=1)
        # two-pass variance: sum P (V - m1)^2 avoids cancelling E[V^2] - m1^2
        var = np.sum(P * (V - m1[:, None]) ** 2, axis=1)
        return m1, var

    def relative_central_moment(self, p, G, order):
        p = _as_p_array(p)
        G = _check_group(G)
        n = np.arange(G + 1)
        P = binom.pmf(n[None, :], G, p[:, None])
        V = self.values_relative(p, G)
        m1 = np.sum(P * V, axis=1)
        return np.sum(P * (V - m1[:, None]) ** order, axis=1)

    def effective_weight(self, p, G):
        p = _as_p_array(p)
        G = _check_group(G)
        b = np.arange(G)
        P = binom.pmf(b[None, :], G - 1, p[:, None])
        return P @ self.values(G)[1:]

    def weights_from_samples(self, idx, K):
        idx = np.asarray(idx)
        G = idx.shape[1]
        counts = empirical_counts(idx, K).astype(np.int64)
        return self.values(G)[counts]


class ClipRule(CountRule):
    """The base paper's Eq. (9): omega(n) = max(n/G, eps)^-alpha."""

    def __init__(self, alpha=1.0, eps=1e-3):
        if not (0.0 < eps <= 1.0):
            raise ValueError("eps must lie in (0, 1]")
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be positive and finite")
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.name = f"clip_eps{eps:g}"
        self.label = rf"clip $\epsilon$={eps:g}"

    def values(self, G):
        G = _check_group(G)
        return _pow_neg_alpha(np.maximum(np.arange(G + 1) / G, self.eps), self.alpha)


class AddLambdaRule(CountRule):
    """Additive (Laplace/Jeffreys) smoothing: omega(n) = ((n+lam)/(G+K lam))^-alpha.

    lam = 1 is Laplace's rule of succession, lam = 1/2 is Jeffreys/KT. The
    estimator is bounded without any clipping, but by Eq. (5.9) its dynamic
    range is only (G+lam)/(1+lam), strictly below the clipped rule's G."""

    def __init__(self, alpha=1.0, lam=1.0, K=2):
        if not np.isfinite(lam) or lam <= 0.0:
            raise ValueError("lam must be positive and finite")
        if not isinstance(K, (int, np.integer)) or K < 2:
            raise ValueError("K must be an integer >= 2")
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.K = int(K)
        self.name = f"addlam{lam:g}"
        self.label = rf"add-$\lambda$={lam:g}"

    def values(self, G):
        G = _check_group(G)
        return _pow_neg_alpha((np.arange(G + 1) + self.lam) / (G + self.K * self.lam),
                              self.alpha)


class OffsetRule(CountRule):
    """
    omega(n) = max((n - c)/(G - c), eps)^-alpha, with the offset chosen to make
    the *dynamics-level* distortion second order (derivation_exp5.md, §5.1):

        c* = (1 - alpha) / 2 .

    The point is that the mean-field dynamics do not see E[omega(n)] with
    n ~ Binomial(G, p); by the size-biasing identity (5.5) they see
    E[omega(1+B)] with B ~ Binomial(G-1, p). The extra guaranteed hit shifts the
    argument upward by about (1-p)/G, which partly cancels the convexity bias.
    Balancing the two exactly gives c*, and the residual distortion drops from
    O(1/G) to O(1/G^2).

    c* = 0 at alpha = 1: the base paper's own rule is already the optimal member
    of this family at exactly the exponent IPS uses, where in fact
    w_G(p) = (1 - (1-p)^G)/p holds exactly (`effective_weight_alpha1`).
    """

    def __init__(self, alpha=1.0, eps=1e-3, c=None):
        if not (0.0 < eps <= 1.0):
            raise ValueError("eps must lie in (0, 1]")
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be positive and finite")
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.c = float((1.0 - alpha) / 2.0 if c is None else c)
        self.name = f"offset_c{self.c:g}"
        self.label = rf"$\alpha$-matched offset $c$={self.c:g}"

    def values(self, G):
        G = _check_group(G)
        if G <= self.c:
            raise ValueError("offset c must be smaller than the group size G")
        x = (np.arange(G + 1) - self.c) / (G - self.c)
        return _pow_neg_alpha(np.maximum(x, self.eps), self.alpha)


def effective_weight_alpha1(p, G):
    """
    Closed form of the clipped rule's effective weight at alpha = 1 with
    eps <= 1/G (so the clip never fires on a non-empty count):

        w_G(p) = E[G / (1 + B)] = (1 - (1-p)^G) / p,     B ~ Binomial(G-1, p),

    using sum_b C(G-1,b) p^b q^(G-1-b) / (1+b) = (1 - q^G) / (G p).

    Two consequences (derivation_exp5.md, §5.1):
      * w_G(0) = G and w_G(1) = 1, so the dynamic range is exactly G;
      * w_G(p) p = 1 - (1-p)^G, i.e. the dynamics-level distortion of the
        paper's own weight rule at its own exponent is exactly minus the
        probability that the group misses the outcome entirely -- exponentially
        small in G p, not merely O(1/G).
    It also turns the finite-G mean field of Experiments 2-4 at alpha = 1 into a
    closed-form root problem, with no binomial sums.
    """
    p = _as_p_array(p)
    G = _check_group(G)
    out = np.empty_like(p)
    small = p < 1e-6
    out[~small] = (1.0 - (1.0 - p[~small]) ** G) / p[~small]
    # -expm1(G log1p(-p))/p keeps full precision where 1 - (1-p)^G cancels
    with np.errstate(divide="ignore", invalid="ignore"):
        out[small] = np.where(p[small] > 0.0,
                              -np.expm1(G * np.log1p(-p[small])) / np.where(p[small] > 0, p[small], 1.0),
                              float(G))
    return out


# --------------------------------------------------------------------------
# split-half Richardson extrapolation in 1/G
# --------------------------------------------------------------------------

class RichardsonRule(WeightRule):
    """
    Richardson extrapolation of the O(1/G) estimator bias (derivation_exp5.md
    section 4). The group is split into halves A, B of size M = G/2 and

        omega_R(a, b) = 2 f((a+b)/G) - 1/2 [ f(a/M) + f(b/M) ],
        f(x) = max(x, eps)^-alpha.

    E[omega_R] = 2 E_G[f] - E_{G/2}[f], so the 1/G bias term cancels and the
    bias becomes O(1/G^2); because 2 p_hat = p_hat_A + p_hat_B, the leading
    variance is unchanged.

    guard = None reproduces the naive rule, whose weight for a singleton,
    2 G^alpha - (G/2)^alpha/2 - eps^-alpha/2, is *negative* for realistic eps:
    a sign-flipped reward on exactly the rare outcomes IPS exists to protect.

    guard = m0 >= 1 falls back to the plain clipped weight f((a+b)/G) unless
    BOTH halves are informative (min(a, b) >= m0) AND the extrapolated value is
    still positive. The second condition is not redundant: even with both
    halves non-empty, a lopsided split gives a negative correction (for
    alpha = 1 the rule is negative whenever 8ab < (a+b)^2, i.e. whenever one
    half has more than 3 + 2 sqrt(2) = 5.83 times the other), which happens
    with probability up to ~3% at small G p. Rejecting an extrapolation that
    disagrees this violently with the unextrapolated value is the standard
    Richardson/Romberg convergence check.

    The guarded rule therefore satisfies 0 < omega <= 2 f((a+b)/G) everywhere,
    and keeps omega(1) = min(G, 1/eps)^alpha and omega(G) = 1 -- so by Eq. (5.8)
    it inherits the paper's exact extinction threshold.
    """

    def __init__(self, alpha=1.0, eps=1e-3, guard=1):
        if not (0.0 < eps <= 1.0):
            raise ValueError("eps must lie in (0, 1]")
        if guard is not None and (not isinstance(guard, (int, np.integer)) or guard < 1):
            raise ValueError("guard must be None or an integer >= 1")
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.guard = None if guard is None else int(guard)
        self.name = "richardson_naive" if guard is None else f"richardson_g{guard}"
        self.label = "Richardson (naive)" if guard is None else "Richardson (guarded)"

    # -- the rule as a matrix over (a, b) ----------------------------------

    def values(self, G):
        """omega_R(a, b) for a, b in 0..M, shape (M+1, M+1)."""
        G = _check_group(G, even=True)
        M = G // 2
        a = np.arange(M + 1)[:, None]
        b = np.arange(M + 1)[None, :]
        f = lambda x: _pow_neg_alpha(np.maximum(x, self.eps), self.alpha)
        plain = f((a + b) / G)
        out = 2.0 * plain - 0.5 * (f(a / M) + f(b / M))
        if self.guard is not None:
            accept = (np.minimum(a, b) >= self.guard) & (out > 0.0)
            out = np.where(accept, out, plain)
        return out

    def _half_pmfs(self, p, G):
        M = G // 2
        k = np.arange(M + 1)
        return binom.pmf(k[None, :], M, p[:, None])          # (n_p, M+1)

    def moments_relative(self, p, G):
        p = _as_p_array(p)
        G = _check_group(G, even=True)
        W = self.values(G)
        P = self._half_pmfs(p, G)
        scale = p ** self.alpha
        m1 = np.einsum("ia,ab,ib->i", P, W, P) * scale
        # two-pass form in relative units (E[(V-m1)^2] with V = W * p^alpha)
        var = np.empty_like(m1)
        for i in range(len(p)):
            D = W * scale[i] - m1[i]
            var[i] = P[i] @ (D * D) @ P[i]
        return m1, var

    def relative_central_moment(self, p, G, order):
        p = _as_p_array(p)
        G = _check_group(G, even=True)
        W = self.values(G)
        P = self._half_pmfs(p, G)
        scale = p ** self.alpha
        m1 = np.einsum("ia,ab,ib->i", P, W, P) * scale
        out = np.empty_like(m1)
        for i in range(len(p)):
            D = W * scale[i] - m1[i]
            out[i] = P[i] @ (D ** order) @ P[i]
        return out

    def effective_weight(self, p, G):
        """Eq. (5.7): the designated sample falls in half A or half B with
        probability 1/2 each; the half it lands in has one guaranteed hit."""
        p = _as_p_array(p)
        G = _check_group(G, even=True)
        M = G // 2
        W = self.values(G)
        Pm1 = binom.pmf(np.arange(M)[None, :], M - 1, p[:, None])   # (n_p, M)
        PM = self._half_pmfs(p, G)                                   # (n_p, M+1)
        w_a = np.einsum("ia,ab,ib->i", Pm1, W[1:, :], PM)
        w_b = np.einsum("ia,ab,ib->i", PM, W[:, 1:], Pm1)
        return 0.5 * (w_a + w_b)

    def weights_from_samples(self, idx, K):
        idx = np.asarray(idx)
        G = _check_group(idx.shape[1], even=True)
        M = G // 2
        a = empirical_counts(idx[:, :M], K).astype(np.int64)
        b = empirical_counts(idx[:, M:], K).astype(np.int64)
        return self.values(G)[a, b]

    def fallback_probability(self, p, G):
        """Exact probability that the guard rejects the extrapolation and the
        plain clipped weight is used instead. Exponentially small in G p, which
        is why the guard does not spoil the O(G^-2) bias order."""
        p = _as_p_array(p)
        G = _check_group(G, even=True)
        if self.guard is None:
            return np.zeros_like(p)
        M = G // 2
        a = np.arange(M + 1)[:, None]
        b = np.arange(M + 1)[None, :]
        f = lambda x: _pow_neg_alpha(np.maximum(x, self.eps), self.alpha)
        raw = 2.0 * f((a + b) / G) - 0.5 * (f(a / M) + f(b / M))
        rejected = ~((np.minimum(a, b) >= self.guard) & (raw > 0.0))
        P = self._half_pmfs(p, G)
        return np.einsum("ia,ab,ib->i", P, rejected.astype(np.float64), P)

    def negative_weight_probability(self, p, G):
        """Exact probability that the rule returns a negative weight for an
        outcome of probability p -- i.e. that the update flips the sign of its
        reward."""
        p = _as_p_array(p)
        G = _check_group(G, even=True)
        W = self.values(G)
        P = self._half_pmfs(p, G)
        neg = (W < 0.0).astype(np.float64)
        return np.einsum("ia,ab,ib->i", P, neg, P)


# --------------------------------------------------------------------------
# the unreachable reference
# --------------------------------------------------------------------------

class IdealRule(WeightRule):
    """W_hat = p^-alpha exactly: no bias, no variance, infinite dynamic range.
    Not implementable (the algorithm does not know p); used as the reference
    curve and as the G -> infinity limit."""

    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.name = "ideal"
        self.label = "ideal $p^{-\\alpha}$"

    def moments_relative(self, p, G):
        p = _as_p_array(p)
        return np.ones_like(p), np.zeros_like(p)

    def relative_central_moment(self, p, G, order):
        return np.zeros_like(_as_p_array(p))

    def effective_weight(self, p, G):
        p = _as_p_array(p)
        with np.errstate(divide="ignore"):
            return _pow_neg_alpha(p, self.alpha)

    def dynamic_range(self, G):
        return np.inf

    def weights_from_samples(self, idx, K):
        raise NotImplementedError("the ideal rule cannot be evaluated from samples")


# --------------------------------------------------------------------------
# asymptotics (the delta method / truncation-error prediction)
# --------------------------------------------------------------------------

def delta_relative_bias(p, G, alpha):
    """Leading term of Eq. (5.2): alpha(alpha+1)(1-p) / (2 G p)."""
    p = np.asarray(p, dtype=np.float64)
    return alpha * (alpha + 1.0) * (1.0 - p) / (2.0 * G * p)


def delta_relative_sd(p, G, alpha):
    """Leading term of Eq. (5.2): alpha sqrt((1-p) / (G p))."""
    p = np.asarray(p, dtype=np.float64)
    return alpha * np.sqrt((1.0 - p) / (G * p))


def empty_group_bias(p, G, alpha, eps):
    """Eq. (5.3): the contribution of the p_hat = 0 draw to the relative bias,
    (1-p)^G ((p/eps)^alpha - 1). This is what makes the delta method fail for
    G p <~ 1, and what the clip exists to bound."""
    p = np.asarray(p, dtype=np.float64)
    return (1.0 - p) ** G * ((p / eps) ** alpha - 1.0)


# --------------------------------------------------------------------------
# the sampled update with a pluggable weight rule
# --------------------------------------------------------------------------

def sampled_step(z, r, rule, G, rng):
    """
    One stochastic policy-gradient step, identical to
    `dynamics.rhs_sampled` except that the reward-rescaling weight comes from
    `rule` instead of being hard-wired to max(p_hat, eps)^-alpha:

        ghat_i = p_hat_i r_i W_hat_i  -  p_i sum_k p_hat_k r_k W_hat_k

    The sampling-frequency multiplier p_hat_i is *structural* (it is the
    empirical average over the trajectories actually drawn, straight out of the
    score-function estimator), so it is never replaced; only the weight is a
    design choice. Outcome sampling uses the same inverse-transform routine as
    Experiments 1-3.

    z : (S, K) logits for S independent seeds
    Returns (ghat, p_hat), both (S, K).
    """
    z = np.asarray(z, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    S, K = z.shape
    p = softmax(z, axis=-1)
    idx = inverse_transform_sample_batch(p, G, rng)
    p_hat = empirical_counts(idx, K) / G
    w_hat = rule.weights_from_samples(idx, K)
    term = r[None, :] * p_hat * w_hat
    return term - p * np.sum(term, axis=-1, keepdims=True), p_hat
