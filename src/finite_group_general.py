"""
Finite-group mean field for any number of outcomes K (Experiment 3). Builds on
src/finite_group.py, which holds the K = 2 theory used by Experiment 2.

The finite-G stationary point has no closed form for K > 2
(derivation_exp3.md, "Finite group size for K > 2"):
  - effective_weight: w_G(p) = phi_G(p)/p as a size-biased binomial mean,
    E[max((1+B)/G, eps)^-alpha] with B ~ Binomial(G-1, p) -- strictly
    decreasing in p, with an exact derivative;
  - meanfield_stationary: nested scalar root-finding (inner: invert w_G per
    outcome; outer: normalize), by bisection or by safeguarded Newton;
  - extinction_alpha: the alpha below which a given outcome is driven extinct,
    and kept_at_large_alpha: whether any alpha keeps it at all;
  - meanfield_log_residual / meanfield_log_jacobian: the same stationary point
    as a logit-space Newton problem, to show where that formulation breaks.
"""

import numpy as np
from scipy.stats import binom

from .rootfinding import bisection
from .newton import newton_bracketed


def _clipped_power(n, G, alpha, eps):
    return np.maximum(np.asarray(n, dtype=np.float64) / G, eps) ** (-alpha)


def effective_weight(p, G, alpha, eps=1e-3):
    """
    w_G(p) = phi_G(p) / p = E[max((1+B)/G, eps)^-alpha],  B ~ Binomial(G-1, p).
    (Size-biasing: n Binom(n; G, p) / (G p) = Binom(n-1; G-1, p).) Well
    defined at p = 0, where it equals the ceiling min(G, 1/eps)^alpha.
    p: scalar or 1-D array. Returns a 1-D array.
    """
    p = np.atleast_1d(np.asarray(p, dtype=np.float64))
    n = np.arange(G)
    return binom.pmf(n[None, :], G - 1, p[:, None]) @ _clipped_power(n + 1, G, alpha, eps)


def effective_weight_prime(p, G, alpha, eps=1e-3):
    """dw_G/dp by d/dp E_{Bin(m,p)}[f(B)] = m E_{Bin(m-1,p)}[f(B+1) - f(B)], m = G-1."""
    p = np.atleast_1d(np.asarray(p, dtype=np.float64))
    n = np.arange(G - 1)
    diff = _clipped_power(n + 2, G, alpha, eps) - _clipped_power(n + 1, G, alpha, eps)
    return (G - 1) * (binom.pmf(n[None, :], G - 2, p[:, None]) @ diff)


class _Counter:
    """Counts evaluations of w_G and w_G' so bisection and Newton can be
    compared by work done, not just by iteration count."""

    def __init__(self, G, alpha, eps):
        self.G, self.alpha, self.eps = G, alpha, eps
        self.n_w = 0
        self.n_dw = 0

    def w(self, p):
        self.n_w += 1
        return effective_weight(p, self.G, self.alpha, self.eps)[0]

    def dw(self, p):
        self.n_dw += 1
        return effective_weight_prime(p, self.G, self.alpha, self.eps)[0]


def meanfield_stationary(r, G, alpha, eps=1e-3, method="newton", tol_p=1e-14, tol_c=1e-12):
    """
    Finite-G mean-field stationary distribution for any K.

    Interior outcomes satisfy r_i w_G(p_i) = S; an outcome with r_i w_G(0) <= S
    sits at p_i = 0 (extinct: the drift there vanishes and cannot be invaded).
    Because w_G is strictly decreasing on [0, 1] with w_G(1) = 1, each p_i(S)
    is unique (inner root), and T(S) = sum_i p_i(S) - 1 is non-increasing,
    positive at S = r_max and -1 at S = r_max * ceiling (outer root).

    method: "bisection" (both levels) or "newton" (both levels, safeguarded
    by the bracket, outer slope dT/dS = sum_interior 1 / (r_i w_G'(p_i))).

    Returns (p, S, stats) with stats = {"outer_iter", "inner_iter",
    "n_w", "n_dw"} (inner_iter summed over all inner solves).
    """
    r = np.asarray(r, dtype=np.float64)
    ctr = _Counter(G, alpha, eps)
    # w_G(0) through the same binomial sum the inner solver evaluates, so the
    # clamp and the bracket agree to the last bit ((1/G)^-alpha and G^alpha
    # can differ in the last bit)
    ceil = effective_weight(0.0, G, alpha, eps)[0]
    inner_iter = [0]

    def p_of_S(S):
        p = np.zeros(len(r))
        for i, ri in enumerate(r):
            v = S / ri
            if v >= ceil:
                p[i] = 0.0
            elif v <= 1.0:
                p[i] = 1.0
            elif method == "bisection":
                p[i], it = bisection(lambda x: ctr.w(x) - v, 0.0, 1.0, tol=tol_p)
                inner_iter[0] += it
            else:
                p[i], it = newton_bracketed(lambda x: ctr.w(x) - v, ctr.dw, 0.0, 1.0, tol=tol_p)
                inner_iter[0] += it
        return p

    cache = {}

    def T(S):
        if S not in cache:
            cache[S] = p_of_S(S)
        return cache[S].sum() - 1.0

    def dT(S):
        p = cache[S] if S in cache else p_of_S(S)
        interior = (p > 0.0) & (p < 1.0)
        slope = np.sum(1.0 / (r[interior] * np.array([ctr.dw(x) for x in p[interior]])))
        return slope if interior.any() else 0.0

    lo, hi = r.max(), r.max() * ceil
    if method == "bisection":
        S, outer = bisection(T, lo, hi, tol=tol_c)
    else:
        S, outer = newton_bracketed(T, dT, lo, hi, tol=tol_c)
    p = p_of_S(S)
    stats = {"outer_iter": int(outer), "inner_iter": int(inner_iter[0]),
             "n_w": int(ctr.n_w), "n_dw": int(ctr.n_dw)}
    return p, S, stats


def kept_at_large_alpha(r, j, G):
    """
    Whether outcome j survives in the limit alpha -> infinity (G < 1/eps).
    There w_G(p) / G^alpha -> (1-p)^(G-1), so interior outcomes satisfy
    r_i (1-p_i)^(G-1) = s, and j is kept iff s < r_j. At the margin s = r_j the
    support is A = {i : r_i >= r_j}, and normalization gives the criterion
        sum_{i in A, i != j} (r_j / r_i)^(1/(G-1))  >  |A| - 2.
    Always true for K = 2 (the paper's two-outcome picture), but for K > 2 a
    small group can lose an outcome at every alpha (derivation_exp3.md).
    """
    r = np.asarray(r, dtype=np.float64)
    others = r[(r >= r[j]) & (np.arange(len(r)) != j)]
    return bool(np.sum((r[j] / others) ** (1.0 / (G - 1))) > len(others) - 1)


def extinction_alpha(r, j, G, eps=1e-3, alpha_lo=1e-3, alpha_hi=40.0, tol=1e-9):
    """
    The alpha at which outcome j is driven to extinction in the finite-G mean
    field: the root of  h(alpha) = ln(r_j ceiling(alpha)) - ln S*(alpha),
    by bisection (h < 0: extinct, h > 0: kept). For K = 2 this is exactly
    finite_group.critical_alpha(r_max / r_j, G, eps). Returns (np.inf, 0)
    when h is still negative at alpha_hi (outcome j is not kept at any alpha
    in range; compare kept_at_large_alpha).
    Returns (alpha_c, n_bisection_iterations).
    """
    r = np.asarray(r, dtype=np.float64)
    log_m = np.log(min(G, 1.0 / eps))

    def h(alpha):
        _, S, _ = meanfield_stationary(r, G, alpha, eps, method="newton")
        return np.log(r[j]) + alpha * log_m - np.log(S)

    if h(alpha_hi) <= 0:
        return np.inf, 0
    return bisection(h, alpha_lo, alpha_hi, tol=tol)


def meanfield_log_residual(y, r, G, alpha, eps=1e-3):
    """Gauge-fixed log-balance residual of the mean field in logits (z_K = 0):
    F_i = ln(r_i w_G(p_i)) - ln(r_K w_G(p_K)). A finite root exists only if
    no outcome is extinct; used to show where logit-space Newton breaks."""
    z = np.append(np.asarray(y, dtype=np.float64), 0.0)
    p = np.exp(z - z.max())
    p /= p.sum()
    lv = np.log(np.asarray(r)) + np.log(effective_weight(p, G, alpha, eps))
    return lv[:-1] - lv[-1]


def meanfield_log_jacobian(y, r, G, alpha, eps=1e-3):
    """d F_i / d z_j = s_i (delta_ij - p_j) p_i - s_K (delta_Kj - p_j) p_K,
    with s_i = w_G'(p_i) / w_G(p_i)."""
    z = np.append(np.asarray(y, dtype=np.float64), 0.0)
    p = np.exp(z - z.max())
    p /= p.sum()
    s = effective_weight_prime(p, G, alpha, eps) / effective_weight(p, G, alpha, eps)
    a = s * p                                            # d ln w(p_i) / d z_i part
    J_full = np.diag(a) - np.outer(a, p)                 # d ln w(p_i) / d z_j
    return (J_full[:-1] - J_full[-1])[:, :-1]
