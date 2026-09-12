"""
Mean-field (expected-update) theory for alpha-IPS estimated from a FINITE
group of G samples, as in src/dynamics.py::rhs_sampled.

The sampled update's reward-rescaling term for outcome i is
    p_hat_i * max(p_hat_i, eps)^(-alpha),     G * p_hat_i ~ Binomial(G, p_i).
Its expectation phi_G(p_i) replaces the ideal p_i^(1-alpha) in the mean
dynamics, so the finite-G stationary point solves
    r_i * phi_G(p_i) / p_i = const   for all i
instead of r_i * p_i^(-alpha) = const. The effective inverse-probability
weight phi_G(p)/p equals 1 at p = 1 but saturates as p -> 0,
    phi_G(p)/p -> min(G, 1/eps)^alpha,
because a rare outcome is seen at most once, at frequency 1/G. IPS therefore
cannot protect an outcome whose reward is below r_max / min(G, 1/eps)^alpha.
See derivation.md, "Finite group size".
"""

import numpy as np
from scipy.stats import binom

from .rootfinding import bisection


def expected_weight(p, G, alpha, eps=1e-3):
    """phi_G(p) = E[p_hat * max(p_hat, eps)^-alpha] as an exact binomial sum.
    p: scalar or 1-D array. Returns a 1-D array."""
    p = np.atleast_1d(np.asarray(p, dtype=np.float64))
    n = np.arange(G + 1)
    ph = n / G
    val = ph * np.clip(ph, eps, 1.0) ** (-alpha)
    return binom.pmf(n[None, :], G, p[:, None]) @ val


def weight_ceiling(G, alpha, eps=1e-3):
    """Largest effective inverse-probability weight a group of G can apply."""
    return min(G, 1.0 / eps) ** alpha


def critical_alpha(reward_ratio, G, eps=1e-3):
    """alpha below which the mean dynamics drive extinct an outcome whose
    reward is r_max / reward_ratio:  reward_ratio = min(G, 1/eps)^alpha_c."""
    return np.log(reward_ratio) / np.log(min(G, 1.0 / eps))


def meanfield_stationary_K2(r, G, alpha, eps=1e-3, tol=1e-12):
    """
    Finite-G mean-field stationary probability of outcome 1 for K=2 with
    r[0] > r[1]: the root of g(p) = r1 phi(p)/p - r2 phi(1-p)/(1-p), by
    bisection. Returns 1.0 when there is no interior root (alpha at or below
    critical_alpha): the minority outcome is driven extinct.
    """
    def g(p):
        return (r[0] * expected_weight(p, G, alpha, eps)[0] / p
                - r[1] * expected_weight(1.0 - p, G, alpha, eps)[0] / (1.0 - p))

    lo, hi = 1e-12, 1.0 - 1e-12
    if g(hi) >= 0:
        return 1.0
    root, _ = bisection(g, lo, hi, tol=tol)
    return root
