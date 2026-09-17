"""
Exponential moving averages of the outcome frequency: the third variance-control
device in the project pitch, and the only one that carries state across training
steps (Experiment 5, `derivation_exp5.md` section 6).

    p_bar_t = (1 - beta) p_bar_{t-1} + beta * p_hat_t,
    W_hat_t = max(p_bar_t, eps)^-alpha .

Three exact results are implemented here and checked in the experiment:

  1. Static policy -- pure variance reduction, with no steady-state bias:
         Var(p_bar) = beta/(2-beta) Var(p_hat)     =>  G_eff = G (2-beta)/beta.
  2. Moving policy -- a lag bias (1-beta)/beta per unit of per-step drift.
  3. The mean-field learning dynamics become SECOND ORDER. With kappa = beta/h,
         x'' + kappa x' + kappa * lambda_j x = 0,
     where lambda_j are exactly Experiment 2's linear rates. The policy rings
     when kappa < 4 lambda, and the decay rate is maximized at kappa = 4 lambda,
     where it is 2 lambda -- twice the rate without an EMA.
"""

from __future__ import annotations

import numpy as np

from .dynamics import softmax, stationary_p, reward_norm


# --------------------------------------------------------------------------
# 1-2. estimator-level theory
# --------------------------------------------------------------------------

def variance_factor(beta):
    """Var(p_bar_infinity) / Var(p_hat) = beta / (2 - beta)."""
    beta = np.asarray(beta, dtype=np.float64)
    if np.any(beta <= 0.0) or np.any(beta > 1.0):
        raise ValueError("beta must lie in (0, 1]")
    return beta / (2.0 - beta)


def effective_group_size(G, beta):
    """The group size an i.i.d. estimator would need to match the EMA's
    variance: G_eff = G (2 - beta) / beta."""
    return G / variance_factor(beta)


def lag_bias(beta, delta):
    """Steady-state lag of the EMA behind a policy drifting by `delta` per
    step: E[p_t - p_bar_t] = (1 - beta) delta / beta."""
    beta = np.asarray(beta, dtype=np.float64)
    return (1.0 - beta) / beta * np.asarray(delta, dtype=np.float64)


def frequency_mse(beta, delta, G, p):
    """Eq. (5.14): lag bias squared plus reduced sampling variance, the total
    per-step error of p_bar as an estimate of the moving p."""
    return lag_bias(beta, delta) ** 2 + variance_factor(beta) * p * (1.0 - p) / G


# --------------------------------------------------------------------------
# 3. the coupled mean-field dynamics
# --------------------------------------------------------------------------

def coupled_rhs(state, r, alpha, kappa, p_floor=1e-12):
    """
    Right-hand side of the h -> 0 mean field of EMA-weighted training, for a
    batch of S independent states.

        z_dot_i   = p_i (r_i p_bar_i^-alpha - S),   S = sum_k p_k r_k p_bar_k^-alpha
        p_bar_dot = kappa (p - p_bar)

    state : (S, 2K) = [z | p_bar]. Returns the same shape.
    """
    state = np.asarray(state, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    K = len(r)
    z, pbar = state[:, :K], state[:, K:]
    p = softmax(z, axis=-1)
    v = r[None, :] * np.clip(pbar, p_floor, None) ** (-alpha)
    S = np.sum(p * v, axis=-1, keepdims=True)
    return np.concatenate([p * (v - S), kappa * (p - pbar)], axis=-1)


def coupled_jacobian(r, alpha, kappa):
    """
    Analytical Jacobian of `coupled_rhs` at the stationary point
    (z* = log p*, p_bar = p*), in block form

        [[ 0 ,          M    ],
         [ kappa C , -kappa I ]],     M = -alpha ||r||_{1/alpha} (I - p* 1^T),
                                      C = diag(p*) - p* p*^T.

    The upper-left block is exactly zero because the only z-dependence of the
    drift multiplies the advantage, which vanishes at p*. That is what makes
    the linearized system second order (derivation_exp5.md, Eq. 5.15).
    """
    r = np.asarray(r, dtype=np.float64)
    K = len(r)
    ps = stationary_p(r, alpha)
    c = reward_norm(r, alpha)
    M = -alpha * c * (np.eye(K) - np.outer(ps, np.ones(K)))
    C = np.diag(ps) - np.outer(ps, ps)
    J = np.zeros((2 * K, 2 * K))
    J[:K, K:] = M
    J[K:, :K] = kappa * C
    J[K:, K:] = -kappa * np.eye(K)
    return J


def mode_roots(lam, kappa):
    """Both roots of s^2 + kappa s + kappa lambda = 0 (complex when
    kappa < 4 lambda)."""
    lam = np.asarray(lam, dtype=np.float64)
    disc = np.asarray(kappa ** 2 - 4.0 * kappa * lam, dtype=np.complex128)
    sq = np.sqrt(disc)
    return 0.5 * (-kappa + sq), 0.5 * (-kappa - sq)


def decay_rate(lam, kappa):
    """Asymptotic decay rate Re(-s) of the slow branch: kappa/2 while
    underdamped (kappa <= 4 lambda), then (kappa/2)(1 - sqrt(1 - 4 lambda/kappa))
    which falls back to lambda as kappa -> infinity."""
    lam = np.asarray(lam, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    under = kappa <= 4.0 * lam
    ratio = np.where(under, 0.0, 1.0 - 4.0 * lam / np.where(kappa > 0, kappa, 1.0))
    return np.where(under, kappa / 2.0, 0.5 * kappa * (1.0 - np.sqrt(np.maximum(ratio, 0.0))))


def critical_kappa(lam):
    """kappa* = 4 lambda: the critically damped EMA, where the decay rate is
    maximal and equal to 2 lambda."""
    return 4.0 * np.asarray(lam, dtype=np.float64)


def is_oscillatory(lam, kappa):
    """True when the mode rings rather than relaxing monotonically."""
    return np.asarray(kappa, dtype=np.float64) < 4.0 * np.asarray(lam, dtype=np.float64)
