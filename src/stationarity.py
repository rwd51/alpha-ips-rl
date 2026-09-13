"""
The stationarity condition z_dot = 0 of the alpha-scaled flow (Eq. * of
derivation.md), written as a root-finding problem in three equivalent forms.

    drift     F_i = z_dot_i = p_i * (r_i p_i^-alpha - S)        (the ODE itself)
    balance   F_i = r_i p_i^-alpha - r_K p_K^-alpha            (drift / p_i, differenced)
    log       F_i = ln(r_i p_i^-alpha) - ln(r_K p_K^-alpha)    (log of the balance terms)

All three share the root p* ~ r^(1/alpha) but have very different global
shapes, which decides whether Newton-Raphson converges (Experiment 3,
derivation_exp3.md "Root-finding formulations").

z_dot is invariant to shifting every logit, so the Jacobian of the full
K-dimensional system is singular (J @ 1 = 0). We fix the gauge z_K = 0 and
solve for y = z_1..z_{K-1}; since the drift sums to zero, dropping its last
component loses nothing.

K = 2 has the scalar unknown u = z_1 - z_2 (so p_1 = sigmoid(u)), and each
form gets a closed-form derivative for scalar Newton.
"""

import numpy as np
from scipy.special import expit, log_expit

from .dynamics import rhs_alpha, jacobian_alpha


# --------------------------------------------------------------------------
# K = 2: scalar residuals in u = z_1 - z_2, root u* = ln(r_1/r_2) / alpha
# --------------------------------------------------------------------------

def root_K2(r, alpha):
    """Analytical root u* = ln(r_1/r_2)/alpha of every form below."""
    return np.log(r[0] / r[1]) / alpha


def drift_K2(u, r, alpha):
    """g(u) = u_dot = z_dot_1 - z_dot_2 = 2 (r_1 p^(1-a) q - r_2 p q^(1-a))."""
    p, q = expit(u), expit(-u)
    return 2.0 * (r[0] * p ** (1.0 - alpha) * q - r[1] * p * q ** (1.0 - alpha))


def drift_K2_prime(u, r, alpha):
    """dg/du, using dp/du = pq and dq/du = -pq."""
    p, q = expit(u), expit(-u)
    return 2.0 * (r[0] * p ** (1.0 - alpha) * q * ((1.0 - alpha) * q - p)
                  - r[1] * p * q ** (1.0 - alpha) * (q - (1.0 - alpha) * p))


def balance_K2(u, r, alpha):
    """b(u) = r_1 p^-a - r_2 q^-a = g(u) / (2pq). Strictly decreasing in u."""
    return r[0] * np.exp(-alpha * log_expit(u)) - r[1] * np.exp(-alpha * log_expit(-u))


def balance_K2_prime(u, r, alpha):
    p, q = expit(u), expit(-u)
    return -alpha * (r[0] * np.exp(-alpha * log_expit(u)) * q
                     + r[1] * np.exp(-alpha * log_expit(-u)) * p)


def log_K2(u, r, alpha):
    """l(u) = ln(r_1 p^-a) - ln(r_2 q^-a) = ln(r_1/r_2) - alpha u: linear."""
    return np.log(r[0] / r[1]) - alpha * u


def log_K2_prime(u, r, alpha):
    return -alpha


FORMS_K2 = {"drift": (drift_K2, drift_K2_prime),
            "balance": (balance_K2, balance_K2_prime),
            "log": (log_K2, log_K2_prime)}


# --------------------------------------------------------------------------
# general K: gauge-fixed residuals in y = z_1..z_{K-1} (z_K = 0)
# --------------------------------------------------------------------------

def _logits(y):
    return np.append(np.asarray(y, dtype=np.float64), 0.0)


def _log_softmax(z):
    zmax = np.max(z)
    return z - zmax - np.log(np.sum(np.exp(z - zmax)))


def drift_residual(y, r, alpha):
    return rhs_alpha(_logits(y), r, alpha)[:-1]


def drift_jacobian(y, r, alpha):
    return jacobian_alpha(_logits(y), r, alpha)[:-1, :-1]


def balance_residual(y, r, alpha):
    v = np.asarray(r) * np.exp(-alpha * _log_softmax(_logits(y)))
    return v[:-1] - v[-1]


def balance_jacobian(y, r, alpha):
    """With v_i = r_i p_i^-a and d ln p_i / d z_j = delta_ij - p_j:
    d(v_i - v_K)/d z_j = -a v_i delta_ij + a p_j (v_i - v_K)   (j < K)."""
    lp = _log_softmax(_logits(y))
    p = np.exp(lp)
    v = np.asarray(r) * np.exp(-alpha * lp)
    return -alpha * np.diag(v[:-1]) + alpha * np.outer(v[:-1] - v[-1], p[:-1])


def log_residual(y, r, alpha):
    lr = np.log(np.asarray(r, dtype=np.float64))
    return (lr[:-1] - lr[-1]) - alpha * np.asarray(y, dtype=np.float64)


def log_jacobian(y, r, alpha):
    return -alpha * np.eye(len(r) - 1)


FORMS = {"drift": (drift_residual, drift_jacobian),
         "balance": (balance_residual, balance_jacobian),
         "log": (log_residual, log_jacobian)}


def gauge_root(p):
    """Gauge-fixed logits y* = ln(p_i / p_K), i < K, of a distribution p."""
    lp = np.log(np.asarray(p, dtype=np.float64))
    return lp[:-1] - lp[-1]
