"""
Core learning-dynamics model for the outcome-selection bandit.

Setting (matches the base paper, Sinha, Elango & Liu 2026, arXiv:2601.21669,
and the course project's alpha-generalization):

  - K possible terminal outcomes with fixed rewards r_1,...,r_K.
  - Policy is softmax over logits z in R^K:  p_i = exp(z_i) / sum_k exp(z_k).
  - Training scales the terminal reward by inverse probability to a tunable
    power alpha:  r_tilde_alpha(o) = r(o) / p(o)^alpha.
      * alpha = 0  -> standard expected-return objective (provably collapses)
      * alpha = 1  -> Inverse Probability Scaling (IPS) from the paper
      * alpha > 1  -> over-corrects, flattens further towards uniform
  - Under gradient flow on the alpha-scaled objective (stop-gradient on the
    probability appearing in the denominator, exactly as in the paper's
    Appendix A.2 proof for alpha=1), the logits evolve as

        z_i_dot = r_i * p_i^(1-alpha)  -  p_i * sum_k r_k * p_k^(1-alpha)     (*)

    Sanity checks (see derivation.md for the full derivation):
      alpha=0:  z_i_dot = p_i*(r_i - sum_k p_k r_k)   = p_i * a_i    (paper's Eq. 2, collapse)
      alpha=1:  z_i_dot = r_i - p_i * sum_k r_k                       (paper's Eq. 4, IPS)

    Stationary point (all i):  p_i^alpha  ~  r_i   =>   p*_i  ~  r_i^(1/alpha)
    which is exactly the project's Eq. (7) target law.

This module provides:
  - softmax(z)
  - rhs_alpha(z, r, alpha):  right-hand side f(z) of the IDEALIZED (noise-free)
    ODE z_dot = f(z), used for Euler / RK4 integration.
  - rhs_sampled(z, r, alpha, G, rng, eps): ONE realistic stochastic training
    step, using a finite group of G Monte-Carlo samples (inverse-transform
    sampling) to estimate outcome frequencies p_hat, exactly as a real
    GRPO-style algorithm would (Algorithm 1 / Eq. 9 of the paper). This is
    what "training" actually looks like: no ODE integrator is applied to it,
    a single stochastic gradient step is taken per iteration, exactly as SGD/
    policy-gradient training does.
"""

from __future__ import annotations
import numpy as np


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along the last axis by default."""
    z = np.asarray(z, dtype=np.float64)
    z_shift = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z_shift)
    return ez / np.sum(ez, axis=axis, keepdims=True)


def rhs_alpha(z: np.ndarray, r: np.ndarray, alpha: float, p_floor: float = 1e-12) -> np.ndarray:
    """
    Idealized (deterministic, noise-free) gradient-flow right-hand side (*).

    z : (..., K) logits (any leading batch shape, e.g. many independent seeds)
    r : (K,) fixed reward vector
    alpha : diversity-knob exponent (0 = standard/collapse, 1 = IPS)

    Returns dz/dt with the same shape as z.
    """
    z = np.asarray(z, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    p = softmax(z, axis=-1)
    # p^(1-alpha) can blow up as p->0 when alpha>1 (negative exponent); clip the
    # BASE only inside this power term to keep the ODE well defined, mirroring
    # the paper's own epsilon-clipping motivation for the same singularity.
    p_c = np.clip(p, p_floor, 1.0)
    w = r * p_c ** (1.0 - alpha)          # r_i * p_i^(1-alpha)
    S = np.sum(w, axis=-1, keepdims=True)  # sum_k r_k * p_k^(1-alpha)
    return w - p * S


def inverse_transform_sample_batch(p: np.ndarray, G: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw G categorical samples per row of p using inverse-transform sampling
    (explicit CDF + uniform draw, as in the course RNG lab) -- vectorized over
    an arbitrary number of independent rows (seeds).

    p : (S, K) rows are probability distributions (sum to 1 along axis=-1)
    Returns idx : (S, G) integer outcome indices in [0, K-1].
    """
    p = np.asarray(p, dtype=np.float64)
    S, K = p.shape
    cdf = np.cumsum(p, axis=-1)             # (S, K)
    cdf[:, -1] = 1.0                         # guard against fp round-off
    u = rng.random((S, G))                   # uniform(0,1) draws
    # idx[s,g] = # of cdf entries strictly less than u[s,g]  == inverse-CDF index
    idx = np.sum(cdf[:, None, :] < u[:, :, None], axis=-1)
    return np.clip(idx, 0, K - 1)


def empirical_counts(idx: np.ndarray, K: int) -> np.ndarray:
    """idx: (S,G) sampled outcome indices -> counts: (S,K)."""
    S, G = idx.shape
    onehot = (idx[:, :, None] == np.arange(K)[None, None, :])
    return onehot.sum(axis=1).astype(np.float64)  # (S,K)


def rhs_sampled(z: np.ndarray, r: np.ndarray, alpha: float, G: int,
                 rng: np.random.Generator, eps: float = 1e-3) -> np.ndarray:
    """
    ONE realistic stochastic policy-gradient step estimated from a finite
    group of G Monte-Carlo rollouts (as GRPO/IPS-GRPO actually do), for a
    batch of S independent seeds/runs.

    Derivation (matches the deterministic case but with p replaced by the
    empirical frequency p_hat *only* inside the reward-rescaling term, since
    a real algorithm never has access to the true p -- it only has samples;
    the outer probability-shift term p_i still uses the true current policy,
    since that comes from the exact softmax log-derivative, not an estimate):

        r_tilde_i   = r_i / max(p_hat_i, eps)^alpha
        ghat_i      = p_hat_i * r_tilde_i  -  p_i * sum_k p_hat_k * r_tilde_k
                    = r_i * p_hat_i^(1-alpha)  -  p_i * sum_k r_k * p_hat_k^(1-alpha)

    which reduces to the correct unbiased batched-REINFORCE estimator when
    alpha=0, and to the paper's Eq. (9) frequency-clipped IPS when alpha=1.

    z : (S, K) logits, S independent seeds
    Returns ghat : (S, K), the stochastic update direction (to be used as
    z <- z + h * ghat, a single SGD-style step -- NOT integrated with RK4,
    since this is a noisy discrete training step, not a smooth ODE).
    """
    z = np.asarray(z, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    S, K = z.shape
    p = softmax(z, axis=-1)
    idx = inverse_transform_sample_batch(p, G, rng)
    counts = empirical_counts(idx, K)
    p_hat = counts / G
    p_hat_c = np.clip(p_hat, eps, 1.0)
    w = r[None, :] * p_hat_c ** (1.0 - alpha)   # (S,K)
    S_sum = np.sum(w, axis=-1, keepdims=True)
    ghat = w - p * S_sum
    return ghat, p_hat


def stationary_p(r: np.ndarray, alpha: float) -> np.ndarray:
    """Analytical stationary distribution p*_i ~ r_i^(1/alpha) (Eq. 7)."""
    r = np.asarray(r, dtype=np.float64)
    if alpha == 0:
        # limit alpha->0 collapses all mass onto argmax(r); ties split evenly.
        out = np.zeros_like(r)
        out[r == r.max()] = 1.0 / np.sum(r == r.max())
        return out
    w = r ** (1.0 / alpha)
    return w / np.sum(w)
