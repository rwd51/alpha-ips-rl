"""
Shared library for case 3: alpha-IPS training on REAL bandit logs
(Open Bandit Dataset, ZOZOTOWN, uniform-random logging policy).

Model (same as the project, src/dynamics.py): a softmax policy p = softmax(z)
over the K items. One training step draws a group of G items from p; each drawn
item receives a REAL logged reward (a click, 0/1) by resampling one logged
impression of that item from the uniform-random log ("bootstrap replay"); the
reward is scaled by max(p_hat, eps)^-alpha with p_hat the group frequency
(stop-gradient), and the logits move by z <- z + h * ghat.

Two update forms are implemented (S seeds at once, shapes (S, K)):

  plain (the project's / paper's form, = src.dynamics.rhs_sampled with a
         per-sample reward):
      ghat_i = p_hat_i * rt_i  -  p_i * sum_k p_hat_k * rt_k
  grpo  (group-mean baseline, as GRPO does; no std normalisation):
      ghat_i = p_hat_i * (rt_i - mean_group(rt))

  where rt_i = rbar_i * max(p_hat_i, eps)^-alpha and rbar_i is the mean of the
  logged clicks drawn for item i in this group (0 if item i was not drawn).

Mean field. Because E[rbar_i | counts] = c_i (the item's CTR in the replayed
log), the expected plain update equals the deterministic one with r = c, so the
project's finite-group theory (src.finite_group_general.meanfield_stationary)
applies unchanged. For the grpo form the multinomial correlation between
p_hat_i and the group mean changes the effective weight to

    w~_G(p) = (1 - 1/G) * E[max((1+B)/G, eps)^-alpha],   B ~ Bin(G-2, p)

(derivation in the case-3 write-up), whose boost ceiling is (G-1)^alpha
instead of G^alpha; GroupMeanBaselineRule plugs it into src.meanfield_rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.dynamics import softmax, inverse_transform_sample_batch, stationary_p  # noqa: E402
from src.finite_group_general import (meanfield_stationary, extinction_alpha,  # noqa: E402
                                      effective_weight)
from src.integrators import rk4_step                                             # noqa: E402
from src import meanfield_rule                                                   # noqa: E402

DATA = HERE / "data"
FIG = HERE / "figures"
RES = HERE / "results"
EPS = 1e-3            # the project's default clip; it never binds here since 1/G > eps
TAU_KEPT = 1e-3       # an item counts as "kept" if its probability exceeds this
J_JUMP = 0.05         # step-size rule: one click on an item seen once moves its logit by J


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_log(campaign: str = "all") -> dict:
    """Row-level random-policy log written by prepare_data.py (time-sorted)."""
    d = np.load(DATA / f"obd_random_{campaign}.npz")
    out = {k: d[k] for k in d.files}
    out["item"] = out["item"].astype(np.int64)
    out["click"] = out["click"].astype(np.int64)
    out["K"] = int(out["item"].max() + 1)
    return out


def time_split(log: dict, frac: float = 0.5):
    """First `frac` of the rows (by time) and the rest. The log is time-sorted."""
    assert np.all(np.diff(log["ts_us"]) >= 0)
    cut = int(round(frac * len(log["item"])))
    a = {k: (v[:cut] if isinstance(v, np.ndarray) else v) for k, v in log.items()}
    b = {k: (v[cut:] if isinstance(v, np.ndarray) else v) for k, v in log.items()}
    return a, b


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (good for small CTRs)."""
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    ph = k / n
    den = 1.0 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def item_stats(item, click, K):
    """Per-item impressions n, clicks k, CTR and 95% Wilson interval."""
    n = np.bincount(item, minlength=K).astype(np.int64)
    k = np.bincount(item, weights=click, minlength=K).astype(np.int64)
    ctr = k / n
    lo, hi = wilson(k, n)
    return {"n": n, "k": k, "ctr": ctr, "lo": lo, "hi": hi}


# --------------------------------------------------------------------------
# replay world: rewards are real logged clicks
# --------------------------------------------------------------------------

class ReplayWorld:
    """
    mode="click": the reward of a drawn item is the click of a uniformly
        resampled logged impression of that item (bootstrap replay). Valid
        because the logging policy was uniform random and independent of the
        user, so an item's logged clicks are i.i.d. draws of "show this item to
        a random visitor in a random slot". Equivalent in distribution to
        Bernoulli(c_i) with c_i the item's CTR in this log.
    mode="ctr": the deterministic "real reward landscape" variant, r_i = c_i.
    """

    def __init__(self, item, click, K, mode="click"):
        order = np.argsort(item, kind="stable")
        self.clicks_by_item = click[order].astype(np.float64)
        self.n = np.bincount(item, minlength=K).astype(np.int64)
        self.start = np.concatenate([[0], np.cumsum(self.n)[:-1]]).astype(np.int64)
        self.ctr = np.bincount(item, weights=click, minlength=K) / self.n
        self.K = K
        self.mode = mode

    def rewards(self, idx, rng):
        if self.mode == "ctr":
            return self.ctr[idx]
        j = rng.integers(0, self.n[idx])            # one logged impression of each drawn item
        return self.clicks_by_item[self.start[idx] + j]


# --------------------------------------------------------------------------
# sampling and the update
# --------------------------------------------------------------------------

def sample_items(p, G, rng):
    """
    Inverse-transform sampling of G items per row, exactly the rule of
    src.dynamics.inverse_transform_sample_batch (idx = #{k: cdf_k < u}), but
    with one searchsorted over row-offset CDFs instead of an (S, G, K)
    comparison tensor. tests in check_sampler() confirm identical draws.
    """
    S, K = p.shape
    cdf = np.cumsum(p, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random((S, G))
    off = np.arange(S)[:, None]
    idx = np.searchsorted((cdf + off).ravel(), (u + off).ravel(), side="left").reshape(S, G) - off * K
    return np.clip(idx, 0, K - 1)


def check_sampler(S=64, G=64, K=80, n=20, seed=0):
    """Fraction of draws on which sample_items agrees with the project's sampler
    when both consume the same uniforms (should be 1.0 up to fp boundary ties)."""
    agree = []
    for t in range(n):
        p = softmax(np.random.default_rng(seed + t).normal(0, 2, (S, K)))
        a = sample_items(p, G, np.random.default_rng(1000 + t))
        b = inverse_transform_sample_batch(p, G, np.random.default_rng(1000 + t))
        agree.append(np.mean(a == b))
    return float(np.mean(agree))


def ips_step(z, world, alpha, G, rng, eps=EPS, variant="plain"):
    """One group-REINFORCE step for S seeds. Returns (ghat, p, p_hat, R)."""
    S, K = z.shape
    p = softmax(z, axis=-1)
    idx = sample_items(p, G, rng)                          # (S, G) drawn items
    R = world.rewards(idx, rng)                            # (S, G) logged clicks
    flat = (idx + K * np.arange(S)[:, None]).ravel()
    counts = np.bincount(flat, minlength=S * K).reshape(S, K)
    rsum = np.bincount(flat, weights=R.ravel(), minlength=S * K).reshape(S, K)
    p_hat = counts / G
    W = np.maximum(p_hat, eps) ** (-alpha)                 # stop-gradient IPS weight
    term = (rsum / G) * W                                  # p_hat_i * rbar_i * W_i (0 if unseen)
    M = term.sum(axis=1, keepdims=True)                    # group mean of scaled rewards
    if variant == "plain":
        ghat = term - p * M
    elif variant == "grpo":
        ghat = term - p_hat * M
    else:
        raise ValueError(variant)
    return ghat, p, p_hat, R


def step_size(alpha, G, J=J_JUMP):
    """h such that one click on an item drawn once in the group moves that
    item's logit by J, for every alpha and G: the jump is h * G^(alpha-1)."""
    return J * float(G) ** (1.0 - alpha)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def norm_entropy(p):
    p = np.asarray(p, dtype=np.float64)
    K = p.shape[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=-1)
    return h / np.log(K)


def n_kept(p, tau=TAU_KEPT):
    return (np.asarray(p) > tau).sum(axis=-1)


def l1(p, q):
    return np.abs(np.asarray(p) - np.asarray(q)).sum(axis=-1)


def policy_ctr(p, stats):
    """Expected CTR sum_i p_i c_i of a fixed policy p under a log's per-item CTR
    estimates, with the standard error from the binomial noise of those
    estimates: Var = sum_i p_i^2 c_i (1 - c_i) / n_i (items independent)."""
    p = np.asarray(p, dtype=np.float64)
    c, n = stats["ctr"], stats["n"]
    v = p @ c
    se = np.sqrt((p ** 2) @ (c * (1 - c) / n))
    return v, se


# --------------------------------------------------------------------------
# training by replay
# --------------------------------------------------------------------------

def train(world, alpha, G, T, S=64, seed=0, h=None, eps=EPS, variant="plain",
          n_rec=200, avg_frac=0.25, trace_seeds=3, eval_ctrs=()):
    """
    Run S independent seeds for T steps. Returns a dict with
      p_final (S,K), p_avg (S,K) = time average over the last avg_frac of steps,
      rec_t (n_rec,), rec_maxp / rec_H / rec_kept (n_rec, S),
      rec_V (len(eval_ctrs), n_rec, S) expected CTR under each given CTR vector,
      trace (n_rec, trace_seeds, K) full policies of the first seeds,
      online_ctr = mean logged reward collected in the averaging window.
    """
    K = world.K
    h = step_size(alpha, G) if h is None else h
    rng = np.random.default_rng(seed)
    z = np.zeros((S, K))
    rec_every = max(1, T // n_rec)
    t_avg0 = int(T * (1 - avg_frac))
    p_sum = np.zeros((S, K))
    n_avg = 0
    online = 0.0
    rec = {"t": [], "maxp": [], "H": [], "kept": [], "V": [], "trace": []}
    C = np.array(eval_ctrs) if len(eval_ctrs) else np.zeros((0, K))
    for t in range(T):
        ghat, p, _, R = ips_step(z, world, alpha, G, rng, eps, variant)
        if t >= t_avg0:
            p_sum += p
            n_avg += 1
            online += R.mean()
        if t % rec_every == 0:
            rec["t"].append(t)
            rec["maxp"].append(p.max(axis=1))
            rec["H"].append(norm_entropy(p))
            rec["kept"].append(n_kept(p))
            rec["V"].append(C @ p.T)
            rec["trace"].append(p[:trace_seeds].astype(np.float32))
        z += h * ghat
        z -= z.max(axis=1, keepdims=True)          # gauge fix: softmax is shift-invariant
    p_final = softmax(z, axis=-1)
    return {
        "alpha": alpha, "G": G, "T": T, "S": S, "h": h, "eps": eps, "variant": variant,
        "mode": world.mode, "seed": seed,
        "p_final": p_final, "p_avg": p_sum / n_avg,
        "rec_t": np.array(rec["t"]), "rec_maxp": np.array(rec["maxp"]),
        "rec_H": np.array(rec["H"]), "rec_kept": np.array(rec["kept"]),
        "rec_V": np.transpose(np.array(rec["V"]), (1, 0, 2)) if len(eval_ctrs) else None,
        "trace": np.array(rec["trace"]),
        "online_ctr": online / max(n_avg, 1),
        "draws": T * G * S,
    }


# --------------------------------------------------------------------------
# theory (project code) on the real CTRs
# --------------------------------------------------------------------------

class GroupMeanBaselineRule:
    """Effective weight of the grpo (group-mean baseline) update, for
    src.meanfield_rule.stationary:  w~_G(p) = (1 - 1/G) E[omega(1 + B)],
    B ~ Bin(G-2, p), omega(n) = max(n/G, eps)^-alpha. Ceiling (1-1/G) G^alpha,
    w~_G(1) = (1-1/G)^(1-alpha): dynamic range (G-1)^alpha."""

    def __init__(self, alpha, eps=EPS):
        self.alpha, self.eps = alpha, eps

    def effective_weight(self, p, G):
        p = np.atleast_1d(np.asarray(p, dtype=np.float64))
        b = np.arange(G - 1)
        omega = np.maximum((1.0 + b) / G, self.eps) ** (-self.alpha)
        return (1.0 - 1.0 / G) * (binom.pmf(b[None, :], G - 2, p[:, None]) @ omega)


def predicted_policy(ctr, alpha, G, variant="plain", eps=EPS):
    """Finite-G mean-field stationary policy of the replay training (alpha > 0)."""
    if variant == "plain":
        p, S, _ = meanfield_stationary(ctr, G, alpha, eps, method="newton")
    else:
        p, S, _ = meanfield_rule.stationary(ctr, GroupMeanBaselineRule(alpha, eps), G)
    return p, S


def ideal_policy(ctr, alpha):
    """Infinite-group target p* ~ ctr^(1/alpha) (point mass on the best at alpha=0)."""
    return stationary_p(ctr, alpha)


def mf_drift(ctr, G, alpha, eps=EPS, variant="plain"):
    """Mean drift of the replay update as an ODE right-hand side f(z), z (S,K):
        dz_i/dt = p_i (c_i w_G(p_i) - S(p)),  S(p) = sum_k p_k c_k w_G(p_k),
    with w_G the plain (src.finite_group_general.effective_weight) or the
    group-mean-baseline effective weight. Time t corresponds to t/h replay steps."""
    rule = GroupMeanBaselineRule(alpha, eps) if variant == "grpo" else None
    ctr = np.asarray(ctr, dtype=np.float64)

    def f(z):
        p = softmax(z, axis=-1)
        if alpha == 0:
            w = np.ones_like(p)
        elif rule is None:
            w = effective_weight(p.ravel(), G, alpha, eps).reshape(p.shape)
        else:
            w = rule.effective_weight(p.ravel(), G).reshape(p.shape)
        cw = ctr * w
        return p * (cw - (p * cw).sum(axis=-1, keepdims=True))
    return f


def mf_trajectory(ctr, G, alpha, h, T, n_rec=200, avg_frac=0.25, variant="plain",
                  n_rk4=8000, eps=EPS):
    """Integrate the mean-field ODE from the uniform policy over the same
    training budget as the replay (time T*h) with RK4 (src.integrators.rk4_step)
    in n_rk4 steps. Returns (t_steps (n_rec,), p_traj (n_rec, K), p_avg (K,)),
    t_steps in replay steps and p_avg the time average over the last avg_frac."""
    K = len(ctr)
    f = mf_drift(ctr, G, alpha, eps, variant)
    dt = h * T / n_rk4
    z = np.zeros((1, K))
    rec_every = max(1, n_rk4 // n_rec)
    ts, ps = [], []
    p_sum, n_avg = np.zeros(K), 0
    for i in range(n_rk4):
        if i % rec_every == 0:
            ts.append(i * T / n_rk4)
            ps.append(softmax(z)[0])
        if i >= n_rk4 * (1 - avg_frac):
            p_sum += softmax(z)[0]
            n_avg += 1
        z = rk4_step(f, z, dt)
        z -= z.max()
    return np.array(ts), np.array(ps), p_sum / max(n_avg, 1)
