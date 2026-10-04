"""
Check 11 -- do the bandit results transfer to multi-step policies with shared
parameters?  For the VANILLA update they do, at the level of the target:

  ideal stop-gradient IPS:  E_tau[ r_o p_o^-alpha grad log pi(tau) ] = grad_theta F_alpha(p(theta))
  finite group, count rule: E[(1/G) sum_g r_{o_g} omega(n_{o_g}) grad log pi(tau_g)]
                            = grad_theta F_G(p(theta)),   F_G = sum_o r_o Phi_G(p_o)
(size-biasing per trajectory).  So for ANY differentiable parameterization the
expected update is gradient ascent on the same concave outcome-level potential,
and the target (the KKT point) does not depend on path multiplicity.

Tree MDP: s0 --a--> s1 --{x,y,z}--> {A, A, B};  s0 --b--> s2 --{u,v}--> {B, C}.
Outcome A has two paths, B two (in different subtrees), C one.  Tabular softmax
per state, 7 parameters.  Rewards (A,B,C) = (3, 2, 1).

(a) exact expectation (enumerating all 5^G trajectory tuples) vs finite differences.
(b) Monte Carlo training of the tree policy with the vanilla update vs the bandit
    KKT target for the same (r, G, alpha).

Run:  python3 -u next_phase/case2_checks/c11_mdp_portability.py      (~6-8 min)
"""
import itertools
import os
import sys

import numpy as np
from scipy.stats import binom

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.finite_group_general import meanfield_stationary  # noqa: E402

EPS = 1e-3
rng = np.random.default_rng(12)
r_out = np.array([3.0, 2.0, 1.0])                      # A, B, C
# trajectories: (state-0 action, state-1/2 action index, outcome)
TRAJ = [(0, 0, 0), (0, 1, 0), (0, 2, 1), (1, 0, 1), (1, 1, 2)]
SL = {0: slice(0, 2), 1: slice(2, 5), 2: slice(5, 7)}


def sm(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def traj_probs(th):
    p0, p1, p2 = sm(th[SL[0]]), sm(th[SL[1]]), sm(th[SL[2]])
    out = []
    for a, b, o in TRAJ:
        out.append(p0[a] * (p1[b] if a == 0 else p2[b]))
    return np.array(out)


def grad_log(th, k):
    a, b, o = TRAJ[k]
    g = np.zeros(7)
    p0 = sm(th[SL[0]])
    g[SL[0]] = -p0
    g[SL[0].start + a] += 1
    s = 1 if a == 0 else 2
    ps = sm(th[SL[s]])
    g[SL[s]] = -ps
    g[SL[s].start + b] += 1
    return g


def outcome_probs(th):
    pt = traj_probs(th)
    return np.array([pt[0] + pt[1], pt[2] + pt[3], pt[4]])


def omega(n, G, alpha):
    return np.maximum(n / G, EPS) ** (-alpha)


def Phi(p, G, alpha):
    k = np.arange(1, G + 1)
    cum = np.concatenate([[0.0], np.cumsum(omega(k, G, alpha))])
    return binom.pmf(np.arange(G + 1), G, p) @ cum / G


def F_G(th, G, alpha):
    return float(np.sum([r_out[o] * Phi(po, G, alpha) for o, po in enumerate(outcome_probs(th))]))


def F_ideal(th, alpha):
    p = outcome_probs(th)
    return float(np.sum(r_out * np.log(p))) if alpha == 1 else float(np.sum(r_out * p ** (1 - alpha)) / (1 - alpha))


def num_grad(F, th, h=1e-6):
    return np.array([(F(th + h * e) - F(th - h * e)) / (2 * h) for e in np.eye(7)])


worst_i, worst_g = 0.0, 0.0
for trial in range(20):
    th = rng.normal(0, 1, 7)
    pt = traj_probs(th)
    po = outcome_probs(th)
    for alpha in [0.5, 1.0, 2.0]:
        exact = sum(pt[k] * r_out[TRAJ[k][2]] * po[TRAJ[k][2]] ** (-alpha) * grad_log(th, k) for k in range(5))
        fd = num_grad(lambda t: F_ideal(t, alpha), th)
        worst_i = max(worst_i, np.max(np.abs(exact - fd)) / (1 + np.max(np.abs(fd))))
    for G, alpha in [(3, 1.0), (4, 0.5), (4, 2.0)]:
        Eg = np.zeros(7)
        for tup in itertools.product(range(5), repeat=G):
            prob = np.prod(pt[list(tup)])
            outs = np.array([TRAJ[k][2] for k in tup])
            n = np.bincount(outs, minlength=3)
            g = sum(r_out[TRAJ[k][2]] * omega(n[TRAJ[k][2]], G, alpha) * grad_log(th, k) for k in tup) / G
            Eg += prob * g
        fd = num_grad(lambda t: F_G(t, G, alpha), th)
        worst_g = max(worst_g, np.max(np.abs(Eg - fd)) / (1 + np.max(np.abs(fd))))
print(f"(a) ideal IPS: max rel |E[update] - grad F_alpha(p(theta))| = {worst_i:.1e}")
print(f"    finite G (exact 5^G enumeration): max rel |E[update] - grad F_G(p(theta))| = {worst_g:.1e}")

# (b) Monte Carlo training on the tree vs the bandit KKT target (vectorized over seeds)
OUT = np.array([TRAJ[k][2] for k in range(5)])          # outcome of each trajectory
A0 = np.array([TRAJ[k][0] for k in range(5)])           # state-0 action
A1 = np.array([TRAJ[k][1] for k in range(5)])           # second action (in s1 or s2)


def sm_rows(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def tree_mc(G, alpha, h=0.05, steps=40000, seeds=32):
    TH = np.zeros((seeds, 7))
    acc = np.zeros(3)
    cnt = 0
    idx = np.arange(seeds)[:, None]
    for t in range(steps):
        P0, P1, P2 = sm_rows(TH[:, 0:2]), sm_rows(TH[:, 2:5]), sm_rows(TH[:, 5:7])
        PT = np.stack([P0[:, 0] * P1[:, 0], P0[:, 0] * P1[:, 1], P0[:, 0] * P1[:, 2],
                       P0[:, 1] * P2[:, 0], P0[:, 1] * P2[:, 1]], axis=1)        # (S, 5)
        cdf = np.cumsum(PT, axis=1)
        cdf[:, -1] = 1.0
        u = rng.random((seeds, G))
        k = np.sum(cdf[:, None, :] < u[:, :, None], axis=2)                       # (S, G) inverse transform
        o = OUT[k]
        n = np.stack([np.sum(o == c, axis=1) for c in range(3)], axis=1)        # (S, 3)
        Rt = r_out[o] * omega(n[idx, o].astype(float), G, alpha)               # (S, G)
        a0, a1 = A0[k], A1[k]
        g = np.zeros_like(TH)
        g[:, 0:2] = np.einsum("sg,sgc->sc", Rt, np.eye(2)[a0] - P0[:, None, :])
        in1 = (a0 == 0).astype(float)
        in2 = 1.0 - in1
        g[:, 2:5] = np.einsum("sg,sgc->sc", Rt * in1, np.eye(3)[np.minimum(a1, 2)] - P1[:, None, :])
        g[:, 5:7] = np.einsum("sg,sgc->sc", Rt * in2, np.eye(2)[np.minimum(a1, 1)] - P2[:, None, :])
        TH += h * g / G
        if t >= steps // 2 and t % 10 == 0:
            p_out = np.stack([PT[:, 0] + PT[:, 1], PT[:, 2] + PT[:, 3], PT[:, 4]], axis=1)
            acc += p_out.mean(axis=0)
            cnt += 1
    return acc / cnt


for G, alpha in [(4, 1.0), (8, 1.0), (4, 2.0)]:
    pi_bandit, _, _ = meanfield_stationary(r_out, G, alpha, EPS)
    p_tree = tree_mc(G, alpha)
    print(f"(b) G={G}, alpha={alpha}: tree MC long-run outcome dist {np.round(p_tree, 4)} | "
          f"bandit KKT target {np.round(pi_bandit, 4)} | ideal r^(1/alpha) {np.round(r_out**(1/alpha)/np.sum(r_out**(1/alpha)), 4)}")

# (c) the alpha = 2 gap: finite-h bias or a different target?  Shrink h (same horizon h*steps).
pi_bandit, _, _ = meanfield_stationary(r_out, 4, 2.0, EPS)
for h_small, steps in [(0.05, 40000), (0.02, 100000), (0.01, 200000)]:
    p_tree = tree_mc(4, 2.0, h=h_small, steps=steps)
    print(f"(c) G=4, alpha=2, h={h_small}: tree MC {np.round(p_tree, 4)} | bandit KKT {np.round(pi_bandit, 4)} | "
          f"l1 gap {np.abs(p_tree - pi_bandit).sum():.4f}")
