"""
Check 05 -- does the survival limit hold for the update IPS-GRPO actually computes?

Bandit, deterministic rewards, IPS weight omega(n) = max(n/G, eps)^-alpha,
R~_k = r_k omega(n_k) for every outcome present in the group.  Three updates:

  V  vanilla group-REINFORCE (= src/dynamics.py::rhs_sampled):
        g_i = p^_i R~_i - p_i sum_k p^_k R~_k
  B  group-mean baseline (GRPO without std; "Dr. GRPO"):
        g_i = p^_i (R~_i - mu),            mu = sum_k p^_k R~_k
  N  full GRPO advantage (mean and std), one on-policy step (ratio = 1, no KL):
        g_i = p^_i (R~_i - mu) / sigma,    sigma^2 = sum_k p^_k (R~_k - mu)^2   (g = 0 if sigma = 0)

K = 2 boundary analysis (p2 -> 0; only groups with one minority sample matter):
  V : minority survives iff  r2 omega(1) > r1 omega(G)    -> alpha > ln rho / ln min(G, 1/eps)
  B,N: minority survives iff r2 omega(1) > r1 omega(G-1)  -> alpha > ln rho / ln(G-1)  (eps <= 1/G)
For K = 2 the GRPO step is g_1 = sqrt(p^_1 p^_2) sign(R~_1 - R~_2): reward magnitudes drop out.

(a) exact K=2 mean field: alpha_c by bisection for each update vs both formulas.
(b) Monte Carlo, r = (4,1), alpha = 1: long-run p2 for each update and G.
(c) exact K=5 mean field (multinomial enumeration) for r = (5,4,3,2,1), alpha = 1.

Run:  python3 -u next_phase/case2_checks/c05_grpo_variants.py      (~3-5 min)
"""
import itertools
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammaln
from scipy.stats import binom

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.dynamics import softmax  # noqa: E402
from src.finite_group import meanfield_stationary_K2  # noqa: E402
from src.finite_group_general import meanfield_stationary  # noqa: E402

EPS = 1e-3
rng = np.random.default_rng(11)


def omega(n, G, alpha, eps=EPS):
    n = np.asarray(n, dtype=float)
    return np.where(n > 0, np.maximum(n / G, eps) ** (-alpha), 0.0)


def updates_from_counts(counts, r, p, G, alpha, eps=EPS):
    """counts (M, K) -> dict of the three update vectors (M, K)."""
    ph = counts / G
    Rt = r[None, :] * omega(counts, G, alpha, eps)
    mu = np.sum(ph * Rt, axis=1, keepdims=True)
    V = ph * Rt - p[None, :] * mu if p.ndim == 1 else ph * Rt - p * mu
    B = ph * (Rt - mu)
    var = np.sum(ph * (Rt - mu) ** 2, axis=1, keepdims=True)
    sd = np.sqrt(var)
    scale = np.max(np.abs(Rt), axis=1, keepdims=True) + 1e-300
    N = np.where(sd > 1e-12 * scale, B / np.where(sd > 0, sd, 1.0), 0.0)
    return {"V": V, "B": B, "N": N}


# ---------------------------------------------------------------- (a) K = 2 exact
def m1_K2(p, r, G, alpha, eps=EPS):
    n = np.arange(G + 1)
    counts = np.stack([n, G - n], axis=1).astype(float)
    pr = binom.pmf(n, G, p)
    U = updates_from_counts(counts, r, np.array([p, 1 - p]), G, alpha, eps)
    return {k: float(pr @ v[:, 0]) for k, v in U.items()}


def survives(r, G, alpha, var, eps=EPS):
    """minority survives iff the drift of u = z1 - z2 is negative as p1 -> 1."""
    p = 1 - 1e-7
    return m1_K2(p, r, G, alpha, eps)[var] < 0


def alpha_c(r, G, var, lo=1e-3, hi=40.0):
    if not survives(r, G, hi, var):
        return np.inf
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if survives(r, G, mid, var):
            hi = mid
        else:
            lo = mid
    return hi


def interior_roots(r, G, alpha, var, eps=EPS):
    ps = np.linspace(0.5, 1 - 1e-9, 4001)
    f = np.array([m1_K2(p, r, G, alpha, eps)[var] for p in ps])
    sc = np.flatnonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)
    return [0.5 * (ps[i] + ps[i + 1]) for i in sc]


r2 = np.array([4.0, 1.0])
print("(a) K=2, r=(4,1), eps=1e-3: critical alpha (minority survives above it)")
print("     G | ln4/lnG | V (exact) | ln4/ln(G-1) | B (exact) | N (exact)")
for G in [2, 3, 4, 5, 6, 8, 16, 32, 64]:
    f1 = np.log(4) / np.log(G)
    f2 = np.log(4) / np.log(G - 1) if G > 2 else np.inf
    print(f"    {G:2d} | {f1:7.4f} | {alpha_c(r2, G, 'V'):9.4f} | {f2:11.4f} | "
          f"{alpha_c(r2, G, 'B'):9.4f} | {alpha_c(r2, G, 'N'):9.4f}")
nroots = {v: max(len(interior_roots(r2, G, a, v)) for G in [3, 4, 6, 8, 16] for a in [0.7, 1.0, 1.5, 2.5])
          for v in "VBN"}
print(f"    max number of interior roots of the K=2 drift on a grid (G in 3..16, alpha in 0.7..2.5): {nroots}")
print("    K=2 stationary p2 at alpha = 1 (exact mean field):")
for G in [4, 5, 6, 8, 16, 64, 256]:
    row = []
    for v in "VBN":
        rt = interior_roots(r2, G, 1.0, v)
        row.append(f"{v}: {1 - rt[0]:.4f}" if rt else f"{v}: 0 (extinct)")
    print(f"      G={G:3d}: " + ", ".join(row) + f"   [repo K2 solver, V: {1 - meanfield_stationary_K2(r2, G, 1.0):.4f}]")


# ---------------------------------------------------------------- (b) Monte Carlo
def mc(r, G, alpha, var, h=0.05, steps=40000, seeds=32):
    K = len(r)
    z = np.zeros((seeds, K))
    acc = np.zeros((seeds, K))
    for t in range(steps):
        p = softmax(z)
        counts = rng.multinomial(G, p).astype(float)
        g = updates_from_counts(counts, r, p, G, alpha)[var]
        z = z + h * g
        if t >= steps // 2:
            acc += softmax(z)
    return acc.mean(axis=0) / (steps - steps // 2)


print("(b) Monte Carlo, r=(4,1), alpha=1, h=0.05, 40k steps, 32 seeds; long-run p2 (2nd half):")
for G in [3, 4, 5, 6, 8, 16]:
    row = []
    for v in "VBN":
        p2 = mc(r2, G, 1.0, v)[1]
        rt = interior_roots(r2, G, 1.0, v)
        mf = 1 - rt[0] if rt else 0.0
        row.append(f"{v}: MC {p2:.4f} / MF {mf:.4f}")
    print(f"    G={G:2d}: " + " | ".join(row))


# ---------------------------------------------------------------- (c) K = 5 exact
def compositions(G, K):
    out = []
    for c in itertools.combinations(range(G + K - 1), K - 1):
        prev, parts = -1, []
        for x in c:
            parts.append(x - prev - 1)
            prev = x
        parts.append(G + K - 1 - prev - 1)
        out.append(parts)
    return np.array(out, dtype=float)


def mean_drift_exact(z, r, G, alpha, var, comps, logcoef):
    p = softmax(z)
    logp = np.log(np.clip(p, 1e-300, None))
    lp = logcoef + comps @ logp
    w = np.exp(lp - lp.max())
    w *= np.exp(lp.max())
    U = updates_from_counts(comps, r, p, G, alpha)[var]
    return w @ U


r5 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
print("(c) K=5, r=(5,4,3,2,1), alpha=1, exact mean field (ODE to t=2e4 from uniform):")
for G in [8, 16, 32]:
    comps = compositions(G, 5)
    logcoef = gammaln(G + 1) - np.sum(gammaln(comps + 1), axis=1)
    pi_V, _, _ = meanfield_stationary(r5, G, 1.0, EPS, method="newton")
    row = [f"V(KKT): {np.array2string(pi_V, precision=4)}"]
    for v in "BN":
        sol = solve_ivp(lambda t, z: mean_drift_exact(z, r5, G, 1.0, v, comps, logcoef), (0, 2e4),
                        np.zeros(5), method="LSODA", rtol=1e-8, atol=1e-10)
        row.append(f"{v}: {np.array2string(softmax(sol.y[:, -1]), precision=4)}")
    print(f"    G={G:2d}: " + "  ".join(row))
