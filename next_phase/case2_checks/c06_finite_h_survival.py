"""
Check 06 -- mean-field survival (h -> 0) versus survival of the constant-step
stochastic process (the Monte Carlo agent), K = 2, vanilla update (rhs_sampled).

Boundary analysis (p2 = q -> 0, u = z1 - z2 ~ -ln q):
  * a homogeneous group (prob ~ 1 - Gq) moves u UP by  c q,   c = 2 h r1 omega(G)
  * a group with one minority sample (prob ~ Gq) moves u DOWN by J = 2 h r2 omega(1) / G
Both rates are proportional to q, so in the operational time d tau = q dt the gap is
a spectrally negative Levy process; its stationary tail is exp(-theta* u) with
      c theta* = G (1 - exp(-theta* J)).
Real time spends 1/q = e^u steps per unit tau, so the real-time stationary
density ~ exp((1 - theta*) u) is normalisable iff theta* > 1, i.e. iff

      G (1 - exp(-2 h r2 omega(1) / G)) > 2 h r1 omega(G)          (*)

As h -> 0, (*) becomes r2 omega(1) > r1 omega(G): the report's mean-field
criterion.  At finite h it needs margin m > ~ h r2 omega(1) / G  (= h r2 G^(alpha-1)).
If (*) fails but m > 0, the walk is null recurrent: the time-average of p2 -> 0.

(a) r = (4,1), alpha = 1, G = 5 (m = ln 1.25 > 0, mean-field p2 = 0.0735):
    (*) predicts survival only for h < 0.233.  Time-averaged p2 over growing windows.
(b) exact tie r = (1,1), G = 16, h = 0.5 (the Exp. 1/2 setting): (*) predicts that
    the noise-driven collapse persists for alpha < 0.0116 although every alpha > 0
    has a uniform mean-field attractor.  Long-run share of time with max p >= 0.95.

Each (h) or (alpha) job runs in its own process (multiprocessing, 7 workers).
Run:  python3 -u next_phase/case2_checks/c06_finite_h_survival.py      (~6-10 min on 8 cores)
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.finite_group import meanfield_stationary_K2  # noqa: E402

EPS = 1e-3


def omega(n, G, alpha):
    return np.maximum(np.asarray(n, float) / G, EPS) ** (-alpha)


def criterion(r, G, alpha, h):
    J = 2 * h * r[1] * omega(1, G, alpha) / G
    c = 2 * h * r[0] * omega(G, G, alpha)
    lhs = G * (1 - np.exp(-J))
    psi = lambda th: c * th - G * (1 - np.exp(-th * J))
    theta = brentq(psi, 1e-9, 1e4) if c < G * J else 0.0
    return lhs > c, theta


def run(args):
    """vanilla K=2 update (same distribution as rhs_sampled); window averages of
    p2 and of the share of time with max p >= 0.95, over (W/2, W] for each W."""
    r, G, alpha, h, windows, seeds, seed = args
    rng = np.random.default_rng(seed)
    r = np.asarray(r, float)
    u = np.zeros(seeds)
    n_steps = max(windows)
    W = np.array(sorted(windows))
    s_p2 = np.zeros(len(W))
    s_col = np.zeros(len(W))
    cnt = np.zeros(len(W))
    first = W.min() // 2
    for t in range(1, n_steps + 1):
        p = 1.0 / (1.0 + np.exp(-u))
        q = 1.0 - p
        n = rng.binomial(G, p).astype(float)
        w1 = np.where(n > 0, omega(n, G, alpha), 0.0)
        w2 = np.where(n < G, omega(G - n, G, alpha), 0.0)
        u += 2 * h * (q * r[0] * (n / G) * w1 - p * r[1] * (1 - n / G) * w2)
        if t > first:
            act = (t > W // 2) & (t <= W)
            if act.any():
                pp = 1.0 / (1.0 + np.exp(-u))
                m2 = float(np.mean(1 - pp))
                mc = float(np.mean(np.maximum(pp, 1 - pp) >= 0.95))
                s_p2[act] += m2
                s_col[act] += mc
                cnt[act] += 1
    return {int(w): (s_p2[i] / cnt[i], s_col[i] / cnt[i]) for i, w in enumerate(W)}


if __name__ == "__main__":
    rA = (4.0, 1.0)
    hs = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    winA = [20000, 80000, 320000, 1280000]
    rB = (1.0, 1.0)
    alphas = [0.0, 0.005, 0.01, 0.015, 0.025, 0.05, 0.25]
    winB = [50000, 200000, 800000]
    jobs = [(rA, 5, 1.0, h, winA, 48, 100 + i) for i, h in enumerate(hs)]
    jobs += [(rB, 16, a, 0.5, winB, 48, 200 + i) for i, a in enumerate(alphas)]
    with Pool(7) as pool:
        res = pool.map(run, jobs)

    mf = 1 - meanfield_stationary_K2(np.array(rA), 5, 1.0)
    print(f"(a) r=(4,1), alpha=1, G=5: margin m = {np.log(5/4):.3f}, mean-field p2 = {mf:.4f}")
    for i, h in enumerate(hs):
        ok, th = criterion(np.array(rA), 5, 1.0, h)
        trend = "  ".join(f"T={w:>7d}: <p2>={res[i][w][0]:.4f}" for w in winA)
        print(f"    h={h:4.2f}: (*) {'holds' if ok else 'FAILS'} (theta*={th:5.2f})  |  {trend}")
    G, h = 16, 0.5
    a_h = np.log(-G * np.log(1 - 2 * h * rB[0] / G) / (2 * h * rB[1])) / np.log(G)
    print(f"(b) tie r=(1,1), G=16, h=0.5: (*) predicts persistent noise-collapse for alpha < {a_h:.4f}")
    for k, a in enumerate(alphas):
        ok, th = criterion(np.array(rB), G, a, h)
        rr = res[len(hs) + k]
        trend = "  ".join(f"T={w:>6d}: share(max p>=.95)={rr[w][1]:.3f}" for w in winB)
        print(f"    alpha={a:5.3f}: (*) {'holds' if ok else 'FAILS'} (theta*={th:6.2f}) | {trend}")
