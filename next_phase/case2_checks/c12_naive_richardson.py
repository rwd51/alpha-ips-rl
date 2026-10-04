"""
Check 12 -- the report's corollary "naive split-half Richardson gives a singleton a
negative weight, so the minority dies unconditionally" uses the K=2 survival
theorem, whose proof needs w_G to be monotone.  For the naive rule w_G is NOT
monotone and can be negative in the middle of [0, 1], so the theorem does not apply.

K = 2 mean field:  u_dot = 2 p1 p2 f(p1),  f(p1) = r1 w_G(p1) - r2 w_G(1 - p1).
We list every sign change of f on (0, 1) (stable rest point: f goes + -> -),
integrate the mean field from p1 = 0.5, and run the Monte Carlo agent
(src.estimators.sampled_step with RichardsonRule(guard=None)).

Run:  python3 -u next_phase/case2_checks/c12_naive_richardson.py      (~1-2 min)
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.dynamics import softmax  # noqa: E402
from src.estimators import RichardsonRule, sampled_step  # noqa: E402

rng = np.random.default_rng(13)
for G in [8, 16]:
    rule = RichardsonRule(1.0, 1e-3, guard=None)
    ps = np.linspace(1e-6, 1 - 1e-6, 20001)
    w = rule.effective_weight(ps, G)
    neg = ps[w < 0]
    print(f"G={G}, alpha=1, naive Richardson: w_G(0)={rule.effective_weight(np.array([0.0]), G)[0]:.1f}, "
          f"w_G(1)={rule.effective_weight(np.array([1.0]), G)[0]:.2f}, w_G<0 on p in "
          f"[{neg.min() if len(neg) else np.nan:.3f}, {neg.max() if len(neg) else np.nan:.3f}]")
    for r in [np.array([1.5, 1.0]), np.array([4.0, 1.0])]:
        f = r[0] * rule.effective_weight(ps, G) - r[1] * rule.effective_weight(1 - ps, G)
        sc = np.flatnonzero(np.sign(f[:-1]) != np.sign(f[1:]))
        rests = [(round(float(ps[i]), 4), "stable" if f[i] > 0 else "unstable") for i in sc]

        def rhs(t, z):
            p = softmax(z)
            ww = rule.effective_weight(p, G)
            S = float(np.sum(p * r * ww))
            return p * (r * ww - S)

        sol = solve_ivp(rhs, (0, 5000), np.zeros(2), method="LSODA", rtol=1e-9, atol=1e-12)
        p_end = softmax(sol.y[:, -1])
        # Monte Carlo agent
        z = np.zeros((32, 2))
        h, steps = 0.01, 40000
        acc = np.zeros(2)
        for t in range(steps):
            g, _ = sampled_step(z, r, rule, G, rng)
            z = z + h * g
            if t >= steps // 2:
                acc += softmax(z).mean(axis=0)
        acc /= steps - steps // 2
        print(f"   r={tuple(r)}: rest points of f in (0,1): {rests}; mean field from p1=0.5 -> p1={p_end[0]:.4f}; "
              f"MC (h=0.01, 32 seeds) long-run p1 = {acc[0]:.4f}")
