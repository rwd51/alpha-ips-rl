"""
Check 13 -- many tied valid outcomes (K >> G), the regime of the base paper's
HypoSpace tasks (binary reward: every valid answer has r = 1).

With all r equal, the survival criterion omega(1)/omega(G) > r1/r2 = 1 is always met,
so (S4) says nothing.  What changes is the RESTORING FORCE toward the uniform
policy.  Linearizing the finite-G mean field m_i = p_i (r w_G(p_i) - S) at p = 1/K
gives rates  lambda_G = -r w_G'(1/K) / K^2  (K-1 fold), against the ideal
lambda_inf = alpha r K^(alpha-1).  For alpha = 1 and K >> G:
      lambda_G / lambda_inf ~ G (G-1) / (2 K^2),
because w_G is nearly flat (saturated at G^alpha) once p << 1/G.

Run:  python3 -u next_phase/case2_checks/c13_many_outcomes.py      (seconds)
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.dynamics import softmax  # noqa: E402
from src.finite_group_general import effective_weight  # noqa: E402

EPS = 1e-3


def mf(z, r, G, alpha):
    p = softmax(z)
    w = effective_weight(p, G, alpha, EPS)
    S = float(np.sum(p * r * w))
    return p * (r * w - S)


print("  K   G  alpha | slowest restoring rate: finite-G MF | ideal IPS | ratio | G(G-1)/(2K^2)")
for alpha in [1.0, 2.0]:
    for G in [8, 16]:
        for K in [4, 16, 64, 256]:
            r = np.ones(K)
            z0 = np.zeros(K)
            d = 1e-6
            J = np.zeros((K, K))
            for j in range(K):
                e = np.zeros(K)
                e[j] = d
                J[:, j] = (mf(z0 + e, r, G, alpha) - mf(z0 - e, r, G, alpha)) / (2 * d)
            ev = np.sort(np.linalg.eigvalsh(-0.5 * (J + J.T)))[1:]
            lam_G = ev.min()
            lam_inf = alpha * K ** (alpha - 1)
            print(f"  {K:3d} {G:3d}  {alpha:3.1f}  | {lam_G:12.5g}                        | {lam_inf:9.4g} | "
                  f"{lam_G/lam_inf:7.4f} | {G*(G-1)/(2*K*K) if alpha == 1 else float('nan'):7.4f}")
