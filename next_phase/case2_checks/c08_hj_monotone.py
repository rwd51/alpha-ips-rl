"""
Check 08 -- the "single sign change of h_j(alpha)" (listed as unproven in the
report) follows from a monotonicity argument.

Write m_G = min(G, 1/eps) and the normalized weight
    wt_alpha(p) = w_G(p) / m_G^alpha = E[ (max(1/G,eps) / max((1+B)/G,eps))^alpha ],
    B ~ Bin(G-1, p).
The base of the power lies in (0, 1], so wt_alpha(p) is non-increasing in alpha,
with wt_alpha(0) = 1 for every alpha.  Hence every p_i(s) solving r_i wt(p_i) = s
is non-increasing in alpha, the normalized level s*(alpha) = S*(alpha)/m_G^alpha
is non-increasing in alpha, and
    h_j(alpha) = ln r_j + alpha ln m_G - ln S*(alpha) = ln r_j - ln s*(alpha)
is NON-DECREASING in alpha: at most one sign change, from - to +.
(For eps <= 1/G, B ~ Bin(G-1,p) is stochastically increasing in G, so the same
argument shows the support can only grow with G.)

Numerics: a fast nested solver (log w_G tabulated on a 40001-point grid, outer
bisection on ln S) is first validated against the repository's
finite_group_general.meanfield_stationary, then used on random reward vectors.

Run:  python3 -u next_phase/case2_checks/c08_hj_monotone.py      (~2-4 min)
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.finite_group_general import effective_weight, meanfield_stationary, kept_at_large_alpha  # noqa: E402

rng = np.random.default_rng(8)
tq = np.linspace(0.0, 1.0, 40001)
P_GRID = 0.5 - 0.5 * np.cos(np.pi * tq)          # clustered at both ends


_CACHE = {}


def _logw(G, alpha, eps):
    """log w_G on P_GRID; depends only on (G, alpha, eps), so it is cached."""
    key = (G, float(alpha), float(eps))
    if key not in _CACHE:
        _CACHE[key] = np.log(effective_weight(P_GRID, G, alpha, eps))
    return _CACHE[key]


def fast_stationary(r, G, alpha, eps):
    logw = _logw(G, alpha, eps)                                  # decreasing in p
    lw0, lw1 = logw[0], logw[-1]
    lr = np.log(r)

    def p_of(lS):
        v = lS - lr
        p = np.interp(-v, -logw, P_GRID)
        return np.where(v >= lw0, 0.0, np.where(v <= lw1, 1.0, p))

    lo, hi = lr.max() + lw1, lr.max() + lw0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if p_of(mid).sum() > 1.0:
            lo = mid
        else:
            hi = mid
    lS = 0.5 * (lo + hi)
    p = p_of(lS)
    return p / p.sum(), lS


# validation against the repository solver
dmax_p, dmax_S = 0.0, 0.0
for t in range(30):
    K = int(rng.integers(3, 7))
    r = np.sort(np.exp(rng.normal(0, 1.0, K)))[::-1]
    G = int(rng.choice([2, 3, 5, 8, 16]))
    a = float(np.exp(rng.uniform(np.log(0.05), np.log(8))))
    e = float(rng.choice([1e-3, 0.15]))
    p1, lS1 = fast_stationary(r, G, a, e)
    p2, S2, _ = meanfield_stationary(r, G, a, e, method="newton")
    dmax_p = max(dmax_p, np.max(np.abs(p1 - p2)))
    dmax_S = max(dmax_S, abs(lS1 - np.log(S2)))
print(f"fast solver vs repository solver (30 random cases): max |dp| = {dmax_p:.1e}, max |d ln S| = {dmax_S:.1e}")

alphas = np.geomspace(0.02, 40.0, 60)
viol_s, viol_h, multi, tested, agree, agree_n, curves = 0, 0, 0, 0, 0, 0, 0
for trial in range(60):
    K = int(rng.integers(3, 9))
    r = np.sort(np.exp(rng.normal(0, rng.choice([0.3, 1.0, 2.0]), K)))[::-1]
    r = r / r.max()
    for G in [2, 3, 5, 8, 16]:
        for eps in [1e-3, 0.15]:
            mG = min(G, 1 / eps)
            s = np.array([fast_stationary(r, G, a, eps)[1] - a * np.log(mG) for a in alphas])
            tested += 1
            viol_s += int(np.any(np.diff(s) > 1e-7))
            H = np.log(r)[None, :] - s[:, None]
            for j in range(1, K):
                curves += 1
                sg = np.sign(H[:, j])
                sg = sg[sg != 0]
                multi += int(np.sum(sg[1:] != sg[:-1]) > 1)
                viol_h += int(np.any(np.diff(H[:, j]) < -1e-7))
                if eps < 1 / G and abs(H[-1, j]) > 1e-6:
                    agree_n += 1
                    agree += int(kept_at_large_alpha(r, j, G) == (H[-1, j] > 0))
print(f"{tested} (r, G, eps) cases x 60 alphas in [0.02, 40], {curves} outcome curves:")
print(f"  cases where s*(alpha) = S*/m_G^alpha increased:      {viol_s}")
print(f"  curves h_j(alpha) that decreased anywhere:          {viol_h}")
print(f"  curves with more than one sign change:              {multi}")
print(f"  alpha->inf criterion vs sign of h_j(40) (eps<1/G):   {agree}/{agree_n} agree")
