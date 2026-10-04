"""
Check 09 -- independent re-derivation checks of the remaining formulas and numbers.

 1  alpha = 0 exact solution: collapse time 53.89/Delta, p2 ~ 1/(2 Delta t)
 2  rates: lambda(r=(4,1), alpha=1) = 1.6;  tie: lambda = alpha rbar K^(alpha-1)
 3  eps* = p (brute force) and add-lambda lambda* = Kp(1-p)/(1-Kp)^2 (brute force)
 4  size-biasing identity and the alpha = 1 closed form w_G = (1-(1-p)^G)/p
 5  distortion coefficients at G = 4096: clip alpha(alpha-1)/2, offset c* ~ 0
 6  naive Richardson: is w_G monotone?  does the K=2 drift have an interior root?
    guarded Richardson: monotonicity of w_G (the theorem needs it)
 7  general-K numbers quoted in the report: alpha_c(O5, G=16) = 0.886,
    minimum G for O4/O5, alpha = 4 peak share 0.445 at G = 12
 8  clip kink jump Delta_k = -alpha (k/G)^-alpha P[B = k-1] (finite differences)
 9  group-mean baseline at alpha = 1: mean field at G == vanilla mean field at G-1 (any K)
10  the "factor 555" EMA comparison: what part of the clip's RMSE can reach the update?
11  the paper's Table 4: where can the clip fire at all (eps > 1/G)?

Run:  python3 -u next_phase/case2_checks/c09_misc_formulas.py      (~1-2 min)
"""
import os
import sys

import numpy as np
from scipy.stats import binom

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.dynamics import linear_rates  # noqa: E402
from src.estimators import ClipRule, AddLambdaRule, RichardsonRule, OffsetRule, effective_weight_alpha1  # noqa: E402
from src.finite_group_general import (meanfield_stationary, kept_at_large_alpha, extinction_alpha,  # noqa: E402
                                      effective_weight)
from src.finite_group import meanfield_stationary_K2  # noqa: E402
from src.meanfield_rule import check_monotone, stationary_K2  # noqa: E402

rng = np.random.default_rng(9)

# 1 ---------------------------------------------------------------------------
u0, u1 = 0.1, np.log(99)
print(f"1  (u+sinh u) from 0.1 to ln 99 = {(u1 + np.sinh(u1)) - (u0 + np.sinh(u0)):.4f}  (report: 53.89)")

# 2 ---------------------------------------------------------------------------
print(f"2  lambda(r=(4,1), alpha=1) = {linear_rates(np.array([4.0, 1.0]), 1.0)[0]:.6f}  (report: 1.6)")
for K, a in [(3, 0.5), (5, 2.0)]:
    lam = linear_rates(np.full(K, 1.7), a)
    print(f"   tie K={K}, alpha={a}: rates {np.round(lam, 6)} vs alpha*rbar*K^(alpha-1) = {a*1.7*K**(a-1):.6f}")

# 3 ---------------------------------------------------------------------------
worst = 0.0
for alpha, G, p in [(0.5, 8, 0.07), (1.0, 16, 0.3), (2.0, 32, 0.11), (3.0, 64, 0.52)]:
    eps = np.geomspace(1e-4, 1.0, 400001)
    n = np.arange(G + 1)
    pm = binom.pmf(n, G, p)
    x = n / G
    mse = np.array([pm @ ((np.maximum(x, e) ** (-alpha) * p ** alpha - 1) ** 2) for e in eps[::200]])
    e_best = eps[::200][np.argmin(mse)]
    worst = max(worst, abs(e_best / p - 1))
print(f"3  eps* = p by brute force (grid ratio 1.0046): max |eps*/p - 1| = {worst:.2e}")
worst = 0.0
for G, K, p in [(16, 2, 0.1), (32, 5, 0.05), (8, 3, 0.6)]:
    lam = np.linspace(0, 50, 2000001)
    D = G + K * lam
    mse = (lam ** 2 * (1 - K * p) ** 2 + G * p * (1 - p)) / D ** 2
    lb = lam[np.argmin(mse)]
    th = K * p * (1 - p) / (1 - K * p) ** 2
    worst = max(worst, abs(lb - th) / th)
print(f"   add-lambda lambda* = Kp(1-p)/(1-Kp)^2 by brute force: max rel diff {worst:.1e}")

# 4 ---------------------------------------------------------------------------
G, a, e = 12, 1.7, 1e-3
p = np.linspace(1e-4, 1, 501)
n = np.arange(G + 1)
phi = binom.pmf(n[None, :], G, p[:, None]) @ ((n / G) * np.where(n > 0, np.maximum(n / G, e) ** (-a), 0))
sb = effective_weight(p, G, a, e)
print(f"4  size-biasing: max rel |phi_G/p - E[omega(1+B)]| = {np.max(np.abs(phi / p - sb) / sb):.1e}")
print(f"   alpha=1 closed form vs binomial sum (G=12): "
      f"{np.max(np.abs(effective_weight(p, G, 1.0, e) - effective_weight_alpha1(p, G)) / effective_weight_alpha1(p, G)):.1e}")
print(f"   ... with eps = 0.2 > 1/G the identity fails by up to "
      f"{np.max(np.abs(effective_weight(p, G, 1.0, 0.2) - effective_weight_alpha1(p, G)) / effective_weight_alpha1(p, G)):.2f} (needs eps <= 1/G)")

# 5 ---------------------------------------------------------------------------
G, p = 4096, 0.3
for alpha in [0.5, 2.0, 3.0]:
    D_clip = ClipRule(alpha, 1e-6).effective_weight(np.array([p]), G)[0] * p ** alpha - 1
    D_off = OffsetRule(alpha, 1e-6).effective_weight(np.array([p]), G)[0] * p ** alpha - 1
    coef = D_clip * G * p / (1 - p)
    print(f"5  alpha={alpha}: G p D/(1-p): clip {coef:+.4f} (theory {alpha*(alpha-1)/2:+.4f}); "
          f"offset c*: {D_off*G*p/(1-p):+.5f} (theory 0)")

# 6 ---------------------------------------------------------------------------
for G in [8, 16, 64]:
    for alpha in [0.5, 1.0, 2.0]:
        naive = RichardsonRule(alpha, 1e-3, guard=None)
        guarded = RichardsonRule(alpha, 1e-3, guard=1)
        okn, risen = check_monotone(naive, G)
        okg, riseg = check_monotone(guarded, G)
        # K=2 drift sign changes in p1 in (0.5,1) for rho = 1.5, 4
        roots = []
        for rho in [1.5, 4.0]:
            ps = np.linspace(0.5, 1 - 1e-9, 2001)
            w1 = naive.effective_weight(ps, G)
            w2 = naive.effective_weight(1 - ps, G)
            f = rho * w1 - w2
            roots.append(int(np.sum(np.sign(f[:-1]) * np.sign(f[1:]) < 0)))
        print(f"6  G={G:2d} alpha={alpha}: naive w_G monotone={okn} (max rise {risen:.2e}), "
              f"interior roots rho=1.5/4: {roots}; guarded monotone={okg}")

# 7 ---------------------------------------------------------------------------
r5 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
ac, _ = extinction_alpha(r5, 4, 16, 1e-3)
print(f"7  alpha_c(O5, G=16) = {ac:.4f} (report 0.886); pairwise ln5/ln16 = {np.log(5)/np.log(16):.4f}")
for j, name in [(3, 'O4'), (4, 'O5')]:
    Gmin = next(G for G in range(2, 40) if kept_at_large_alpha(r5, j, G))
    print(f"   minimum G keeping {name} at some alpha: {Gmin}")
shares = {G: 1 - meanfield_stationary_K2(np.array([4.0, 1.0]), G, 4.0) for G in [8, 10, 12, 14, 16]}
print("   alpha=4, r=(4,1): p2 by G " + ", ".join(f"G={G}: {s:.4f}" for G, s in shares.items())
      + f"  (G->inf: {1/(1+4**0.25):.4f})")

# 8 ---------------------------------------------------------------------------
G, alpha, p = 16, 1.0, 0.33
for k in [1, 2, 3]:
    le = np.log(k / G)
    d = 1e-6
    w = lambda x: effective_weight(np.array([p]), G, alpha, np.exp(x))[0]
    s_left = (w(le - d) - w(le - 2 * d)) / d
    s_right = (w(le + 2 * d) - w(le + d)) / d
    pred = -alpha * (k / G) ** (-alpha) * binom.pmf(k - 1, G - 1, p)
    print(f"8  kink at eps={k}/{G}: slope jump {s_right - s_left:+.5f} vs Delta_k {pred:+.5f}")

# 9 ---------------------------------------------------------------------------
def m_vanilla_a1(p, r, G):
    a = 1 - (1 - p) ** G
    return r * a - p * np.sum(r * a)


def m_baseline_a1(p, r, G):
    # E[p^_i (R~_i - mu)] at alpha = 1, eps <= 1/G:  p^_i R~_i = r_i 1[n_i>0], mu = sum_k r_k 1[n_k>0]
    K = len(r)
    out = np.empty(K)
    for i in range(K):
        s = r[i] * p[i]                                       # E[p^_i 1[n_i>0]] r_i = r_i p_i
        for k in range(K):
            if k != i:
                s += r[k] * p[i] * (1 - (1 - p[k]) ** (G - 1))
        out[i] = r[i] * (1 - (1 - p[i]) ** G) - s
    return out


worst = 0.0
for t in range(2000):
    K = int(rng.integers(2, 7))
    r = rng.uniform(0.2, 5, K)
    p = rng.dirichlet(np.ones(K))
    G = int(rng.integers(2, 40))
    worst = max(worst, np.max(np.abs(m_baseline_a1(p, r, G) - m_vanilla_a1(p, r, G - 1))))
print(f"9  alpha=1: max |m_baseline(G) - m_vanilla(G-1)| over 2000 random (K,r,p,G) = {worst:.1e}")
pi15, _, _ = meanfield_stationary(r5, 15, 1.0, 1e-3)
print(f"   => K=5, r=(5,4,3,2,1): baseline at G=16 has the vanilla G=15 target {np.round(pi15, 4)}")

# 10 --------------------------------------------------------------------------
G, p, alpha = 16, 0.1, 1.0
rule = ClipRule(alpha, 1e-3)
m1, var = rule.moments_relative(np.array([p]), G)
n = np.arange(G + 1)
pm = binom.pmf(n, G, p)
x = n / G
om = np.where(n > 0, np.maximum(x, 1e-3) ** (-alpha), 1e-3 ** (-alpha))
rel = om * p ** alpha
cond = n >= 1
rmse_cond = np.sqrt(np.sum(pm[cond] * (rel[cond] - 1) ** 2) / pm[cond].sum())
b = np.arange(G)
pb = binom.pmf(b, G - 1, p)
relsb = np.maximum((1 + b) / G, 1e-3) ** (-alpha) * p ** alpha
rmse_sb = np.sqrt(np.sum(pb * (relsb - 1) ** 2))
term = x * np.where(n > 0, np.maximum(x, 1e-3) ** (-alpha), 0) / p ** (1 - alpha)   # p^ omega(n)/p^(1-alpha)
print(f"10 clip, G=16, p=0.1, alpha=1: relative RMSE of omega(n) incl. n=0 = {np.sqrt((m1[0]-1)**2+var[0]):.1f} (report 42.6);"
      f" share of MSE from n=0: {pm[0]*(rel[0]-1)**2/((m1[0]-1)**2+var[0]):.3f}")
print(f"   conditional on n>=1: {rmse_cond:.3f};  size-biased (what the mean field sees): {rmse_sb:.3f};"
      f"  update term p^ omega(n): mean {pm @ term:.4f}, sd {np.sqrt(pm @ term**2 - (pm @ term)**2):.4f}")

# 11 --------------------------------------------------------------------------
table4 = {4: (17.63, 17.63, 17.63), 8: (27.50, 27.50, 41.99), 16: (41.02, 33.05, 43.91),
          32: (32.13, 35.27, 25.62), 64: (36.52, 39.12, 42.19)}
print("11 paper Table 4: clip can fire (eps > 1/G)?  entries")
for G, vals in table4.items():
    fires = [e > 1 / G for e in (0.01, 0.1, 0.2)]
    print(f"   G={G:2d}: fires {fires}  entries {vals}")
