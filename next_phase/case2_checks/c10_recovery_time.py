"""
Check 10 -- with a finite group, a nearly extinct outcome recovers only
hyperbolically, even when the survival criterion holds.

Ideal IPS (alpha = 1): z_dot_j ~ r_j for p_j -> 0, so ln p_j grows linearly and the
recovery time is ~ ln(1/p_j(0)) / rate.  Finite G: the weight saturates at
w_G(0) = omega(1), so z_dot_j ~ p_j (r_j w_G(0) - S): the probability multiplier
p_j that IPS was meant to remove comes back for p_j << 1/G, and
      dp_j/dt ~ c p_j^2,   recovery time ~ 1 / (c p_j(0)),
      K = 2: c = 2 (r2 w_G(0) - r1 w_G(1)).

(a) K=2, r=(4,1), alpha=1, G=16 (survival regime, p2* = 0.195): time for p2 to
    reach p2*/2 from p2(0) = 1e-2 ... 1e-8, finite-G mean field vs ideal IPS.
(b) the Monte Carlo agent (rhs_sampled), h = 0.05, 64 seeds, from p2(0) = 1e-3, 1e-4.
(c) K=5, r=(5,4,3,2,1), G=16, alpha=1: recovery of O5 (r=1) from 1e-6.

Run:  python3 -u next_phase/case2_checks/c10_recovery_time.py      (~2-4 min)
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.dynamics import rhs_alpha, rhs_sampled, softmax, stationary_p  # noqa: E402
from src.finite_group import meanfield_stationary_K2  # noqa: E402
from src.finite_group_general import effective_weight, meanfield_stationary  # noqa: E402

EPS = 1e-3
rng = np.random.default_rng(10)


def mf_rhs(z, r, G, alpha):
    p = softmax(z)
    w = effective_weight(p, G, alpha, EPS)
    S = float(np.sum(p * r * w))
    return p * (r * w - S)


def time_to(f, z0, j, target, T):
    ev = lambda t, z: softmax(z)[j] - target
    ev.terminal, ev.direction = True, 1
    sol = solve_ivp(f, (0, T), z0, method="LSODA", rtol=1e-9, atol=1e-12, events=ev)
    return sol.t_events[0][0] if len(sol.t_events[0]) else np.inf


r = np.array([4.0, 1.0])
G, alpha = 16, 1.0
p2s = 1 - meanfield_stationary_K2(r, G, alpha)
c = 2 * (r[1] * effective_weight(0.0, G, alpha, EPS)[0] - r[0])
print(f"(a) K=2, r=(4,1), alpha=1, G=16: mean-field p2* = {p2s:.4f}; target p2*/2; c = {c:.1f}")
for p20 in [1e-2, 1e-4, 1e-6, 1e-8]:
    z0 = np.log(np.array([1 - p20, p20]))
    t_fin = time_to(lambda t, z: mf_rhs(z, r, G, alpha), z0, 1, p2s / 2, 1e10)
    t_ide = time_to(lambda t, z: rhs_alpha(z, r, alpha, p_floor=0.0), z0, 1, 0.1, 1e6)
    print(f"    p2(0)={p20:.0e}: finite-G mean field t = {t_fin:12.1f}  (1/(c p2(0)) = {1/(c*p20):12.1f});"
          f"  ideal IPS t = {t_ide:6.2f}")

print("(b) Monte Carlo agent (rhs_sampled), h=0.05, 64 seeds: time to reach p2 >= p2*/2")
h = 0.05
for p20 in [1e-3, 1e-4]:
    z = np.tile(np.log(np.array([1 - p20, p20])), (64, 1))
    hit = np.full(64, np.nan)
    max_steps = int(20 / (c * p20) / h)
    for step in range(1, max_steps + 1):
        g, _ = rhs_sampled(z, r, alpha, G, rng, EPS)
        z = z + h * g
        p2 = softmax(z)[:, 1]
        new = (p2 >= p2s / 2) & np.isnan(hit)
        hit[new] = step * h
        if np.all(np.isfinite(hit)):
            break
    print(f"    p2(0)={p20:.0e}: median t = {np.nanmedian(hit):8.1f}, recovered {np.mean(np.isfinite(hit))*100:.0f}% "
          f"(mean-field prediction {time_to(lambda t, zz: mf_rhs(zz, r, G, alpha), np.log(np.array([1-p20, p20])), 1, p2s/2, 1e9):.1f})")

r5 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
pi, S, _ = meanfield_stationary(r5, 16, 1.0, EPS)
p0 = pi.copy()
p0[4] = 1e-6
p0[:4] *= (1 - 1e-6) / p0[:4].sum()
t5 = time_to(lambda t, z: mf_rhs(z, r5, 16, 1.0), np.log(p0), 4, pi[4] / 2, 1e10)
ps = stationary_p(r5, 1.0)
q0 = ps.copy()
q0[4] = 1e-6
q0[:4] *= (1 - 1e-6) / q0[:4].sum()
t5i = time_to(lambda t, z: rhs_alpha(z, r5, 1.0, p_floor=0.0), np.log(q0), 4, ps[4] / 2, 1e6)
print(f"(c) K=5, G=16, alpha=1: O5 from 1e-6 to p5*/2 = {pi[4]/2:.4f}: finite-G mean field t = {t5:.0f} "
      f"(invasion rate r5 w_G(0) - S* = {16 - S:.2f}); ideal IPS t = {t5i:.2f}")
