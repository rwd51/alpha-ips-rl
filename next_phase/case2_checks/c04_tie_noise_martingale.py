"""
Check 04 -- the noise-driven collapse at an exact tie (alpha = 0) is NOT a
"Polya-urn rich-get-richer" mechanism: the logit gap is an exact martingale.

For K = 2, r1 = r2 = r, the sampled update gives
      Delta u = 2 h r (p_hat_1 - p_1),   E[Delta u | u] = 0,   Var = 4 h^2 r^2 p(1-p)/G.
Diffusion approximation (generator  (1/2) sigma^2(u) d^2/du^2,  sigma^2 = D p(1-p),
D = 4 h^2 r^2 / G):  mean first-passage time from u = 0 to |u| = U = ln(th/(1-th)):

      E[t_c] = (G / (h r^2)) * (U^2/2 + cosh U - 1)      (time units t = steps * h)

-> exactly linear in G (the report fits G^1.07).  Because u is a martingale with
bounded increments whose conditional variance never vanishes in the interior,
u cannot converge: limsup u = +inf and liminf u = -inf a.s. -> the "collapse" is
not absorbing; the winner switches on ever longer time scales.

(a) E[Delta u] = 0 check with the repository's rhs_sampled.
(b) mean / median first passage to max p >= 0.98 (Exp. 1, Fig. 2c setting:
    h = 0.5, r = 1), 2000 seeds per G, vs the formula and the report's medians.
(c) recurrence: G = 4, 400 seeds, 2e5 steps; count switches of the collapsed side.

Run:  python3 -u next_phase/case2_checks/c04_tie_noise_martingale.py      (~1-2 min)
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.dynamics import rhs_sampled, softmax  # noqa: E402

rng = np.random.default_rng(7)
h, r = 0.5, 1.0

# ---------------------------------------------------------------- (a)
z = np.tile(np.array([0.8, 0.0]), (200000, 1))
g, _ = rhs_sampled(z, np.array([r, r]), 0.0, 16, rng)
du = h * (g[:, 0] - g[:, 1])
print(f"(a) u0=0.8, G=16: mean Delta u = {du.mean():+.2e} +- {du.std()/np.sqrt(len(du)):.1e} (SE); "
      f"Var = {du.var():.5f} vs 4h^2r^2p(1-p)/G = {4*h*h*r*r*softmax(z[0])[0]*softmax(z[0])[1]/16:.5f}")


# ---------------------------------------------------------------- (b)
def formula(G, th):
    U = np.log(th / (1 - th))
    return (G / (h * r * r)) * (U * U / 2 + np.cosh(U) - 1)


def first_passage(G, th, n_seeds, max_steps):
    u = np.zeros(n_seeds)
    t = np.full(n_seeds, np.nan)
    U = np.log(th / (1 - th))
    alive = np.ones(n_seeds, bool)
    for step in range(1, max_steps + 1):
        p = 1 / (1 + np.exp(-u[alive]))
        n = rng.binomial(G, p)
        u[alive] += 2 * h * r * (n / G - p)
        hit = np.abs(u[alive]) >= U
        idx = np.flatnonzero(alive)[hit]
        t[idx] = step * h
        alive[idx] = False
        if not alive.any():
            break
    return t


report_medians = {4: 177, 8: 383, 16: 654, 32: 2051, 64: 3066}    # Exp.1 re-run with fix (RESULTS_exp2)
print("(b) first passage to max p >= 0.98, h=0.5, r=1 (time units):")
print("     G | formula mean | MC mean (SE)      | MC median | report median (80 seeds, censored)")
Gs, means = [], []
for G in [4, 8, 16, 32, 64]:
    pred = formula(G, 0.98)
    t = first_passage(G, 0.98, 2000, int(40 * pred / h))
    ok = np.isfinite(t)
    Gs.append(G)
    means.append(np.mean(t[ok]))
    print(f"    {G:2d} | {pred:12.0f} | {np.mean(t[ok]):8.0f} ({np.std(t[ok])/np.sqrt(ok.sum()):4.0f})  | "
          f"{np.median(t[ok]):9.0f} | {report_medians[G]}   (uncensored {ok.mean()*100:.1f}%)")
slope = np.polyfit(np.log(Gs), np.log(means), 1)[0]
print(f"    log-log slope of MC mean vs G = {slope:.3f}  (diffusion theory: exactly 1)")

# cross-check one G with the repository sampler
G = 16
zz = np.zeros((1000, 2))
tt = np.full(1000, np.nan)
U = np.log(0.98 / 0.02)
for step in range(1, 40000):
    g, _ = rhs_sampled(zz, np.array([r, r]), 0.0, G, rng)
    zz = zz + h * g
    u = zz[:, 0] - zz[:, 1]
    new = (np.abs(u) >= U) & np.isnan(tt)
    tt[new] = step * h
    if np.all(np.isfinite(tt)):
        break
print(f"    G=16 with src.dynamics.rhs_sampled (1000 seeds): mean {np.nanmean(tt):.0f}, median {np.nanmedian(tt):.0f}")

# ---------------------------------------------------------------- (c)
G, n_seeds, n_steps = 4, 400, 200000
u = np.zeros(n_seeds)
side = np.zeros(n_seeds, int)            # +1: p1 >= 0.95, -1: p1 <= 0.05, 0: not yet collapsed
switches = np.zeros(n_seeds, int)
U95 = np.log(0.95 / 0.05)
time_collapsed = np.zeros(n_seeds)
for step in range(n_steps):
    p = 1 / (1 + np.exp(-u))
    n = rng.binomial(G, p)
    u += 2 * h * r * (n / G - p)
    s_now = np.where(u >= U95, 1, np.where(u <= -U95, -1, 0))
    flip = (s_now != 0) & (side != 0) & (s_now != side)
    switches += flip
    side = np.where(s_now != 0, s_now, side)
    if step >= n_steps // 2:
        time_collapsed += (s_now != 0)
print(f"(c) G=4, h=0.5, {n_seeds} seeds, {n_steps} steps: seeds that switched winner at least once: "
      f"{np.mean(switches > 0)*100:.1f}%; mean switches per seed {switches.mean():.2f}; "
      f"time share with max p >= 0.95 in 2nd half: {time_collapsed.mean()/(n_steps//2)*100:.1f}%")
