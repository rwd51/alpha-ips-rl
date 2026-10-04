"""
Check 01 -- the alpha-flow (and the finite-G mean field) are exact gradient flows
of a potential that is strictly concave in p. This gives a global-convergence
argument that the report lacks (it only linearizes at p*).

Claims tested
  (a) rhs_alpha(z) == grad_z F_alpha(z),
        F_alpha(z) = sum_k r_k p_k^(1-alpha)/(1-alpha)   (alpha != 1)
        F_1(z)     = sum_k r_k ln p_k
  (b) jacobian_alpha(z) is symmetric at EVERY z (it is the Hessian of F_alpha),
      so its eigenvalues are real along the whole path, not only at p*.
  (c) the base paper's potential Psi = KL(p*||p) (alpha = 1) does NOT extend:
      dPsi/dt <= 0 holds for K = 2 (any alpha) and alpha = 1 (any K), but fails
      at some states for K >= 3, alpha != 1.  F_alpha is the right Lyapunov function.
  (d) global convergence from near-boundary starts (adaptive stiff integrator),
      with F_alpha non-decreasing along every trajectory.
  (e) finite-G mean field: m(z) == grad_z F_G(z) with
        F_G = (1/G) sum_i r_i E_{n_i~Bin(G,p_i)}[ sum_{k=1}^{n_i} omega(k) ],
      and convergence of the mean field from near-boundary starts to the KKT point.

Run:  python3 -u next_phase/case2_checks/c01_potential_lyapunov.py      (~1-2 min)
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import binom

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.dynamics import rhs_alpha, jacobian_alpha, softmax, stationary_p, linear_rates  # noqa: E402
from src.finite_group_general import effective_weight, meanfield_stationary  # noqa: E402

rng = np.random.default_rng(20261003)


def F_alpha_p(p, r, a):
    if abs(a - 1.0) < 1e-12:
        return float(np.sum(r * np.log(p)))
    return float(np.sum(r * p ** (1.0 - a)) / (1.0 - a))


def num_grad(F, z, h=1e-6):
    g = np.zeros_like(z)
    for i in range(len(z)):
        e = np.zeros_like(z)
        e[i] = h
        g[i] = (F(z + e) - F(z - e)) / (2 * h)
    return g


# ---------------------------------------------------------------- (a), (b)
worst_grad, worst_sym = 0.0, 0.0
for trial in range(300):
    K = int(rng.integers(2, 8))
    r = rng.uniform(0.2, 5.0, K)
    a = float(rng.choice([0.0, 0.3, 0.7, 1.0, 1.5, 2.5, 4.0]))
    z = rng.normal(0, 1.2, K)
    g_true = rhs_alpha(z, r, a)
    g_num = num_grad(lambda zz: F_alpha_p(softmax(zz), r, a), z)
    worst_grad = max(worst_grad, np.max(np.abs(g_true - g_num)) / (1 + np.max(np.abs(g_true))))
    J = jacobian_alpha(z, r, a)
    worst_sym = max(worst_sym, np.max(np.abs(J - J.T)) / (1 + np.max(np.abs(J))))
print("(a) max rel |rhs_alpha - grad_z F_alpha|      =", f"{worst_grad:.2e}", "(finite-difference level)")
print("(b) max rel asymmetry of jacobian_alpha(z)     =", f"{worst_sym:.2e}", "(J is a Hessian everywhere)")

# ---------------------------------------------------------------- (c)
print("(c) sign of dKL(p*||p)/dt at random states (20000 per cell):")
for K in [2, 3, 5]:
    row = []
    for a in [0.3, 0.7, 1.0, 1.5, 3.0]:
        N = 20000
        r = np.exp(rng.normal(0, 1.5, (N, K)))
        w = (r / r.max(axis=1, keepdims=True)) ** (1.0 / a)
        ps = w / w.sum(axis=1, keepdims=True)
        conc = rng.choice([0.2, 1.0, 5.0], size=(N, 1))
        p = rng.gamma(np.broadcast_to(conc, (N, K)))
        p = np.clip(p / p.sum(axis=1, keepdims=True), 1e-12, None)
        p /= p.sum(axis=1, keepdims=True)
        v = r * p ** (-a)
        S = np.sum(p * v, axis=1, keepdims=True)
        zdot = p * (v - S)
        dpsi = np.sum((p - ps) * zdot, axis=1)
        dF = np.sum(zdot ** 2, axis=1)                      # dF/dt = |grad F|^2 >= 0 always
        bad = np.sum(dpsi > 1e-12 * np.sum(r, axis=1))
        row.append(f"a={a}: {bad:4d} KL-increases")
        assert np.all(dF >= 0)
    print(f"    K={K}: " + " | ".join(row))

# ---------------------------------------------------------------- (d)
print("(d) global convergence from near-boundary starts (LSODA, rtol 1e-10), r=(5,4,3,2,1):")
r5 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
for a in [0.2, 0.5, 1.0, 2.0, 3.0]:
    ps = stationary_p(r5, a)
    lam_min = linear_rates(r5, a)[0]
    T = 30.0 / lam_min
    worst_final, F_dec, KL_inc = 0.0, 0, 0
    for s in range(4):
        p0 = rng.dirichlet(np.full(5, 0.3))
        p0 = np.clip(p0, 1e-9, None)
        p0 /= p0.sum()
        z0 = np.log(p0)
        f = lambda t, z: rhs_alpha(z, r5, a, p_floor=0.0)
        te = np.concatenate([[0.0], np.geomspace(1e-6, T, 3000)])
        sol = solve_ivp(f, (0, T), z0, method="LSODA", rtol=1e-10, atol=1e-12, t_eval=te)
        P = softmax(sol.y.T)
        Fv = np.array([F_alpha_p(pp, r5, a) for pp in P])
        psi = np.sum(ps * np.log(ps / P), axis=1)
        F_dec += int(np.sum(np.diff(Fv) < -1e-9 * (1 + np.abs(Fv[:-1]))))
        KL_inc += int(np.sum(np.diff(psi) > 1e-9 * (1 + np.abs(psi[:-1]))))
        worst_final = max(worst_final, np.abs(P[-1] - ps).sum())
    print(f"    alpha={a:3.1f}: T=30/lambda_min={T:9.1f}: max l1(p(T),p*) = {worst_final:.1e};"
          f" F decreases: {F_dec}; KL increases: {KL_inc}")


# ---------------------------------------------------------------- (e)
def Phi_G(p, G, alpha, eps):
    """Phi_G(p) = int_0^p w_G = (1/G) E_{n~Bin(G,p)}[sum_{k<=n} omega(k)]."""
    k = np.arange(1, G + 1)
    om = np.maximum(k / G, eps) ** (-alpha)
    cum = np.concatenate([[0.0], np.cumsum(om)])          # cum[n] = sum_{k<=n} omega(k)
    n = np.arange(G + 1)
    return binom.pmf(n[None, :], G, np.atleast_1d(p)[:, None]) @ cum / G


def mean_drift_G(z, r, G, alpha, eps):
    p = softmax(z)
    w = effective_weight(p, G, alpha, eps)
    S = float(np.sum(p * r * w))
    return p * (r * w - S)


worst = 0.0
for trial in range(150):
    K = int(rng.integers(2, 6))
    r = rng.uniform(0.3, 4, K)
    G = int(rng.choice([2, 3, 4, 8, 16]))
    alpha = float(rng.choice([0.5, 1.0, 2.0]))
    eps = float(rng.choice([1e-3, 0.2]))
    z = rng.normal(0, 1, K)
    FG = lambda zz: float(np.sum(r * Phi_G(softmax(zz), G, alpha, eps)))
    g_num = num_grad(FG, z)
    g = mean_drift_G(z, r, G, alpha, eps)
    worst = max(worst, np.max(np.abs(g - g_num)) / (1 + np.max(np.abs(g))))
print(f"(e) max rel |finite-G mean drift - grad_z F_G| = {worst:.2e}")

print("    finite-G mean field from near-boundary starts, r=(5,4,3,2,1), eps=1e-3:")
for G, alpha in [(4, 1.0), (16, 1.0), (8, 0.5), (16, 2.0)]:
    pi, S, _ = meanfield_stationary(r5, G, alpha, 1e-3, method="newton")
    worst_final, F_dec = 0.0, 0
    for s in range(3):
        p0 = rng.dirichlet(np.full(5, 0.3))
        p0 = np.clip(p0, 1e-9, None)
        p0 /= p0.sum()
        f = lambda t, z: mean_drift_G(z, r5, G, alpha, 1e-3)
        T = 4000.0
        te = np.concatenate([[0.0], np.geomspace(1e-4, T, 600)])
        sol = solve_ivp(f, (0, T), np.log(p0), method="LSODA", rtol=1e-9, atol=1e-11, t_eval=te)
        P = softmax(sol.y.T)
        Fv = np.array([float(np.sum(r5 * Phi_G(pp, G, alpha, 1e-3))) for pp in P])
        F_dec += int(np.sum(np.diff(Fv) < -1e-9 * (1 + np.abs(Fv[:-1]))))
        worst_final = max(worst_final, np.abs(P[-1] - pi).sum())
    print(f"      G={G:2d}, alpha={alpha}: support {int(np.sum(pi > 0))}/5, "
          f"max l1(p(T), pi_KKT) = {worst_final:.1e}, F_G decreases: {F_dec}")
