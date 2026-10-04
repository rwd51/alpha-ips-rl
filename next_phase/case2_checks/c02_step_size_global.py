"""
Check 02 -- (S3) "an explicit update is stable iff h*lambda_max < 2 (Euler) /
< 2.7853 (RK4)" is a LOCAL statement at p*.  For alpha > 1 the local stiffness
lambda_max(-J(z)) is unbounded near the simplex boundary, so no fixed h is
globally safe.

  (a) RK4 real-axis constant: root of x^3 - 4x^2 + 12x - 24 = 0, and |R(-x)| <= 1 on [0, x*].
  (b) r=(4,1), alpha=2 (lambda* = 8).  Euler with h = 0.05 (h*lambda* = 0.4, "stable")
      and h = 0.2 (h*lambda* = 1.6) from starts p2(0) in {0.5, 1e-1, 1e-2, 1e-3, 1e-4}.
  (c) largest local rate lambda_max(-J(z)) along the exact path from each start.

Run:  python3 -u next_phase/case2_checks/c02_step_size_global.py      (seconds)
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.dynamics import rhs_alpha, jacobian_alpha, softmax, stationary_p, linear_rates  # noqa: E402

# ---------------------------------------------------------------- (a)
roots = np.roots([1, -4, 12, -24])
xstar = float(np.real(roots[np.argmin(np.abs(np.imag(roots)))]))
x = np.linspace(0, xstar, 100001)
R = 1 - x + x ** 2 / 2 - x ** 3 / 6 + x ** 4 / 24
print(f"(a) RK4 real-axis limit x* = {xstar:.10f};  max |R(-x)| on [0,x*] = {np.max(np.abs(R)):.12f};"
      f" min R(-x) = {R.min():.4f}")

# ---------------------------------------------------------------- (b), (c)
r = np.array([4.0, 1.0])
a = 2.0
ps = stationary_p(r, a)
lam = linear_rates(r, a)[-1]
print(f"(b) r=(4,1), alpha=2: p* = {ps.round(4)}, lambda* = {lam:.4f}")


def euler_run(z0, h, n=4000, p_floor=0.0):
    z = z0.copy()
    with np.errstate(all="ignore"):
        for k in range(n):
            g = rhs_alpha(z, r, a, p_floor=p_floor)
            if not np.all(np.isfinite(g)):
                return np.nan, k
            z = z + h * g
            z = z - z.max()
    p = softmax(z)
    return float(np.abs(p - ps).sum()), n


for p20 in [0.5, 1e-1, 1e-2, 1e-3, 1e-4]:
    z0 = np.log(np.array([1 - p20, p20]))
    # exact path and its local stiffness
    sol = solve_ivp(lambda t, z: rhs_alpha(z, r, a, p_floor=0.0), (0, 5.0), z0, method="LSODA",
                    rtol=1e-10, atol=1e-12, t_eval=np.linspace(0, 5.0, 2001))
    lam_path = max(np.max(np.linalg.eigvalsh(-jacobian_alpha(zz, r, a, p_floor=0.0))) for zz in sol.y.T)
    out = []
    for h in [0.05, 0.2]:
        err, steps = euler_run(z0, h)
        err_f, _ = euler_run(z0, h, p_floor=1e-12)          # the floor used by the experiments
        tag = f"overflow at step {steps}" if np.isnan(err) else f"l1 err {err:.0e}"
        out.append(f"Euler h={h}: {tag} (with p_floor=1e-12: l1 err after 4000 steps {err_f:.2f})")
    print(f"    p2(0)={p20:7.0e}: sup_path lambda_max(-J) = {lam_path:10.1f}  ->  2/sup = {2/lam_path:.2e};  "
          + "; ".join(out))
