"""
Check 03 -- is "collapse at equal rewards comes only from sampling noise" right?

The report tests only FULL ties (all K rewards equal), where the alpha=0 drift is
identically zero.  The base paper's sentence is about two outcomes with equal
reward (equal, not necessarily zero, advantage).  If any worse outcome exists,
the common advantage a = r_top - rbar > 0 and

      d/dt log(p1/p2) = (p1 - p2) * a ,

so the deterministic flow amplifies any initial asymmetry between the tied
outcomes.  Here: alpha = 0, K = 3, r = (1, 1, r3), RK45/LSODA to t = 1e7.

Prediction (r3 = 0): once p1 ~ 1, p3 ~ 1/(2t) and d log(p1/p2)/dt ~ 1/(2t),
so p2/p1 ~ t^(-1/2): deterministic but algebraic collapse inside the tie.

Run:  python3 -u next_phase/case2_checks/c03_partial_tie.py      (seconds)
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.dynamics import rhs_alpha, softmax  # noqa: E402

T = 1e7
te = np.geomspace(1e-2, T, 400)
for r3, p0 in [(1.0, (0.36, 0.34, 0.30)),      # full tie: frozen
               (0.0, (0.36, 0.34, 0.30)),      # tied top pair + a worse outcome
               (0.5, (0.36, 0.34, 0.30)),
               (0.9, (0.36, 0.34, 0.30)),
               (0.0, (0.35, 0.35, 0.30))]:     # exactly symmetric pair: stays symmetric
    r = np.array([1.0, 1.0, r3])
    z0 = np.log(np.array(p0))
    sol = solve_ivp(lambda t, z: rhs_alpha(z, r, 0.0), (0, T), z0, method="LSODA",
                    rtol=1e-11, atol=1e-13, t_eval=te)
    P = softmax(sol.y.T)
    ratio = P[:, 1] / P[:, 0]
    # tail slope of log(p2/p1) vs log t over the last two decades
    m = te > T / 100
    slope = np.polyfit(np.log(te[m]), np.log(ratio[m]), 1)[0] if r3 < 1 else 0.0
    idx = [np.searchsorted(te, x) for x in (1e1, 1e3, 1e5, 1e7 - 1)]
    vals = ", ".join(f"t=1e{int(round(np.log10(te[i])))}: p2/p1={ratio[i]:.3e}" for i in idx)
    print(f"r=(1,1,{r3}), p0={p0}: {vals};  tail slope d log(p2/p1)/d log t = {slope:+.3f}")
