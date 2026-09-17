"""
One-dimensional minimization (Chapra & Canale ch. 13), added for Experiment 5.

Experiment 5 repeatedly has to answer "which value of this knob minimizes the
total error?" -- the clip eps, the smoothing lam, the moving-average beta. Those
objectives are continuous but only piecewise smooth (the clipped estimator has
kinks at eps = n/G) and have no closed form, so a derivative-free bracketing
minimizer is the right tool:

  golden_section       the course's golden-section search
  golden_section_log   the same search in log x, for knobs spanning decades
  grid_minimum         a dense scan, used only to audit the two above

The search is only valid on a unimodal bracket, so every experiment call is
cross-checked against `grid_minimum` on the same interval and the disagreement
is reported rather than assumed away.
"""

from __future__ import annotations

import numpy as np


GOLDEN_INV = (np.sqrt(5.0) - 1.0) / 2.0        # 1/phi = 0.6180339887...


def golden_section(f, lo, hi, tol=1e-10, max_iter=200):
    """
    Minimize a unimodal f on [lo, hi] by golden-section search.

    Each iteration keeps two interior points at the golden ratio and discards
    the sub-interval that cannot contain the minimum, so the bracket shrinks by
    a constant factor 1/phi per function evaluation (one new evaluation per
    iteration after the first two).

    Returns (x_min, f_min, n_iterations, n_evaluations).
    """
    lo, hi = float(lo), float(hi)
    if not (hi > lo):
        raise ValueError("golden_section: need hi > lo")
    d = GOLDEN_INV * (hi - lo)
    x1, x2 = hi - d, lo + d
    f1, f2 = f(x1), f(x2)
    n_eval = 2
    for it in range(1, max_iter + 1):
        if hi - lo < tol:
            break
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - GOLDEN_INV * (hi - lo)
            f1 = f(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + GOLDEN_INV * (hi - lo)
            f2 = f(x2)
        n_eval += 1
    if f1 < f2:
        return x1, float(f1), it, n_eval
    return x2, float(f2), it, n_eval


def golden_section_log(f, lo, hi, rel_tol=1e-10, max_iter=300):
    """Golden-section search in log x, for positive knobs spanning decades.
    Returns (x_min, f_min, n_iterations, n_evaluations)."""
    if lo <= 0.0 or hi <= lo:
        raise ValueError("golden_section_log: need 0 < lo < hi")
    x, fx, it, nev = golden_section(lambda t: f(np.exp(t)), np.log(lo), np.log(hi),
                                    tol=rel_tol, max_iter=max_iter)
    return float(np.exp(x)), fx, it, nev


def grid_minimum(f, lo, hi, n=2001, geometric=False):
    """Dense scan, used to audit the golden-section result (and to detect a
    multimodal objective, where the search would not be justified).
    Returns (x_min, f_min, values, grid)."""
    grid = np.geomspace(lo, hi, n) if geometric else np.linspace(lo, hi, n)
    values = np.array([f(x) for x in grid], dtype=np.float64)
    j = int(np.argmin(values))
    return float(grid[j]), float(values[j]), values, grid


def is_unimodal(values, slack=0):
    """True when `values` decreases then increases (allowing `slack` local
    violations, e.g. from round-off on a flat stretch)."""
    v = np.asarray(values, dtype=np.float64)
    j = int(np.argmin(v))
    left_bad = int(np.count_nonzero(np.diff(v[:j + 1]) > 0))
    right_bad = int(np.count_nonzero(np.diff(v[j:]) < 0))
    return left_bad + right_bad <= slack
