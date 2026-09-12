"""
Bracketing root-finding (Chapra & Canale ch. 5), shared by any experiment
that needs a root or a stability boundary. Newton-Raphson and friends for
Experiment 3 belong here too.
"""

import numpy as np


def bisection(f, lo, hi, tol=1e-12, max_iter=200):
    """
    Root of a scalar function f on [lo, hi] by bisection; f(lo) and f(hi)
    must have opposite signs. Stops when the bracket half-width < tol.
    Returns (root, n_iterations).
    """
    flo, fhi = f(lo), f(hi)
    if flo == 0:
        return lo, 0
    if fhi == 0:
        return hi, 0
    if np.sign(flo) == np.sign(fhi):
        raise ValueError("bisection: f(lo) and f(hi) must have opposite signs")
    for it in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if fmid == 0 or 0.5 * (hi - lo) < tol:
            return mid, it
        if np.sign(fmid) == np.sign(flo):
            lo, flo = mid, fmid
        else:
            hi = mid
    return 0.5 * (lo + hi), max_iter


def bisection_boundary(pred, lo, hi, n_iter=40, geometric=False):
    """
    Vectorized bisection for the switch point of a monotone predicate, for
    many independent problems at once. pred(x) takes an array x (one entry
    per problem) and returns a boolean array that is True on the lo side and
    False on the hi side of each problem's boundary (e.g. "the explicit
    scheme is stable at step size x"). geometric=True bisects in log(x), for
    brackets spanning several decades.
    Returns the bracket midpoints after n_iter halvings.
    """
    lo = np.array(lo, dtype=np.float64, copy=True)
    hi = np.array(hi, dtype=np.float64, copy=True)
    for _ in range(n_iter):
        mid = np.sqrt(lo * hi) if geometric else 0.5 * (lo + hi)
        ok = np.asarray(pred(mid), dtype=bool)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return np.sqrt(lo * hi) if geometric else 0.5 * (lo + hi)
