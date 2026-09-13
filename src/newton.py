"""
Open and Newton-type root-finding (Chapra & Canale ch. 6), added for
Experiment 3. The bracketing methods (bisection, bisection_boundary) stay in
src/rootfinding.py.

  - newton_raphson         open method, quadratic convergence near a simple root
  - secant                 open method, derivative-free, order (1+sqrt 5)/2
  - newton_bracketed       Newton safeguarded by a bracket (falls back to a
                           bisection step whenever Newton would leave it)
  - newton_system          multivariate Newton-Raphson (optionally damped by
                           backtracking on the residual norm)
  - traced                 wraps f so the points a solver evaluates are recorded,
                           e.g. to plot bisection's iterates

Every scalar routine accepts an optional `trace` list; when given, each new
iterate is appended to it, so convergence histories can be plotted. The open
methods return (root, n_iterations, converged) instead of raising, because
divergence is itself a result that Experiment 3 measures.
"""

import numpy as np


def traced(f, trace):
    """
    Return a wrapper of f that appends every argument it is called with to
    `trace`. rootfinding.bisection evaluates f(lo), f(hi) and then one
    midpoint per iteration, so trace[2:] is its sequence of midpoints.
    """
    def wrapper(x):
        trace.append(x)
        return f(x)
    return wrapper


def newton_raphson(f, df, x0, tol=1e-12, max_iter=100, x_max=700.0, trace=None):
    """
    Scalar Newton-Raphson x <- x - f(x)/f'(x), stopped when |step| < tol
    (Chapra's approximate-error criterion, so the last iteration is the one
    that confirms convergence). Declared diverged when the iterate or the
    derivative stops being finite, the derivative is exactly zero, or
    |x| > x_max (for logits, beyond x_max = 700 exp() overflows anyway).
    Returns (root, n_iterations, converged).
    """
    x = float(x0)
    with np.errstate(all="ignore"):
        for it in range(1, max_iter + 1):
            fx, dfx = f(x), df(x)
            if fx == 0:
                return x, it, True
            if dfx == 0 or not (np.isfinite(fx) and np.isfinite(dfx)):
                return x, it, False
            step = fx / dfx
            x = x - step
            if trace is not None:
                trace.append(x)
            if not np.isfinite(x) or abs(x) > x_max:
                return x, it, False
            if abs(step) < tol:
                return x, it, True
    return x, max_iter, False


def secant(f, x0, x1, tol=1e-12, max_iter=100, x_max=700.0, trace=None):
    """
    Secant method: Newton with f' replaced by the slope through the last two
    iterates. Same stopping rule and return convention as newton_raphson.
    """
    x_prev, x = float(x0), float(x1)
    with np.errstate(all="ignore"):
        f_prev, fx = f(x_prev), f(x)
        for it in range(1, max_iter + 1):
            if fx == 0:
                return x, it, True
            denom = fx - f_prev
            if denom == 0 or not np.isfinite(denom):
                return x, it, False
            step = fx * (x - x_prev) / denom
            x_prev, f_prev = x, fx
            x = x - step
            if trace is not None:
                trace.append(x)
            if not np.isfinite(x) or abs(x) > x_max:
                return x, it, False
            if abs(step) < tol:
                return x, it, True
            fx = f(x)
    return x, max_iter, False


def newton_bracketed(f, df, lo, hi, tol=1e-12, max_iter=200, trace=None):
    """
    Safeguarded Newton ("rtsafe"): keep a sign-change bracket [lo, hi] and take
    the Newton step only when it lands strictly inside the bracket; otherwise
    take a bisection step. Converges whenever bisection would, and
    quadratically once Newton's steps are accepted.
    Returns (root, n_iterations).
    """
    flo, fhi = f(lo), f(hi)
    if flo == 0:
        return lo, 0
    if fhi == 0:
        return hi, 0
    if np.sign(flo) == np.sign(fhi):
        raise ValueError("newton_bracketed: f(lo) and f(hi) must have opposite signs")
    x = 0.5 * (lo + hi)
    for it in range(1, max_iter + 1):
        fx, dfx = f(x), df(x)
        if fx == 0:
            return x, it
        if np.sign(fx) == np.sign(flo):
            lo, flo = x, fx
        else:
            hi = x
        x_new = x - fx / dfx if dfx != 0 else np.nan
        if not (lo < x_new < hi):
            x_new = 0.5 * (lo + hi)
        step = x_new - x
        x = x_new
        if trace is not None:
            trace.append(x)
        if abs(step) < tol:
            return x, it
    return x, max_iter


def newton_system(F, J, x0, tol=1e-12, max_iter=100, damped=False, x_max=700.0, trace=None):
    """
    Multivariate Newton-Raphson: solve J(x) dx = -F(x), x <- x + lam dx.
    With damped=True, lam is halved until the residual norm decreases
    (Armijo backtracking, sufficient-decrease constant 1e-4). Stops when
    ||lam dx|| < tol (1 + ||x||). The Jacobian must be nonsingular near the
    root -- fix any continuous symmetry (e.g. a logit gauge) before calling.
    Returns (root, n_iterations, converged).
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    with np.errstate(all="ignore"):
        for it in range(1, max_iter + 1):
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return x, it, False
            try:
                dx = np.linalg.solve(J(x), -Fx)
            except np.linalg.LinAlgError:
                return x, it, False
            lam = 1.0
            if damped:
                norm0 = np.linalg.norm(Fx)
                while lam > 1e-10 and not np.linalg.norm(F(x + lam * dx)) <= (1.0 - 1e-4 * lam) * norm0:
                    lam *= 0.5
            x = x + lam * dx
            if trace is not None:
                trace.append(x.copy())
            if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > x_max:
                return x, it, False
            if np.linalg.norm(lam * dx) < tol * (1.0 + np.linalg.norm(x)):
                return x, it, True
    return x, max_iter, False
