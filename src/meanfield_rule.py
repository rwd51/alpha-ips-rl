"""
The finite-group mean-field stationary point for an ARBITRARY weight rule
(Experiment 5). `src/finite_group.py` (K=2) and `src/finite_group_general.py`
(any K) solve the same problem for the one rule the base paper uses,
max(p_hat, eps)^-alpha; this module takes the rule as an argument so that
Laplace smoothing, Richardson extrapolation and the ideal weight can be pushed
through the identical theory.

Everything rests on derivation_exp5.md Eq. (5.5)/(5.7): whatever the rule, the
mean drift is m_i = p_i (r_i w_G(p_i) - S) with S = sum_k p_k r_k w_G(p_k) and
w_G = rule.effective_weight. So the stationary conditions are unchanged,

    r_i w_G(p_i) = S      (interior),      r_i w_G(0) <= S      (extinct),

and are solved by the same nested scalar root-finding as Experiment 3: an inner
solve inverts the monotone w_G for each outcome at a trial S, an outer solve
picks S so the probabilities sum to one. Both levels are bracketed, so
bisection is guaranteed; the inner level is vectorized over outcomes with
`rootfinding.bisection_boundary`.

Monotonicity of w_G is an assumption of the inner solve. It is automatic for
the clipped and add-lambda rules (both decrease in n), but NOT structurally
guaranteed for the Richardson rules, so `check_monotone` is provided and the
experiment calls it before trusting a solve.
"""

from __future__ import annotations

import numpy as np

from .rootfinding import bisection, bisection_boundary


def check_monotone(rule, G, n=801, tol=1e-12):
    """Verify numerically that w_G is non-increasing on [0, 1].
    Returns (ok, largest_increase)."""
    p = np.linspace(0.0, 1.0, n)
    w = rule.effective_weight(p, G)
    rise = float(np.max(np.diff(w)))
    return bool(rise <= tol), rise


def stationary_K2(r, rule, G, tol=1e-15):
    """
    K=2 mean-field stationary probability of the higher-reward outcome.

    The interior condition is r_1 w_G(p) = r_2 w_G(1-p). The left-minus-right
    side is positive at p = 1/2 and equals r_1 w_G(1) - r_2 w_G(0) at p = 1, so
    an interior root exists exactly when w_G(0)/w_G(1) > r_1/r_2
    (derivation_exp5.md, Eq. 5.8). Returns 1.0 when it does not: the minority
    outcome is extinct.
    """
    r = np.asarray(r, dtype=np.float64)
    if r.shape != (2,) or r[0] < r[1] or np.any(r <= 0):
        raise ValueError("stationary_K2 expects r = (r1, r2) with r1 >= r2 > 0")

    def g(p):
        w = rule.effective_weight(np.array([p, 1.0 - p]), G)
        return r[0] * w[0] - r[1] * w[1]

    if g(1.0 - 1e-15) >= 0.0:
        return 1.0
    root, _ = bisection(g, 0.5, 1.0 - 1e-15, tol=tol)
    return float(root)


def stationary(r, rule, G, outer_tol=1e-13, inner_iter=60, max_outer=200):
    """
    Mean-field stationary distribution for any K and any rule.

    Returns (p, S, stats). `stats` records the iteration counts of both levels
    and the worst residual max_i |r_i w_G(p_i) - S| / S over the support, which
    the caller should check.
    """
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 1 or np.any(r <= 0.0):
        raise ValueError("r must be a positive one-dimensional reward vector")
    r_max = float(r.max())
    rn = r / r_max                                     # scale-free: p* is invariant
    ceil = float(rule.effective_weight(np.array([0.0]), G)[0])
    w_one = float(rule.effective_weight(np.array([1.0]), G)[0])
    if not np.isfinite(ceil) or ceil <= 0.0:
        raise ValueError("this rule has a non-positive weight ceiling w_G(0); "
                         "the mean field has no interior stationary point")

    def p_of_S(S):
        """Invert w_G(p_i) = S / r_i for every outcome at once."""
        v = S / rn
        p = np.zeros_like(rn)
        interior = (v < ceil) & (v > w_one)
        p[v <= w_one] = 1.0
        if np.any(interior):
            target = v[interior]

            def pred(x):
                full = np.zeros_like(rn)
                full[interior] = x
                w = rule.effective_weight(full, G)[interior]
                return w > target                      # True on the low-p side
            p[interior] = bisection_boundary(pred, np.zeros(interior.sum()),
                                             np.ones(interior.sum()), n_iter=inner_iter)
        return p

    def T(S):
        return float(p_of_S(S).sum() - 1.0)

    lo, hi = w_one, ceil                               # S/r_max in [w(1), w(0)]
    if T(hi * (1.0 - 1e-15)) > 0.0:
        raise ValueError("no stationary point inside the bracket")
    S, outer = bisection(T, lo, hi * (1.0 - 1e-15), tol=outer_tol, max_iter=max_outer)
    p = p_of_S(S)
    p = p / p.sum()                                    # normalize away bisection slack
    support = p > 0.0
    w = rule.effective_weight(p, G)
    residual = float(np.max(np.abs(rn[support] * w[support] - S)) / S) if support.any() else np.inf
    extinct_ok = bool(np.all(rn[~support] * ceil <= S * (1.0 + 1e-9))) if (~support).any() else True
    stats = {"outer_iter": int(outer), "inner_iter": int(inner_iter),
             "scaled_residual": residual, "extinct_condition_ok": extinct_ok,
             "ceiling": ceil, "w_one": w_one, "S_scaled": float(S)}
    return p, float(S * r_max), stats


def extinction_alpha_K2(rule_factory, r, G, alpha_lo=1e-3, alpha_hi=40.0, tol=1e-10):
    """
    The exponent below which the K=2 minority outcome dies, for a family of
    rules parameterized by alpha. `rule_factory(alpha)` must return the rule at
    that alpha. Bisection on h(alpha) = ln(w_G(0)/w_G(1)) - ln(r1/r2), whose
    sign is exactly the survival criterion (Eq. 5.8).
    Returns (alpha_c, n_iterations), or (inf, 0) if the minority never survives.
    """
    ratio = float(r[0] / r[1])

    def h(alpha):
        return float(np.log(rule_factory(alpha).dynamic_range(G)) - np.log(ratio))

    if h(alpha_hi) <= 0.0:
        return np.inf, 0
    if h(alpha_lo) > 0.0:
        return alpha_lo, 0
    return bisection(h, alpha_lo, alpha_hi, tol=tol)
