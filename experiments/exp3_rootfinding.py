"""
EXPERIMENT 3 -- Root-finding for the stationary point: bisection vs
Newton-Raphson, and where the stationary point has no closed form.

Exp. 2 reached p* ~ r^(1/alpha) by integrating the ODE for many e-folding
times. Here we solve z_dot = 0 directly (Methodology step 4), compare the
course's root-finding methods on it, check them against the analytical p*,
and then use them on the finite-group mean field, whose stationary point has
no closed form for K > 2.

The stationarity condition is written in three equivalent forms
(src/stationarity.py, derivation_exp3.md "Root-finding formulations"):
    drift     z_dot_i itself            = p_i (r_i p_i^-alpha - S)
    balance   drift / p_i, differenced  = r_i p_i^-alpha - r_K p_K^-alpha
    log       log of the balance terms  = ln(r_i/r_K) - alpha (z_i - z_K)
They share the root but not the global shape, and Newton notices.

  1. FIG 1 -- K=2, one unknown u = z_1 - z_2, root u* = ln(r_1/r_2)/alpha:
     (a) Convergence histories at alpha = 1: bisection is linear (factor 1/2),
         secant superlinear (~1.62), Newton quadratic; empirical orders by
         least squares on ln e_{n+1} vs ln e_n.
     (b) Newton's basin of attraction on the drift, over (alpha, u0): bounded
         for alpha <= 1 (the drift is bounded, so far away it is flat), global
         for alpha above ~1.1. The balance and log forms converge from every start.
     (c) Far from the root Newton is only linear: in the exponential tails each
         step moves u by 1/alpha (balance) or 1/(alpha-1) (drift), so the
         iteration count grows like alpha |u0 - u*|. Bisection pays a fixed
         log2(width/tol); the log form needs a single step.

  2. FIG 2 -- K=5, r=(5,4,3,2,1), multivariate Newton with the gauge z_5 = 0:
     (a) Quadratic convergence from the uniform policy for every form at
         alpha = 1; at alpha = 0.5 the drift form runs off to the boundary.
     (b) Success rate from 200 random starts vs alpha: plain drift Newton
         fails for alpha < 1, and backtracking does not rescue it -- for
         alpha < 1 the drift also vanishes on the simplex boundary, so the
         residual-norm line search is attracted there. Balance/log: 100%.
     (c) Conditioning: without the gauge the Jacobian is singular (cond ~ 1e16);
         with it, the drift Jacobian's condition number grows as alpha -> 0,
         like Exp. 2's stiffness ratio, while the balance and log forms are
         perfectly conditioned (cond = 1) at the root.

  3. FIG 3 -- Finite group size G, K=5 (no closed form):
     (a) The effective weight w_G(p) = E[max((1+B)/G, eps)^-alpha],
         B ~ Binomial(G-1, p), saturates at G^alpha; it equals Exp. 2's binomial
         expectation phi_G(p)/p to machine precision.
     (b) Stationary distribution vs G by nested root-finding (inner: invert w_G
         per outcome; outer: normalize), bisection and safeguarded Newton
         agreeing, confirmed by the Monte-Carlo agent. Newton needs 8-20x less
         work (evaluations of w_G and w_G'). Small groups drive the weak outcomes
         extinct.
     (c) Extinction thresholds alpha_c(G) for each outcome by bisection on
         alpha: only the runner-up follows Exp. 2's two-outcome formula
         ln(r_1/r_j)/ln G; weaker outcomes need a larger alpha, and below a
         minimum G no alpha keeps them at all.

Outputs: results/exp3_rootfinding/figures/fig*.png (300 dpi) and
results/exp3_rootfinding/data/*.csv, exp3_summary.json.
"""

from __future__ import annotations
import sys
import os
import csv
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dynamics import softmax, rhs_sampled, stationary_p, jacobian_alpha, linear_rates
from src.stationarity import FORMS_K2, FORMS, root_K2, drift_K2, gauge_root
from src.rootfinding import bisection
from src.newton import newton_raphson, secant, newton_system, traced
from src.fitting import least_squares_line
from src.finite_group import expected_weight, critical_alpha
from src.finite_group_general import (effective_weight, meanfield_stationary, extinction_alpha,
                                      kept_at_large_alpha, meanfield_log_residual,
                                      meanfield_log_jacobian)
from src.plotstyle import apply_style, savefig, panel_label, PALETTE

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP_NAME = "exp3_rootfinding"
FIG_DIR = os.path.join(ROOT, "results", EXP_NAME, "figures")
DATA_DIR = os.path.join(ROOT, "results", EXP_NAME, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

apply_style()
SUMMARY = {}

R41 = np.array([4.0, 1.0])                  # the proposal's running example
R5 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
INK = "0.25"                                 # annotation text colour
ERR_FLOOR = 1e-17                            # exact zeros drawn here on log axes
FORM_STYLE = {"drift": (PALETTE[1], "o"), "balance": (PALETTE[0], "s"), "log": (PALETTE[2], "^")}


def _write_csv(name, header, rows):
    with open(os.path.join(DATA_DIR, name), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        w.writerows(rows)


def _alpha_axis(ax, ticks):
    """Log-scale alpha axis with plain-number tick labels."""
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def _empirical_order(err, lo=1e-14, hi=1e-1):
    """Order q of e_{n+1} ~ C e_n^q by least squares on (ln e_n, ln e_{n+1}),
    using consecutive pairs inside [lo, hi] (past the start, above round-off).
    Returns (q, r2, n_pairs); q is nan with fewer than 2 pairs."""
    e = np.asarray(err, dtype=np.float64)
    sel = (e[:-1] > lo) & (e[:-1] < hi) & (e[1:] > lo) & (e[1:] < hi)
    if sel.sum() < 2:
        return np.nan, np.nan, int(sel.sum())
    q, _, r2 = least_squares_line(np.log(e[:-1][sel]), np.log(e[1:][sel]))
    return q, r2, int(sel.sum())


# --------------------------------------------------------------------------
# FIGURE 1 -- K = 2: bisection vs secant vs Newton, three formulations
# --------------------------------------------------------------------------

def fig1_scalar_rootfinding():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    r = R41

    # --- panel (a): convergence histories at alpha = 1 ---------------------
    a = 1.0
    u_star = root_K2(r, a)
    g = lambda u: drift_K2(u, r, a)
    runs = {}
    calls = []
    _, n_bis = bisection(traced(g, calls), -20.0, 20.0, tol=1e-15)
    runs["bisection (drift)"] = calls[2:]          # drop f(lo), f(hi): the rest are the midpoints
    tr = []
    secant(g, 0.0, 0.5, tol=1e-15, trace=tr)
    runs["secant (drift)"] = [0.5] + tr
    for form in ["drift", "balance", "log"]:
        f, df = FORMS_K2[form]
        tr = []
        newton_raphson(lambda u: f(u, r, a), lambda u: df(u, r, a), 0.0, tol=1e-15, trace=tr)
        runs[f"Newton ({form})"] = tr
    styles = {"bisection (drift)": ("0.45", "."), "secant (drift)": (PALETTE[3], "D"),
              "Newton (drift)": FORM_STYLE["drift"], "Newton (balance)": FORM_STYLE["balance"],
              "Newton (log)": FORM_STYLE["log"]}
    rows_a = []
    for name, trace in runs.items():
        # iteration 0 is the start u = 0 (bisection's first midpoint on [-20, 20] is that start)
        start = [] if name.startswith("bisection") else [0.0]
        err = np.abs(np.asarray(start + list(trace)) - u_star)
        color, marker = styles[name]
        if name.startswith("bisection"):
            slope, _, r2 = least_squares_line(np.arange(len(err))[err > 1e-14], np.log2(err[err > 1e-14]))
            label = f"{name}: $e_{{n+1}}/e_n\\approx${2 ** slope:.2f}"
            SUMMARY["fig1a_bisection_rate_per_iter"] = float(2 ** slope)
            SUMMARY["fig1a_bisection_iters"] = int(n_bis)
        else:
            q, r2, npairs = _empirical_order(err)
            label = f"{name}: order {q:.2f}" if np.isfinite(q) else f"{name}: exact in 1 step"
            SUMMARY.setdefault("fig1a_empirical_order", {})[name] = None if not np.isfinite(q) else float(q)
        # drift and balance Newton nearly coincide: draw drift with large hollow markers underneath
        hollow = name == "Newton (drift)"
        axes[0].semilogy(np.arange(len(err)), np.maximum(err, ERR_FLOOR), "-", marker=marker,
                         ms=6.0 if hollow else 3.2, mfc="white" if hollow else color, mew=1.0,
                         lw=1.6 if hollow else 0.9, color=color, label=label, zorder=1 if hollow else 2)
        SUMMARY.setdefault("fig1a_iterations", {})[name] = len(trace)
        for n, e in enumerate(err):
            rows_a.append([name, n, e])
    axes[0].axhspan(ERR_FLOOR / 3, 1e-15, color="0.5", alpha=0.12, lw=0)
    axes[0].text(0.97, 0.06, "machine precision", transform=axes[0].transAxes,
                 fontsize=6.3, color="0.4", ha="right")
    axes[0].set_xlim(-0.5, 25)
    axes[0].set_ylim(ERR_FLOOR / 3, 1e2)
    axes[0].set_xlabel("iteration $n$")
    axes[0].set_ylabel("$|u_n - u^*|$ (log scale)")
    axes[0].set_title("Convergence at $\\alpha$=1, $r$=(4, 1)")
    axes[0].legend(loc="upper right", fontsize=5.8)
    panel_label(axes[0], "a")
    _write_csv("fig1a_convergence_histories.csv", ["method", "iteration", "abs_error"], rows_a)

    # --- panel (b): Newton basin of attraction on the drift ----------------
    alphas = np.geomspace(0.2, 4.0, 90)
    u0s = np.linspace(-15.0, 15.0, 241)
    f, df = FORMS_K2["drift"]
    iters = np.full((len(u0s), len(alphas)), np.nan)
    frac_conv = {"drift": [], "balance": []}
    for ai, a in enumerate(alphas):
        u_star = root_K2(r, a)
        for form in ["drift", "balance"]:
            f, df = FORMS_K2[form]
            ok_count = 0
            for ui, u0 in enumerate(u0s):
                x, n, ok = newton_raphson(lambda u: f(u, r, a), lambda u: df(u, r, a), u0)
                good = ok and abs(x - u_star) < 1e-8
                ok_count += good
                if form == "drift" and good:
                    iters[ui, ai] = n
            frac_conv[form].append(ok_count / len(u0s))
    a_edges = np.geomspace(alphas[0] / np.sqrt(alphas[1] / alphas[0]),
                           alphas[-1] * np.sqrt(alphas[1] / alphas[0]), len(alphas) + 1)
    du = u0s[1] - u0s[0]
    u_edges = np.concatenate([u0s - du / 2, [u0s[-1] + du / 2]])
    cmap = plt.get_cmap("cividis_r").copy()
    cmap.set_bad("white")
    axes[1].grid(False)
    im = axes[1].pcolormesh(a_edges, u_edges, np.ma.masked_invalid(iters), cmap=cmap,
                            vmin=1, vmax=30, shading="flat", rasterized=True)
    a_line = np.geomspace(0.25, 4.0, 100)
    axes[1].plot(a_line, np.log(4.0) / a_line, color=PALETTE[1], lw=1.4, label="root $u^*=\\ln 4/\\alpha$")
    axes[1].axvline(1.0, color="0.3", lw=0.8, ls="--")
    for a_txt in [0.5, 1.0, 2.0]:
        i = np.argmin(np.abs(alphas - a_txt))
        SUMMARY.setdefault("fig1b_drift_newton_frac_converged", {})[f"{a_txt:g}"] = float(frac_conv["drift"][i])
    SUMMARY["fig1b_balance_newton_min_frac_converged"] = float(np.min(frac_conv["balance"]))
    # smallest alpha from which drift Newton converges from every start in the grid
    full = np.array(frac_conv["drift"]) == 1.0
    a_global = float(alphas[np.argmax(full)]) if full.any() else None
    SUMMARY["fig1b_drift_newton_global_from_alpha"] = a_global
    axes[1].text(0.22, -13.8, "white: Newton on the drift\ndiverges or stalls", fontsize=6.0, color=INK)
    axes[1].text(0.97, 0.97, f"balance, log forms:\nconverge from all starts",
                 transform=axes[1].transAxes, fontsize=6.0, color=INK, ha="right", va="top",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    _alpha_axis(axes[1], [0.2, 0.5, 1, 2, 4])
    axes[1].set_xlim(a_edges[0], a_edges[-1])
    axes[1].set_ylim(-15, 15)
    axes[1].set_xlabel("exponent $\\alpha$ (log scale)")
    axes[1].set_ylabel("Newton start $u_0 = z_1 - z_2$")
    axes[1].set_title("Newton basin on the drift")
    axes[1].legend(loc="lower right", fontsize=6.2, frameon=True, framealpha=0.9, edgecolor="none")
    cb = fig.colorbar(im, ax=axes[1], shrink=0.92, pad=0.02, extend="max")
    cb.set_label("Newton iterations", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    panel_label(axes[1], "b")
    _write_csv("fig1b_drift_newton_frac_converged.csv", ["alpha", "frac_drift", "frac_balance"],
               [[a, fd, fb] for a, fd, fb in zip(alphas, frac_conv["drift"], frac_conv["balance"])])

    # --- panel (c): iterations vs distance from the root -------------------
    d0 = np.linspace(-30.0, 30.0, 121)
    rows_c = []
    cases = [("balance", 0.5, PALETTE[5]), ("balance", 1.0, PALETTE[0]), ("balance", 2.0, "#003f6b"),
             ("drift", 2.0, PALETTE[1])]
    for form, a, color in cases:
        f, df = FORMS_K2[form]
        u_star = root_K2(r, a)
        its = []
        for d in d0:
            x, n, ok = newton_raphson(lambda u: f(u, r, a), lambda u: df(u, r, a), u_star + d)
            its.append(n if ok and abs(x - u_star) < 1e-8 else np.nan)
        its = np.array(its)
        far = (np.abs(d0) > 8) & np.isfinite(its)
        slope, _, r2 = least_squares_line(np.abs(d0[far]), its[far])
        theory = a if form == "balance" else a - 1.0
        axes[2].plot(d0, its, "-", color=color, lw=1.1,
                     label=f"Newton ({form}), $\\alpha$={a:g}: slope {slope:.2f}")
        SUMMARY.setdefault("fig1c_tail_slope", {})[f"{form}_alpha{a:g}"] = {"fit": float(slope), "theory": theory,
                                                                            "r2": float(r2)}
        rows_c += [[form, a, d, n] for d, n in zip(d0, its)]
    # bisection on a bracket that contains every start above: count is start-independent
    _, n_b = bisection(lambda u: drift_K2(u, r, 1.0), -40.0, 40.0, tol=1e-12)
    axes[2].axhline(n_b, color="0.45", lw=1.1, ls="--", label=f"bisection on [-40, 40]: {n_b}")
    axes[2].axhline(2, color=PALETTE[2], lw=1.1, ls="-.", label="Newton (log): 2 (1 step + check)")
    SUMMARY["fig1c_bisection_iters_bracket80"] = int(n_b)
    axes[2].set_ylim(0, 75)
    axes[2].set_xlabel("start offset $u_0 - u^*$")
    axes[2].set_ylabel("iterations to $|\\Delta u| < 10^{-12}$")
    axes[2].set_title("Far from the root Newton is linear")
    axes[2].legend(loc="upper center", fontsize=5.8)
    panel_label(axes[2], "c")
    _write_csv("fig1c_iterations_vs_start.csv", ["form", "alpha", "u0_minus_ustar", "iterations"], rows_c)

    savefig(fig, os.path.join(FIG_DIR, "fig1_scalar_rootfinding.png"))
    return {"orders": SUMMARY["fig1a_empirical_order"], "a_global": a_global,
            "frac": SUMMARY["fig1b_drift_newton_frac_converged"]}


# --------------------------------------------------------------------------
# FIGURE 2 -- K = 5: multivariate Newton-Raphson
# --------------------------------------------------------------------------

def _newton_K(form, y0, r, a, damped=False, trace=None):
    F, J = FORMS[form]
    return newton_system(lambda y: F(y, r, a), lambda y: J(y, r, a), y0, tol=1e-12,
                         max_iter=100, damped=damped, trace=trace)


def fig2_multivariate_newton():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    K = len(R5)

    # --- panel (a): convergence from the uniform policy --------------------
    runs = [("drift", 1.0, "-"), ("balance", 1.0, "-"), ("log", 1.0, "-"), ("drift", 0.5, ":")]
    rows_a = []
    for form, a, ls in runs:
        y_star = gauge_root(stationary_p(R5, a))
        tr = []
        x, n, ok = _newton_K(form, np.zeros(K - 1), R5, a, trace=tr)
        err = np.array([np.max(np.abs(y_star))] + [np.max(np.abs(t - y_star)) for t in tr])
        color, marker = FORM_STYLE[form]
        if ok:
            q, _, _ = _empirical_order(err, lo=1e-13)
            tag = f"order {q:.2f}" if np.isfinite(q) else "exact in 1 step"
            SUMMARY.setdefault("fig2a_empirical_order_alpha1", {})[form] = None if not np.isfinite(q) else float(q)
        else:
            tag = "diverges"
            p_last = softmax(np.append(tr[-1], 0.0))
            SUMMARY["fig2a_drift_alpha0.5_final_p"] = p_last.tolist()
        SUMMARY.setdefault("fig2a_iterations", {})[f"{form}_alpha{a:g}"] = {"n": int(n), "converged": bool(ok)}
        axes[0].semilogy(np.arange(len(err)), np.maximum(err, ERR_FLOOR), ls, marker=marker, ms=3.2,
                         lw=0.9, color=color, label=f"{form}, $\\alpha$={a:g}: {tag}")
        rows_a += [[form, a, i, e] for i, e in enumerate(err)]
    J_full = jacobian_alpha(np.log(stationary_p(R5, 1.0)), R5, 1.0)
    sv = np.linalg.svd(J_full, compute_uv=False)
    SUMMARY["fig2a_full_jacobian_smallest_singular_value"] = float(sv[-1])
    axes[0].text(0.20, 0.20, f"without the gauge: $\\sigma_{{\\min}}(J)$={sv[-1]:.0e}\n"
                 "(singular, since $J\\,\\mathbf{1}=0$)",
                 transform=axes[0].transAxes, fontsize=6.3, color=INK)
    axes[0].set_ylim(ERR_FLOOR / 3, 1e3)
    axes[0].set_xlabel("Newton iteration $n$ (start: uniform policy)")
    axes[0].set_ylabel("$\\|y_n - y^*\\|_\\infty$ (log scale)")
    axes[0].set_title("$K$=5, gauge $z_5$=0")
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.93), fontsize=5.8)
    panel_label(axes[0], "a")
    _write_csv("fig2a_convergence_histories.csv", ["form", "alpha", "iteration", "max_abs_error"], rows_a)

    # --- panel (b): success rate from random starts ------------------------
    alphas = np.geomspace(0.2, 4.0, 15)
    n_starts = 200
    rng = np.random.default_rng(303)
    Y0 = rng.normal(0.0, 3.0, size=(n_starts, K - 1))
    methods = [("drift", False, "drift"), ("drift", True, "drift + backtracking"),
               ("balance", False, "balance"), ("log", False, "log")]
    success = {m[2]: [] for m in methods}
    rows_b = []
    for a in alphas:
        y_star = gauge_root(stationary_p(R5, a))
        for form, damped, name in methods:
            good, its, boundary = 0, [], 0
            for y0 in Y0:
                x, n, ok = _newton_K(form, y0, R5, a, damped=damped)
                if ok and np.max(np.abs(x - y_star)) < 1e-8:
                    good += 1
                    its.append(n)
                elif np.max(np.abs(x)) > 30:
                    boundary += 1                  # some p_i < e^-30: ended at the simplex boundary
            success[name].append(good / n_starts)
            rows_b.append([name, a, good / n_starts, np.median(its) if its else np.nan,
                           boundary / n_starts])
    styles_b = {"drift": (PALETTE[1], "o", "-"), "drift + backtracking": (PALETTE[4], "v", "--"),
                "balance": (PALETTE[0], "s", "-"), "log": (PALETTE[2], "^", ":")}
    for name, vals in success.items():
        color, marker, ls = styles_b[name]
        hollow = name == "balance"                  # balance and log coincide at 100%: balance drawn open, larger
        axes[1].plot(alphas, 100 * np.array(vals), ls, marker=marker, ms=6.0 if hollow else 3.5,
                     mfc="white" if hollow else color, mew=1.0, color=color, lw=1.1, label=name,
                     zorder=1 if hollow else 2)
    axes[1].axvline(1.0, color="0.5", lw=0.8, ls="--")
    _alpha_axis(axes[1], [0.2, 0.5, 1, 2, 4])
    axes[1].set_ylim(-4, 104)
    axes[1].set_xlabel("exponent $\\alpha$ (log scale)")
    axes[1].set_ylabel(f"starts converged to $p^*$ (%, $n$={n_starts})")
    axes[1].set_title("Newton from random logits")
    axes[1].legend(loc="center right", fontsize=6.3)
    panel_label(axes[1], "b")
    for name in success:
        SUMMARY.setdefault("fig2b_success_rate", {})[name] = {f"{a:.3g}": float(v) for a, v in zip(alphas, success[name])}
    _write_csv("fig2b_success_vs_alpha.csv",
               ["method", "alpha", "success_rate", "median_iterations_converged", "frac_ended_at_boundary"], rows_b)

    # --- panel (c): conditioning of the Newton system at the root ----------
    a_dense = np.geomspace(0.1, 5.0, 80)
    conds = {"drift": [], "balance": [], "log": [], "full": [], "stiff": []}
    for a in a_dense:
        ps = stationary_p(R5, a)
        y_star = gauge_root(ps)
        for form in ["drift", "balance", "log"]:
            conds[form].append(np.linalg.cond(FORMS[form][1](y_star, R5, a)))
        sv = np.linalg.svd(jacobian_alpha(np.log(ps), R5, a), compute_uv=False)
        conds["full"].append(sv[0] / sv[-1])
        lam = linear_rates(R5, a)
        conds["stiff"].append(lam[-1] / lam[0])
    axes[2].semilogy(a_dense, conds["full"], color="0.55", lw=1.1, label="drift, no gauge ($K\\times K$)")
    axes[2].semilogy(a_dense, conds["drift"], color=FORM_STYLE["drift"][0], lw=1.4, label="drift, gauge $z_5$=0")
    # at the root the balance Jacobian is exactly -alpha c I, so balance and log both sit at 1
    axes[2].semilogy(a_dense, conds["log"], color=FORM_STYLE["log"][0], lw=3.0, label="log, gauge $z_5$=0")
    axes[2].semilogy(a_dense, conds["balance"], "--", color=FORM_STYLE["balance"][0], lw=1.4,
                     label="balance, gauge $z_5$=0")
    axes[2].semilogy(a_dense, conds["stiff"], "--", color="k", lw=0.8,
                     label="$\\lambda_{\\max}/\\lambda_{\\min}$ (Exp. 2)")
    _alpha_axis(axes[2], [0.1, 0.2, 0.5, 1, 2, 5])
    axes[2].set_ylim(0.3, 1e19)
    axes[2].set_xlabel("exponent $\\alpha$ (log scale)")
    axes[2].set_ylabel("condition number at the root (log scale)")
    axes[2].set_title("Conditioning of the Newton system")
    axes[2].legend(loc="center right", fontsize=6.0)
    panel_label(axes[2], "c")
    for key in conds:
        SUMMARY.setdefault("fig2c_condition_number", {})[key] = {
            f"{a:g}": float(conds[key][int(np.argmin(np.abs(a_dense - a)))]) for a in [0.1, 1.0, 5.0]}
    _write_csv("fig2c_condition_numbers.csv", ["alpha", "cond_drift", "cond_balance", "cond_log",
                                                 "cond_full_nogauge", "stiffness_ratio"],
               np.column_stack([a_dense, conds["drift"], conds["balance"], conds["log"],
                                conds["full"], conds["stiff"]]).tolist())

    savefig(fig, os.path.join(FIG_DIR, "fig2_multivariate_newton.png"))
    return {"success": success, "alphas": alphas}


# --------------------------------------------------------------------------
# FIGURE 3 -- finite group size, K = 5: stationary point without closed form
# --------------------------------------------------------------------------

def _mc_long_run(r, alpha, G, h, n_steps, n_seeds, rng):
    """Monte-Carlo agent from the uniform policy; policy averaged over the
    second half of training, then over seeds. Returns (mean, sd over seeds)."""
    z = np.zeros((n_seeds, len(r)))
    acc = np.zeros_like(z)
    for step in range(n_steps):
        ghat, _ = rhs_sampled(z, r, alpha, G, rng)
        z = z + h * ghat
        if step >= n_steps // 2:
            acc += softmax(z)
    p_avg = acc / (n_steps - n_steps // 2)
    return p_avg.mean(axis=0), p_avg.std(axis=0)


def fig3_finite_group():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    K = len(R5)
    rng = np.random.default_rng(3003)

    # --- panel (a): the effective weight w_G(p) -----------------------------
    a = 1.0
    p = np.geomspace(1e-4, 1.0, 300)
    axes[0].loglog(p, p ** (-a), "--", color="k", lw=1.0, label="ideal $p^{-\\alpha}$ ($G\\to\\infty$)")
    max_rel = 0.0
    for i, G in enumerate([2, 4, 16, 64]):
        w = effective_weight(p, G, a)
        direct = expected_weight(p, G, a) / p
        max_rel = max(max_rel, float(np.max(np.abs(w - direct) / w)))
        axes[0].loglog(p, w, color=PALETTE[i], label=f"$w_G$, $G$={G}")
        axes[0].scatter(p[::25], direct[::25], s=14, facecolor="white", edgecolor=PALETTE[i], lw=0.9, zorder=3)
        axes[0].axhline(G ** a, color=PALETTE[i], lw=0.6, ls=":")
    axes[0].scatter([], [], s=14, facecolor="white", edgecolor="0.3", lw=0.9, label="$\\varphi_G(p)/p$ (Exp. 2)")
    axes[0].text(0.56, 0.50, f"max rel. difference {max_rel:.1e}\ndotted: ceiling $G^\\alpha$",
                 transform=axes[0].transAxes, fontsize=6.3, color=INK, va="bottom")
    axes[0].set_ylim(0.7, 2e4)
    axes[0].set_xlabel("outcome probability $p$ (log scale)")
    axes[0].set_ylabel("effective weight $w_G(p)$, $\\alpha$=1")
    axes[0].set_title("A finite group caps the weight")
    axes[0].legend(loc="upper right", fontsize=6.0)
    panel_label(axes[0], "a")
    SUMMARY["fig3a_max_rel_diff_sizebiased_vs_direct"] = max_rel

    # --- solver comparison: nested bisection vs nested safeguarded Newton ---
    rows_s = []
    max_dp = 0.0
    for G in [4, 16, 64]:
        for a_s in [0.5, 1.0, 2.0]:
            out = {}
            for method in ["bisection", "newton"]:
                t0 = time.time()
                p_s, S, st = meanfield_stationary(R5, G, a_s, method=method)
                out[method] = (p_s, st, time.time() - t0)
            dp = float(np.max(np.abs(out["bisection"][0] - out["newton"][0])))
            max_dp = max(max_dp, dp)
            sb, sn = out["bisection"][1], out["newton"][1]
            rows_s.append([G, a_s, sb["outer_iter"], sb["inner_iter"], sb["n_w"], out["bisection"][2],
                           sn["outer_iter"], sn["inner_iter"], sn["n_w"], sn["n_dw"], out["newton"][2], dp,
                           int(np.sum(out["newton"][0] > 0))])
    rows_s = np.array(rows_s, dtype=float)
    work_ratio = rows_s[:, 4] / (rows_s[:, 8] + rows_s[:, 9])
    SUMMARY["fig3_nested_max_abs_diff_bisection_vs_newton"] = max_dp
    SUMMARY["fig3_nested_work_ratio_bisection_over_newton"] = {"min": float(work_ratio.min()),
                                                               "max": float(work_ratio.max())}
    SUMMARY["fig3_nested_outer_iters"] = {"bisection": [int(rows_s[:, 2].min()), int(rows_s[:, 2].max())],
                                          "newton": [int(rows_s[:, 6].min()), int(rows_s[:, 6].max())]}
    _write_csv("fig3_nested_solver_comparison.csv",
               ["G", "alpha", "bis_outer_iter", "bis_inner_iter", "bis_n_w", "bis_seconds",
                "newton_outer_iter", "newton_inner_iter", "newton_n_w", "newton_n_dw", "newton_seconds",
                "max_abs_p_diff", "support_size"], rows_s.tolist())

    # logit-space Newton on the mean field: fine with full support, breaks with extinction
    for G in [16, 4]:
        p_nest, _, _ = meanfield_stationary(R5, G, 1.0)
        x, n, ok = newton_system(lambda y: meanfield_log_residual(y, R5, G, 1.0),
                                 lambda y: meanfield_log_jacobian(y, R5, G, 1.0), np.zeros(K - 1))
        z = np.append(x, 0.0)
        p_newton = softmax(z) if np.all(np.isfinite(z)) else np.full(K, np.nan)
        SUMMARY.setdefault("fig3_logit_newton_meanfield", {})[str(G)] = {
            "converged": bool(ok), "iterations": int(n), "support_size": int(np.sum(p_nest > 0)),
            "max_abs_diff_vs_nested": float(np.nanmax(np.abs(p_newton - p_nest))) if ok else None}

    # --- panel (b): stationary distribution vs G ----------------------------
    a = 1.0
    G_line = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    P = np.array([meanfield_stationary(R5, G, a)[0] for G in G_line])
    p_ideal = stationary_p(R5, a)
    G_mc = [2, 4, 8, 16, 32, 64]
    h_mc, n_mc, seeds_mc = 0.01, 100_000, 16
    P_mc, P_sd = [], []
    for G in G_mc:
        m, s = _mc_long_run(R5, a, G, h_mc, n_mc, seeds_mc, rng)
        P_mc.append(m)
        P_sd.append(s)
    P_mc, P_sd = np.array(P_mc), np.array(P_sd)
    P_mf_at_mc = P[[G_line.index(G) for G in G_mc]]
    SUMMARY["fig3b_max_abs_mc_minus_meanfield"] = float(np.max(np.abs(P_mc - P_mf_at_mc)))
    SUMMARY["fig3b_max_abs_mc_minus_meanfield_by_G"] = {str(G): float(np.max(np.abs(P_mc[i] - P_mf_at_mc[i])))
                                                        for i, G in enumerate(G_mc)}
    for i in range(K):
        color = PALETTE[i]
        axes[1].plot(G_line, P[:, i], color=color, lw=1.3, label=f"O{i+1}, $r$={R5[i]:g}")
        axes[1].errorbar(G_mc, P_mc[:, i], yerr=P_sd[:, i], fmt="o", ms=3.5, color=color,
                         mfc="white", mew=1.0, capsize=0)
        axes[1].axhline(p_ideal[i], color=color, lw=0.6, ls=":")
    axes[1].errorbar([], [], yerr=[], fmt="o", ms=3.5, color="0.3", mfc="white", mew=1.0,
                     label=f"MC ($h$={h_mc:g})")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks([2, 4, 8, 16, 32, 64, 128, 256])
    axes[1].set_xticklabels(["2", "4", "8", "16", "32", "64", "128", "256"])
    axes[1].xaxis.set_minor_locator(mticker.NullLocator())
    axes[1].set_ylim(-0.02, 0.72)
    axes[1].text(0.97, 0.62, "lines: nested root-finding\ndotted: ideal $p^* = r/\\sum r$",
                 transform=axes[1].transAxes, fontsize=6.2, color=INK, ha="right", va="top")
    axes[1].set_xlabel("group size $G$ (log scale)")
    axes[1].set_ylabel("mean-field stationary $p_i$, $\\alpha$=1")
    axes[1].set_title("$K$=5: small groups lose outcomes")
    axes[1].legend(loc="upper right", fontsize=6.0, ncol=2)
    panel_label(axes[1], "b")
    SUMMARY["fig3b_support_size_by_G"] = {str(G): int(np.sum(P[i] > 0)) for i, G in enumerate(G_line)}
    _write_csv("fig3b_stationary_vs_G.csv", ["G"] + [f"p{i+1}_meanfield" for i in range(K)],
               np.column_stack([G_line, P]).tolist())
    _write_csv("fig3b_mc_vs_meanfield.csv",
               ["G"] + [f"p{i+1}_mc" for i in range(K)] + [f"p{i+1}_mc_sd" for i in range(K)]
               + [f"p{i+1}_meanfield" for i in range(K)],
               np.column_stack([G_mc, P_mc, P_sd, P_mf_at_mc]).tolist())

    # finite learning rate: the MC - mean-field gap at G = 4 shrinks with h
    gap_rows = []
    for h, n in [(0.05, 30_000), (0.02, 75_000), (0.01, 150_000)]:
        m, _ = _mc_long_run(R5, a, 4, h, n, seeds_mc, rng)
        gap = float(np.max(np.abs(m - P[G_line.index(4)])))
        gap_rows.append([h, n, gap])
        SUMMARY.setdefault("fig3b_G4_gap_vs_h", {})[f"{h:g}"] = gap
    _write_csv("fig3b_G4_gap_vs_learning_rate.csv", ["h", "n_steps", "max_abs_mc_minus_meanfield"], gap_rows)

    # --- panel (c): extinction thresholds -----------------------------------
    G_c = [2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 64, 128, 256]
    alpha_c = np.full((K, len(G_c)), np.nan)
    rows_c = []
    for gi, G in enumerate(G_c):
        for j in range(1, K):
            ac, _ = extinction_alpha(R5, j, G)
            kept_inf = kept_at_large_alpha(R5, j, G)
            if np.isfinite(ac) != kept_inf:
                raise RuntimeError(f"alpha->inf criterion disagrees with bisection at G={G}, j={j}")
            alpha_c[j, gi] = ac
            rows_c.append([G, j + 1, R5[j], ac, critical_alpha(R5[0] / R5[j], G), kept_inf])
    G_dense = np.geomspace(2, 256, 200)
    g_min = {}
    for j in range(1, K):
        color = PALETTE[j]
        finite = np.isfinite(alpha_c[j])
        axes[2].plot(np.array(G_c)[finite], alpha_c[j, finite], "-o", ms=3.2, color=color, lw=1.3,
                     label=f"O{j+1} ($r$={R5[j]:g})")
        axes[2].plot(G_dense, np.log(R5[0] / R5[j]) / np.log(G_dense), "--", color=color, lw=0.8)
        g_min[str(j + 1)] = int(np.array(G_c)[finite][0])
        if not finite.all():
            axes[2].scatter(np.array(G_c)[~finite], np.full((~finite).sum(), 6.3 - 0.3 * (j - 3)),
                            marker="x", s=22, color=color, lw=1.2, zorder=3)
    axes[2].plot([], [], "--", color="0.3", lw=0.8, label="$K$=2 formula $\\ln(r_1/r_j)/\\ln G$")
    axes[2].text(7.0, 6.15, "$\\times$: not kept at any $\\alpha$", fontsize=6.2, color=INK, va="center")
    axes[2].axhline(1.0, color="0.5", lw=0.8, ls=":")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks([2, 4, 8, 16, 32, 64, 128, 256])
    axes[2].set_xticklabels(["2", "4", "8", "16", "32", "64", "128", "256"])
    axes[2].xaxis.set_minor_locator(mticker.NullLocator())
    axes[2].set_xlim(1.8, 290)
    axes[2].set_ylim(0, 6.6)
    axes[2].set_xlabel("group size $G$ (log scale)")
    axes[2].set_ylabel("extinction threshold $\\alpha_c$")
    axes[2].set_title("Keeping outcome $j$ needs $\\alpha > \\alpha_c$")
    axes[2].legend(loc="center right", fontsize=6.0)
    panel_label(axes[2], "c")
    SUMMARY["fig3c_min_G_kept_at_some_alpha"] = g_min
    SUMMARY["fig3c_alpha_c_G16"] = {f"O{j+1}": float(alpha_c[j, G_c.index(16)]) for j in range(1, K)}
    SUMMARY["fig3c_pairwise_formula_G16"] = {f"O{j+1}": float(critical_alpha(R5[0] / R5[j], 16))
                                             for j in range(1, K)}
    runner_up_err = np.nanmax(np.abs(alpha_c[1] - np.log(R5[0] / R5[1]) / np.log(np.array(G_c))))
    SUMMARY["fig3c_runner_up_max_abs_diff_vs_pairwise"] = float(runner_up_err)
    _write_csv("fig3c_extinction_thresholds.csv",
               ["G", "outcome", "reward", "alpha_c_K5_meanfield", "alpha_c_pairwise_formula",
                "kept_as_alpha_to_inf"], rows_c)

    savefig(fig, os.path.join(FIG_DIR, "fig3_finite_group.png"))
    return {"max_rel": max_rel, "max_dp": max_dp, "work_ratio": (work_ratio.min(), work_ratio.max()),
            "mc_vs_mf": SUMMARY["fig3b_max_abs_mc_minus_meanfield"], "g_min": g_min,
            "runner_up_err": runner_up_err}


# --------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EXPERIMENT 3: root-finding for the stationary point")
    print("=" * 70)

    t0 = time.time()
    print("\n[Figure 1] K=2: bisection, secant, Newton on three formulations ...")
    r1 = fig1_scalar_rootfinding()
    print("  empirical orders at alpha=1: "
          + ", ".join(f"{k}: {v:.2f}" if v is not None else f"{k}: 1 step" for k, v in r1["orders"].items()))
    print("  drift-Newton basin, fraction of starts in [-15, 15] converging: "
          + ", ".join(f"alpha={k}: {v * 100:.0f}%" for k, v in r1["frac"].items()))
    print(f"  drift Newton converges from every start for alpha >= {r1['a_global']:.3f}")
    print(f"  ({time.time() - t0:.0f}s)")

    t0 = time.time()
    print("\n[Figure 2] K=5: multivariate Newton ...")
    r2_ = fig2_multivariate_newton()
    for name, vals in r2_["success"].items():
        print(f"  {name:22s} success: "
              + " ".join(f"{a:.2g}:{v * 100:.0f}%" for a, v in zip(r2_["alphas"][::2], vals[::2])))
    print(f"  ({time.time() - t0:.0f}s)")

    t0 = time.time()
    print("\n[Figure 3] finite group size, K=5 ...")
    r3 = fig3_finite_group()
    print(f"  size-biased w_G vs direct binomial phi_G/p: max rel. diff {r3['max_rel']:.1e}")
    print(f"  nested bisection vs Newton: max |dp| {r3['max_dp']:.1e}, "
          f"work ratio {r3['work_ratio'][0]:.0f}-{r3['work_ratio'][1]:.0f}x")
    print(f"  max |MC - mean field| (h=0.01): {r3['mc_vs_mf']:.4f}")
    print(f"  runner-up threshold vs K=2 formula: max |diff| {r3['runner_up_err']:.1e}")
    print("  smallest G that keeps each outcome at some alpha: "
          + ", ".join(f"O{k}: {v}" for k, v in r3["g_min"].items()))
    print(f"  ({time.time() - t0:.0f}s)")

    with open(os.path.join(DATA_DIR, "exp3_summary.json"), "w") as fp:
        json.dump(SUMMARY, fp, indent=2, default=float)

    print(f"\nAll figures written to results/{EXP_NAME}/figures/, "
          f"data to results/{EXP_NAME}/data/.")
    print(f"Summary JSON: results/{EXP_NAME}/data/exp3_summary.json")


if __name__ == "__main__":
    main()
