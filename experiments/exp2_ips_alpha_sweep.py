"""
EXPERIMENT 2 -- The alpha sweep: the inverse-probability exponent as a
tunable diversity knob.

Exp. 1 established the control condition (alpha = 0 collapses). Here we sweep
the exponent of the generalized scaling r_tilde = r / p^alpha (derivation.md)
and test every prediction of the alpha-generalization, first on the idealized
ODE and then on the finite-sample Monte-Carlo agent.

  1. FIG 1 -- The knob (RK4 on the ODE):
     (a) K=2, r=(4,1): trajectories converge to p* ~ r^(1/alpha)
         (94/6, 80/20, 67/33 at alpha = 0.5, 1, 2); alpha = 0 still collapses.
     (b) Stationary majority probability vs alpha for three reward ratios:
         RK4 endpoints sit on the analytical curve to machine precision.
     (c) K=5, r=(5,4,3,2,1): full stationary distributions -- alpha moves
         the policy from concentrated to near-uniform.

  2. FIG 2 -- Convergence rates (least squares on log||p(t) - p*||):
     (a) The error decays exponentially; in rescaled time lambda*t every
         alpha falls on the same slope.
     (b) Fitted rates match the linearization J* = -alpha ||r||_{1/alpha} Cov(p*)
         for every alpha, reward gap, and for K=5.
     (c) alpha -> 0 is a singular limit: the rate vanishes like
         (r2/r1)^(1/alpha), and at alpha = 0 exactly the approach is only
         algebraic, p2 ~ 1/(2 Delta t).

  3. FIG 3 -- Stiffness and explicit-integrator stability:
     (a) The fastest rate grows like alpha * K^(alpha-1) (tie) -- alpha > 1
         makes the ODE stiff -- while the slowest rate vanishes as alpha -> 0.
     (b) The critical step size found by vectorized bisection on the actual
         nonlinear iteration matches the linear stability limits
         2/lambda_max (Euler) and 2.785/lambda_max (RK4).
     (c) At alpha = 2 a step size RK4 handles sends Euler into a 0 <-> 1
         oscillation. (Exp. 1 found no such regime at alpha = 0.)

  4. FIG 4 -- Monte-Carlo agent (finite group G; each SGD step is a noisy
     Euler step):
     (a) K=5 exact tie: alpha = 0 collapses (Exp. 1) but any alpha > 0 keeps
         the policy diverse under the same sampling noise.
     (b) K=2, r=(4,1): long-run MC policy vs alpha for several G, against the
         ideal p* and the exact finite-G mean-field prediction.
     (c) (alpha, G) phase diagram: a finite group caps the inverse-probability
         weight at min(G, 1/eps)^alpha, so IPS protects the minority outcome
         only above alpha_c = ln(r1/r2) / ln min(G, 1/eps).

Outputs: results/exp2_ips_alpha_sweep/figures/fig*.png (300 dpi) and
results/exp2_ips_alpha_sweep/data/*.csv, exp2_summary.json.
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
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dynamics import (softmax, rhs_alpha, rhs_sampled, stationary_p,
                          jacobian_alpha, linear_rates)
from src.integrators import euler_step, rk4_step, integrate, integrate_at
from src.fitting import least_squares_line
from src.rootfinding import bisection, bisection_boundary
from src.finite_group import meanfield_stationary_K2, critical_alpha
from src.plotstyle import apply_style, savefig, panel_label, PALETTE

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP_NAME = "exp2_ips_alpha_sweep"
FIG_DIR = os.path.join(ROOT, "results", EXP_NAME, "figures")
DATA_DIR = os.path.join(ROOT, "results", EXP_NAME, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

apply_style()
SUMMARY = {}

R41 = np.array([4.0, 1.0])                  # the proposal's running example
R5 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
STEPPERS = {"euler": euler_step, "rk4": rk4_step}
INK = "0.25"                                 # annotation text colour


def _write_csv(name, header, rows):
    with open(os.path.join(DATA_DIR, name), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        w.writerows(rows)


def _norm_entropy(p):
    """Shannon entropy / log K along the last axis: 1 = uniform, 0 = collapsed."""
    p = np.asarray(p, dtype=np.float64)
    return -np.sum(p * np.log(np.clip(p, 1e-300, 1.0)), axis=-1) / np.log(p.shape[-1])


def _alpha_axis(ax, ticks):
    """Log-scale alpha axis with plain-number tick labels."""
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def _run_to_stationary(r, alpha, n_efolds=40.0, h_max=0.05):
    """RK4 from the uniform policy for n_efolds e-folding times of the slowest
    linear mode, with h well inside RK4's stability limit (Fig. 3). The rates
    only set the time budget; p* itself is never used. Returns p(T)."""
    lam = linear_rates(r, alpha)
    h = min(h_max, 0.5 / lam[-1])
    n = int(np.ceil(n_efolds / lam[0] / h))
    f = lambda z: rhs_alpha(z, r, alpha)
    _, traj = integrate(f, np.zeros((1, len(r))), h, n, method="rk4", record_every=n)
    return softmax(traj[-1, 0, :])


# --------------------------------------------------------------------------
# FIGURE 1 -- the diversity knob on the idealized ODE
# --------------------------------------------------------------------------

def fig1_diversity_knob():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))

    # --- panel (a): trajectories, K=2, r=(4,1), all alphas in one batch ------
    alphas = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    h, T = 0.005, 200.0                      # h * lambda_max = 0.33 at alpha=4
    n = int(T / h)
    steps = np.concatenate([[0], np.unique(np.geomspace(1, n, 500).astype(int))])
    f = lambda z: rhs_alpha(z, R41, alphas[:, None])
    ts, traj = integrate_at(f, np.zeros((len(alphas), 2)), h, steps, method="rk4")
    p1 = softmax(traj)[:, :, 0]
    for i, a in enumerate(alphas):
        ps = stationary_p(R41, a)[0]
        axes[0].axhline(ps, color=PALETTE[i], lw=0.7, ls=":")
        axes[0].plot(ts[1:], p1[1:, i], color=PALETTE[i],
                     label=f"$\\alpha$={a:g}   $p_1^*$={ps:.3f}")
        SUMMARY.setdefault("fig1a_p1_at_T200", {})[f"{a:g}"] = float(p1[-1, i])
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.2, 1.03)
    axes[0].set_xlabel("time $t$ (log scale)")
    axes[0].set_ylabel("$p_1(t)$, rewards $r$=(4, 1)")
    axes[0].set_title("Trajectories settle at $p^*\\propto r^{1/\\alpha}$")
    axes[0].legend(loc="lower right", fontsize=6.5)
    panel_label(axes[0], "a")

    # --- panel (b): stationary majority probability vs alpha ---------------
    rewards = [np.array([4.0, 1.0]), np.array([2.0, 1.0]), np.array([1.25, 1.0])]
    a_dense = np.geomspace(0.1, 8.0, 300)
    a_mark = np.geomspace(0.3, 5.0, 9)
    rows_b, max_l1_b = [], 0.0
    for j, r in enumerate(rewards):
        axes[1].plot(a_dense, [stationary_p(r, a)[0] for a in a_dense], color=PALETTE[j],
                     label=f"$p^*_1$, $r$=({r[0]:g}, {r[1]:g})")
        p_mark = []
        for a in a_mark:
            p_ode = _run_to_stationary(r, a)
            ps = stationary_p(r, a)
            l1 = float(np.sum(np.abs(p_ode - ps)))
            max_l1_b = max(max_l1_b, l1)
            p_mark.append(p_ode[0])
            rows_b.append([r[0], r[1], a, p_ode[0], ps[0], l1])
        axes[1].scatter(a_mark, p_mark, s=18, facecolor="white", edgecolor=PALETTE[j],
                        lw=1.0, zorder=3)
    axes[1].scatter([], [], s=18, facecolor="white", edgecolor="0.3", lw=1.0,
                    label="RK4 endpoint")
    for a, txt in [(0.5, "94/6"), (1.0, "80/20"), (2.0, "67/33")]:
        axes[1].annotate(txt, (a, stationary_p(R41, a)[0]), xytext=(5, 4),
                         textcoords="offset points", fontsize=6.5, color=INK)
    axes[1].axhline(0.5, color="0.6", lw=0.8, ls=":")
    axes[1].axvline(1.0, color="0.6", lw=0.8, ls="--")
    axes[1].text(1.05, 0.975, "IPS (paper)", fontsize=6.3, color="0.4", va="top")
    axes[1].text(0.03, 0.05, f"max $\\ell_1$ error = {max_l1_b:.1e}",
                 transform=axes[1].transAxes, fontsize=6.8, color=INK)
    _alpha_axis(axes[1], [0.1, 0.2, 0.5, 1, 2, 5])
    axes[1].set_ylim(0.45, 1.02)
    axes[1].set_xlabel("exponent $\\alpha$ (log scale)")
    axes[1].set_ylabel("stationary $p_1^*$ (higher-reward outcome)")
    axes[1].set_title("The diversity knob")
    axes[1].legend(loc="upper right", fontsize=6.5)
    panel_label(axes[1], "b")
    SUMMARY["fig1b_max_l1_ode_vs_theory"] = max_l1_b

    _write_csv("fig1b_stationary_vs_alpha.csv",
               ["r1", "r2", "alpha", "p1_rk4", "p1_theory", "l1_error"], rows_b)

    # --- panel (c): K=5 stationary distributions ---------------------------
    alphas_c = [0.5, 1.0, 2.0, 4.0]
    x = np.arange(len(R5))
    width = 0.2
    rows_c, max_l1_c = [], 0.0
    for i, a in enumerate(alphas_c):
        p_ode = _run_to_stationary(R5, a)
        ps = stationary_p(R5, a)
        H = float(_norm_entropy(ps))
        l1 = float(np.sum(np.abs(p_ode - ps)))
        max_l1_c = max(max_l1_c, l1)
        off = (i - (len(alphas_c) - 1) / 2) * width
        axes[2].bar(x + off, p_ode, width=0.88 * width, color=PALETTE[i],
                    label=f"$\\alpha$={a:g}  ($H/\\ln K$={H:.2f})")
        axes[2].scatter(x + off, ps, marker="_", s=30, color="k", lw=0.9, zorder=3,
                        label="analytic $p^*$" if i == 0 else None)
        rows_c.append([a, H, l1] + list(p_ode))
        SUMMARY.setdefault("fig1c_K5_norm_entropy", {})[f"{a:g}"] = H
    axes[2].axhline(1.0 / len(R5), color="0.5", lw=0.8, ls=":")
    axes[2].text(len(R5) - 0.55, 1.0 / len(R5) + 0.01, "uniform", fontsize=6.3,
                 color="0.4", ha="right", va="bottom")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"O{i+1}\n$r$={v:g}" for i, v in enumerate(R5)])
    axes[2].set_ylabel("stationary probability (RK4)")
    axes[2].set_title("$K$=5: raising $\\alpha$ spreads the mass")
    axes[2].legend(loc="upper right", fontsize=6.5)
    panel_label(axes[2], "c")
    SUMMARY["fig1c_max_l1_ode_vs_theory"] = max_l1_c

    _write_csv("fig1c_K5_stationary.csv",
               ["alpha", "norm_entropy", "l1_error"] + [f"p{i+1}" for i in range(len(R5))], rows_c)

    savefig(fig, os.path.join(FIG_DIR, "fig1_diversity_knob.png"))
    return {"max_l1": max(max_l1_b, max_l1_c)}


# --------------------------------------------------------------------------
# FIGURE 2 -- convergence rates by least squares
# --------------------------------------------------------------------------

def _decay_curve(r, alpha, n_efolds=60.0):
    """RK4 from the uniform policy; returns t, ||p(t) - p*||_1, and the
    slowest linear rate (used only for the time budget and the x-rescaling)."""
    lam = linear_rates(r, alpha)
    h = min(0.05, 0.2 / lam[-1])
    n = int(np.ceil(n_efolds / lam[0] / h))
    f = lambda z: rhs_alpha(z, r, alpha)
    ts, traj = integrate(f, np.zeros((1, len(r))), h, n, method="rk4",
                         record_every=max(1, n // 4000))
    err = np.sum(np.abs(softmax(traj[:, 0, :]) - stationary_p(r, alpha)), axis=1)
    return ts, err, lam[0]


def _fit_rate(ts, err, window=(1e-12, 1e-6)):
    """Least-squares slope of ln||p - p*||_1 against t inside the asymptotic
    window (past the nonlinear transient, above round-off).
    Returns (rate, intercept, r2, mask)."""
    sel = (err > window[0]) & (err < window[1])
    slope, intercept, r2 = least_squares_line(ts[sel], np.log(err[sel]))
    return -slope, intercept, r2, sel


def fig2_convergence_rates():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))

    # --- panel (a): exponential decay, rescaled time -----------------------
    r_a = np.array([2.0, 1.0])
    for i, a in enumerate([0.5, 1.0, 2.0, 4.0]):
        ts, err, lam = _decay_curve(r_a, a)
        lam_fit, b, _, sel = _fit_rate(ts, err)
        axes[0].semilogy(lam * ts, err, color=PALETTE[i],
                         label=f"$\\alpha$={a:g}  ($\\lambda$={lam:.3g})")
        axes[0].semilogy(lam * ts[sel], np.exp(b - lam_fit * ts[sel]), "--", color="k", lw=0.7)
    axes[0].semilogy([], [], "--", color="k", lw=0.7, label="least-squares fit")
    axes[0].axhspan(1e-12, 1e-6, color="0.5", alpha=0.1, lw=0)
    axes[0].text(44, 3e-9, "fit window", fontsize=6.3, color="0.4", ha="right", va="center")
    axes[0].set_xlim(0, 45)
    axes[0].set_ylim(1e-16, 3)
    axes[0].set_xlabel("rescaled time $\\lambda_{\\mathrm{lin}}\\,t$")
    axes[0].set_ylabel("$\\|p(t)-p^*\\|_1$ (log scale)")
    axes[0].set_title("Exponential convergence, $r$=(2, 1)")
    axes[0].legend(loc="lower left", fontsize=6.3)
    panel_label(axes[0], "a")

    # --- panel (b): fitted rate vs linearized theory -----------------------
    cases = [("$K$=2, $r$=(1.25, 1)", np.array([1.25, 1.0]), np.geomspace(0.3, 4.0, 9)),
             ("$K$=2, $r$=(2, 1)", np.array([2.0, 1.0]), np.geomspace(0.3, 4.0, 9)),
             ("$K$=2, $r$=(4, 1)", np.array([4.0, 1.0]), np.geomspace(0.3, 4.0, 9)),
             ("$K$=5, $r$=(5,4,3,2,1)", R5, np.geomspace(0.5, 4.0, 7))]
    a_dense = np.geomspace(0.27, 4.4, 200)
    rows, max_rel, min_r2 = [], 0.0, 1.0
    for j, (label, r, a_grid) in enumerate(cases):
        axes[1].plot(a_dense, [linear_rates(r, a)[0] for a in a_dense], color=PALETTE[j], lw=1.1)
        fits = []
        for a in a_grid:
            ts, err, lam = _decay_curve(r, a)
            lam_fit, _, r2, _ = _fit_rate(ts, err)
            rel = abs(lam_fit - lam) / lam
            max_rel, min_r2 = max(max_rel, rel), min(min_r2, r2)
            fits.append(lam_fit)
            rows.append([label.replace("$", ""), a, lam_fit, lam, rel, r2])
        axes[1].scatter(a_grid, fits, s=18, color=PALETTE[j], zorder=3, label=label)
    axes[1].plot([], [], color="0.3", lw=1.1, label="theory $\\alpha\\|r\\|_{1/\\alpha}\\mu_{\\min}$")
    _alpha_axis(axes[1], [0.3, 0.5, 1, 2, 4])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("exponent $\\alpha$ (log scale)")
    axes[1].set_ylabel("convergence rate $\\lambda$ (log scale)")
    axes[1].set_title("Fitted rate = linearized rate")
    axes[1].legend(loc="upper left", fontsize=6.2)
    axes[1].text(0.97, 0.05, f"max rel. error {max_rel:.1e}\nmin $R^2$ = {min_r2:.6f}",
                 transform=axes[1].transAxes, fontsize=6.5, color=INK, ha="right")
    panel_label(axes[1], "b")
    SUMMARY["fig2b_max_rel_rate_error"] = float(max_rel)
    SUMMARY["fig2b_min_r2"] = float(min_r2)
    _write_csv("fig2b_rate_fit_vs_theory.csv",
               ["case", "alpha", "rate_fit", "rate_theory", "rel_error", "r2"], rows)

    # --- panel (c): alpha -> 0 singular limit ------------------------------
    alphas_c = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
    r_c, delta = np.array([2.0, 1.0]), 1.0
    h, T = 0.2, 2.0e4
    steps = np.unique(np.geomspace(1, int(T / h), 600).astype(int))
    f = lambda z: rhs_alpha(z, r_c, alphas_c[:, None])
    ts, traj = integrate_at(f, np.zeros((len(alphas_c), 2)), h, steps, method="rk4")
    p2 = softmax(traj)[:, :, 1]
    for i, a in enumerate(alphas_c):
        axes[2].loglog(ts, p2[:, i], color=PALETTE[i], label=f"$\\alpha$={a:g}")
        if a > 0:
            axes[2].axhline(stationary_p(r_c, a)[1], color=PALETTE[i], lw=0.7, ls=":")
    tail = ts > 100
    slope0, _, r2_0 = least_squares_line(np.log10(ts[tail]), np.log10(p2[tail, 0]))
    t_guide = ts[ts > 5]
    axes[2].loglog(t_guide, 1.0 / (2 * delta * t_guide), "--", color="k", lw=0.8,
                   label="$1/(2\\Delta t)$")
    axes[2].text(0.97, 0.62, f"$\\alpha$=0 tail slope {slope0:.3f}\n(dotted: $p_2^*$)",
                 transform=axes[2].transAxes, fontsize=6.5, color=INK, ha="right", va="top")
    axes[2].set_ylim(1e-5, 1)
    axes[2].set_xlabel("time $t$ (log scale)")
    axes[2].set_ylabel("minority probability $p_2(t)$ (log scale)")
    axes[2].set_title("$\\alpha\\to0$ is a singular limit")
    axes[2].legend(loc="lower left", fontsize=6.3)
    panel_label(axes[2], "c")
    SUMMARY["fig2c_alpha0_loglog_slope"] = float(slope0)
    SUMMARY["fig2c_alpha0_r2"] = float(r2_0)
    SUMMARY["fig2c_rate_theory"] = {f"{a:g}": float(linear_rates(r_c, a)[0]) for a in alphas_c[1:]}
    _write_csv("fig2c_minority_prob_vs_t.csv", ["t"] + [f"p2_alpha{a:g}" for a in alphas_c],
               np.column_stack([ts, p2]).tolist())

    savefig(fig, os.path.join(FIG_DIR, "fig2_convergence_rates.png"))
    return {"max_rel": max_rel, "slope0": slope0}


# --------------------------------------------------------------------------
# FIGURE 3 -- stiffness and explicit stability
# --------------------------------------------------------------------------

def fig3_stiffness_stability():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))

    # RK4's real-axis stability limit: R(-x) = 1 <=> x^3 - 4x^2 + 12x - 24 = 0
    x_rk4, it_rk4 = bisection(lambda x: x**3 - 4 * x**2 + 12 * x - 24, 2.0, 3.0, tol=1e-13)
    SUMMARY["rk4_real_axis_limit"] = float(x_rk4)
    SUMMARY["rk4_real_axis_limit_bisection_iters"] = int(it_rk4)

    # --- panel (a): spectrum of the stationary Jacobian vs alpha -----------
    a_dense = np.geomspace(0.1, 5.0, 200)
    lam_max = np.array([linear_rates(R5, a)[-1] for a in a_dense])
    lam_min = np.array([linear_rates(R5, a)[0] for a in a_dense])
    axes[0].fill_between(a_dense, lam_min, lam_max, color=PALETTE[0], alpha=0.12, lw=0)
    axes[0].plot(a_dense, lam_max, color=PALETTE[0], label="$\\lambda_{\\max}$, $r$=(5,4,3,2,1)")
    axes[0].plot(a_dense, lam_min, "--", color=PALETTE[0], label="$\\lambda_{\\min}$, $r$=(5,4,3,2,1)")
    axes[0].plot(a_dense, a_dense * 5.0 ** (a_dense - 1), color=PALETTE[1],
                 label="exact tie: $\\alpha K^{\\alpha-1}$")
    a_mark = np.geomspace(0.15, 5.0, 10)
    num_max, num_min, max_rel = [], [], 0.0
    for a in a_mark:
        J = jacobian_alpha(np.log(stationary_p(R5, a)), R5, a)
        ev = np.sort(np.abs(np.linalg.eigvals(J).real))      # ev[0] is the shift mode (0)
        num_min.append(ev[1])
        num_max.append(ev[-1])
        th = linear_rates(R5, a)
        max_rel = max(max_rel, abs(ev[1] - th[0]) / th[0], abs(ev[-1] - th[-1]) / th[-1])
    axes[0].scatter(a_mark, num_max, s=16, facecolor="white", edgecolor=PALETTE[0], lw=1.0, zorder=3,
                    label="eig. of Jacobian at $z^*$")
    axes[0].scatter(a_mark, num_min, s=16, facecolor="white", edgecolor=PALETTE[0], lw=1.0, zorder=3)
    axes[0].axhline(2.0 / 0.05, color="0.5", lw=0.8, ls=":")
    axes[0].text(0.11, 2.0 / 0.05 * 1.4, "Euler limit at $h$=0.05", fontsize=6.3, color="0.4")
    _alpha_axis(axes[0], [0.1, 0.2, 0.5, 1, 2, 5])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("exponent $\\alpha$ (log scale)")
    axes[0].set_ylabel("linear rate at $p^*$ (log scale)")
    axes[0].set_title("Spectrum of $J^*$ across $\\alpha$")
    axes[0].legend(loc="lower right", fontsize=6.2)
    panel_label(axes[0], "a")
    SUMMARY["fig3a_max_rel_err_closed_form_vs_numeric_eig"] = float(max_rel)

    # --- panel (b): critical step size by vectorized bisection -------------
    alphas_b = np.geomspace(0.5, 4.0, 8)
    zstar = np.log(np.array([stationary_p(R5, a) for a in alphas_b]))
    rng = np.random.default_rng(7)
    v = rng.standard_normal(zstar.shape)
    v -= v.mean(axis=1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    delta0 = 1e-4
    z0 = zstar + delta0 * v
    f = lambda z: rhs_alpha(z, R5, alphas_b[:, None])

    def stable_at(method, n_steps=4000):
        """pred(h): does a small perturbation of z* shrink after n_steps?
        The deviation is measured in logit space with the shift mode projected
        out -- J* is symmetric, so each mode's size is monotone in this norm."""
        step = STEPPERS[method]

        def pred(h):
            z = z0.copy()
            with np.errstate(all="ignore"):
                for _ in range(n_steps):
                    z = step(f, z, h[:, None])
                dev = z - zstar
                dev -= dev.mean(axis=1, keepdims=True)
                growth = np.linalg.norm(dev, axis=1) / delta0
            return np.isfinite(growth) & (growth < 1.0)
        return pred

    h_lo, h_hi = np.full(len(alphas_b), 1e-4), np.full(len(alphas_b), 10.0)
    lam_max_b = np.array([linear_rates(R5, a)[-1] for a in alphas_b])
    h_crit, rows = {}, []
    for method, limit in [("euler", 2.0), ("rk4", x_rk4)]:
        pred = stable_at(method)
        if not (np.all(pred(h_lo)) and not np.any(pred(h_hi))):
            raise RuntimeError(f"step-size bracket does not straddle the {method} boundary")
        h_crit[method] = bisection_boundary(pred, h_lo, h_hi, n_iter=30, geometric=True)
        rel = np.abs(h_crit[method] * lam_max_b / limit - 1.0)
        SUMMARY[f"fig3b_{method}_max_rel_err_hcrit"] = float(rel.max())
        for a, hc, lm in zip(alphas_b, h_crit[method], lam_max_b):
            rows.append([method, a, hc, limit / lm])

    a_line = np.geomspace(0.45, 4.4, 200)
    lmax_line = np.array([linear_rates(R5, a)[-1] for a in a_line])
    axes[1].plot(a_line, x_rk4 / lmax_line, color=PALETTE[0],
                 label=f"RK4 theory ${x_rk4:.3f}/\\lambda_{{\\max}}$")
    axes[1].plot(a_line, 2.0 / lmax_line, color=PALETTE[1], label="Euler theory $2/\\lambda_{\\max}$")
    axes[1].scatter(alphas_b, h_crit["rk4"], marker="s", s=18, facecolor="white",
                    edgecolor=PALETTE[0], lw=1.0, zorder=3, label="RK4, bisection")
    axes[1].scatter(alphas_b, h_crit["euler"], marker="o", s=18, facecolor="white",
                    edgecolor=PALETTE[1], lw=1.0, zorder=3, label="Euler, bisection")

    # alpha at which Exp. 1's step size h=0.05 stops being stable
    lam_max_of = lambda a: linear_rates(R5, a)[-1]
    a_euler, _ = bisection(lambda a: 0.05 * lam_max_of(a) - 2.0, 0.5, 4.0, tol=1e-10)
    a_rk4, _ = bisection(lambda a: 0.05 * lam_max_of(a) - x_rk4, 0.5, 4.0, tol=1e-10)
    axes[1].axhline(0.05, color="0.5", lw=0.8, ls=":")
    axes[1].text(0.47, 0.05 * 0.72, f"$h$=0.05 (Exp. 1) is unstable above\n"
                 f"$\\alpha$={a_euler:.2f} (Euler), {a_rk4:.2f} (RK4)",
                 fontsize=6.0, color="0.4", va="top")
    _alpha_axis(axes[1], [0.5, 1, 2, 4])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("exponent $\\alpha$ (log scale)")
    axes[1].set_ylabel("largest stable step $h_{\\mathrm{crit}}$ (log scale)")
    axes[1].set_title("Stability limit, $r$=(5,4,3,2,1)")
    axes[1].legend(loc="upper right", fontsize=6.2)
    panel_label(axes[1], "b")
    SUMMARY["fig3b_alpha_where_h0.05_unstable"] = {"euler": float(a_euler), "rk4": float(a_rk4)}
    _write_csv("fig3b_critical_step.csv", ["method", "alpha", "h_crit_bisection", "h_crit_theory"], rows)

    # --- panel (c): Euler vs RK4 at alpha=2, r=(4,1) -----------------------
    a_c = 2.0
    lam_c = linear_rates(R41, a_c)[-1]
    f_c = lambda z: rhs_alpha(z, R41, a_c)
    T = 6.0
    # Start near p* (p1 = 0.6): linear stability is a local statement. From the
    # uniform policy the local rate is 10, not lambda* = 8, so h = 0.3 would take
    # even RK4 outside its stability interval (3 > 2.785) during the transient.
    z0_c = np.array([np.log(0.6 / 0.4), 0.0])
    lam_uniform = np.max(np.abs(np.linalg.eigvals(jacobian_alpha(np.zeros(2), R41, a_c)).real))
    SUMMARY["fig3c_local_rate_at_uniform_policy"] = float(lam_uniform)
    sol = solve_ivp(lambda t, z: rhs_alpha(z[None, :], R41, a_c)[0], [0, T], z0_c,
                    method="DOP853", rtol=1e-12, atol=1e-13, dense_output=True)
    tt = np.linspace(0, T, 600)
    axes[2].plot(tt, softmax(sol.sol(tt).T)[:, 0], color="0.2", lw=2.4, alpha=0.3,
                 label="reference (DOP853)")
    runs = [("Euler", "euler", 0.2, PALETTE[2], "o"),
            ("Euler", "euler", 0.3, PALETTE[1], "o"),
            ("RK4", "rk4", 0.3, PALETTE[0], "s")]
    for name, method, h, color, marker in runs:
        n = int(round(T / h))
        ts, tr = integrate(f_c, z0_c[None, :], h, n, method=method)
        p1 = softmax(tr[:, 0, :])[:, 0]
        axes[2].plot(ts, p1, "-", marker=marker, ms=3.2, lw=0.9, color=color,
                     label=f"{name}, $h$={h}  ($h\\lambda$={h * lam_c:.1f})")
    axes[2].axhline(stationary_p(R41, a_c)[0], color="0.5", lw=0.8, ls=":")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_xlabel("time $t$ (start $p_1$=0.6)")
    axes[2].set_ylabel("$p_1(t)$, $\\alpha$=2, $r$=(4, 1)")
    axes[2].set_title("Euler oscillates where RK4 converges")
    axes[2].legend(loc="lower right", fontsize=6.2)
    panel_label(axes[2], "c")
    SUMMARY["fig3c_lambda"] = float(lam_c)

    savefig(fig, os.path.join(FIG_DIR, "fig3_stiffness_stability.png"))
    return {"x_rk4": x_rk4, "euler_err": SUMMARY["fig3b_euler_max_rel_err_hcrit"],
            "rk4_err": SUMMARY["fig3b_rk4_max_rel_err_hcrit"], "a_euler": a_euler, "a_rk4": a_rk4}


# --------------------------------------------------------------------------
# FIGURE 4 -- Monte-Carlo agent: the knob under sampling noise
# --------------------------------------------------------------------------

def fig4_monte_carlo():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    rng = np.random.default_rng(2026)

    # --- panel (a): K=5 exact tie, entropy over training -------------------
    K, G, h, n_steps, n_seeds, rec_every = 5, 16, 0.5, 6000, 100, 20
    alphas_a = np.array([0.0, 0.25, 0.5, 1.0])     # h*lambda <= 0.5 for all (tie: alpha K^(alpha-1))
    alpha_col = np.repeat(alphas_a, n_seeds)[:, None]
    z = np.zeros((alpha_col.shape[0], K))
    H = []
    for step in range(n_steps):
        ghat, _ = rhs_sampled(z, np.ones(K), alpha_col, G, rng)
        z = z + h * ghat
        if (step + 1) % rec_every == 0:
            H.append(_norm_entropy(softmax(z)).reshape(len(alphas_a), n_seeds))
    H = np.array(H)                                   # (n_rec, n_alpha, n_seeds)
    t = np.arange(1, H.shape[0] + 1) * rec_every * h
    p_end = softmax(z).reshape(len(alphas_a), n_seeds, K)
    rows_a = []
    for i, a in enumerate(alphas_a):
        frac = float(np.mean(p_end[i].max(axis=1) > 0.95))
        q1, med, q3 = np.percentile(H[:, i, :], [25, 50, 75], axis=1)
        axes[0].fill_between(t, q1, q3, color=PALETTE[i], alpha=0.18, lw=0)
        axes[0].plot(t, med, color=PALETTE[i], label=f"$\\alpha$={a:g}  ({frac * 100:.0f}% collapsed)")
        SUMMARY.setdefault("fig4a_frac_collapsed", {})[f"{a:g}"] = frac
        SUMMARY.setdefault("fig4a_final_median_entropy", {})[f"{a:g}"] = float(med[-1])
        rows_a.append([a, frac, med[-1], q1[-1], q3[-1]])
    axes[0].set_ylim(-0.03, 1.05)
    axes[0].set_xlabel("training step $\\times h$")
    axes[0].set_ylabel("policy entropy $H/\\ln K$")
    axes[0].set_title(f"Exact tie, $K$=5, $G$={G}")
    axes[0].legend(loc="lower left", fontsize=6.3)
    panel_label(axes[0], "a")
    _write_csv("fig4a_tie_entropy.csv",
               ["alpha", "frac_collapsed", "median_entropy_end", "q1_end", "q3_end"], rows_a)

    # --- MC sweep over (alpha, G) for K=2, r=(4,1) -------------------------
    alphas = np.geomspace(0.3, 2.0, 12)
    Gs = [2, 4, 8, 16, 32, 64, 128]
    n_seeds, h, n_steps = 16, 0.05, 30000             # h*lambda <= 0.4 up to alpha=2
    alpha_col = np.repeat(alphas, n_seeds)[:, None]
    p1_mc = np.zeros((len(Gs), len(alphas)))
    p1_sd = np.zeros_like(p1_mc)
    for gi, G in enumerate(Gs):
        z = np.zeros((alpha_col.shape[0], 2))
        acc = np.zeros_like(z)
        for step in range(n_steps):
            ghat, _ = rhs_sampled(z, R41, alpha_col, G, rng)
            z = z + h * ghat
            if step >= n_steps // 2:
                acc += softmax(z)
        p_avg = (acc[:, 0] / (n_steps - n_steps // 2)).reshape(len(alphas), n_seeds)
        p1_mc[gi], p1_sd[gi] = p_avg.mean(axis=1), p_avg.std(axis=1)
    p1_ideal = np.array([stationary_p(R41, a)[0] for a in alphas])
    p1_mf = np.array([[meanfield_stationary_K2(R41, G, a) for a in alphas] for G in Gs])
    retained = (1.0 - p1_mc) / (1.0 - p1_ideal[None, :])

    rows = []
    for gi, G in enumerate(Gs):
        for ai, a in enumerate(alphas):
            rows.append([G, a, p1_mc[gi, ai], p1_sd[gi, ai], p1_ideal[ai], p1_mf[gi, ai],
                         retained[gi, ai], critical_alpha(4.0, G)])
    _write_csv("fig4bc_mc_alpha_G_sweep.csv",
               ["G", "alpha", "p1_mc", "p1_mc_sd_seeds", "p1_ideal", "p1_meanfield",
                "minority_retained", "alpha_crit"], rows)
    SUMMARY["fig4bc_max_abs_mc_minus_meanfield"] = float(np.max(np.abs(p1_mc - p1_mf)))

    # --- panel (b): slices at G = 4, 16, 64 ---------------------------------
    a_dense = np.geomspace(0.28, 2.1, 90)
    axes[1].plot(a_dense, [stationary_p(R41, a)[0] for a in a_dense], "--", color="k", lw=1.0,
                 label="ideal $p^*$ ($G\\to\\infty$)")
    for j, G in enumerate([4, 16, 64]):
        gi = Gs.index(G)
        color = PALETTE[j]
        axes[1].plot(a_dense, [meanfield_stationary_K2(R41, G, a) for a in a_dense],
                     color=color, lw=1.1)
        axes[1].errorbar(alphas, p1_mc[gi], yerr=p1_sd[gi], fmt="o", ms=3.5, color=color,
                         mfc="white", mew=1.0, capsize=0, label=f"MC, $G$={G}")
        axes[1].axvline(critical_alpha(4.0, G), color=color, lw=0.8, ls=":")
    axes[1].plot([], [], color="0.3", lw=1.1, label="finite-$G$ mean field")
    axes[1].plot([], [], color="0.3", lw=0.8, ls=":", label="$\\alpha_c=\\ln 4/\\ln G$")
    _alpha_axis(axes[1], [0.3, 0.5, 1, 2])
    axes[1].set_ylim(0.6, 1.015)
    axes[1].set_xlabel("exponent $\\alpha$ (log scale)")
    axes[1].set_ylabel("long-run $\\bar p_1$, $r$=(4, 1)")
    axes[1].set_title("Finite-$G$ mean field predicts MC")
    axes[1].legend(loc="lower left", fontsize=6.2)
    panel_label(axes[1], "b")

    # --- panel (c): (alpha, G) phase diagram --------------------------------
    ratio = alphas[1] / alphas[0]
    a_edges = np.concatenate([[alphas[0] / np.sqrt(ratio)],
                              np.sqrt(alphas[1:] * alphas[:-1]),
                              [alphas[-1] * np.sqrt(ratio)]])
    G_arr = np.array(Gs, dtype=float)
    G_edges = np.concatenate([[G_arr[0] / np.sqrt(2)], np.sqrt(G_arr[1:] * G_arr[:-1]),
                              [G_arr[-1] * np.sqrt(2)]])
    im = axes[2].pcolormesh(a_edges, G_edges, np.clip(retained, 0, 1), cmap="Blues",
                            vmin=0, vmax=1, shading="flat")
    G_line = np.geomspace(G_edges[0], G_edges[-1], 200)
    axes[2].plot(np.log(4.0) / np.log(G_line), G_line, "--", color=PALETTE[1], lw=1.5,
                 label="$\\alpha_c = \\ln(r_1/r_2)/\\ln G$")
    _alpha_axis(axes[2], [0.3, 0.5, 1, 2])
    axes[2].set_yscale("log", base=2)
    axes[2].set_xlim(a_edges[0], a_edges[-1])
    axes[2].set_ylim(G_edges[0], G_edges[-1])
    axes[2].set_yticks(Gs)
    axes[2].set_yticklabels([str(g) for g in Gs])
    axes[2].grid(False)
    axes[2].set_xlabel("exponent $\\alpha$ (log scale)")
    axes[2].set_ylabel("group size $G$")
    axes[2].set_title("Collapse boundary")
    axes[2].legend(loc="upper right", fontsize=6.3, frameon=True, framealpha=0.9, edgecolor="none")
    cb = fig.colorbar(im, ax=axes[2], shrink=0.92, pad=0.02, extend="max")
    cb.set_label("minority mass kept  $\\bar p_2 / p_2^*$", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    panel_label(axes[2], "c")

    savefig(fig, os.path.join(FIG_DIR, "fig4_monte_carlo.png"))
    return {"frac_collapsed": SUMMARY["fig4a_frac_collapsed"],
            "mc_vs_mf": SUMMARY["fig4bc_max_abs_mc_minus_meanfield"]}


# --------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EXPERIMENT 2: the alpha sweep -- IPS exponent as a diversity knob")
    print("=" * 70)

    t0 = time.time()
    print("\n[Figure 1] Stationary law p* ~ r^(1/alpha) on the ODE ...")
    r1 = fig1_diversity_knob()
    print(f"  max l1 |p_RK4(T) - p*| over all alphas and rewards: {r1['max_l1']:.2e}")
    print(f"  ({time.time() - t0:.0f}s)")

    t0 = time.time()
    print("\n[Figure 2] Convergence rates by least squares ...")
    r2_ = fig2_convergence_rates()
    print(f"  max relative error, fitted vs linearized rate: {r2_['max_rel']:.2e}")
    print(f"  alpha=0 minority-probability log-log tail slope: {r2_['slope0']:.4f} (theory -1)")
    print(f"  ({time.time() - t0:.0f}s)")

    t0 = time.time()
    print("\n[Figure 3] Stiffness and explicit stability ...")
    r3 = fig3_stiffness_stability()
    print(f"  RK4 real-axis stability limit (bisection): {r3['x_rk4']:.10f}")
    print(f"  critical step by bisection vs linear theory -- max rel. error "
          f"Euler: {r3['euler_err']:.1e}, RK4: {r3['rk4_err']:.1e}")
    print(f"  h=0.05 becomes unstable above alpha = {r3['a_euler']:.3f} (Euler), "
          f"{r3['a_rk4']:.3f} (RK4) for r=(5,4,3,2,1)")
    print(f"  ({time.time() - t0:.0f}s)")

    t0 = time.time()
    print("\n[Figure 4] Monte-Carlo agent ...")
    r4 = fig4_monte_carlo()
    print("  K=5 exact tie, fraction collapsed: "
          + ", ".join(f"alpha={a}: {v * 100:.0f}%" for a, v in r4["frac_collapsed"].items()))
    print(f"  max |MC - finite-G mean field| over the (alpha, G) grid: {r4['mc_vs_mf']:.3f}")
    print(f"  ({time.time() - t0:.0f}s)")

    with open(os.path.join(DATA_DIR, "exp2_summary.json"), "w") as fp:
        json.dump(SUMMARY, fp, indent=2, default=float)

    print(f"\nAll figures written to results/{EXP_NAME}/figures/, "
          f"data to results/{EXP_NAME}/data/.")
    print(f"Summary JSON: results/{EXP_NAME}/data/exp2_summary.json")


if __name__ == "__main__":
    main()
