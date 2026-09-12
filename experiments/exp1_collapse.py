"""
EXPERIMENT 1 -- Outcome-level mode collapse under the standard objective.

This is the baseline/control condition for the whole project: before any
Inverse-Probability-Scaling correction or alpha-generalization is applied,
we must show *precisely* when and how the standard expected-return
objective (alpha = 0) causes outcome-level mode collapse, using the
outcome-selection bandit of Sinha, Elango & Liu (2026, arXiv:2601.21669).

Three findings are established numerically, each backed by course numerical
methods (RNG/Monte-Carlo, Euler & RK4 integration, least-squares fitting):

  1. FIG 1 -- Deterministic (idealized, noise-free) gradient flow:
     (a) Under an EXACT reward tie, the flow is provably flat (dz/dt == 0
         everywhere) -- nothing moves, confirmed to machine precision.
         This sharpens the paper's Theorem 3.1: outcome-level collapse in
         the noiseless limit needs a nonzero reward gap, however small.
     (b) For any nonzero reward gap, the flow deterministically collapses
         onto the higher-reward outcome; smaller gaps collapse more slowly.
     (c) The collapse time follows a clean power law in the reward gap,
         t_collapse ~ C * gap^k, extracted by least-squares fit in log-log
         space (k should come out close to -1).

  2. FIG 2 -- Realistic stochastic training (finite Monte-Carlo group size G,
     inverse-transform sampling): even with an EXACT reward tie and an
     EXACTLY symmetric initialization (z0 = 0), sampling noise alone breaks
     the tie and the outcome-probability multiplier amplifies it into full
     collapse -- a Polya-urn-like rich-get-richer mechanism. This is the
     correct, rigorous version of the paper's informal claim ("even with
     equal rewards ... small asymmetries are unavoidable").
     (a) Individual-seed trajectories fan out from p=0.5 to 0 or 1.
     (b) Winner-outcome frequency is unbiased/uniform across outcomes.
     (c) Larger Monte-Carlo group size G (less sampling noise) delays
         collapse -- a genetic-drift-style scaling law in a real RL
         hyperparameter.

  3. FIG 3 -- Numerical integrator validation on the collapse ODE itself:
     (a) Euler is first-order and RK4 is fourth-order accurate on this
         system (empirical order recovered via least-squares log-log fit),
         with a visible machine round-off floor below the truncation-error
         regime (this ODE is a bounded, self-damping monotone gradient flow
         on a linear objective -- we checked a wide grid and confirmed it
         has no classical Euler oscillation/blow-up regime to showcase, so
         we report the floor honestly instead of manufacturing instability).
     (b) The step size used throughout Figs. 1-2 (h=0.05) is validated
         against a high-precision reference solution.

Outputs: results/exp1_collapse/figures/fig1_*.png, fig2_*.png, fig3_*.png
(300 dpi, Nature-journal-like style) and results/exp1_collapse/data/*.csv
summary tables.
"""

from __future__ import annotations
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dynamics import softmax, rhs_alpha, rhs_sampled
from src.integrators import euler_step, rk4_step, integrate, time_to_event
from src.fitting import least_squares_line
from src.plotstyle import apply_style, savefig, panel_label, PALETTE

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP_NAME = "exp1_collapse"
FIG_DIR = os.path.join(ROOT, "results", EXP_NAME, "figures")
DATA_DIR = os.path.join(ROOT, "results", EXP_NAME, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

apply_style()
SUMMARY = {}


# --------------------------------------------------------------------------
# FIGURE 1 -- deterministic (idealized) collapse
# --------------------------------------------------------------------------

def fig1_deterministic_collapse():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))

    # --- panel (a): exact tie -> flow is provably flat ------------------
    r_tie = np.array([1.0, 1.0])
    f_tie = lambda z: rhs_alpha(z, r_tie, alpha=0.0)
    p1_0_list = [0.5, 0.65, 0.80, 0.95]
    max_drift = 0.0
    for i, p1_0 in enumerate(p1_0_list):
        z0 = np.array([[np.log(p1_0 / (1 - p1_0)), 0.0]])
        ts, traj = integrate(f_tie, z0, h=0.05, n_steps=6000, method="rk4", record_every=200)
        p = softmax(traj[:, 0, :])
        drift = np.max(np.abs(p[:, 0] - p1_0))
        max_drift = max(max_drift, drift)
        axes[0].plot(ts, p[:, 0], color=PALETTE[i], label=f"$p_1(0)$={p1_0:.2f}")
    axes[0].set_xlabel("time $t$")
    axes[0].set_ylabel("$p_1(t)$")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("Exact tie: flow is flat")
    axes[0].legend(loc="center right", fontsize=6.5)
    axes[0].text(0.03, 0.06, f"max drift = {max_drift:.1e}\n(machine precision)",
                 transform=axes[0].transAxes, fontsize=6.8, color="0.35")
    panel_label(axes[0], "a")
    SUMMARY["fig1a_max_drift_exact_tie"] = float(max_drift)

    # --- panel (b): reward-gap collapse trajectories ---------------------
    deltas_b = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    z0_b = np.array([[0.05, -0.05]])
    for i, delta in enumerate(deltas_b):
        r = np.array([1.0 + delta, 1.0])
        f = lambda z, r=r: rhs_alpha(z, r, alpha=0.0)
        T = max(80.0, 3.0 * 54.0 / delta)
        n_steps = int(T / 0.05)
        ts, traj = integrate(f, z0_b, h=0.05, n_steps=n_steps, method="rk4",
                              record_every=max(1, n_steps // 400))
        p = softmax(traj[:, 0, :])
        axes[1].plot(ts, p[:, 0], color=PALETTE[i % len(PALETTE)],
                     label=f"$\\Delta$={delta}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("time $t$ (log scale)")
    axes[1].set_ylabel("$p_1(t)$ (higher-reward outcome)")
    axes[1].set_title("Reward-gap collapse")
    axes[1].legend(loc="lower right", fontsize=6.5, ncol=1)
    panel_label(axes[1], "b")

    # --- panel (c): collapse-time power law -------------------------------
    deltas_c = np.logspace(np.log10(0.02), np.log10(4.0), 12)
    t_collapse = []
    for delta in deltas_c:
        r = np.array([1.0 + delta, 1.0])
        f = lambda z, r=r: rhs_alpha(z, r, alpha=0.0)
        z0 = np.array([0.05, -0.05])
        event = lambda z: np.max(softmax(z)) >= 0.99
        T_budget = 3.0 * 54.0 / delta + 60.0
        t = time_to_event(f, z0, h=0.05, event=event, method="rk4",
                           max_steps=int(T_budget / 0.05))
        t_collapse.append(t)
    t_collapse = np.array(t_collapse, dtype=np.float64)
    logD, logT = np.log10(deltas_c), np.log10(t_collapse)
    slope, intercept, r2 = least_squares_line(logD, logT)
    fit_line = 10 ** (intercept) * deltas_c ** slope

    axes[2].scatter(deltas_c, t_collapse, s=22, color=PALETTE[0], zorder=3,
                     label="simulated $t_{\\mathrm{collapse}}$")
    axes[2].plot(deltas_c, fit_line, "--", color=PALETTE[1], lw=1.3,
                 label=f"fit: $t\\propto\\Delta^{{{slope:.2f}}}$\n$R^2$={r2:.4f}")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("reward gap $\\Delta$ (log scale)")
    axes[2].set_ylabel("time to collapse (log scale)")
    axes[2].set_title("Collapse-time scaling law")
    axes[2].legend(loc="upper right", fontsize=6.8)
    panel_label(axes[2], "c")

    savefig(fig, os.path.join(FIG_DIR, "fig1_deterministic_collapse.png"))

    SUMMARY["fig1c_power_law_exponent"] = float(slope)
    SUMMARY["fig1c_power_law_constant_C"] = float(10 ** intercept)
    SUMMARY["fig1c_r2"] = float(r2)

    import csv
    with open(os.path.join(DATA_DIR, "fig1c_collapse_time_vs_delta.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["delta", "t_collapse"])
        for d, t in zip(deltas_c, t_collapse):
            w.writerow([d, t])

    return {"exponent": slope, "C": 10 ** intercept, "r2": r2, "max_drift": max_drift}


# --------------------------------------------------------------------------
# FIGURE 2 -- stochastic (Monte-Carlo) tie-breaking collapse
# --------------------------------------------------------------------------

def _run_mc_bandit(K, G, n_seeds, n_steps, h, rng, eps=1e-3, threshold=0.95):
    """Run stochastic training (alpha=0) for n_seeds independent runs on a
    K-outcome EXACT tie (all rewards = 1), starting from z=0 exactly.
    Returns: p1_traj (n_recorded, n_seeds) [only meaningful for K==2 plotting],
    winner index per seed (n_seeds,), collapse step per seed (n_seeds,)."""
    r = np.ones(K)
    z = np.zeros((n_seeds, K))
    collapse_step = np.full(n_seeds, np.nan)
    record_every = max(1, n_steps // 400)
    recorded_p = []
    for step in range(n_steps):
        ghat, _ = rhs_sampled(z, r, alpha=0.0, G=G, rng=rng, eps=eps)
        z = z + h * ghat
        p = softmax(z)
        newly = (np.max(p, axis=1) >= threshold) & np.isnan(collapse_step)
        collapse_step[newly] = step
        if step % record_every == 0:
            recorded_p.append(p.copy())
    recorded_p = np.array(recorded_p)  # (n_recorded, n_seeds, K)
    winner = np.argmax(p, axis=1)
    return recorded_p, winner, collapse_step * h  # convert steps -> "time" units


def fig2_stochastic_tiebreak():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    rng = np.random.default_rng(12345)

    # --- panel (a): spaghetti plot, K=2, exact tie, exact symmetric init --
    K, G, n_seeds, n_steps, h = 2, 16, 36, 3000, 0.5
    recorded_p, winner, collapse_t = _run_mc_bandit(K, G, n_seeds, n_steps, h, rng)
    ts = np.arange(recorded_p.shape[0]) * (n_steps // (recorded_p.shape[0])) * h
    for s in range(n_seeds):
        color = PALETTE[0] if winner[s] == 0 else PALETTE[1]
        axes[0].plot(ts, recorded_p[:, s, 0], color=color, alpha=0.55, lw=0.8)
    axes[0].axhline(0.5, color="0.6", lw=0.8, ls=":")
    axes[0].set_xlabel("training step $\\times h$")
    axes[0].set_ylabel("$p_1$ (outcome 1 probability)")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title(f"Noise-driven fan-out ($G$={G})")
    frac1 = np.mean(winner == 0)
    axes[0].text(0.5, 0.94, f"outcome 1 wins {frac1*100:.0f}% of {n_seeds} seeds",
                 transform=axes[0].transAxes, fontsize=7, va="top", ha="center",
                 color="0.15",
                 bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.92,
                           boxstyle="round,pad=0.3", linewidth=0.6))
    panel_label(axes[0], "a")
    SUMMARY["fig2a_frac_outcome1_wins"] = float(frac1)

    # --- panel (b): winner-frequency histogram, K=5, exact tie -----------
    K5 = 5
    n_seeds_hist = 200
    _, winner5, _ = _run_mc_bandit(K5, G=16, n_seeds=n_seeds_hist, n_steps=6000, h=0.5, rng=rng)
    counts = np.bincount(winner5, minlength=K5)
    freq = counts / n_seeds_hist
    err = np.sqrt((1.0 / K5) * (1 - 1.0 / K5) / n_seeds_hist)  # binomial SE under H0: uniform
    x = np.arange(K5)
    axes[1].bar(x, freq, color=PALETTE[2], width=0.6, label="observed")
    axes[1].axhline(1.0 / K5, color=PALETTE[1], lw=1.3, ls="--", label="uniform $1/K$")
    axes[1].fill_between([-0.5, K5 - 0.5], 1.0 / K5 - 1.96 * err, 1.0 / K5 + 1.96 * err,
                          color=PALETTE[1], alpha=0.15, label="95% CI under $H_0$")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"O{i+1}" for i in x])
    axes[1].set_xlabel("outcome (K=5, all rewards equal)")
    axes[1].set_ylabel("fraction of seeds that collapse here")
    axes[1].set_title(f"Collapse target is arbitrary ($n$={n_seeds_hist})")
    axes[1].legend(loc="upper right", fontsize=6.5)
    panel_label(axes[1], "b")
    SUMMARY["fig2b_K5_winner_freq"] = freq.tolist()

    import csv
    with open(os.path.join(DATA_DIR, "fig2b_winner_freq_K5.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["outcome", "frequency", "count"])
        for i in x:
            w.writerow([i, freq[i], counts[i]])

    # --- panel (c): collapse time vs group size G -------------------------
    Gs = [4, 8, 16, 32, 64]
    n_seeds_g = 80
    n_steps_g = 15000
    med_t, q1_t, q3_t = [], [], []
    for G in Gs:
        _, _, collapse_t_g = _run_mc_bandit(2, G, n_seeds_g, n_steps_g, h=0.5, rng=rng,
                                             threshold=0.98)
        valid = collapse_t_g[~np.isnan(collapse_t_g)]
        med_t.append(np.median(valid))
        q1_t.append(np.percentile(valid, 25))
        q3_t.append(np.percentile(valid, 75))
        SUMMARY.setdefault("fig2c_collapse_frac_by_G", {})[str(G)] = float(len(valid) / n_seeds_g)
    med_t, q1_t, q3_t = map(np.array, (med_t, q1_t, q3_t))
    yerr = np.vstack([med_t - q1_t, q3_t - med_t])
    axes[2].errorbar(Gs, med_t, yerr=yerr, fmt="o-", color=PALETTE[3], capsize=3,
                      label="median $\\pm$ IQR")
    logG, logT = np.log10(Gs), np.log10(med_t)
    slope_g, intercept_g, r2_g = least_squares_line(logG, logT)
    fit_g = 10 ** intercept_g * np.array(Gs, dtype=float) ** slope_g
    axes[2].plot(Gs, fit_g, "--", color=PALETTE[1], lw=1.2,
                 label=f"fit slope={slope_g:.2f} ($R^2$={r2_g:.3f})")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xticks(Gs)
    axes[2].set_xticklabels([str(g) for g in Gs])
    axes[2].xaxis.set_minor_locator(mticker.NullLocator())
    axes[2].set_xlabel("Monte-Carlo group size $G$ (log scale)")
    axes[2].set_ylabel("time to collapse (log scale)")
    axes[2].set_title("Larger $G$ delays collapse")
    axes[2].legend(loc="upper left", fontsize=6.8)
    panel_label(axes[2], "c")
    SUMMARY["fig2c_group_size_scaling_exponent"] = float(slope_g)

    savefig(fig, os.path.join(FIG_DIR, "fig2_stochastic_tiebreak.png"))

    with open(os.path.join(DATA_DIR, "fig2c_collapse_time_vs_G.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["G", "median_t", "q1_t", "q3_t"])
        for G, m, q1, q3 in zip(Gs, med_t, q1_t, q3_t):
            w.writerow([G, m, q1, q3])

    return {"frac_outcome1_wins": frac1, "K5_freq": freq, "group_scaling_exponent": slope_g}


# --------------------------------------------------------------------------
# FIGURE 3 -- Euler vs RK4 accuracy and stability on the collapse ODE
# --------------------------------------------------------------------------

def fig3_integrator_accuracy():
    """
    Note on scope: the collapse ODE z_dot = p*(r - r_bar) is a bounded,
    self-damping gradient flow (softmax keeps p in (0,1) and the RHS
    vanishes as any p_i -> 0 or 1) -- it is a monotone ascent on a LINEAR
    objective over the simplex. We checked a wide grid of reward scales and
    step sizes and confirmed it has no classical Euler oscillation/blow-up
    regime (unlike a stiff or logistic-map-type system): the RHS itself is
    globally bounded by max|r|, so no step size can make an explicit step
    "overshoot" past an equilibrium in the way that produces instability.
    So instead of manufacturing an instability that this system does not
    have, panel (b) reports the two things that ARE genuinely true and
    useful here: (i) the truncation-error regime cleanly gives the
    textbook orders 1 and 4, with a visible machine-precision round-off
    floor below it (Week-3 "approximation & error analysis" content), and
    (ii) our production step size (h=0.05, used throughout Figs. 1-2) is
    validated as visually indistinguishable from a high-precision reference.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))

    # --- panel (a): empirical order + truncation/round-off floor ----------
    r = np.array([1.2, 1.0])
    z0 = np.array([0.02, -0.02])
    T_eval = 15.0  # inside the steep transient, most sensitive to error

    from scipy.integrate import solve_ivp

    def f_scipy(t, z):
        return rhs_alpha(z[None, :], r, alpha=0.0)[0]

    sol = solve_ivp(f_scipy, [0, T_eval], z0, method="DOP853", rtol=1e-13, atol=1e-14)
    z_ref = sol.y[:, -1]
    f = lambda z: rhs_alpha(z, r, alpha=0.0)

    # step counts chosen so that n*h == T_eval EXACTLY (avoids a phase-
    # alignment artifact we found: a leftover partial step lands at a
    # different point of the steep transition and corrupts the order fit)
    ns = np.array([30, 50, 60, 100, 150, 200, 300, 600, 1200, 2400, 6000])
    hs = T_eval / ns
    err_euler, err_rk4 = [], []
    for n, h in zip(ns, hs):
        _, tr_e = integrate(f, z0[None, :], h, int(n), method="euler", record_every=int(n))
        _, tr_r = integrate(f, z0[None, :], h, int(n), method="rk4", record_every=int(n))
        err_euler.append(np.linalg.norm(tr_e[-1, 0, :] - z_ref))
        err_rk4.append(np.linalg.norm(tr_r[-1, 0, :] - z_ref))
    err_euler, err_rk4 = np.array(err_euler), np.array(err_rk4)

    # fit the order ONLY on the truncation-dominated regime (RK4 floors out
    # at machine precision, ~1e-14, once h is small enough -- fitting past
    # that would measure noise, not the scheme's order)
    clean = err_rk4 > 1e-11
    slope_e, _, r2_e = least_squares_line(np.log10(hs), np.log10(err_euler))
    slope_r, _, r2_r = least_squares_line(np.log10(hs[clean]), np.log10(err_rk4[clean]))

    axes[0].loglog(hs, err_euler, "o-", ms=4, color=PALETTE[1],
                    label=f"Euler (order$\\approx${slope_e:.2f})")
    axes[0].loglog(hs, err_rk4, "s-", ms=4, color=PALETTE[0],
                    label=f"RK4 (order$\\approx${slope_r:.2f} above floor)")
    axes[0].axhspan(1e-16, 1e-11, color="0.5", alpha=0.12)
    axes[0].text(hs[-1], 1.3e-15, "round-off floor", fontsize=6.3, color="0.35",
                 ha="right", va="bottom")
    axes[0].set_xlabel("step size $h$ (log scale)")
    axes[0].set_ylabel(f"global error at $t$={T_eval:.0f} (log scale)")
    axes[0].set_title("Truncation error $\\rightarrow$ round-off floor")
    axes[0].legend(loc="upper left", fontsize=6.8)
    panel_label(axes[0], "a")
    SUMMARY["fig3a_euler_order"] = float(slope_e)
    SUMMARY["fig3a_rk4_order"] = float(slope_r)
    SUMMARY["fig3a_note"] = ("order fitted only where rk4_err > 1e-11; below that the "
                              "curve is flat machine round-off, not truncation error")

    # --- panel (b): validate the production step size (h=0.05) ------------
    r_b = np.array([1.0 + 0.25, 1.0])
    z0_b = np.array([0.05, -0.05])
    T_b = 60.0

    def f_scipy_b(t, z):
        return rhs_alpha(z[None, :], r_b, alpha=0.0)[0]

    tt = np.linspace(0, T_b, 4000)
    sol_b = solve_ivp(f_scipy_b, [0, T_b], z0_b, method="DOP853",
                       rtol=1e-12, atol=1e-13, dense_output=True)
    p_true = softmax(sol_b.sol(tt).T)[:, 0]

    f_b = lambda z: rhs_alpha(z, r_b, alpha=0.0)
    h_prod = 0.05
    n_prod = int(T_b / h_prod)
    _, tr_e = integrate(f_b, z0_b[None, :], h_prod, n_prod, method="euler")
    _, tr_r = integrate(f_b, z0_b[None, :], h_prod, n_prod, method="rk4")
    ts_prod = np.arange(n_prod + 1) * h_prod
    p_e = softmax(tr_e[:, 0, :])[:, 0]
    p_r = softmax(tr_r[:, 0, :])[:, 0]

    axes[1].plot(tt, p_true, "-", color="0.2", lw=2.2, alpha=0.35, label="reference (DOP853)")
    axes[1].plot(ts_prod, p_r, "--", color=PALETTE[0], lw=1.3, label=f"RK4, $h$={h_prod}")
    axes[1].plot(ts_prod, p_e, ":", color=PALETTE[1], lw=1.3, label=f"Euler, $h$={h_prod}")
    axes[1].set_xlabel("time $t$")
    axes[1].set_ylabel("$p_1(t)$")
    axes[1].set_title("Production step size ($h$=0.05) validated")
    axes[1].legend(loc="lower right", fontsize=6.8)
    panel_label(axes[1], "b")

    p_true_at_prod = np.interp(ts_prod, tt, p_true)
    max_err_rk4 = float(np.max(np.abs(p_r - p_true_at_prod)))
    max_err_euler = float(np.max(np.abs(p_e - p_true_at_prod)))
    axes[1].text(0.03, 0.55,
                 f"max$|p-p_{{true}}|$\nRK4: {max_err_rk4:.1e}\nEuler: {max_err_euler:.1e}",
                 transform=axes[1].transAxes, fontsize=6.5, color="0.3", va="top")
    SUMMARY["fig3b_max_err_rk4_h0.05"] = max_err_rk4
    SUMMARY["fig3b_max_err_euler_h0.05"] = max_err_euler

    savefig(fig, os.path.join(FIG_DIR, "fig3_integrator_accuracy.png"))

    import csv
    with open(os.path.join(DATA_DIR, "fig3a_error_vs_h.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["h", "euler_error", "rk4_error"])
        for h, ee, er in zip(hs, err_euler, err_rk4):
            w.writerow([h, ee, er])

    return {"euler_order": slope_e, "rk4_order": slope_r,
            "max_err_rk4": max_err_rk4, "max_err_euler": max_err_euler}


# --------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EXPERIMENT 1: outcome-level mode collapse under alpha=0 (standard RL)")
    print("=" * 70)

    print("\n[Figure 1] Deterministic collapse + reward-gap scaling law ...")
    r1 = fig1_deterministic_collapse()
    print(f"  max drift under exact tie (T=300, RK4): {r1['max_drift']:.2e}")
    print(f"  t_collapse ~ C * delta^k :  k = {r1['exponent']:.3f},  "
          f"C = {r1['C']:.2f},  R^2 = {r1['r2']:.5f}")

    print("\n[Figure 2] Monte-Carlo noise-driven tie-breaking collapse ...")
    r2_ = fig2_stochastic_tiebreak()
    print(f"  K=2 exact-tie MC run: outcome-1 wins {r2_['frac_outcome1_wins']*100:.1f}% "
          f"of seeds (expect ~50%)")
    print(f"  K=5 exact-tie MC run winner frequencies: "
          + ", ".join(f"{f*100:.1f}%" for f in r2_['K5_freq']) + "  (expect ~20% each)")
    print(f"  collapse-time vs group-size G power-law exponent: "
          f"{r2_['group_scaling_exponent']:.3f}")

    print("\n[Figure 3] Euler vs RK4 accuracy validation ...")
    r3 = fig3_integrator_accuracy()
    print(f"  empirical order -- Euler: {r3['euler_order']:.2f} (theory 1), "
          f"RK4: {r3['rk4_order']:.2f} (theory 4, fit above round-off floor)")
    print(f"  production step size h=0.05 max abs error vs true solution -- "
          f"RK4: {r3['max_err_rk4']:.1e}, Euler: {r3['max_err_euler']:.1e}")

    with open(os.path.join(DATA_DIR, "exp1_summary.json"), "w") as fp:
        json.dump(SUMMARY, fp, indent=2, default=float)

    print(f"\nAll figures written to results/{EXP_NAME}/figures/, "
          f"data to results/{EXP_NAME}/data/.")
    print(f"Summary JSON: results/{EXP_NAME}/data/exp1_summary.json")


if __name__ == "__main__":
    main()
