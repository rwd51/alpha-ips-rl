"""
EXPERIMENT 5 -- Estimator error: bias, variance, and what they cost downstream.

Experiments 1-4 assumed one particular way of turning a group of G rollouts into
the inverse-probability weight the alpha-IPS update needs: the base paper's
clipped frequency, max(p_hat, eps)^-alpha. This experiment is the project
pitch's Section 4.4 -- an approximation-and-error-analysis study of that step
itself -- and asks how eps-clipping, Laplace smoothing, Richardson extrapolation
and a moving average trade bias against variance, and which of those trades the
learning dynamics actually notice. The theory is in `derivation_exp5.md`.

  1. FIGURE 1 -- How far is p_hat from p, and what that does to the weight.
     (a) Exact relative bias / sd / RMSE of max(p_hat, eps)^-alpha against p,
         with the delta-method (truncation-error) asymptotes and a Monte-Carlo
         validation using the same inverse-transform sampler as Exps. 1-3.
     (b) The expansion parameter is G p, not G: rescaled bias collapses onto
         alpha(alpha+1)/2, and the empty-group term (1-p)^G eps^-alpha explains
         exactly where the expansion fails. The validity threshold in G p is
         located by bisection.
     (c) The course's total-error curve: MSE(eps) falls (tail truncation) then
         rises (systematic under-weighting). Golden-section search finds the
         minimum; the empirical law is eps* ~ p, so no single clip is right for
         both a majority and a minority outcome.

  2. FIGURE 2 -- Three variance-control devices on the bias-variance plane.
     (a) |relative bias| against relative sd as each knob sweeps (clip eps,
         add-lambda, EMA beta), with iso-RMSE contours.
     (b) Richardson extrapolation in 1/G raises the bias order from 1 to 2 at
         no leading-order variance cost (least-squares fitted orders).
     (c) ... but the naive version returns NEGATIVE weights for rare outcomes,
         with the exact probability computed here; the guard removes them and
         restores the paper's singleton weight exactly.

  3. FIGURE 3 -- Downstream: which trades the policy notices.
     (a) The effective weight w_G(p) of each rule against the ideal p^-alpha,
         with the dynamic range omega(1)/omega(G) that decides survival.
     (b) K=2, r=(4,1): mean-field minority mass against G for each rule, and
         the extinction exponent alpha_c by bisection.
     (c) K=5: Monte-Carlo training under each rule against its mean-field
         prediction and the ideal p* ~ r^(1/alpha).

  4. FIGURE 4 -- Moving averages: the device with a memory.
     (a) Variance reduction beta/(2-beta), i.e. an effective group size
         G (2-beta)/beta bought with no extra rollouts, and no steady-state bias.
     (b) The price: a lag (1-beta)/beta per unit of policy drift. Total error
         has an interior optimum, located by golden-section search.
     (c) The dynamics become second order, s^2 + kappa s + kappa lambda = 0:
         training rings for kappa < 4 lambda and converges fastest, at twice
         Experiment 2's rate, at kappa = 4 lambda.

Outputs: results/exp5_bias_variance/figures/fig*.png (300 dpi) and
results/exp5_bias_variance/data/*.csv, exp5_summary.json.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dynamics import (inverse_transform_sample_batch, linear_rates,
                          rhs_sampled, softmax, stationary_p)
from src.eigen import deflated_power_method, power_method, qr_algorithm
from src.ema import (coupled_jacobian, coupled_rhs, critical_kappa, decay_rate,
                     effective_group_size, frequency_mse, is_oscillatory,
                     lag_bias, mode_roots, variance_factor)
from src.estimators import (AddLambdaRule, ClipRule, IdealRule, OffsetRule,
                            RichardsonRule, delta_relative_bias,
                            delta_relative_sd, effective_weight_alpha1,
                            empty_group_bias, sampled_step)
from src.finite_group_general import effective_weight as clip_effective_weight
from src.fitting import least_squares_line
from src.integrators import integrate
from src.meanfield_rule import (check_monotone, extinction_alpha_K2, stationary,
                                stationary_K2)
from src.optimize1d import golden_section_log, grid_minimum, is_unimodal
from src.plotstyle import PALETTE, apply_style, panel_label, savefig
from src.rootfinding import bisection


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP_NAME = "exp5_bias_variance"
FIG_DIR = os.path.join(ROOT, "results", EXP_NAME, "figures")
DATA_DIR = os.path.join(ROOT, "results", EXP_NAME, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

apply_style()

SUMMARY: dict[str, object] = {}
ALPHA = 1.0                      # the paper's IPS exponent, unless stated
EPS = 1e-3                       # the paper's clipping threshold
R41 = np.array([4.0, 1.0])
R5 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
INK = "0.25"
SEEDS = {"fig1": 5101, "fig2": 5102, "fig3": 5103, "fig4": 5104, "fig5": 5105}


def _write_csv(name, header, rows):
    with open(os.path.join(DATA_DIR, name), "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerows(rows)


def _log_ticks(ax, axis, ticks):
    setter = ax.set_xticks if axis == "x" else ax.set_yticks
    labeller = ax.set_xticklabels if axis == "x" else ax.set_yticklabels
    setter(ticks)
    labeller([f"{t:g}" for t in ticks])
    minor = ax.xaxis if axis == "x" else ax.yaxis
    minor.set_minor_formatter(mticker.NullFormatter())


# --------------------------------------------------------------------------
# Monte-Carlo validation of the exact binomial moments
# --------------------------------------------------------------------------

def mc_weight_moments(rule, p, G, n_groups, rng, shift=1.0, chunk=20000, K=2):
    """
    Monte-Carlo estimate of (E[W_hat], Var(W_hat)) in units of p^-alpha, using
    the same inverse-transform categorical sampler as Experiments 1-3 (so this
    validates the simulator as well as the algebra).

    A K=2 policy (p, 1-p) is enough: the statistics of outcome i depend only on
    its marginal count, which is Binomial(G, p) whatever the other outcomes do.
    Sums are accumulated around `shift` (the exact relative mean) so the
    variance is not formed as a difference of two nearly equal large numbers.

    Returns (relative_mean, relative_variance, standard_error_of_the_mean).
    """
    p = float(p)
    scale = p ** rule.alpha
    row = np.array([p, 1.0 - p])
    n_done, s1, s2 = 0, 0.0, 0.0
    while n_done < n_groups:
        m = min(chunk, n_groups - n_done)
        idx = inverse_transform_sample_batch(np.tile(row, (m, 1)), G, rng)
        v = rule.weights_from_samples(idx, K)[:, 0] * scale - shift
        s1 += float(v.sum())
        s2 += float((v * v).sum())
        n_done += m
    mean_d = s1 / n_done
    var = s2 / n_done - mean_d ** 2
    return shift + mean_d, var, np.sqrt(max(var, 0.0) / n_done)


# --------------------------------------------------------------------------
# FIGURE 1 -- the estimator error budget
# --------------------------------------------------------------------------

def fig1_error_budget():
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    rng = np.random.default_rng(SEEDS["fig1"])
    out = {}

    # --- panel (a): exact bias / sd / rmse vs p, with MC validation --------
    G = 16
    rule = ClipRule(ALPHA, EPS)
    p_dense = np.geomspace(1e-3, 0.999, 400)
    tab = rule.error_table(p_dense, G)
    # The clip makes the bias negative for p below roughly eps (the estimate is
    # floored at eps^-alpha, which under-weights an even rarer outcome); above
    # that, convexity of x^-alpha makes it positive. Locate the crossing.
    bias_of = lambda x: float(rule.error_table(np.array([x]), G)["rel_bias"][0])
    p_zero = None
    if bias_of(1e-3) < 0.0 < bias_of(1e-2):
        p_zero, _ = bisection(bias_of, 1e-3, 1e-2, tol=1e-14)
    if np.any(tab["rel_bias"][p_dense > 2e-2] <= 0.0):
        raise AssertionError("relative bias changed sign where it should be positive")

    axes[0].plot(p_dense, tab["rel_rmse"], color=PALETTE[0], label="relative RMSE")
    axes[0].plot(p_dense, tab["rel_sd"], color=PALETTE[1], label="relative sd")
    axes[0].plot(p_dense, np.abs(tab["rel_bias"]), color=PALETTE[2], label="|relative bias|")
    if p_zero is not None:
        neg = p_dense < p_zero
        axes[0].plot(p_dense[neg], np.abs(tab["rel_bias"][neg]), color=PALETTE[2], lw=2.8,
                     alpha=0.35, solid_capstyle="butt")
        axes[0].axvline(p_zero, color=PALETTE[2], lw=0.7, ls=":")
        axes[0].text(p_zero * 1.2, 3.0, r"bias $<0$ below $p\approx\epsilon$",
                     fontsize=5.8, color=PALETTE[2], rotation=90, va="center")
        out["bias_sign_change_p"] = float(p_zero)
    axes[0].plot(p_dense, delta_relative_sd(p_dense, G, ALPHA), ":", color=PALETTE[1], lw=1.0,
                 label=r"delta method $\alpha\sqrt{(1-p)/Gp}$")
    axes[0].plot(p_dense, delta_relative_bias(p_dense, G, ALPHA), ":", color=PALETTE[2], lw=1.0,
                 label=r"delta method $\frac{\alpha(\alpha+1)(1-p)}{2Gp}$")

    p_mc = np.geomspace(0.02, 0.9, 10)
    n_groups = 400_000
    rows_a, worst_bias_z, worst_var_z, worst_sd_rel = [], 0.0, 0.0, 0.0
    for p in p_mc:
        m1x, varx = rule.moments_relative(np.array([p]), G)
        m1x, varx = float(m1x[0]), float(varx[0])
        mu4 = float(rule.relative_central_moment(np.array([p]), G, 4)[0])
        m1m, varm, se = mc_weight_moments(rule, p, G, n_groups, rng, shift=m1x)
        z = abs(m1m - m1x) / se if se > 0 else 0.0
        # exact standard error of a Monte-Carlo variance estimate
        se_var = np.sqrt(max(mu4 - varx ** 2, 0.0) / n_groups)
        z_var = abs(varm - varx) / se_var if se_var > 0 else 0.0
        # how much of the exact variance is carried by the single empty-group draw?
        p0 = (1.0 - p) ** G
        v0 = EPS ** (-ALPHA) * p ** ALPHA
        share0 = p0 * (v0 - m1x) ** 2 / varx
        sd_rel = abs(np.sqrt(varm) - np.sqrt(varx)) / np.sqrt(varx)
        worst_bias_z = max(worst_bias_z, z)
        worst_var_z = max(worst_var_z, z_var)
        worst_sd_rel = max(worst_sd_rel, sd_rel)
        rows_a.append([G, p, m1x - 1.0, m1m - 1.0, se, np.sqrt(varx), np.sqrt(varm), z,
                       share0, n_groups * p0, sd_rel, z_var,
                       se_var / (2.0 * np.sqrt(varx)) / np.sqrt(varx)])
    # a MC sd is only meaningful where its own (exact) statistical error is small
    well = np.array([r[12] < 0.05 for r in rows_a])
    axes[0].scatter(p_mc, [abs(r[3]) for r in rows_a], s=16, marker="o", facecolor="white",
                    edgecolor=PALETTE[2], lw=0.9, zorder=3)
    axes[0].scatter(p_mc[well], [r[6] for r, w in zip(rows_a, well) if w], s=16, marker="s",
                    facecolor="white", edgecolor=PALETTE[1], lw=0.9, zorder=3)
    axes[0].scatter(p_mc[~well], [r[6] for r, w in zip(rows_a, well) if not w], s=16,
                    marker="s", facecolor="white", edgecolor="0.65", lw=0.9, zorder=3)
    axes[0].scatter([], [], s=16, marker="o", facecolor="white", edgecolor="0.3", lw=0.9,
                    label=f"Monte Carlo ({n_groups // 1000}k groups)")
    axes[0].scatter([], [], s=16, marker="s", facecolor="white", edgecolor="0.65", lw=0.9,
                    label="MC sd, rare-event dominated")
    axes[0].axvline(1.0 / G, color="0.6", lw=0.8, ls="--")
    axes[0].text(1.0 / G * 1.1, 4e-3, "$p=1/G$", fontsize=6.3, color="0.4")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylim(3e-3, 2e2)
    axes[0].set_xlabel("outcome probability $p$")
    axes[0].set_ylabel(r"error relative to $p^{-\alpha}$")
    axes[0].set_title(rf"Weight estimate, $G$={G}, $\alpha$={ALPHA:g}, $\epsilon$={EPS:g}")
    axes[0].legend(loc="lower left", fontsize=5.8)
    panel_label(axes[0], "a")
    _write_csv("fig1a_weight_error_vs_p.csv",
               ["G", "p", "rel_bias_exact", "rel_bias_mc", "mc_sem",
                "rel_sd_exact", "rel_sd_mc", "bias_z_score", "variance_share_empty_group",
                "expected_empty_groups", "sd_rel_error", "variance_z_score",
                "predicted_sd_rel_error"], rows_a)
    out["mc_worst_bias_z"] = float(worst_bias_z)
    out["mc_worst_variance_z"] = float(worst_var_z)
    out["mc_worst_sd_rel_error"] = float(worst_sd_rel)
    out["mc_worst_sd_rel_error_where_predicted_small"] = float(
        max([r[10] for r, w in zip(rows_a, well) if w], default=0.0))
    out["max_variance_share_of_empty_group"] = float(max(r[8] for r in rows_a))

    # --- panel (b): the expansion parameter is G p ------------------------
    Gs_b = [8, 16, 32, 64, 128, 256]
    rows_b = []
    for j, G in enumerate(Gs_b):
        p_b = np.geomspace(0.5 / G, 0.95, 220)
        b = ClipRule(ALPHA, EPS).error_table(p_b, G)["rel_bias"]
        scaled = b * G * p_b / (1.0 - p_b)
        axes[1].plot(G * p_b, scaled, color=PALETTE[j % len(PALETTE)], lw=1.2,
                     label=f"$G$={G}")
        for pp, ss in zip(p_b[::30], scaled[::30]):
            rows_b.append([G, pp, G * pp, ss])
    axes[1].axhline(ALPHA * (ALPHA + 1) / 2.0, color="k", lw=1.0, ls="--",
                    label=r"$\alpha(\alpha+1)/2$")

    # where does the asymptote become valid? bisection on the 10% level
    def rel_excess(gp, G):
        p = gp / G
        b = float(ClipRule(ALPHA, EPS).error_table(np.array([p]), G)["rel_bias"][0])
        return b / delta_relative_bias(p, G, ALPHA) - 1.1

    thresholds, threshold_status = {}, {}
    for G in Gs_b:
        lo, hi = 1.0, 0.9 * G
        f_lo, f_hi = rel_excess(lo, G), rel_excess(hi, G)
        if f_lo * f_hi < 0:
            gp_c, _ = bisection(lambda x: rel_excess(x, G), lo, hi, tol=1e-9)
            thresholds[G] = float(gp_c)
            threshold_status[G] = "found"
        elif f_hi > 0:
            # even the largest p at this group size is more than 10% off: the
            # (Gp)^-2 term is still ~20% at G = 8, p = 0.9
            threshold_status[G] = f"never within 10% (ratio {f_hi + 1.1:.3f} at Gp={hi:g})"
        else:
            # already inside 10% at Gp = 1, because there p ~ eps and the clip's
            # negative bias partly cancels the positive convexity bias
            threshold_status[G] = f"already within 10% at Gp=1 (ratio {f_lo + 1.1:.3f})"
    if thresholds:
        gp_line = float(np.max(list(thresholds.values())))
        axes[1].axvline(gp_line, color=PALETTE[1], lw=0.9, ls=":")
        axes[1].text(gp_line * 1.12, 20.0, f"10% accurate\nabove $Gp$={gp_line:.1f}",
                     fontsize=6.0, color="0.35")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.2, 2e2)
    axes[1].set_xlabel(r"expected hits $Gp$")
    axes[1].set_ylabel(r"$\mathrm{rel.\ bias}\times Gp/(1-p)$")
    axes[1].set_title("One expansion parameter, not two")
    axes[1].legend(loc="upper right", fontsize=6.0, ncol=2)
    panel_label(axes[1], "b")
    _write_csv("fig1b_bias_collapse.csv", ["G", "p", "Gp", "scaled_rel_bias"], rows_b)
    out["delta_validity_threshold_Gp"] = thresholds
    out["delta_validity_threshold_status"] = threshold_status

    # below the threshold the delta method alone fails by an order of magnitude,
    # and the single empty-group term (5.3) accounts for the rest
    breakdown = {}
    for p, G in [(0.2, 16), (0.1, 16), (0.3, 16), (0.05, 32)]:
        exact = float(ClipRule(ALPHA, EPS).error_table(np.array([p]), G)["rel_bias"][0])
        delta = float(delta_relative_bias(p, G, ALPHA))
        both = delta + float(empty_group_bias(p, G, ALPHA, EPS))
        breakdown[f"p={p:g}, G={G}"] = {"Gp": p * G, "exact": exact,
                                        "delta_only_ratio": exact / delta,
                                        "delta_plus_empty_group_ratio": exact / both}
    out["empty_group_explains_the_breakdown"] = breakdown

    # --- panel (c): total error vs eps, golden-section minimum ------------
    G = 16
    p_panel = [0.05, 0.2, 0.5]
    eps_dense = np.geomspace(1e-4, 0.95, 260)
    rows_c = []
    for j, p in enumerate(p_panel):
        mse = np.array([ClipRule(ALPHA, e).error_table(np.array([p]), G)["rel_mse"][0]
                        for e in eps_dense])
        axes[2].plot(eps_dense, mse, color=PALETTE[j], label=f"$p$={p:g}")
        objective = lambda e: float(ClipRule(ALPHA, e).error_table(np.array([p]), G)["rel_mse"][0])
        e_star, f_star, n_it, n_ev = golden_section_log(objective, 1e-4, 0.95, rel_tol=1e-12)
        e_grid, f_grid, vals, _ = grid_minimum(objective, 1e-4, 0.95, n=1201, geometric=True)
        axes[2].scatter([e_star], [f_star], s=26, marker="*", color=PALETTE[j], zorder=4)
        rows_c.append([G, p, e_star, f_star, e_grid, f_grid, n_it, n_ev,
                       bool(is_unimodal(vals, slack=0)), e_star / p])
    axes[2].scatter([], [], s=26, marker="*", color="0.3",
                    label="golden-section minimum")
    # the bias / variance split at p = 0.2
    p_split = 0.2
    split = [ClipRule(ALPHA, e).error_table(np.array([p_split]), G) for e in eps_dense]
    axes[2].plot(eps_dense, [s["rel_bias"][0] ** 2 for s in split], "--", lw=0.9,
                 color=PALETTE[1], label=r"bias$^2$, $p$=0.2")
    axes[2].plot(eps_dense, [s["rel_mse"][0] - s["rel_bias"][0] ** 2 for s in split], ":",
                 lw=0.9, color=PALETTE[1], label=r"variance, $p$=0.2")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-3, 3e5)
    axes[2].set_xlabel(r"clipping threshold $\epsilon$")
    axes[2].set_ylabel(r"relative MSE of $\widehat W$")
    axes[2].set_title(rf"Total-error curve, $G$={G}")
    axes[2].legend(loc="lower left", fontsize=5.8)
    panel_label(axes[2], "c")

    # the eps* ~ p law over a (G, p) grid
    law_rows, ratios, slopes = [], [], {}
    for G in [8, 16, 32, 64, 128]:
        ps = np.geomspace(0.01, 0.5, 9)
        stars = []
        for p in ps:
            obj = lambda e: float(ClipRule(ALPHA, e).error_table(np.array([p]), G)["rel_mse"][0])
            e_star, f_star, _, _ = golden_section_log(obj, 1e-5, 0.95, rel_tol=1e-12)
            stars.append(e_star)
            ratios.append(e_star / p)
            law_rows.append([G, p, e_star, e_star / p, f_star])
        slope, intercept, r2 = least_squares_line(np.log(ps), np.log(stars))
        slopes[G] = {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}
    _write_csv("fig1c_optimal_eps.csv",
               ["G", "p", "eps_star_golden", "mse_star", "eps_star_grid", "mse_grid",
                "golden_iters", "golden_evals", "grid_unimodal", "eps_star_over_p"], rows_c)
    _write_csv("fig1c_optimal_eps_law.csv",
               ["G", "p", "eps_star", "eps_star_over_p", "rel_mse_star"], law_rows)
    out["eps_star_law"] = slopes
    out["eps_star_over_p"] = {"min": float(np.min(ratios)), "max": float(np.max(ratios)),
                              "median": float(np.median(ratios))}
    out["golden_vs_grid_max_rel_diff"] = float(
        max(abs(r[2] - r[4]) / r[4] for r in rows_c))
    out["all_objectives_unimodal"] = bool(all(r[8] for r in rows_c))
    if not out["all_objectives_unimodal"]:
        raise AssertionError("MSE(eps) was not unimodal on the scanned bracket, so "
                             "golden-section search is not justified there")
    out["panel_c_minima"] = {f"p={r[1]:g}": {"eps_star": r[2], "rel_mse": r[3],
                                             "eps_star_over_p": r[9]} for r in rows_c}

    savefig(fig, os.path.join(FIG_DIR, "fig1_error_budget.png"))
    SUMMARY["figure1"] = out
    return out


# --------------------------------------------------------------------------
# FIGURE 2 -- the three devices compared
# --------------------------------------------------------------------------

def fig2_device_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    rng = np.random.default_rng(SEEDS["fig2"])
    out = {}

    # --- panel (a): the bias-variance plane -------------------------------
    G, p = 16, 0.1
    rows_a = []
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")

    def point(rule):
        t = rule.error_table(np.array([p]), G)
        return abs(float(t["rel_bias"][0])), float(t["rel_sd"][0])

    eps_sweep = np.geomspace(1e-4, 0.9, 60)
    xy = np.array([point(ClipRule(ALPHA, e)) for e in eps_sweep])
    axes[0].plot(xy[:, 0], xy[:, 1], color=PALETTE[0], label=r"clip, $\epsilon$ sweep")
    for e in [1e-3, 0.01, 0.1, 0.2]:
        b, s = point(ClipRule(ALPHA, e))
        axes[0].scatter([b], [s], s=18, color=PALETTE[0], zorder=3)
        axes[0].annotate(rf"$\epsilon$={e:g}", (b, s), xytext=(3, -7),
                         textcoords="offset points", fontsize=5.6, color=PALETTE[0])
        rows_a.append(["clip", e, b, s, np.hypot(b, s)])

    lam_sweep = np.geomspace(1e-3, 30.0, 60)
    xy = np.array([point(AddLambdaRule(ALPHA, l, 2)) for l in lam_sweep])
    axes[0].plot(xy[:, 0], xy[:, 1], color=PALETTE[1], label=r"add-$\lambda$, $\lambda$ sweep")
    for l in [0.5, 1.0, 5.0]:
        b, s = point(AddLambdaRule(ALPHA, l, 2))
        axes[0].scatter([b], [s], s=18, color=PALETTE[1], zorder=3)
        axes[0].annotate(rf"$\lambda$={l:g}", (b, s), xytext=(3, 4),
                         textcoords="offset points", fontsize=5.6, color=PALETTE[1])
        rows_a.append(["add_lambda", l, b, s, np.hypot(b, s)])

    # EMA: p_bar is a weighted sum of binomials, so its moments are estimated by
    # simulating the recursion; the dashed curve is the Binomial(G_eff) proxy.
    betas = np.array([1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02])
    ema_pts, proxy_pts, rows_ema = [], [], []
    for beta in betas:
        m, v = _ema_weight_moments(p, G, beta, ALPHA, EPS, rng, n_chains=40000,
                                   n_burn=int(20.0 / beta))
        ema_pts.append((abs(m - 1.0), np.sqrt(max(v, 0.0))))
        G_eff = effective_group_size(G, beta)
        b_pr, s_pr = _proxy_moments(p, G_eff, ALPHA, EPS)
        proxy_pts.append((abs(b_pr), s_pr))
        rows_ema.append([beta, G_eff, m - 1.0, np.sqrt(max(v, 0.0)), b_pr, s_pr])
    ema_pts = np.array(ema_pts)
    proxy_pts = np.array(proxy_pts)
    axes[0].plot(ema_pts[:, 0], ema_pts[:, 1], "-o", ms=3, color=PALETTE[2],
                 label=r"EMA, $\beta$ sweep (MC)")
    axes[0].plot(proxy_pts[:, 0], proxy_pts[:, 1], "--", lw=0.9, color=PALETTE[2],
                 label=r"EMA $\mathrm{Binomial}(G_{\mathrm{eff}})$ proxy")
    for rule, colour, marker in [(RichardsonRule(ALPHA, EPS, guard=1), PALETTE[3], "D"),
                                 (RichardsonRule(ALPHA, EPS, guard=None), PALETTE[4], "X")]:
        b, s = point(rule)
        axes[0].scatter([b], [s], s=34, marker=marker, color=colour, zorder=4,
                        label=rule.label)
        rows_a.append([rule.name, np.nan, b, s, np.hypot(b, s)])

    # axis limits from the data, then iso-RMSE arcs sqrt(bias^2 + sd^2) = const
    all_x = np.array([r[2] for r in rows_a] + list(ema_pts[:, 0]) + list(proxy_pts[:, 0]))
    all_y = np.array([r[3] for r in rows_a] + list(ema_pts[:, 1]) + list(proxy_pts[:, 1]))
    xlim = (10 ** np.floor(np.log10(all_x.min())), 10 ** np.ceil(np.log10(all_x.max())))
    ylim = (10 ** np.floor(np.log10(all_y.min())), 10 ** np.ceil(np.log10(all_y.max())))
    for level in 10.0 ** np.arange(-1, 4):
        gx = np.geomspace(xlim[0], min(level * 0.9999, xlim[1]), 300)
        gy = np.sqrt(np.maximum(level ** 2 - gx ** 2, 0.0))
        keep = (gy >= ylim[0]) & (gy <= ylim[1])
        if keep.any():
            axes[0].plot(gx[keep], gy[keep], color="0.82", lw=0.6, ls=":", zorder=0)
            axes[0].annotate(f"RMSE={level:g}", (gx[keep][0], gy[keep][0]), xytext=(1, 1),
                             textcoords="offset points", fontsize=5.0, color="0.62")
    axes[0].set_xlim(*xlim)
    axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("|relative bias|")
    axes[0].set_ylabel("relative standard deviation")
    axes[0].set_title(rf"Bias-variance plane, $G$={G}, $p$={p:g}")
    axes[0].legend(loc="lower left", fontsize=5.6)
    panel_label(axes[0], "a")
    _write_csv("fig2a_bias_variance_plane.csv",
               ["rule", "knob", "abs_rel_bias", "rel_sd", "rel_rmse"], rows_a)
    _write_csv("fig2a_ema_points.csv",
               ["beta", "G_eff", "rel_bias_mc", "rel_sd_mc", "rel_bias_proxy", "rel_sd_proxy"],
               rows_ema)
    prox_err = np.abs(ema_pts[:, 0] - proxy_pts[:, 0]) / np.maximum(proxy_pts[:, 0], 1e-300)
    slow = betas <= 0.2
    out["ema_proxy_max_bias_rel_error"] = float(np.max(prox_err))
    out["ema_proxy_max_bias_rel_error_beta_le_0.2"] = float(np.max(prox_err[slow]))
    out["ema_rmse_at_smallest_beta"] = {"beta": float(betas[-1]),
                                        "rel_rmse": float(np.hypot(*ema_pts[-1]))}
    out["clip_rmse_reference"] = {"eps": 1e-3,
                                  "rel_rmse": float(np.hypot(*point(ClipRule(ALPHA, EPS))))}

    # --- panel (b): Richardson raises the bias order ----------------------
    cases = [(0.3, [64, 128, 256, 512, 1024, 2048]),
             (0.1, [256, 512, 1024, 2048])]
    rows_b, orders = [], {}
    for j, (p_b, Gs) in enumerate(cases):
        for k, (rule_of, tag, ls, marker) in enumerate(
                [(lambda G: ClipRule(ALPHA, EPS), "plain", "-", "o"),
                 (lambda G: RichardsonRule(ALPHA, EPS, guard=1), "guarded Richardson", "--", "s")]):
            bias, sd = [], []
            for G in Gs:
                t = rule_of(G).error_table(np.array([p_b]), G)
                bias.append(abs(float(t["rel_bias"][0])))
                sd.append(float(t["rel_sd"][0]))
            slope, _, r2 = least_squares_line(np.log(Gs), np.log(bias))
            orders[f"p={p_b:g}, {tag}"] = {"order": float(-slope), "r2": float(r2)}
            colour = PALETTE[j * 2 + k]
            axes[1].loglog(Gs, bias, ls, marker=marker, ms=3.4, color=colour,
                           label=f"$p$={p_b:g}, {tag}: order {-slope:.2f}")
            for G, b, s in zip(Gs, bias, sd):
                rows_b.append([p_b, tag, G, b, s])
    # variance ratio, the other half of the claim
    ratios = []
    for p_b, Gs in cases:
        for G in Gs:
            s_plain = float(ClipRule(ALPHA, EPS).error_table(np.array([p_b]), G)["rel_sd"][0])
            s_rich = float(RichardsonRule(ALPHA, EPS, guard=1)
                           .error_table(np.array([p_b]), G)["rel_sd"][0])
            ratios.append(s_rich / s_plain)
    axes[1].text(0.03, 0.05, f"sd ratio (Richardson / plain)\n"
                 f"{min(ratios):.3f} to {max(ratios):.3f}",
                 transform=axes[1].transAxes, fontsize=6.2, color=INK)
    axes[1].set_xlabel("group size $G$")
    axes[1].set_ylabel("|relative bias|")
    axes[1].set_title("Richardson extrapolation in $1/G$")
    axes[1].legend(loc="upper right", fontsize=5.9)
    panel_label(axes[1], "b")
    _write_csv("fig2b_richardson_order.csv", ["p", "rule", "G", "abs_rel_bias", "rel_sd"], rows_b)
    out["richardson_orders"] = orders
    out["richardson_sd_ratio"] = {"min": float(min(ratios)), "max": float(max(ratios))}

    # --- panel (c): the naive rule's negative weights ---------------------
    rows_c = []
    for j, G in enumerate([8, 16, 32, 64, 128]):
        naive = RichardsonRule(ALPHA, EPS, guard=None)
        p_c = np.geomspace(1e-3, 0.95, 200)
        q = naive.negative_weight_probability(p_c, G)
        axes[2].plot(p_c, np.maximum(q, 1e-12), color=PALETTE[j % len(PALETTE)],
                     label=f"$G$={G}")
        guarded = RichardsonRule(ALPHA, EPS, guard=1)
        q_guard = float(np.max(guarded.negative_weight_probability(p_c, G)))
        for pp, qq in zip(p_c[::25], q[::25]):
            rows_c.append([G, pp, qq, q_guard])
        out.setdefault("naive_singleton_weight", {})[str(G)] = float(naive.values(G)[1, 0])
        out.setdefault("guarded_max_negative_probability", {})[str(G)] = q_guard
        out.setdefault("dynamic_range", {})[str(G)] = {
            "clip": ClipRule(ALPHA, EPS).dynamic_range(G),
            "add_lambda_1": AddLambdaRule(ALPHA, 1.0, 2).dynamic_range(G),
            "richardson_guarded": guarded.dynamic_range(G),
            "richardson_naive": naive.dynamic_range(G)}
    axes[2].axhline(0.5, color="0.6", lw=0.8, ls=":")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-6, 1.6)
    axes[2].set_xlabel("outcome probability $p$")
    axes[2].set_ylabel(r"$\Pr[\widehat W<0]$, naive Richardson")
    axes[2].set_title("Why the correction needs a guard")
    axes[2].legend(loc="lower left", fontsize=6.0)
    panel_label(axes[2], "c")
    _write_csv("fig2c_negative_weights.csv",
               ["G", "p", "prob_negative_naive", "max_prob_negative_guarded"], rows_c)

    savefig(fig, os.path.join(FIG_DIR, "fig2_device_comparison.png"))
    SUMMARY["figure2"] = out
    return out


def _proxy_moments(p, G_eff, alpha, eps):
    """Relative bias and sd of max(p_bar, eps)^-alpha when p_bar is treated as
    Binomial(G_eff, p)/G_eff with a (generally non-integer) effective group
    size, rounded to the nearest integer. A proxy only -- the MC curve is the
    measurement."""
    G_int = max(2, int(round(G_eff)))
    t = ClipRule(alpha, eps).error_table(np.array([p]), G_int)
    return float(t["rel_bias"][0]), float(t["rel_sd"][0])


def _ema_weight_moments(p, G, beta, alpha, eps, rng, n_chains=20000, n_burn=400):
    """Simulate the EMA recursion for a static policy and return the relative
    mean and variance of max(p_bar, eps)^-alpha.

    Only the marginal count of one outcome matters here, so the group is drawn
    with `rng.binomial(G, p)` rather than through the policy-level
    inverse-transform sampler; the two give the same distribution, and the
    inverse-transform route is used wherever a full policy is sampled
    (Figure 1's validation and every training run).
    """
    pbar = np.full(n_chains, p)
    for _ in range(n_burn):
        pbar = (1.0 - beta) * pbar + beta * rng.binomial(G, p, size=n_chains) / G
    v = np.maximum(pbar, eps) ** (-alpha) * p ** alpha
    return float(v.mean()), float(v.var())


# --------------------------------------------------------------------------
# FIGURE 3 -- downstream consequences
# --------------------------------------------------------------------------

def _rule_set(alpha=ALPHA, eps=EPS, K=2):
    return [
        (ClipRule(alpha, eps), PALETTE[0], "-"),
        (AddLambdaRule(alpha, 1.0, K), PALETTE[1], "-"),
        (AddLambdaRule(alpha, 0.5, K), PALETTE[4], "-."),
        (RichardsonRule(alpha, eps, guard=1), PALETTE[2], "--"),
        (RichardsonRule(alpha, eps, guard=None), PALETTE[3], ":"),
    ]


def fig3_downstream():
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    out = {}

    # --- panel (a): the effective weight and its dynamic range ------------
    G = 16
    p_dense = np.linspace(1e-4, 1.0, 600)
    rows_a = []
    # A log axis keeps the interesting 1..G band readable; the naive Richardson
    # rule is negative over most of the range, so it is drawn only where it is
    # positive and its sign change is marked instead of rescaling the panel
    # around -472.
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    for rule, colour, ls in _rule_set(K=2):
        w = rule.effective_weight(p_dense, G)
        positive = w > 0.0
        axes[0].plot(p_dense[positive], w[positive], ls, color=colour, label=rule.label)
        ok, rise = check_monotone(rule, G)
        out.setdefault("monotone", {})[rule.name] = {"ok": ok, "max_increase": float(rise)}
        out.setdefault("dynamic_range_G16", {})[rule.name] = rule.dynamic_range(G)
        if not positive.all():
            cross = float(p_dense[positive][0])
            out.setdefault("weight_sign_change_p", {})[rule.name] = cross
            axes[0].axvspan(p_dense[0], cross, color=colour, alpha=0.10, lw=0)
            axes[0].text(np.sqrt(p_dense[0] * cross), 1.6,
                         f"{rule.label}:\n$w_G<0$ here\n(down to $-472$)",
                         fontsize=5.6, color=colour, ha="center", va="bottom")
        for pp, ww in zip(p_dense[::40], w[::40]):
            rows_a.append([rule.name, G, pp, ww])
    axes[0].plot(p_dense, IdealRule(ALPHA).effective_weight(p_dense, G), color="k", lw=1.0,
                 ls=(0, (6, 2)), label=r"ideal $p^{-\alpha}$")
    axes[0].axhline(G ** ALPHA, color="0.6", lw=0.8, ls=":")
    axes[0].text(1.5e-4, G ** ALPHA * 1.2, r"$\omega(1)=G^{\alpha}$ (clip ceiling)",
                 fontsize=6.0, color="0.4")
    axes[0].set_ylim(0.8, 6e2)
    axes[0].set_xlabel("outcome probability $p$")
    axes[0].set_ylabel(r"effective weight $w_G(p)$")
    axes[0].set_title(rf"What the dynamics see, $G$={G}")
    axes[0].legend(loc="lower left", fontsize=5.8)
    panel_label(axes[0], "a")
    _write_csv("fig3a_effective_weight.csv", ["rule", "G", "p", "w_eff"], rows_a)

    # --- panel (b): K=2 minority mass vs G --------------------------------
    Gs = [g for g in range(2, 65)]
    rho = float(R41[0] / R41[1])
    rows_b = []
    for rule_proto, colour, ls in _rule_set(K=2):
        p2, first_alive = [], None
        for G in Gs:
            if isinstance(rule_proto, RichardsonRule) and G % 2 != 0:
                p2.append(np.nan)
                continue
            rule = rule_proto
            try:
                p1 = stationary_K2(R41, rule, G)
            except ValueError:
                p1 = 1.0
            p2.append(1.0 - p1)
            if first_alive is None and 1.0 - p1 > 1e-9:
                first_alive = G
            rows_b.append([rule.name, G, 1.0 - p1, rule.dynamic_range(G),
                           rule.dynamic_range(G) > rho])
        marker = "o" if isinstance(rule_proto, RichardsonRule) else None
        axes[1].plot(Gs, p2, ls, color=colour, label=rule_proto.label, marker=marker,
                     ms=2.6, markevery=4)
        out.setdefault("first_surviving_G", {})[rule_proto.name] = first_alive
    # the split-half rules are only defined for an even G, so their entry above
    # is the first EVEN surviving group size; their criterion is the clip's.
    out["split_rules_evaluated_on_even_G_only"] = True
    axes[1].axhline(1.0 - stationary_p(R41, ALPHA)[0], color="k", lw=1.0, ls=(0, (6, 2)),
                    label=r"ideal $p_2^*=r_2/\sum r$")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks([2, 4, 8, 16, 32, 64])
    axes[1].xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    axes[1].set_ylim(-0.01, 0.23)
    axes[1].set_xlabel("group size $G$")
    axes[1].set_ylabel(r"stationary minority mass $p_2$")
    axes[1].set_title(r"$K$=2, $r$=(4, 1), $\alpha$=1")
    axes[1].legend(loc="lower right", fontsize=5.8)
    panel_label(axes[1], "b")
    _write_csv("fig3b_minority_mass.csv",
               ["rule", "G", "p2_meanfield", "dynamic_range", "survives"], rows_b)

    # extinction exponents by bisection, per rule
    alpha_rows = []
    for G in [4, 8, 16, 32, 64]:
        entry = {}
        for name, factory in [("clip", lambda a: ClipRule(a, EPS)),
                              ("add_lambda_1", lambda a: AddLambdaRule(a, 1.0, 2)),
                              ("add_lambda_0.5", lambda a: AddLambdaRule(a, 0.5, 2)),
                              ("richardson_guarded", lambda a: RichardsonRule(a, EPS, guard=1))]:
            a_c, _ = extinction_alpha_K2(factory, R41, G)
            entry[name] = float(a_c)
            alpha_rows.append([G, name, a_c, np.log(rho) / np.log(
                factory(1.0).dynamic_range(G))])
        out.setdefault("alpha_c", {})[str(G)] = entry
    _write_csv("fig3b_alpha_critical.csv",
               ["G", "rule", "alpha_c_bisection", "alpha_c_closed_form"], alpha_rows)
    out["alpha_c_clip_vs_guarded_richardson_max_diff"] = float(max(
        abs(v["clip"] - v["richardson_guarded"]) for v in out["alpha_c"].values()))
    out["alpha_c_bisection_vs_closed_form_max_diff"] = float(
        max(abs(r[2] - r[3]) for r in alpha_rows if np.isfinite(r[2])))

    # --- panel (c): K=5 Monte-Carlo training ------------------------------
    Gs_c = [8, 16, 32, 64]
    h, n_steps, n_seeds = 0.01, 100_000, 32
    p_ideal = stationary_p(R5, ALPHA)
    rows_c, worst_gap = [], 0.0
    width, offset = 0.18, -1.5
    trained = [(r, c, s) for r, c, s in _rule_set(K=5) if r.name != "richardson_naive"]
    for rule_proto, colour, ls in trained:
        l1_mc, l1_mf, err = [], [], []
        for G in Gs_c:
            p_mf, _, st = stationary(R5, rule_proto, G)
            if st["scaled_residual"] > 1e-9 or not st["extinct_condition_ok"]:
                raise AssertionError(f"mean-field solve failed for {rule_proto.name}, G={G}")
            p_bar, p_sd = _mc_train(R5, rule_proto, G, h, n_steps, n_seeds,
                                    np.random.default_rng(SEEDS["fig3"] + G))
            gap = float(np.sum(np.abs(p_bar - p_mf)))
            worst_gap = max(worst_gap, gap)
            l1_mc.append(float(np.sum(np.abs(p_bar - p_ideal))))
            l1_mf.append(float(np.sum(np.abs(p_mf - p_ideal))))
            # 2 standard errors of the across-seed mean, summed over outcomes
            err.append(float(2.0 * np.sum(p_sd) / np.sqrt(n_seeds)))
            rows_c.append([rule_proto.name, G, *p_bar, *p_mf, l1_mc[-1], l1_mf[-1], gap,
                           err[-1]])
        x = np.arange(len(Gs_c)) + offset * width
        axes[2].bar(x, l1_mc, width, yerr=err, color=colour, alpha=0.85,
                    error_kw={"lw": 0.7, "capsize": 1.2, "ecolor": "0.3"},
                    label=rule_proto.label)
        axes[2].scatter(x, l1_mf, s=14, marker="_", color="k", lw=1.1, zorder=3)
        offset += 1.0
    axes[2].scatter([], [], s=14, marker="_", color="k", lw=1.1, label="mean-field prediction")
    axes[2].set_xticks(np.arange(len(Gs_c)))
    axes[2].set_xticklabels([str(g) for g in Gs_c])
    axes[2].set_xlabel("group size $G$")
    axes[2].set_ylabel(r"$\ell_1$ distance to ideal $p^*$")
    axes[2].set_title(rf"$K$=5 training, $h$={h:g}, {n_seeds} seeds")
    axes[2].legend(loc="upper right", fontsize=5.8)
    panel_label(axes[2], "c")
    _write_csv("fig3c_training.csv",
               ["rule", "G"] + [f"p_mc_{i+1}" for i in range(5)]
               + [f"p_mf_{i+1}" for i in range(5)]
               + ["l1_mc", "l1_meanfield", "l1_mc_vs_mf", "l1_two_sem"], rows_c)
    out["mc_vs_meanfield_max_l1"] = float(worst_gap)
    out["training"] = {"h": h, "n_steps": n_steps, "n_seeds": n_seeds}

    savefig(fig, os.path.join(FIG_DIR, "fig3_downstream.png"))
    SUMMARY["figure3"] = out
    return out


def _mc_train(r, rule, G, h, n_steps, n_seeds, rng, burn_frac=0.5):
    """Run the sampled update under `rule` and return the time-averaged policy
    over the second half of training (mean and across-seed sd)."""
    K = len(r)
    z = np.zeros((n_seeds, K))
    acc = np.zeros((n_seeds, K))
    start = int(n_steps * burn_frac)
    for step in range(n_steps):
        ghat, _ = sampled_step(z, r, rule, G, rng)
        z = z + h * ghat
        if step >= start:
            acc += softmax(z)
    p_avg = acc / (n_steps - start)
    return p_avg.mean(axis=0), p_avg.std(axis=0)


# --------------------------------------------------------------------------
# FIGURE 4 -- moving averages
# --------------------------------------------------------------------------

def fig4_dynamics_objective():
    """
    A minimum-MSE estimator of p^-alpha is not what the learning dynamics want.
    By the size-biasing identity (5.5) the dynamics see w_G(p) = E[omega(1+B)],
    B ~ Binomial(G-1, p) -- the weight given to a group that is already known to
    contain one copy of this outcome. The extra guaranteed hit shifts the
    argument up by about (1-p)/G, which partly cancels the convexity bias, so
    the *dynamics-level distortion* D(p) = w_G(p) p^alpha - 1 obeys

        clip (c=0)              D = alpha(alpha-1)(1-p)/(2 G p) + O(G^-2)
        guarded Richardson      D = -alpha(1-p)/(G p)          + O(G^-2)
        offset c = (1-alpha)/2  D = O(G^-2)

    At alpha = 1 the first coefficient vanishes and in fact the closed form
    w_G(p) = (1-(1-p)^G)/p holds exactly, so the base paper's own rule is
    accurate to within the probability of missing the outcome entirely -- while
    the estimator-optimal Richardson rule introduces an O(1/G) error.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    out = {}
    p_ref = 0.3          # panels (b), (c): well inside the asymptotic regime
    p_a = 0.1            # panel (a): smaller p keeps (1-p)^G above round-off longer

    def distortion(rule, G, p=p_ref):
        return float(rule.effective_weight(np.array([p]), G)[0]) * p ** rule.alpha - 1.0

    # --- panel (a): alpha = 1 ---------------------------------------------
    Gs = np.array([16, 32, 64, 128, 256, 512, 1024, 2048])
    rules_a = [(lambda a: ClipRule(a, 1e-6), "clip (paper)", PALETTE[0], "-", "o"),
               (lambda a: RichardsonRule(a, 1e-6, guard=1), "guarded Richardson",
                PALETTE[2], "--", "s"),
               (lambda a: AddLambdaRule(a, 1.0, 5), r"add-$\lambda$=1", PALETTE[1], "-.", "^")]
    rows_a, orders_a = [], {}
    for factory, label, colour, ls, marker in rules_a:
        d = np.array([abs(distortion(factory(1.0), int(G), p_a)) for G in Gs])
        keep = d > 1e-15
        axes[0].loglog(Gs[keep], d[keep], ls, marker=marker, ms=3.4, color=colour,
                       label=label)
        if label.startswith("clip"):
            # not a power law at all: D = -(1-p)^G exactly. Compare directly,
            # over the range where the law is above the binomial sum's own
            # round-off floor.
            law = (1.0 - p_a) ** Gs.astype(np.float64)
            usable = law > 1e-11
            signed = np.array([distortion(factory(1.0), int(G), p_a) for G in Gs])
            rel = np.abs(signed[usable] + law[usable]) / law[usable]
            orders_a[label] = {"exact_law": "-(1-p)^G",
                               "max_rel_error_vs_law": float(np.max(rel)),
                               "n_points_above_roundoff": int(usable.sum()),
                               "order": float("nan")}
        else:
            # fit only where the asymptotic regime has actually been reached
            fit = keep & (Gs * p_a >= 20.0)
            slope, _, r2 = least_squares_line(np.log(Gs[fit]), np.log(d[fit]))
            orders_a[label] = {"order": float(-slope), "r2": float(r2),
                               "n_points": int(fit.sum()), "fit_from_G": int(Gs[fit][0])}
        for G, dd in zip(Gs, d):
            rows_a.append([label, int(G), p_a, dd])
    axes[0].loglog(Gs, (1.0 - p_a) ** Gs.astype(np.float64), ":", color="k", lw=1.2,
                   label=r"exact clip law $(1-p)^{G}$")
    axes[0].axhline(1e-13, color="0.7", lw=0.7, ls="--")
    axes[0].text(300, 2e-13, "round-off floor of the binomial sum",
                 fontsize=5.6, color="0.45")
    # the closed form itself
    p_chk = np.concatenate([np.geomspace(1e-12, 1e-6, 30), np.linspace(1e-6, 1.0, 400)])
    worst = 0.0
    for G in [2, 4, 16, 64, 256]:
        a = ClipRule(1.0, 1.0 / G).effective_weight(p_chk, G)
        b = effective_weight_alpha1(p_chk, G)
        worst = max(worst, float(np.max(np.abs(a - b) / b)))
    out["alpha1_closed_form_max_rel_error"] = worst
    axes[0].set_ylim(1e-16, 1e-1)
    axes[0].set_xlabel("group size $G$")
    axes[0].set_ylabel(r"$|w_G(p)\,p^{\alpha}-1|$")
    axes[0].set_title(rf"Dynamics-level distortion, $\alpha$=1, $p$={p_a:g}")
    axes[0].legend(loc="lower left", fontsize=6.0)
    panel_label(axes[0], "a")
    _write_csv("fig4a_alpha1_distortion.csv", ["rule", "G", "p", "abs_distortion"], rows_a)
    out["alpha1_orders"] = orders_a

    # --- panel (b): the O(1/G) coefficient as a function of alpha ---------
    alphas = np.linspace(0.25, 3.0, 23)
    G_fit = 4096
    series = [(lambda a: ClipRule(a, 1e-9), "clip (paper)", PALETTE[0], "o",
               lambda a: a * (a - 1.0) / 2.0),
              (lambda a: RichardsonRule(a, 1e-9, guard=1), "guarded Richardson",
               PALETTE[2], "s", lambda a: -a),
              (lambda a: OffsetRule(a, 1e-9), r"offset $c=(1-\alpha)/2$", PALETTE[3], "D",
               lambda a: 0.0 * a)]
    rows_b, worst_coeff = [], 0.0
    a_dense = np.linspace(0.25, 3.0, 300)
    for factory, label, colour, marker, theory in series:
        measured = np.array([distortion(factory(a), G_fit) * G_fit * p_ref / (1.0 - p_ref)
                             for a in alphas])
        axes[1].plot(a_dense, np.atleast_1d(theory(a_dense)) * np.ones_like(a_dense),
                     color=colour, lw=1.0)
        axes[1].scatter(alphas, measured, s=16, marker=marker, facecolor="white",
                        edgecolor=colour, lw=0.9, zorder=3, label=label)
        th = np.atleast_1d(theory(alphas)) * np.ones_like(alphas)
        err = np.abs(measured - th)
        worst_coeff = max(worst_coeff, float(np.max(err)))
        nz = np.abs(th) > 0
        out.setdefault("coefficient_fit", {})[label] = {
            "max_abs_error": float(np.max(err)),
            "max_rel_error": float(np.max(err[nz] / np.abs(th[nz]))) if nz.any() else None,
            "max_abs_measured": float(np.max(np.abs(measured)))}
        for a, m, t in zip(alphas, measured, np.atleast_1d(theory(alphas)) * np.ones_like(alphas)):
            rows_b.append([label, a, G_fit, m, t])
    axes[1].axhline(0.0, color="0.6", lw=0.8, ls=":")
    axes[1].axvline(1.0, color="0.6", lw=0.8, ls="--")
    axes[1].text(1.04, 2.55, r"IPS, $\alpha$=1", fontsize=6.2, color="0.4")
    axes[1].set_xlabel(r"exponent $\alpha$")
    axes[1].set_ylabel(r"$\lim_{G\to\infty} Gp\,D(p)/(1-p)$")
    axes[1].set_title(r"The $O(1/G)$ coefficient (lines: theory)")
    axes[1].legend(loc="lower left", fontsize=6.2)
    panel_label(axes[1], "b")
    _write_csv("fig4b_distortion_coefficient.csv",
               ["rule", "alpha", "G", "measured_coefficient", "theory"], rows_b)
    out["coefficient_max_abs_error"] = worst_coeff
    out["fit_group_size"] = G_fit

    # --- panel (c): the order gain, and its downstream price --------------
    rows_c, orders_c = [], {}
    Gs_c = np.array([64, 128, 256, 512, 1024, 2048])
    for j, a in enumerate([0.5, 2.0, 3.0]):
        for factory, tag, ls, marker in [(lambda al: ClipRule(al, 1e-9), "clip", "-", "o"),
                                         (lambda al: OffsetRule(al, 1e-9), "offset", "--", "D")]:
            d = np.array([abs(distortion(factory(a), int(G))) for G in Gs_c])
            slope, _, r2 = least_squares_line(np.log(Gs_c), np.log(d))
            orders_c[f"alpha={a:g}, {tag}"] = {"order": float(-slope), "r2": float(r2)}
            axes[2].loglog(Gs_c, d, ls, marker=marker, ms=3.2, color=PALETTE[j],
                           label=rf"$\alpha$={a:g}, {tag}: order {-slope:.2f}")
            for G, dd in zip(Gs_c, d):
                rows_c.append([a, tag, int(G), dd])
    axes[2].set_xlabel("group size $G$")
    axes[2].set_ylabel(r"$|w_G(p)\,p^{\alpha}-1|$")
    axes[2].set_title(r"Matching the offset to $\alpha$")
    axes[2].legend(loc="lower left", fontsize=5.6, ncol=2)
    panel_label(axes[2], "c")
    _write_csv("fig4c_offset_order.csv", ["alpha", "rule", "G", "abs_distortion"], rows_c)
    out["offset_orders"] = orders_c

    # downstream price and payoff of the offset, K=5 mean field (no MC needed)
    rows_d, payoff = [], {}
    for a in [0.5, 2.0]:
        ideal = stationary_p(R5, a)
        for G in [8, 16, 32, 64, 128]:
            entry = {}
            for rule in [ClipRule(a, EPS), OffsetRule(a, EPS)]:
                p_mf, _, st = stationary(R5, rule, G)
                if st["scaled_residual"] > 1e-9 or not st["extinct_condition_ok"]:
                    raise AssertionError(f"mean-field solve failed: {rule.name}, G={G}, a={a}")
                entry[rule.name] = float(np.sum(np.abs(p_mf - ideal)))
                rows_d.append([a, rule.name, G, entry[rule.name], rule.dynamic_range(G),
                               int(np.count_nonzero(p_mf > 1e-11))])
            payoff[f"alpha={a:g}, G={G}"] = entry
    _write_csv("fig4d_offset_downstream.csv",
               ["alpha", "rule", "G", "l1_to_ideal", "dynamic_range", "support_size"], rows_d)
    out["offset_downstream_l1"] = payoff

    savefig(fig, os.path.join(FIG_DIR, "fig4_dynamics_objective.png"))
    SUMMARY["figure4"] = out
    return out


def fig5_moving_average():
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    rng = np.random.default_rng(SEEDS["fig5"])
    out = {}

    # --- panel (a): variance reduction, no steady-state bias --------------
    G, p = 32, 0.3
    betas = np.geomspace(0.01, 1.0, 40)
    axes[0].plot(betas, variance_factor(betas), color=PALETTE[0],
                 label=r"theory $\beta/(2-\beta)$")
    beta_mc = np.geomspace(0.02, 1.0, 9)
    n_chains_a = 60_000
    rows_a = []
    for beta in beta_mc:
        # (1-beta)^(2 n_burn) ~ e^-12 is stationary to ~1e-5; the variance
        # estimate's own error is then sqrt(2/n_chains) ~ 0.6%
        n_chains, n_burn = n_chains_a, max(60, int(6.0 / beta))
        pbar = np.full(n_chains, p)
        for _ in range(n_burn):
            pbar = (1.0 - beta) * pbar + beta * rng.binomial(G, p, size=n_chains) / G
        ratio = float(pbar.var() / (p * (1.0 - p) / G))
        bias = float(pbar.mean() - p)
        rows_a.append([beta, ratio, float(variance_factor(beta)),
                       float(effective_group_size(G, beta)), bias,
                       float(np.sqrt(pbar.var() / n_chains))])
        axes[0].scatter([beta], [ratio], s=18, marker="o", facecolor="white",
                        edgecolor=PALETTE[0], lw=0.9, zorder=3)
    axes[0].scatter([], [], s=18, marker="o", facecolor="white", edgecolor="0.3", lw=0.9,
                    label=f"Monte Carlo ({n_chains_a // 1000}k chains)")
    ax0b = axes[0].twinx()
    ax0b.plot(betas, effective_group_size(G, betas), color=PALETTE[1], lw=1.0, ls="--")
    ax0b.set_yscale("log")
    ax0b.set_ylabel(r"$G_{\mathrm{eff}}=G(2-\beta)/\beta$", color=PALETTE[1], fontsize=8.5)
    ax0b.tick_params(axis="y", labelsize=7.5, colors=PALETTE[1])
    ax0b.grid(False)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"EMA rate $\beta$")
    axes[0].set_ylabel(r"$\mathrm{Var}(\bar p)/\mathrm{Var}(\hat p)$")
    axes[0].set_title(rf"Free variance reduction, $G$={G}")
    axes[0].legend(loc="upper left", fontsize=6.2)
    panel_label(axes[0], "a")
    _write_csv("fig5a_variance_reduction.csv",
               ["beta", "var_ratio_mc", "var_ratio_theory", "G_eff", "bias_mc", "mc_sem"],
               rows_a)
    out["variance_factor_max_rel_error"] = float(max(
        abs(r[1] - r[2]) / r[2] for r in rows_a))
    out["max_abs_static_bias"] = float(max(abs(r[4]) for r in rows_a))

    # --- panel (b): lag bias and the total-error optimum ------------------
    # The drift is centred on p = 0.3 so that the sampling variance p(1-p)/G is
    # the same at both ends of the test and the theory curve, drawn at p = 0.3,
    # is the right comparison. delta is chosen to put the optimum at an
    # interior beta rather than at the edge of the sweep.
    # beta >= 0.02 so that (1-beta)^n_steps_lag < 1e-5: the lag has reached its
    # steady state (5.13) and what is measured is the lag, not a transient.
    delta, n_chains, n_steps_lag = 5e-4, 40_000, 600
    rows_b = []
    for beta in np.geomspace(0.02, 1.0, 9):
        pt = 0.3 + delta * (np.arange(n_steps_lag) - 0.5 * n_steps_lag)
        pbar = np.full(n_chains, pt[0])
        for t in range(n_steps_lag):
            pbar = (1.0 - beta) * pbar + beta * rng.binomial(G, pt[t], size=n_chains) / G
        lag_mc = float(pt[-1] - pbar.mean())
        sem = float(np.sqrt(pbar.var() / n_chains))
        rows_b.append([beta, lag_mc, float(lag_bias(beta, delta)), sem,
                       float(np.mean((pbar - pt[-1]) ** 2)),
                       float(frequency_mse(beta, delta, G, pt[-1]))])
    beta_dense = np.geomspace(0.004, 1.0, 300)
    axes[1].plot(beta_dense, lag_bias(beta_dense, delta) ** 2, color=PALETTE[1], ls="--",
                 lw=1.0, label=r"lag$^2=((1-\beta)\delta/\beta)^2$")
    axes[1].plot(beta_dense, variance_factor(beta_dense) * 0.3 * 0.7 / G, color=PALETTE[2],
                 ls=":", lw=1.0, label=r"variance $\frac{\beta}{2-\beta}\frac{p(1-p)}{G}$")
    axes[1].plot(beta_dense, frequency_mse(beta_dense, delta, G, 0.3), color=PALETTE[0],
                 label="total MSE of $\\bar p$")
    obj = lambda b: float(frequency_mse(b, delta, G, 0.3))
    b_star, f_star, n_it, n_ev = golden_section_log(obj, 4e-3, 1.0, rel_tol=1e-13)
    b_grid, f_grid, vals, _ = grid_minimum(obj, 4e-3, 1.0, n=4001, geometric=True)
    axes[1].scatter([b_star], [f_star], s=34, marker="*", color=PALETTE[0], zorder=4,
                    label=rf"golden section: $\beta^*$={b_star:.3f}")
    axes[1].errorbar([r[0] for r in rows_b], [r[4] for r in rows_b],
                     yerr=[2.0 * r[3] for r in rows_b], fmt="o", ms=3.2, lw=0.8,
                     color=PALETTE[3], mfc="white", label="simulated drift test")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"EMA rate $\beta$")
    axes[1].set_ylabel(r"squared error of $\bar p$ against the moving $p$")
    axes[1].set_ylim(3e-6, 4e-2)
    axes[1].set_title(rf"Lag versus noise, drift $\delta$={delta:g}/step")
    axes[1].legend(loc="upper left", fontsize=5.8)
    panel_label(axes[1], "b")
    _write_csv("fig5b_lag_and_total_error.csv",
               ["beta", "lag_mc", "lag_theory", "mc_sem", "mse_mc", "mse_theory"], rows_b)
    out["lag_max_z_score"] = float(max(abs(r[1] - r[2]) / r[3] for r in rows_b))
    out["beta_star"] = {"golden": float(b_star), "grid": float(b_grid),
                        "iters": int(n_it), "evals": int(n_ev),
                        "unimodal": bool(is_unimodal(vals))}
    if not out["beta_star"]["unimodal"]:
        raise AssertionError("the EMA total-error curve was not unimodal on the "
                             "scanned bracket")

    # --- panel (c): the dynamics become second order ----------------------
    lam = linear_rates(R5, ALPHA)
    lam_min, lam_max = float(lam[0]), float(lam[-1])
    kappas = np.geomspace(0.05, 2000.0, 300)
    axes[2].plot(kappas, decay_rate(lam_min, kappas), color=PALETTE[0],
                 label=r"slow mode $\lambda_{\min}$")
    axes[2].plot(kappas, decay_rate(lam_max, kappas), color=PALETTE[1], ls="--",
                 label=r"fast mode $\lambda_{\max}$")
    axes[2].axhline(lam_min, color=PALETTE[0], lw=0.7, ls=":")
    axes[2].axhline(2.0 * lam_min, color=PALETTE[0], lw=0.7, ls=":")
    axes[2].axvline(critical_kappa(lam_min), color="0.6", lw=0.8, ls="--")
    axes[2].text(critical_kappa(lam_min) / 1.3, 0.048,
                 rf"$\kappa^*=4\lambda_{{\min}}$={critical_kappa(lam_min):.2f}",
                 fontsize=6.0, color="0.4", ha="right")
    axes[2].text(200.0, 2.0 * lam_min * 1.08, rf"$2\lambda_{{\min}}$={2*lam_min:.2f}",
                 fontsize=6.0, color=PALETTE[0])
    axes[2].text(200.0, lam_min * 0.70, rf"$\lambda_{{\min}}$={lam_min:.2f}",
                 fontsize=6.0, color=PALETTE[0])

    rows_c, worst_eig = [], 0.0
    for kappa in [0.5, 2.0, 4.599, 10.0, 60.0, 400.0]:
        J = coupled_jacobian(R5, ALPHA, kappa)
        ev = np.linalg.eigvals(J)
        numeric = -float(np.max(ev.real[ev.real < -1e-9]))
        theory = float(decay_rate(lam_min, kappa))
        worst_eig = max(worst_eig, abs(numeric - theory) / theory)
        axes[2].scatter([kappa], [numeric], s=20, marker="o", facecolor="white",
                        edgecolor="0.25", lw=0.9, zorder=4)
        rows_c.append([kappa, numeric, theory, bool(is_oscillatory(lam_min, kappa)),
                       bool(is_oscillatory(lam_max, kappa))])
    axes[2].scatter([], [], s=20, marker="o", facecolor="white", edgecolor="0.25", lw=0.9,
                    label=r"eigenvalues of the $2K\times2K$ Jacobian")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"EMA relaxation rate $\kappa=\beta/h$")
    axes[2].set_ylabel("asymptotic decay rate")
    axes[2].set_ylim(0.02, 9.0)
    axes[2].set_title(r"$s^2+\kappa s+\kappa\lambda=0$: $r$=(5,4,3,2,1)")
    axes[2].legend(loc="upper left", fontsize=5.9)
    panel_label(axes[2], "c")

    # inset: ringing below the critical kappa
    ins = axes[2].inset_axes([0.52, 0.08, 0.46, 0.40])
    p_star = stationary_p(R5, ALPHA)
    z0 = np.log(p_star) + np.array([0.6, -0.3, 0.0, -0.2, -0.1])
    state0 = np.concatenate([z0, softmax(z0)])[None, :]
    for kappa, colour, ls in [(1.0, PALETTE[3], "-"), (200.0, PALETTE[0], "--")]:
        f = lambda s: coupled_rhs(s, R5, ALPHA, kappa)
        hh = min(0.002, 0.2 / max(kappa, 1.0))
        ts, traj = integrate(f, state0, hh, int(12.0 / hh), method="rk4",
                             record_every=max(1, int(0.02 / hh)))
        ins.plot(ts, softmax(traj[:, 0, :5])[:, 0] - p_star[0], ls, lw=1.0, color=colour,
                 label=rf"$\kappa$={kappa:g}")
    ins.axhline(0.0, color="0.6", lw=0.6)
    ins.set_xlabel("time $t$", fontsize=6, labelpad=1.0)
    ins.set_ylabel("$p_1-p_1^*$", fontsize=6)
    ins.tick_params(labelsize=5.5)
    ins.legend(fontsize=5.2, loc="upper right")

    _write_csv("fig5c_second_order.csv",
               ["kappa", "rate_numeric", "rate_theory", "slow_mode_rings", "fast_mode_rings"],
               rows_c)
    out["decay_rate_max_rel_error"] = float(worst_eig)
    out["lambda_min"] = lam_min
    out["lambda_max"] = lam_max
    out["kappa_critical"] = float(critical_kappa(lam_min))
    out["best_rate_ratio"] = float(decay_rate(lam_min, critical_kappa(lam_min)) / lam_min)

    savefig(fig, os.path.join(FIG_DIR, "fig5_moving_average.png"))
    SUMMARY["figure5"] = out
    return out


# --------------------------------------------------------------------------
# self-checks: everything that must hold before a single figure is trusted
# --------------------------------------------------------------------------

def _run_self_checks():
    checks = {}
    rng = np.random.default_rng(9051)

    # (1) the size-biasing identity, against the direct phi_G(p)/p sum
    from scipy.stats import binom
    worst = 0.0
    for G, alpha, eps in [(16, 1.0, 1e-3), (12, 1.7, 1e-2), (8, 0.6, 1e-4)]:
        p = np.linspace(0.05, 0.95, 19)
        for rule in [ClipRule(alpha, eps), AddLambdaRule(alpha, 0.7, 4),
                     RichardsonRule(alpha, eps, guard=1),
                     RichardsonRule(alpha, eps, guard=None)]:
            if isinstance(rule, RichardsonRule):
                M = G // 2
                a = np.arange(M + 1)
                W = rule.values(G)
                direct = np.array([
                    float((binom.pmf(a, M, pp)[:, None] * binom.pmf(a, M, pp)[None, :]
                           * ((a[:, None] + a[None, :]) / G) * W).sum() / pp) for pp in p])
            else:
                n = np.arange(G + 1)
                v = rule.values(G)
                direct = np.array([float((binom.pmf(n, G, pp) * (n / G) * v).sum() / pp)
                                   for pp in p])
            got = rule.effective_weight(p, G)
            worst = max(worst, float(np.max(np.abs(got - direct) / np.abs(direct))))
    checks["size_biasing_max_rel_error"] = worst
    if worst > 1e-12:
        raise AssertionError(f"size-biasing identity failed: {worst:.3e}")

    # (2) the clipped rule reproduces Experiments 2-4's effective weight
    p = np.linspace(0.0, 1.0, 101)
    d = float(np.max(np.abs(ClipRule(1.3, 1e-3).effective_weight(p, 16)
                            - clip_effective_weight(p, 16, 1.3, 1e-3))))
    checks["clip_vs_finite_group_general"] = d
    if d > 1e-12:
        raise AssertionError(f"effective weight disagrees with Experiment 3: {d:.3e}")

    # (3) sampled_step reproduces dynamics.rhs_sampled
    z = np.random.default_rng(3).normal(size=(9, 5))
    g1, h1 = sampled_step(z, R5, ClipRule(1.0, 1e-3), 16, np.random.default_rng(21))
    g2, h2 = rhs_sampled(z, R5, 1.0, 16, np.random.default_rng(21), 1e-3)
    d = float(np.max(np.abs(g1 - g2)))
    checks["sampled_step_vs_rhs_sampled"] = d
    if d > 1e-12 or not np.array_equal(h1, h2):
        raise AssertionError(f"sampled_step disagrees with rhs_sampled: {d:.3e}")

    # (4) golden-section reproduces the closed-form optimal add-lambda for p_hat
    worst = 0.0
    for G, K, p in [(16, 5, 0.1), (32, 3, 0.4), (64, 10, 0.02), (8, 2, 0.25)]:
        closed = K * p * (1.0 - p) / (1.0 - K * p) ** 2

        def mse(lam):
            D = G + K * lam
            return (lam * (1.0 - K * p) / D) ** 2 + G * p * (1.0 - p) / D ** 2

        got, _, _, _ = golden_section_log(mse, 1e-6, 1e4, rel_tol=1e-14)
        worst = max(worst, abs(got - closed) / closed)
    checks["golden_section_vs_closed_form_lambda"] = worst
    if worst > 1e-6:
        raise AssertionError(f"golden-section search missed lambda*: {worst:.3e}")

    # (5) power method and QR against numpy, on Cov(p*)
    p_star = stationary_p(R5, ALPHA)
    C = np.diag(p_star) - np.outer(p_star, p_star)
    ref = np.linalg.eigvalsh(C)
    lam_pow, _, n_pow, conv_pow = power_method(C)
    vals_pow, _ = deflated_power_method(C)
    vals_qr, n_qr, off = qr_algorithm(C)
    e_pow = abs(lam_pow - ref[-1])
    e_defl = float(np.max(np.abs(np.sort(vals_pow) - ref)))
    e_qr = float(np.max(np.abs(vals_qr - ref)))
    checks["power_method"] = {"error": float(e_pow), "iterations": int(n_pow),
                              "converged": bool(conv_pow)}
    checks["deflated_power_method_max_error"] = e_defl
    checks["qr_algorithm"] = {"max_error": e_qr, "iterations": int(n_qr),
                              "off_diagonal_norm": float(off)}
    if max(e_pow, e_defl, e_qr) > 1e-10:
        raise AssertionError("power method / QR disagree with numpy")

    # (6) the EMA Jacobian's analytical block form against finite differences
    kappa = 3.0
    J_an = coupled_jacobian(R5, ALPHA, kappa)
    s0 = np.concatenate([np.log(stationary_p(R5, ALPHA)), stationary_p(R5, ALPHA)])
    step = 1e-6
    J_num = np.zeros_like(J_an)
    for j in range(len(s0)):
        e = np.zeros_like(s0)
        e[j] = step
        J_num[:, j] = (coupled_rhs((s0 + e)[None, :], R5, ALPHA, kappa)[0]
                       - coupled_rhs((s0 - e)[None, :], R5, ALPHA, kappa)[0]) / (2 * step)
    d = float(np.max(np.abs(J_an - J_num)))
    checks["coupled_jacobian_vs_finite_difference"] = d
    if d > 1e-6:
        raise AssertionError(f"coupled Jacobian disagrees with finite differences: {d:.3e}")
    lam = linear_rates(R5, ALPHA)
    pred = np.concatenate([np.array(mode_roots(l, kappa)) for l in lam])
    ev = np.linalg.eigvals(J_an)
    err = float(max(np.min(np.abs(ev - s)) for s in pred))
    checks["mode_roots_vs_jacobian_eigenvalues"] = err
    if err > 1e-9:
        raise AssertionError(f"quadratic mode roots are not Jacobian eigenvalues: {err:.3e}")

    # (7) round-off audit. The alpha=1 closed form w_G(p) = (1 - (1-p)^G)/p
    # cancels catastrophically for small p: at p = 1e-12 the subtraction
    # 1 - (1-p)^G loses about eleven digits. The guarded evaluation
    # -expm1(G log1p(-p))/p keeps them. The exact answer is bracketed by the
    # binomial sum, which does no subtraction at all.
    roundoff = {}
    for p_small in (1e-9, 1e-11, 1e-12, 1e-13):
        for G in (8, 64):
            exact = float(ClipRule(1.0, 1.0 / G).effective_weight(np.array([p_small]), G)[0])
            naive = float((1.0 - (1.0 - p_small) ** G) / p_small)
            guarded = float(effective_weight_alpha1(np.array([p_small]), G)[0])
            roundoff[f"p={p_small:g}, G={G}"] = {
                "binomial_sum": exact, "naive_difference": naive,
                "guarded_expm1": guarded,
                "naive_rel_error": abs(naive - exact) / exact,
                "guarded_rel_error": abs(guarded - exact) / exact}
    worst_naive = max(v["naive_rel_error"] for v in roundoff.values())
    worst_guarded = max(v["guarded_rel_error"] for v in roundoff.values())
    checks["roundoff_audit"] = {"cases": roundoff,
                                "worst_naive_rel_error": worst_naive,
                                "worst_guarded_rel_error": worst_guarded}
    if worst_guarded > 1e-12:
        raise AssertionError(f"guarded closed form lost precision: {worst_guarded:.3e}")

    # (8) the closed-form dynamic ranges
    worst = 0.0
    for G in [8, 16, 32]:
        for lam_v in [0.5, 1.0, 2.0]:
            for K in [2, 5]:
                got = AddLambdaRule(1.0, lam_v, K).dynamic_range(G)
                want = (G + lam_v) / (1.0 + lam_v)
                worst = max(worst, abs(got - want) / want)
        worst = max(worst, abs(ClipRule(1.0, 1e-3).dynamic_range(G) - G) / G)
        worst = max(worst, abs(RichardsonRule(1.0, 1e-3, guard=1).dynamic_range(G) - G) / G)
    checks["dynamic_range_closed_form_max_rel_error"] = float(worst)
    if worst > 1e-12:
        raise AssertionError(f"dynamic range disagrees with the closed form: {worst:.3e}")

    # (9) add-lambda's normalizer G + K lam multiplies every weight by the same
    # constant, so it cancels from r_i w_G(p_i) = S and cannot move the
    # stationary point. Panels that use different K are therefore comparable.
    worst = 0.0
    for G in (8, 16, 32):
        ref, _, _ = stationary(R5, AddLambdaRule(1.0, 1.0, 2), G)
        for K in (5, 17):
            other, _, _ = stationary(R5, AddLambdaRule(1.0, 1.0, K), G)
            worst = max(worst, float(np.max(np.abs(ref - other))))
    checks["add_lambda_K_invariance_max_abs_error"] = worst
    if worst > 1e-12:
        raise AssertionError(f"add-lambda stationary point depends on K: {worst:.3e}")

    SUMMARY["self_checks"] = checks
    return checks


# --------------------------------------------------------------------------

def main():
    total_start = time.perf_counter()
    print("=" * 74)
    print("EXPERIMENT 5: estimator bias, variance, and what they cost downstream")
    print("=" * 74)

    start = time.perf_counter()
    print("\n[Self-checks] identities, solvers, optimizer, eigenvalue routines ...")
    c = _run_self_checks()
    print(f"  size-biasing identity            max rel. error {c['size_biasing_max_rel_error']:.2e}")
    print(f"  effective weight vs Experiment 3 max abs. error {c['clip_vs_finite_group_general']:.2e}")
    print(f"  sampled_step vs rhs_sampled      max abs. error {c['sampled_step_vs_rhs_sampled']:.2e}")
    print(f"  golden section vs closed-form    max rel. error {c['golden_section_vs_closed_form_lambda']:.2e}")
    print(f"  power method / QR vs numpy       {c['power_method']['error']:.2e} / "
          f"{c['qr_algorithm']['max_error']:.2e}")
    print(f"  ({time.perf_counter() - start:.1f}s)")

    start = time.perf_counter()
    print("\n[Figure 1] Estimator error budget ...")
    r1 = fig1_error_budget()
    print(f"  Monte-Carlo vs exact: worst z-score {r1['mc_worst_bias_z']:.2f} (mean), "
          f"{r1['mc_worst_variance_z']:.2f} (variance)")
    print(f"  up to {100 * r1['max_variance_share_of_empty_group']:.1f}% of the weight "
          f"variance is carried by the empty-group draw")
    print(f"  eps* / p over the (G, p) grid: median {r1['eps_star_over_p']['median']:.3f} "
          f"(range {r1['eps_star_over_p']['min']:.3f}-{r1['eps_star_over_p']['max']:.3f})")
    print(f"  ({time.perf_counter() - start:.1f}s)")

    start = time.perf_counter()
    print("\n[Figure 2] Clipping, smoothing, EMA and Richardson compared ...")
    r2 = fig2_device_comparison()
    for key, val in r2["richardson_orders"].items():
        print(f"  bias order, {key:28s}: {val['order']:.2f}  (R^2 {val['r2']:.5f})")
    print(f"  sd ratio Richardson/plain: {r2['richardson_sd_ratio']['min']:.3f} to "
          f"{r2['richardson_sd_ratio']['max']:.3f}")
    print(f"  ({time.perf_counter() - start:.1f}s)")

    start = time.perf_counter()
    print("\n[Figure 3] Downstream: mean field and Monte-Carlo training ...")
    r3 = fig3_downstream()
    print(f"  first surviving G at alpha=1: " +
          ", ".join(f"{k}={v}" for k, v in r3["first_surviving_G"].items()))
    print(f"  alpha_c(clip) vs alpha_c(guarded Richardson): "
          f"{r3['alpha_c_clip_vs_guarded_richardson_max_diff']:.2e}")
    print(f"  max l1 |MC - mean field|: {r3['mc_vs_meanfield_max_l1']:.4f}")
    print(f"  ({time.perf_counter() - start:.1f}s)")

    start = time.perf_counter()
    print("\n[Figure 4] The dynamics objective is not the estimator objective ...")
    r4 = fig4_dynamics_objective()
    print(f"  alpha=1 closed form w_G(p)=(1-(1-p)^G)/p: max rel. error "
          f"{r4['alpha1_closed_form_max_rel_error']:.2e}")
    for key, val in r4["alpha1_orders"].items():
        if "exact_law" in val:
            print(f"  alpha=1 distortion, {key:24s}: matches the exact law "
                  f"{val['exact_law']} to {val['max_rel_error_vs_law']:.2e}")
        else:
            print(f"  alpha=1 distortion, {key:24s}: order {val['order']:.2f}")
    print(f"  O(1/G) coefficient vs theory: max abs. error "
          f"{r4['coefficient_max_abs_error']:.2e}")
    print(f"  ({time.perf_counter() - start:.1f}s)")

    start = time.perf_counter()
    print("\n[Figure 5] Moving averages ...")
    r5 = fig5_moving_average()
    print(f"  variance factor beta/(2-beta): max rel. error "
          f"{r5['variance_factor_max_rel_error']:.2e}")
    print(f"  lag formula: worst z-score {r5['lag_max_z_score']:.2f}")
    print(f"  decay-rate law: max rel. error {r5['decay_rate_max_rel_error']:.2e}; "
          f"kappa* = {r5['kappa_critical']:.3f}, speed-up {r5['best_rate_ratio']:.3f}x")
    print(f"  ({time.perf_counter() - start:.1f}s)")

    SUMMARY["configuration"] = {"alpha": ALPHA, "epsilon": EPS, "seeds": SEEDS,
                                "numpy": np.__version__}
    SUMMARY["runtime_seconds"] = float(time.perf_counter() - total_start)
    with open(os.path.join(DATA_DIR, "exp5_summary.json"), "w", encoding="utf-8") as fp:
        json.dump(SUMMARY, fp, indent=2, default=float)

    print(f"\nAll Experiment 5 figures written to results/{EXP_NAME}/figures/.")
    print(f"Data tables and summary written to results/{EXP_NAME}/data/.")
    print(f"Total runtime: {SUMMARY['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
