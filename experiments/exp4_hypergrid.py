"""
EXPERIMENT 4 -- A multidimensional finite-group diversity atlas.

Experiments 2 and 3 varied one or two parameters at a time.  This experiment
asks whether their finite-group conclusions survive a genuine hyperparameter
grid and whether that grid can be used as a trustworthy numerical surrogate.

  1. FIGURE 1 -- Exact K=2 atlas over (alpha, G, epsilon, reward ratio).
     The numerical stationary point is compared with the analytical boundary
         alpha log(min(G, 1/epsilon)) = log(r1/r2).
     The grid also exposes when probability clipping, rather than group size,
     caps the correction.

  2. FIGURE 2 -- General K atlas.
     For r=(5,4,3,2,1), nested finite-group root-finding maps support size and
     entropy over (alpha, G).  A second grid over outcome count and reward
     spread measures the minimum G needed to retain every outcome at alpha=1.

  3. FIGURE 3 -- Hypergrid interpolation audit.
     A three-dimensional tensor grid in log(alpha), log(reward ratio), and
     log(epsilon) is tested on independently sampled off-grid points.  Grid
     refinement is measured separately in smooth regions and near the
     extinction/clipping kinks, where multilinear interpolation is hardest.

Outputs: results/exp4_hypergrid/figures/fig*.png and
results/exp4_hypergrid/data/*.csv, exp4_summary.json.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.finite_group import meanfield_stationary_K2
from src.finite_group_general import effective_weight, meanfield_stationary
from src.fitting import least_squares_line
from src.hypergrid import TensorGrid, finite_group_margin, stationary_k2_batch
from src.plotstyle import PALETTE, apply_style, panel_label, savefig


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP_NAME = "exp4_hypergrid"
FIG_DIR = os.path.join(ROOT, "results", EXP_NAME, "figures")
DATA_DIR = os.path.join(ROOT, "results", EXP_NAME, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

apply_style()

SUMMARY: dict[str, object] = {}
R5 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
EPS_DEFAULT = 1e-3
PROB_TOL = 1e-11
INK = "0.25"


def _write_csv(name, header, rows):
    path = os.path.join(DATA_DIR, name)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerows(rows)


def _geometric_edges(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or np.any(values <= 0) or np.any(np.diff(values) <= 0):
        raise ValueError("geometric edges require at least two increasing positive values")
    middle = np.sqrt(values[:-1] * values[1:])
    first = values[0] ** 2 / middle[0]
    last = values[-1] ** 2 / middle[-1]
    return np.concatenate([[first], middle, [last]])


def _normalized_entropy(p):
    p = np.asarray(p, dtype=np.float64)
    positive = p > 0.0
    return float(-np.sum(p[positive] * np.log(p[positive])) / np.log(len(p)))


def _geometric_rewards(K, spread):
    """K descending rewards with r_max/r_min=spread and r_max=1."""
    if not isinstance(K, (int, np.integer)) or K < 2:
        raise ValueError("K must be an integer >= 2")
    if not np.isfinite(spread) or spread < 1.0:
        raise ValueError("spread must be finite and >= 1")
    return np.exp(-np.linspace(0.0, np.log(spread), K))


def _stationarity_error(p, rewards, G, alpha, eps, S):
    """Scale-free KKT residual for a finite-group stationary point."""
    p = np.asarray(p, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    value = rewards * effective_weight(p, G, alpha, eps)
    active = p > PROB_TOL
    equality = np.max(np.abs(value[active] - S)) if np.any(active) else np.inf
    invasion = np.max(np.maximum(value[~active] - S, 0.0)) if np.any(~active) else 0.0
    return float(max(equality, invasion) / max(abs(S), np.finfo(float).tiny))


def _solve_general_checked(rewards, G, alpha, eps=EPS_DEFAULT):
    """Call the Experiment 3 solver with normalization and strict checks.

    Reward normalization avoids its absolute outer tolerance depending on the
    arbitrary reward unit.  Safeguarded Newton is attempted first; bisection
    is an independent fallback if normalization or KKT residual checks fail.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or len(rewards) < 2 or not np.all(np.isfinite(rewards)):
        raise ValueError("rewards must be a finite vector of length >= 2")
    if np.any(rewards <= 0.0):
        raise ValueError("Experiment 4 requires strictly positive rewards")
    scaled = rewards / rewards.max()

    fallback = False
    p, S, stats = meanfield_stationary(scaled, int(G), float(alpha), float(eps), method="newton")
    sum_error = abs(float(np.sum(p)) - 1.0)
    residual = _stationarity_error(p, scaled, int(G), float(alpha), float(eps), S)
    if (not np.all(np.isfinite(p)) or np.any(p < -1e-13) or np.any(p > 1.0 + 1e-13)
            or sum_error > 2e-9 or residual > 2e-8):
        fallback = True
        p, S, stats = meanfield_stationary(scaled, int(G), float(alpha), float(eps), method="bisection")
        sum_error = abs(float(np.sum(p)) - 1.0)
        residual = _stationarity_error(p, scaled, int(G), float(alpha), float(eps), S)
    if (not np.all(np.isfinite(p)) or np.any(p < -1e-13) or np.any(p > 1.0 + 1e-13)
            or sum_error > 2e-9 or residual > 2e-8):
        raise RuntimeError(
            f"finite-group solve failed validation: G={G}, alpha={alpha}, "
            f"sum_error={sum_error:.3e}, residual={residual:.3e}"
        )
    p = np.clip(p, 0.0, 1.0)
    return p, float(S), stats, float(sum_error), float(residual), fallback


def _build_k2_atlas(alphas, group_sizes, epsilons, reward_ratios):
    shape = (len(alphas), len(group_sizes), len(epsilons), len(reward_ratios))
    p_majority = np.empty(shape, dtype=np.float64)
    margin = np.empty(shape, dtype=np.float64)
    max_residual = 0.0
    rows = []
    for ai, alpha in enumerate(alphas):
        for gi, G in enumerate(group_sizes):
            for ei, eps in enumerate(epsilons):
                p1 = stationary_k2_batch(reward_ratios, int(G), float(alpha), float(eps), iterations=64)
                m = finite_group_margin(reward_ratios, int(G), float(alpha), float(eps))
                p_majority[ai, gi, ei] = p1
                margin[ai, gi, ei] = m

                active = (m > 0.0) & (np.asarray(reward_ratios) > 1.0)
                if np.any(active):
                    balance = (
                        np.asarray(reward_ratios)[active]
                        * effective_weight(p1[active], int(G), float(alpha), float(eps))
                        - effective_weight(1.0 - p1[active], int(G), float(alpha), float(eps))
                    )
                    scale = np.asarray(reward_ratios)[active] * min(float(G), 1.0 / float(eps)) ** float(alpha)
                    max_residual = max(max_residual, float(np.max(np.abs(balance) / scale)))
                for rho, prob, signed in zip(reward_ratios, p1, m):
                    rows.append([alpha, G, eps, rho, prob, 1.0 - prob, signed, int(signed > 0.0)])
    return p_majority, margin, max_residual, rows


def fig1_k2_hypergrid():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15))
    alphas = np.geomspace(0.15, 4.0, 51)
    group_sizes = np.array([2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])
    epsilons = np.array([1e-3, 1e-2, 3e-2, 1e-1, 3e-1])
    reward_ratios = np.array([1.25, 2.0, 4.0, 8.0, 16.0])

    p1, margin, residual, rows = _build_k2_atlas(alphas, group_sizes, epsilons, reward_ratios)
    _write_csv(
        "fig1_k2_hypergrid.csv",
        ["alpha", "G", "epsilon", "reward_ratio", "p_majority", "p_minority", "margin", "minority_kept"],
        rows,
    )

    # Independent cross-check against the original scalar K=2 routine.
    rng = np.random.default_rng(4041)
    checks = []
    for _ in range(40):
        ai = int(rng.integers(len(alphas)))
        gi = int(rng.integers(len(group_sizes)))
        ei = int(rng.integers(len(epsilons)))
        ri = int(rng.integers(len(reward_ratios)))
        legacy = meanfield_stationary_K2(
            np.array([reward_ratios[ri], 1.0]), int(group_sizes[gi]), float(alphas[ai]), float(epsilons[ei])
        )
        checks.append(abs(legacy - p1[ai, gi, ei, ri]))
    legacy_error = float(np.max(checks))

    numeric_kept = (1.0 - p1) > 2e-14
    theory_kept = margin > 0.0
    away = np.abs(margin) > 1e-9
    classification_mismatches = int(np.count_nonzero((numeric_kept != theory_kept) & away))
    if residual > 5e-12 or legacy_error > 2e-10 or classification_mismatches:
        raise AssertionError(
            f"K=2 atlas validation failed: residual={residual:.3e}, "
            f"legacy_error={legacy_error:.3e}, mismatches={classification_mismatches}"
        )

    # Panel a: one canonical slice, with the exact phase boundary overlaid.
    ei = int(np.where(epsilons == 1e-3)[0][0])
    ri = int(np.where(reward_ratios == 4.0)[0][0])
    mesh = axes[0].pcolormesh(
        _geometric_edges(alphas), _geometric_edges(group_sizes),
        (1.0 - p1[:, :, ei, ri]).T,
        shading="flat", cmap="viridis", vmin=0.0, vmax=0.20,
    )
    alpha_boundary = np.log(4.0) / np.log(np.minimum(group_sizes.astype(float), 1e3))
    axes[0].plot(alpha_boundary, group_sizes, "--", color="white", lw=2.6)
    axes[0].plot(alpha_boundary, group_sizes, "--", color=PALETTE[1], lw=1.2,
                 label=r"$\alpha_c=\ln4/\ln G$")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xticks([0.2, 0.5, 1, 2, 4])
    axes[0].xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    axes[0].set_yticks([2, 4, 8, 16, 32, 64, 128, 256])
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    axes[0].set_xlabel(r"IPS exponent $\alpha$")
    axes[0].set_ylabel("group size $G$")
    axes[0].set_title("Minority mass, $r_1/r_2$=4")
    axes[0].legend(loc="lower right", fontsize=6.2)
    cb = fig.colorbar(mesh, ax=axes[0], pad=0.02, shrink=0.90)
    cb.set_label("stationary $p_2$", fontsize=7.5)
    panel_label(axes[0], "a")

    # Panel b: clipping matters exactly when epsilon exceeds 1/G.
    for eps, color in zip(epsilons, PALETTE[:len(epsilons)]):
        prob = 1.0 - np.array([
            stationary_k2_batch(np.array([4.0]), int(G), 1.0, float(eps))[0]
            for G in group_sizes
        ])
        axes[1].plot(group_sizes, prob, "-o", ms=2.8, color=color, label=fr"$\epsilon$={eps:g}")
    axes[1].axhline(0.2, color="0.25", ls="--", lw=1.0, label=r"ideal $G\to\infty$: 0.2")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks([2, 4, 8, 16, 32, 64, 128, 256])
    axes[1].xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    axes[1].set_ylim(-0.01, 0.215)
    axes[1].set_xlabel(r"group size $G$ ($\alpha$=1)")
    axes[1].set_ylabel("stationary minority mass $p_2$")
    axes[1].set_title("Group-size vs clipping ceiling")
    axes[1].legend(loc="lower right", fontsize=5.8, ncol=2)
    panel_label(axes[1], "b")

    # Panel c: all four dimensions collapse onto one signed boundary margin.
    G_view = np.broadcast_to(group_sizes[None, :, None, None], p1.shape)
    x = margin.ravel()
    y = (1.0 - p1).ravel()
    c = np.log2(G_view).ravel()
    order = np.argsort(x)
    take = order[::max(1, len(order) // 8000)]
    scatter = axes[2].scatter(x[take], y[take], c=c[take], s=4.0, cmap="cividis", alpha=0.38,
                              edgecolors="none", rasterized=True)
    axes[2].axvline(0.0, color=PALETTE[1], ls="--", lw=1.2, label="exact boundary")
    axes[2].set_xlim(-3.0, 14.0)
    axes[2].set_ylim(-0.01, 0.43)
    axes[2].set_xlabel(r"margin $\alpha\ln\min(G,1/\epsilon)-\ln(r_1/r_2)$")
    axes[2].set_ylabel("stationary minority mass $p_2$")
    axes[2].set_title("Four-dimensional boundary collapse")
    axes[2].legend(loc="lower right", fontsize=6.2)
    cb = fig.colorbar(scatter, ax=axes[2], pad=0.02, shrink=0.90)
    cb.set_label(r"$\log_2 G$", fontsize=7.5)
    panel_label(axes[2], "c")

    savefig(fig, os.path.join(FIG_DIR, "fig1_k2_hypergrid.png"))
    out = {
        "grid_shape": list(p1.shape),
        "grid_points": int(p1.size),
        "max_scaled_stationarity_residual": residual,
        "max_abs_batch_vs_scalar_solver": legacy_error,
        "boundary_classification_mismatches": classification_mismatches,
    }
    SUMMARY["figure1"] = out
    return out


def fig2_general_k_atlas():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15))
    alphas = np.geomspace(0.06, 4.0, 45)
    group_sizes = np.array([2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])
    support = np.empty((len(alphas), len(group_sizes)), dtype=int)
    entropy = np.empty_like(support, dtype=np.float64)
    quality = np.empty_like(entropy)
    rows = []
    max_sum_error = 0.0
    max_residual = 0.0
    fallback_count = 0

    for ai, alpha in enumerate(alphas):
        for gi, G in enumerate(group_sizes):
            p, S, stats, sum_error, residual, fallback = _solve_general_checked(R5, int(G), float(alpha))
            support[ai, gi] = int(np.count_nonzero(p > PROB_TOL))
            entropy[ai, gi] = _normalized_entropy(p)
            quality[ai, gi] = float(np.dot(p, R5) / R5.max())
            max_sum_error = max(max_sum_error, sum_error)
            max_residual = max(max_residual, residual)
            fallback_count += int(fallback)
            rows.append([alpha, G, support[ai, gi], entropy[ai, gi], quality[ai, gi], S,
                         sum_error, residual, stats["n_w"], stats["n_dw"], int(fallback), *p])
    _write_csv(
        "fig2_general_k_atlas.csv",
        ["alpha", "G", "support_size", "normalized_entropy", "normalized_expected_reward", "S",
         "sum_error", "stationarity_residual", "n_w", "n_dw", "used_bisection_fallback",
         "p1", "p2", "p3", "p4", "p5"],
        rows,
    )
    support_decreases_with_alpha = int(np.count_nonzero(np.diff(support, axis=0) < 0))
    support_decreases_with_G = int(np.count_nonzero(np.diff(support, axis=1) < 0))
    if support_decreases_with_alpha or support_decreases_with_G:
        raise AssertionError(
            "support-size monotonicity failed on the primary atlas: "
            f"alpha decreases={support_decreases_with_alpha}, G decreases={support_decreases_with_G}"
        )

    xedges, yedges = _geometric_edges(alphas), _geometric_edges(group_sizes)
    support_cmap = plt.get_cmap("viridis", 5)
    support_norm = mcolors.BoundaryNorm(np.arange(0.5, 6.5), support_cmap.N)
    mesh0 = axes[0].pcolormesh(xedges, yedges, support.T, cmap=support_cmap, norm=support_norm,
                               shading="flat")
    axes[0].set_title("How many outcomes survive?")
    cb0 = fig.colorbar(mesh0, ax=axes[0], ticks=np.arange(1, 6), pad=0.02, shrink=0.90)
    cb0.set_label("support size", fontsize=7.5)

    mesh1 = axes[1].pcolormesh(xedges, yedges, entropy.T, cmap="magma", vmin=0.0, vmax=1.0,
                               shading="flat")
    axes[1].set_title("Diversity within the support")
    cb1 = fig.colorbar(mesh1, ax=axes[1], pad=0.02, shrink=0.90)
    cb1.set_label(r"entropy $H/\ln 5$", fontsize=7.5)

    for ax, label in zip(axes[:2], ["a", "b"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 4])
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax.set_yticks([2, 4, 8, 16, 32, 64, 128, 256])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax.set_xlabel(r"IPS exponent $\alpha$")
        ax.set_ylabel("group size $G$")
        panel_label(ax, label)
    axes[0].text(0.03, 0.04, "$r=(5,4,3,2,1)$", transform=axes[0].transAxes,
                 fontsize=6.5, color="white")

    # Minimum G for full support at alpha=1 over a second (K, spread) grid.
    outcome_counts = np.array([2, 3, 4, 5, 6, 8, 10])
    spreads = np.geomspace(1.25, 32.0, 9)
    minimum_G = np.full((len(outcome_counts), len(spreads)), np.nan)
    rows_min = []
    for ki, K in enumerate(outcome_counts):
        for ri, spread in enumerate(spreads):
            rewards = _geometric_rewards(int(K), float(spread))
            last_support = 0
            # Exhaustive integer search: reported values are true minima on
            # 2 <= G < 1/epsilon, not merely minima on a sparse plotting grid.
            for G in range(2, int(1.0 / EPS_DEFAULT)):
                p, _, _, sum_error, residual, fallback = _solve_general_checked(rewards, int(G), 1.0)
                max_sum_error = max(max_sum_error, sum_error)
                max_residual = max(max_residual, residual)
                fallback_count += int(fallback)
                last_support = int(np.count_nonzero(p > PROB_TOL))
                if last_support == K:
                    minimum_G[ki, ri] = G
                    break
            rows_min.append([K, spread, minimum_G[ki, ri], last_support])
    _write_csv("fig2_minimum_group_full_support.csv",
               ["K", "reward_spread_rmax_over_rmin", "minimum_G_for_full_support", "support_at_last_G"],
               rows_min)
    if np.any(~np.isfinite(minimum_G)):
        raise AssertionError("the G search failed to find full support for at least one atlas cell")

    mesh2 = axes[2].pcolormesh(
        _geometric_edges(spreads),
        np.concatenate([[outcome_counts[0] - 0.5], 0.5 * (outcome_counts[:-1] + outcome_counts[1:]),
                        [outcome_counts[-1] + 0.5 * (outcome_counts[-1] - outcome_counts[-2])]]),
        minimum_G,
        shading="flat", cmap="cividis", norm=mcolors.LogNorm(vmin=2, vmax=np.max(minimum_G)),
    )
    for ki, K in enumerate(outcome_counts):
        for ri, spread in enumerate(spreads):
            value = int(minimum_G[ki, ri])
            color = "white" if value >= 32 else "0.15"
            axes[2].text(spread, K, str(value), ha="center", va="center", fontsize=5.3, color=color)
    axes[2].set_xscale("log")
    axes[2].set_xticks([1.25, 2, 4, 8, 16, 32])
    axes[2].xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    axes[2].set_yticks(outcome_counts)
    axes[2].set_xlabel(r"reward spread $r_{\max}/r_{\min}$")
    axes[2].set_ylabel("number of outcomes $K$")
    axes[2].set_title(r"Minimum $G$ for full support ($\alpha$=1)")
    cb2 = fig.colorbar(mesh2, ax=axes[2], pad=0.02, shrink=0.90)
    cb2.set_label("minimum group size", fontsize=7.5)
    panel_label(axes[2], "c")

    if max_sum_error > 2e-9 or max_residual > 2e-8:
        raise AssertionError(
            f"general-K atlas validation failed: sum_error={max_sum_error:.3e}, residual={max_residual:.3e}"
        )
    savefig(fig, os.path.join(FIG_DIR, "fig2_general_k_atlas.png"))
    out = {
        "primary_grid_shape": list(support.shape),
        "primary_grid_points": int(support.size),
        "minimum_group_grid_shape": list(minimum_G.shape),
        "max_probability_sum_error": max_sum_error,
        "max_scaled_stationarity_residual": max_residual,
        "bisection_fallback_count": fallback_count,
        "support_decreases_with_alpha": support_decreases_with_alpha,
        "support_decreases_with_group_size": support_decreases_with_G,
        "minimum_G_range": [int(np.min(minimum_G)), int(np.max(minimum_G))],
        "support_at_alpha1_G16": int(support[np.argmin(np.abs(alphas - 1.0)),
                                                    np.where(group_sizes == 16)[0][0]]),
    }
    SUMMARY["figure2"] = out
    return out


def _interpolation_atlas(n=17, G=16):
    log_alpha = np.linspace(np.log(0.2), np.log(3.0), n)
    log_ratio = np.linspace(np.log(1.1), np.log(16.0), n)
    log_eps = np.linspace(np.log(1e-3), np.log(0.4), n)
    grid = TensorGrid({"log_alpha": log_alpha, "log_ratio": log_ratio, "log_epsilon": log_eps})
    minority = np.empty(grid.shape, dtype=np.float64)
    ratios = np.exp(log_ratio)
    for ai, la in enumerate(log_alpha):
        for ei, le in enumerate(log_eps):
            p1 = stationary_k2_batch(ratios, G, float(np.exp(la)), float(np.exp(le)))
            minority[ai, :, ei] = 1.0 - p1
    return grid, minority


def fig3_interpolation_audit():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15))
    fine_grid, fine_values = _interpolation_atlas(n=17, G=16)
    node_error = float(np.max(np.abs(fine_grid.interpolate(fine_values, fine_grid.points())
                                     - fine_values.ravel())))
    if node_error > 2e-15:
        raise AssertionError(f"interpolation is not exact at grid nodes: {node_error:.3e}")

    rng = np.random.default_rng(4043)
    n_holdout = 800
    low = np.array([axis[0] for axis in fine_grid.axes])
    high = np.array([axis[-1] for axis in fine_grid.axes])
    query = rng.uniform(low, high, size=(n_holdout, fine_grid.ndim))
    alpha_q, ratio_q, eps_q = np.exp(query).T
    exact = np.empty(n_holdout, dtype=np.float64)
    for i in range(n_holdout):
        exact[i] = 1.0 - stationary_k2_batch(
            np.array([ratio_q[i]]), 16, float(alpha_q[i]), float(eps_q[i])
        )[0]

    sizes = [5, 9, 17]
    predictions = {}
    metrics = {}
    margin = finite_group_margin(ratio_q, 16, alpha_q, eps_q)
    clip_distance = np.abs(np.log(eps_q * 16.0))
    smooth = (np.abs(margin) > 0.35) & (clip_distance > 0.35)
    transition = ~smooth
    for n in sizes:
        stride = 16 // (n - 1)
        indices = np.arange(0, 17, stride)
        grid = TensorGrid({name: axis[indices] for name, axis in zip(fine_grid.names, fine_grid.axes)})
        values = fine_values[np.ix_(indices, indices, indices)]
        pred = grid.interpolate(values, query)
        predictions[n] = pred
        error = np.abs(pred - exact)
        metrics[n] = {
            "rmse_all": float(np.sqrt(np.mean(error ** 2))),
            "rmse_smooth": float(np.sqrt(np.mean(error[smooth] ** 2))),
            "rmse_transition": float(np.sqrt(np.mean(error[transition] ** 2))),
            "mae": float(np.mean(error)),
            "max_abs": float(np.max(error)),
        }

    rows = []
    for i in range(n_holdout):
        rows.append([i, alpha_q[i], ratio_q[i], eps_q[i], margin[i], clip_distance[i], int(smooth[i]),
                     exact[i], predictions[5][i], predictions[9][i], predictions[17][i],
                     abs(predictions[5][i] - exact[i]), abs(predictions[9][i] - exact[i]),
                     abs(predictions[17][i] - exact[i])])
    _write_csv(
        "fig3_interpolation_holdout.csv",
        ["point", "alpha", "reward_ratio", "epsilon", "extinction_margin", "clip_log_distance",
         "smooth_region", "exact_p_minority", "interp_n5", "interp_n9", "interp_n17",
         "abs_error_n5", "abs_error_n9", "abs_error_n17"],
        rows,
    )
    _write_csv(
        "fig3_interpolation_convergence.csv",
        ["points_per_axis", "spacing", "rmse_all", "rmse_smooth", "rmse_transition", "mae", "max_abs"],
        [[n, 1.0 / (n - 1), metrics[n]["rmse_all"], metrics[n]["rmse_smooth"],
          metrics[n]["rmse_transition"], metrics[n]["mae"], metrics[n]["max_abs"]] for n in sizes],
    )

    # Panel a: a direct off-grid parity check.
    pred9 = predictions[9]
    scatter = axes[0].scatter(exact, pred9, c=np.abs(margin), s=11, cmap="viridis", alpha=0.72,
                              edgecolors="none", rasterized=True)
    limit = max(float(np.max(exact)), float(np.max(pred9))) * 1.03
    axes[0].plot([0, limit], [0, limit], "--", color="0.25", lw=1.0)
    axes[0].set_xlim(-0.006, limit)
    axes[0].set_ylim(-0.006, limit)
    axes[0].set_xlabel("direct finite-$G$ solve")
    axes[0].set_ylabel("9$^3$-grid interpolation")
    axes[0].set_title(f"Off-grid parity ($n$={n_holdout})")
    cb = fig.colorbar(scatter, ax=axes[0], pad=0.02, shrink=0.90)
    cb.set_label("distance from extinction boundary", fontsize=7.0)
    panel_label(axes[0], "a")

    # Panel b: errors cluster at either the extinction or clip-activation kink.
    error9 = np.abs(pred9 - exact)
    active_clip = eps_q > 1.0 / 16.0
    for flag, label, color, marker in [
        (False, r"$\epsilon<1/G$ (clip inactive)", PALETTE[0], "o"),
        (True, r"$\epsilon>1/G$ (clip active)", PALETTE[1], "s"),
    ]:
        selected = active_clip == flag
        axes[1].scatter(np.maximum(np.abs(margin[selected]), 1e-4),
                        np.maximum(error9[selected], 1e-16), s=8, alpha=0.45,
                        color=color, marker=marker, edgecolors="none", label=label, rasterized=True)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$|\alpha\ln\min(G,1/\epsilon)-\ln\rho|$")
    axes[1].set_ylabel("absolute interpolation error")
    axes[1].set_title("Error concentrates near the kink")
    axes[1].legend(loc="upper right", fontsize=5.8)
    panel_label(axes[1], "b")

    # Panel c: empirical convergence under uniform refinement.
    resolution = np.array(sizes) - 1
    categories = [("all", "rmse_all", PALETTE[0], "o"),
                  ("smooth", "rmse_smooth", PALETTE[2], "s"),
                  ("near a boundary", "rmse_transition", PALETTE[1], "^")]
    orders = {}
    for label, key, color, marker in categories:
        error = np.array([metrics[n][key] for n in sizes])
        slope, _, r2 = least_squares_line(np.log(resolution), np.log(error))
        order = -float(slope)
        orders[label] = {"order": order, "r_squared": float(r2)}
        axes[2].loglog(resolution, error, "-", marker=marker, ms=4, color=color,
                       label=f"{label}: order {order:.2f}")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(resolution)
    axes[2].xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    axes[2].set_xlabel("grid intervals per axis")
    axes[2].set_ylabel("holdout RMSE")
    axes[2].set_title("Uniform hypergrid refinement")
    axes[2].legend(loc="lower left", fontsize=6.0)
    panel_label(axes[2], "c")

    if not (metrics[17]["rmse_all"] < metrics[9]["rmse_all"] < metrics[5]["rmse_all"]):
        raise AssertionError("interpolation RMSE did not decrease under grid refinement")
    if not np.all(np.isfinite(list(metrics[17].values()))):
        raise AssertionError("non-finite interpolation metric")
    savefig(fig, os.path.join(FIG_DIR, "fig3_interpolation_audit.png"))
    out = {
        "fine_grid_shape": list(fine_grid.shape),
        "holdout_points": n_holdout,
        "max_grid_node_error": node_error,
        "metrics": {str(key): value for key, value in metrics.items()},
        "empirical_orders": orders,
        "smooth_holdout_points": int(np.count_nonzero(smooth)),
        "transition_holdout_points": int(np.count_nonzero(transition)),
    }
    SUMMARY["figure3"] = out
    return out


def _run_api_self_checks():
    """Fast checks independent of the plotted parameter selections."""
    grid = TensorGrid({"x": [-1.0, 0.0, 2.0], "y": [0.0, 3.0], "z": [4.0]})
    points = np.array([[-0.4, 0.7, 4.0], [1.3, 2.2, 4.0], [2.0, 3.0, 4.0]])
    values = grid.evaluate(lambda x, y, z: 2.0 + 3.0 * x - 0.5 * y + 0.25 * z)
    exact = 2.0 + 3.0 * points[:, 0] - 0.5 * points[:, 1] + 0.25 * points[:, 2]
    affine_error = float(np.max(np.abs(grid.interpolate(values, points) - exact)))
    if affine_error > 2e-14:
        raise AssertionError(f"multilinear interpolation failed affine exactness: {affine_error:.3e}")
    SUMMARY["api_self_checks"] = {"max_affine_interpolation_error": affine_error}


def main():
    total_start = time.perf_counter()
    print("=" * 72)
    print("EXPERIMENT 4: multidimensional finite-group diversity hypergrid")
    print("=" * 72)

    _run_api_self_checks()

    start = time.perf_counter()
    print("\n[Figure 1] K=2: four-dimensional boundary atlas ...")
    r1 = fig1_k2_hypergrid()
    print(f"  {r1['grid_points']:,} cells; max stationarity residual "
          f"{r1['max_scaled_stationarity_residual']:.2e}")
    print(f"  boundary classification mismatches: {r1['boundary_classification_mismatches']}")
    print(f"  ({time.perf_counter() - start:.1f}s)")

    start = time.perf_counter()
    print("\n[Figure 2] General K: support and entropy atlas ...")
    r2 = fig2_general_k_atlas()
    print(f"  max normalization error {r2['max_probability_sum_error']:.2e}; "
          f"max KKT residual {r2['max_scaled_stationarity_residual']:.2e}")
    print(f"  minimum full-support G spans {r2['minimum_G_range'][0]} to {r2['minimum_G_range'][1]}")
    print(f"  ({time.perf_counter() - start:.1f}s)")

    start = time.perf_counter()
    print("\n[Figure 3] Tensor-grid interpolation holdout audit ...")
    r3 = fig3_interpolation_audit()
    print(f"  node exactness {r3['max_grid_node_error']:.2e}; "
          f"17^3 holdout RMSE {r3['metrics']['17']['rmse_all']:.2e}")
    print(f"  ({time.perf_counter() - start:.1f}s)")

    SUMMARY["runtime_seconds"] = float(time.perf_counter() - total_start)
    SUMMARY["configuration"] = {
        "epsilon_default": EPS_DEFAULT,
        "random_seeds": [4041, 4043],
        "probability_support_tolerance": PROB_TOL,
    }
    summary_path = os.path.join(DATA_DIR, "exp4_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(SUMMARY, fp, indent=2, default=float)

    print(f"\nAll Experiment 4 figures written to results/{EXP_NAME}/figures/.")
    print(f"Data tables and summary written to results/{EXP_NAME}/data/.")
    print(f"Total runtime: {SUMMARY['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
