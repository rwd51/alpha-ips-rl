"""
EXPERIMENT 4 -- A multidimensional finite-group diversity atlas.

Experiments 2 and 3 varied one or two parameters at a time.  This experiment
asks whether their finite-group conclusions survive a genuine hyperparameter
grid and whether that grid can be used as a trustworthy numerical surrogate.

  1. FIGURE 1 -- Exact K=2 atlas over (alpha, G, epsilon, reward ratio).
     The numerical stationary point is compared with the analytical boundary
         alpha log(min(G, 1/epsilon)) = log(r1/r2).
     The batched solver uses that boundary to choose its branch, so every
     point is re-solved with the original scalar solver of Experiment 2,
     which never looks at the margin.  The grid also exposes when
     probability clipping, rather than group size, caps the correction.

  2. FIGURE 2 -- General K atlas.
     For r=(5,4,3,2,1), nested finite-group root-finding maps support size and
     entropy over (alpha, G).  A second grid over outcome count and reward
     spread measures the minimum G needed to retain every outcome at alpha=1.

  3. FIGURE 3 -- Hypergrid interpolation audit.
     A three-dimensional tensor grid in log(alpha), log(reward ratio), and
     log(epsilon) is tested on independently sampled off-grid points.  The
     solution has derivative kinks on the extinction surface m=0 and at every
     clipping threshold epsilon=k/G.  Each error is attributed to the grid
     cell used to interpolate it, and convergence is measured separately for
     points whose cells are kink-free or kinked at every refinement level.

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
from src.hypergrid import (TensorGrid, clip_kink_epsilons, finite_group_margin, k2_cell_kinks,
                           stationary_k2_batch)
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
    # "+ 0.0" turns the -0.0 of a one-point distribution into 0.0.
    return float(-np.sum(p[positive] * np.log(p[positive])) / np.log(len(p))) + 0.0


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


def _k2_balance_residual(p1, reward_ratios, G, alpha, eps):
    """Relative K=2 balance residual |rho w(p1) - w(p2)| / (rho w(p1) + w(p2)).

    Dividing by the size of the two balanced terms (not by the weight ceiling,
    which reaches 7e10 on this atlas) keeps the residual meaningful at large
    alpha and G.
    """
    majority = np.asarray(reward_ratios) * effective_weight(p1, G, alpha, eps)
    minority = effective_weight(1.0 - np.asarray(p1), G, alpha, eps)
    return np.abs(majority - minority) / (majority + minority)


def _build_k2_atlas(alphas, group_sizes, epsilons, reward_ratios):
    shape = (len(alphas), len(group_sizes), len(epsilons), len(reward_ratios))
    p_majority = np.empty(shape, dtype=np.float64)
    margin = np.empty(shape, dtype=np.float64)
    max_residual = 0.0
    for ai, alpha in enumerate(alphas):
        for gi, G in enumerate(group_sizes):
            for ei, eps in enumerate(epsilons):
                p1 = stationary_k2_batch(reward_ratios, int(G), float(alpha), float(eps), iterations=64)
                m = finite_group_margin(reward_ratios, int(G), float(alpha), float(eps))
                p_majority[ai, gi, ei] = p1
                margin[ai, gi, ei] = m

                active = (m > 0.0) & (np.asarray(reward_ratios) > 1.0)
                if np.any(active):
                    residual = _k2_balance_residual(p1[active], np.asarray(reward_ratios)[active],
                                                    int(G), float(alpha), float(eps))
                    max_residual = max(max_residual, float(np.max(residual)))
    return p_majority, margin, max_residual


def _scalar_solver_check(alphas, group_sizes, epsilons, reward_ratios, p_majority, margin):
    """Re-solve every atlas point with the original scalar K=2 solver.

    src/finite_group.py::meanfield_stationary_K2 decides collapse from the
    sign of the balance function next to p1 = 1 and evaluates the weight as
    phi_G(p)/p from the direct binomial sum.  Neither the margin m nor the
    batched solver enters, so its phase classification is an independent
    test of the analytical boundary rather than a restatement of it.
    """
    scalar = np.empty_like(p_majority)
    for ai, gi, ei, ri in np.ndindex(p_majority.shape):
        scalar[ai, gi, ei, ri] = meanfield_stationary_K2(
            np.array([reward_ratios[ri], 1.0]), int(group_sizes[gi]), float(alphas[ai]), float(epsilons[ei])
        )
    kept = scalar < 1.0
    # Points with |m| at rounding level cannot be classified numerically; on
    # this grid they are exactly m = 0 (alpha=4, G=2, ratio=16 at every eps).
    on_boundary = np.abs(margin) <= 1e-9
    check = {
        "max_abs_batch_vs_scalar_solver": float(np.max(np.abs(scalar - p_majority))),
        "boundary_classification_mismatches": int(np.count_nonzero((kept != (margin > 0.0)) & ~on_boundary)),
        "classification_points_compared": int(np.count_nonzero(~on_boundary)),
        "points_on_boundary": int(np.count_nonzero(on_boundary)),
        "points_on_boundary_exactly_zero_margin": int(np.count_nonzero(margin == 0.0)),
        "points_on_boundary_collapsed": int(np.count_nonzero(on_boundary & ~kept)),
    }
    return scalar, check


def fig1_k2_hypergrid():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15))
    alphas = np.geomspace(0.15, 4.0, 51)
    group_sizes = np.array([2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])
    epsilons = np.array([1e-3, 1e-2, 3e-2, 1e-1, 3e-1])
    reward_ratios = np.array([1.25, 2.0, 4.0, 8.0, 16.0])

    p1, margin, residual = _build_k2_atlas(alphas, group_sizes, epsilons, reward_ratios)
    scalar_p1, check = _scalar_solver_check(alphas, group_sizes, epsilons, reward_ratios, p1, margin)
    rows = []
    for index in np.ndindex(p1.shape):
        ai, gi, ei, ri = index
        rows.append([alphas[ai], group_sizes[gi], epsilons[ei], reward_ratios[ri], p1[index], 1.0 - p1[index],
                     margin[index], int(margin[index] > 0.0), scalar_p1[index], int(scalar_p1[index] < 1.0)])
    _write_csv(
        "fig1_k2_hypergrid.csv",
        ["alpha", "G", "epsilon", "reward_ratio", "p_majority", "p_minority", "margin", "minority_kept",
         "p_majority_scalar_solver", "minority_kept_scalar_solver"],
        rows,
    )
    if (residual > 5e-12 or check["max_abs_batch_vs_scalar_solver"] > 2e-10
            or check["boundary_classification_mismatches"]
            or check["points_on_boundary_collapsed"] != check["points_on_boundary"]):
        raise AssertionError(f"K=2 atlas validation failed: residual={residual:.3e}, {check}")

    # Panel a: one canonical slice, with the exact phase boundary overlaid.
    # The colour scale spans the data: for alpha > 1 the minority holds more
    # than the alpha = 1 ideal of 0.2.
    ei = int(np.where(epsilons == 1e-3)[0][0])
    ri = int(np.where(reward_ratios == 4.0)[0][0])
    minority = 1.0 - p1[:, :, ei, ri]
    vmax = float(np.ceil(minority.max() * 20.0) / 20.0)
    mesh = axes[0].pcolormesh(
        _geometric_edges(alphas), _geometric_edges(group_sizes), minority.T,
        shading="flat", cmap="viridis", vmin=0.0, vmax=vmax,
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
    axes[0].legend(loc="lower right", fontsize=6.2, frameon=True, facecolor="white", framealpha=0.85,
                   edgecolor="none")
    cb = fig.colorbar(mesh, ax=axes[0], pad=0.02, shrink=0.90)
    cb.set_label("stationary $p_2$", fontsize=7.5)
    panel_label(axes[0], "a")

    a_peak, g_peak = np.unravel_index(int(np.argmax(minority)), minority.shape)
    top = int(np.argmax(alphas))
    top_alpha = float(alphas[top])
    g_row_peak = int(np.argmax(minority[top]))
    peak_p2 = float(minority[top, g_row_peak])
    peak_G = int(group_sizes[g_row_peak])
    weight_over_ideal = {
        name: float(effective_weight(p, peak_G, top_alpha, 1e-3)[0] * p ** top_alpha)
        for name, p in (("minority", peak_p2), ("majority", 1.0 - peak_p2))
    }

    # Panel b: clipping matters exactly when epsilon exceeds 1/G.  The three
    # smallest epsilons coincide to plotting precision (they differ only where
    # epsilon > 1/G and the minority is essentially never seen once), so they
    # get nested hollow markers instead of hiding one another.
    alpha1 = {}
    styles = {1e-3: ("o", 5.4, "none"), 1e-2: ("s", 3.6, "none"), 3e-2: ("^", 2.4, None),
              1e-1: ("D", 2.4, None), 3e-1: ("v", 2.8, None)}
    for eps, color in zip(epsilons, PALETTE[:len(epsilons)]):
        prob = 1.0 - np.array([
            stationary_k2_batch(np.array([4.0]), int(G), 1.0, float(eps))[0]
            for G in group_sizes
        ])
        alpha1[f"{eps:g}"] = prob.tolist()
        marker, size, face = styles[float(eps)]
        axes[1].plot(group_sizes, prob, "-", marker=marker, ms=size, mfc=face, mew=0.9, lw=1.1,
                     color=color, label=fr"$\epsilon$={eps:g}")
    small = np.array([alpha1[key] for key in ("0.001", "0.01", "0.03")])
    small_gap = float(np.max(small.max(axis=0) - small.min(axis=0)))
    axes[1].axhline(0.2, color="0.25", ls="--", lw=1.0, label=r"ideal $G\to\infty$: 0.2")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks([2, 4, 8, 16, 32, 64, 128, 256])
    axes[1].xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    axes[1].set_ylim(-0.01, 0.215)
    axes[1].set_xlabel(r"group size $G$ ($\alpha$=1)")
    axes[1].set_ylabel("stationary minority mass $p_2$")
    axes[1].set_title("Group-size vs clipping ceiling")
    axes[1].legend(loc="center right", fontsize=5.8, ncol=2, title=r"curves for $\epsilon\leq0.03$ overlap",
                   title_fontsize=5.8)
    panel_label(axes[1], "b")

    # First surviving group size at alpha=1 from an integer search, decided
    # by the margin-free scalar solver (G=5 is not on the plotted grid).
    first_G = next(G for G in range(2, 1000)
                   if meanfield_stationary_K2(np.array([4.0, 1.0]), G, 1.0, 1e-3) < 1.0)
    first_p2 = 1.0 - float(stationary_k2_batch(np.array([4.0]), first_G, 1.0, 1e-3)[0])

    # Panel c: all four dimensions collapse onto one signed boundary margin.
    # Every atlas point is drawn, and the limits cover the full data range.
    G_view = np.broadcast_to(group_sizes[None, :, None, None], p1.shape)
    x = margin.ravel()
    y = (1.0 - p1).ravel()
    c = np.log2(G_view).ravel()
    order = np.argsort(x)
    scatter = axes[2].scatter(x[order], y[order], c=c[order], s=3.0, cmap="cividis", alpha=0.38,
                              edgecolors="none", rasterized=True)
    axes[2].axvline(0.0, color=PALETTE[1], ls="--", lw=1.2, label="exact boundary")
    axes[2].set_xlim(np.floor(x.min()) - 0.5, np.ceil(x.max()) + 0.5)
    axes[2].set_ylim(-0.01, float(np.ceil(y.max() * 20.0) / 20.0) + 0.01)
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
        "max_relative_balance_residual": residual,
        **check,
        "alpha1_rho4_minority_mass": {"group_sizes": group_sizes.tolist(), "by_epsilon": alpha1},
        "alpha1_rho4_max_gap_between_eps_le_0.03": small_gap,
        "alpha1_rho4_eps0.001_first_surviving_G": int(first_G),
        "alpha1_rho4_eps0.001_minority_mass_at_first_surviving_G": first_p2,
        "panel_a_max_minority_mass": float(minority.max()),
        "panel_a_argmax": {"alpha": float(alphas[a_peak]), "G": int(group_sizes[g_peak])},
        "panel_a_colour_scale_max": vmax,
        "panel_a_top_alpha_row": {
            "alpha": top_alpha,
            "group_sizes": group_sizes.tolist(),
            "p_minority": minority[top].tolist(),
            "ideal_G_to_infinity": float(1.0 / (1.0 + 4.0 ** (1.0 / top_alpha))),
            "peak_G": peak_G,
            "peak_p_minority": peak_p2,
            "weight_over_ideal_at_peak": weight_over_ideal,
        },
        "panel_c_points_drawn": int(x.size),
        "margin_range": [float(x.min()), float(x.max())],
        "minority_mass_max": float(y.max()),
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
    primary = {"max_sum_error": 0.0, "max_residual": 0.0, "fallbacks": 0}

    for ai, alpha in enumerate(alphas):
        for gi, G in enumerate(group_sizes):
            p, S, stats, sum_error, residual, fallback = _solve_general_checked(R5, int(G), float(alpha))
            support[ai, gi] = int(np.count_nonzero(p > PROB_TOL))
            entropy[ai, gi] = _normalized_entropy(p)
            quality[ai, gi] = float(np.dot(p, R5) / R5.max())
            primary["max_sum_error"] = max(primary["max_sum_error"], sum_error)
            primary["max_residual"] = max(primary["max_residual"], residual)
            primary["fallbacks"] += int(fallback)
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

    # alpha = 1 is not a node of the alpha grid, so the reported distributions
    # at alpha = 1 are solved (and validated) separately.
    alpha1 = {}
    for G in (2, 4, 8, 16, 64, 256):
        p, _, _, sum_error, residual, fallback = _solve_general_checked(R5, G, 1.0)
        primary["max_sum_error"] = max(primary["max_sum_error"], sum_error)
        primary["max_residual"] = max(primary["max_residual"], residual)
        primary["fallbacks"] += int(fallback)
        alpha1[str(G)] = {"p": p.tolist(), "support_size": int(np.count_nonzero(p > PROB_TOL)),
                          "normalized_entropy": _normalized_entropy(p)}
    ideal = R5 / R5.sum()

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
    search = {"max_sum_error": 0.0, "max_residual": 0.0, "fallbacks": 0, "solves": 0}
    for ki, K in enumerate(outcome_counts):
        for ri, spread in enumerate(spreads):
            rewards = _geometric_rewards(int(K), float(spread))
            last_support = 0
            # Exhaustive integer search: reported values are true minima on
            # 2 <= G < 1/epsilon, not merely minima on a sparse plotting grid.
            for G in range(2, int(1.0 / EPS_DEFAULT)):
                p, _, _, sum_error, residual, fallback = _solve_general_checked(rewards, int(G), 1.0)
                search["max_sum_error"] = max(search["max_sum_error"], sum_error)
                search["max_residual"] = max(search["max_residual"], residual)
                search["fallbacks"] += int(fallback)
                search["solves"] += 1
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
            # Text colour from the cell's own luminance, so dark cells get white digits.
            red, green, blue, _ = mesh2.cmap(mesh2.norm(value))
            color = "white" if 0.299 * red + 0.587 * green + 0.114 * blue < 0.5 else "0.1"
            axes[2].text(spread, K, str(value), ha="center", va="center", fontsize=5.3, color=color)
    axes[2].grid(False)
    axes[2].set_xscale("log")
    axes[2].set_xticks([1.25, 2, 4, 8, 16, 32])
    axes[2].xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    axes[2].set_yticks(outcome_counts)
    axes[2].set_xlabel(r"reward spread $r_{\max}/r_{\min}$")
    axes[2].set_ylabel("number of outcomes $K$")
    axes[2].set_title(r"Minimum $G$ for full support ($\alpha$=1)")
    cb2 = fig.colorbar(mesh2, ax=axes[2], pad=0.02, shrink=0.90, ticks=[2, 4, 8, 16, 32, 64])
    cb2.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    cb2.minorticks_off()
    cb2.set_label("minimum group size", fontsize=7.5)
    panel_label(axes[2], "c")

    for name, stats in (("primary atlas", primary), ("minimum-G search", search)):
        if stats["max_sum_error"] > 2e-9 or stats["max_residual"] > 2e-8:
            raise AssertionError(
                f"general-K {name} validation failed: sum_error={stats['max_sum_error']:.3e}, "
                f"residual={stats['max_residual']:.3e}"
            )
    savefig(fig, os.path.join(FIG_DIR, "fig2_general_k_atlas.png"))
    out = {
        "primary_grid_shape": list(support.shape),
        "primary_grid_points": int(support.size),
        "minimum_group_grid_shape": list(minimum_G.shape),
        "max_probability_sum_error": primary["max_sum_error"],
        "max_scaled_stationarity_residual": primary["max_residual"],
        "bisection_fallback_count": primary["fallbacks"],
        "minimum_group_search": {
            "solves": search["solves"],
            "max_probability_sum_error": search["max_sum_error"],
            "max_scaled_stationarity_residual": search["max_residual"],
            "bisection_fallback_count": search["fallbacks"],
        },
        "support_decreases_with_alpha": support_decreases_with_alpha,
        "support_decreases_with_group_size": support_decreases_with_G,
        "minimum_G_range": [int(np.min(minimum_G)), int(np.max(minimum_G))],
        "alpha1_distributions": alpha1,
        "ideal_reward_proportional": {"p": ideal.tolist(), "normalized_entropy": _normalized_entropy(ideal)},
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


CELL_LABELS = ("extinct", "clean", "clip_kink", "extinction_kink", "both_kinks")
# Plot categories: key, legend label, colour, marker.  Clean includes cells
# where the minority is extinct throughout (interpolation is exact there).
FIG3_CATEGORIES = (
    ("clean", "none", PALETTE[0], "o"),
    ("clip", r"$\epsilon=k/G$ only", PALETTE[2], "s"),
    ("extinction", r"$m=0$, possibly also $\epsilon=k/G$", PALETTE[1], "^"),
)


def _cell_labels(grid, query, G):
    """Label the cell used to interpolate each point of one grid level."""
    lower, upper = grid.cell_bounds(query)
    flags = k2_cell_kinks(np.exp(lower), np.exp(upper), G)
    labels = np.full(len(query), "clean", dtype=object)
    labels[flags["extinct"]] = "extinct"
    labels[flags["clip"] & ~flags["extinction"]] = "clip_kink"
    labels[flags["extinction"] & ~flags["clip"]] = "extinction_kink"
    labels[flags["extinction"] & flags["clip"]] = "both_kinks"
    return labels, flags


def _plot_category(labels):
    category = np.full(len(labels), "clean", dtype=object)
    category[labels == "clip_kink"] = "clip"
    category[(labels == "extinction_kink") | (labels == "both_kinks")] = "extinction"
    return category


def _rmse(error):
    return float(np.sqrt(np.mean(error ** 2)))


def fig3_interpolation_audit():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15))
    G = 16
    fine_grid, fine_values = _interpolation_atlas(n=17, G=G)
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
            np.array([ratio_q[i]]), G, float(alpha_q[i]), float(eps_q[i])
        )[0]
    margin = finite_group_margin(ratio_q, G, alpha_q, eps_q)

    sizes = [5, 9, 17]
    predictions, labels, flags = {}, {}, {}
    for n in sizes:
        stride = 16 // (n - 1)
        indices = np.arange(0, 17, stride)
        grid = TensorGrid({name: axis[indices] for name, axis in zip(fine_grid.names, fine_grid.axes)})
        values = fine_values[np.ix_(indices, indices, indices)]
        predictions[n] = grid.interpolate(values, query)
        labels[n], flags[n] = _cell_labels(grid, query, G)

    # The grids are nested: every finer cell lies inside the coarser cell that
    # holds the same point.  A point whose 5-point cell is kink-free (and not
    # entirely extinct) therefore stays in kink-free cells at every level, and
    # a kink inside a point's 17-point cell lies inside all its coarser cells.
    # These two fixed populations give convergence orders that do not mix
    # smooth and kinked cells as the grid is refined.
    clean_every = flags[5]["clean"] & ~flags[5]["extinct"]
    kink_every = ~flags[17]["clean"]
    for n in sizes:
        if (np.any(~flags[n]["clean"][clean_every]) or np.any(flags[n]["extinct"][clean_every])
                or np.any(flags[n]["clean"][kink_every])):
            raise AssertionError("cell kink classification is not nested across refinement levels")

    extinct_points = exact == 0.0
    metrics = {}
    for n in sizes:
        error = np.abs(predictions[n] - exact)
        worst = int(np.argmax(error))
        own_clean = labels[n] == "clean"                  # kink-free cell, minority alive somewhere
        own_kinked = ~flags[n]["clean"]
        metrics[n] = {
            "rmse_all": _rmse(error),
            "rmse_clean_every_level": _rmse(error[clean_every]),
            "rmse_kink_every_level": _rmse(error[kink_every]),
            "mae": float(np.mean(error)),
            "max_abs": float(np.max(error)),
            "own_cell_counts": {label: int(np.count_nonzero(labels[n] == label)) for label in CELL_LABELS},
            "rmse_own_cell_kink_free": _rmse(error[own_clean]),
            "rmse_own_cell_kinked": _rmse(error[own_kinked]),
            "max_abs_own_cell_kink_free": float(np.max(error[own_clean])),
            "errors_above_1e-2": int(np.count_nonzero(error > 1e-2)),
            "errors_above_1e-2_in_kinked_cells": int(np.count_nonzero((error > 1e-2) & own_kinked)),
            "extinct_predicted_above_1e-3": int(np.count_nonzero(extinct_points & (predictions[n] > 1e-3))),
            "extinct_predicted_above_1e-2": int(np.count_nonzero(extinct_points & (predictions[n] > 1e-2))),
            "surviving_predicted_zero": int(np.count_nonzero(~extinct_points & (predictions[n] == 0.0))),
            "worst_point": {"abs_error": float(error[worst]), "exact": float(exact[worst]),
                            "alpha": float(alpha_q[worst]), "reward_ratio": float(ratio_q[worst]),
                            "epsilon": float(eps_q[worst]), "margin": float(margin[worst]),
                            "cell": str(labels[n][worst])},
        }
    orders = {}
    for name, key in (("all", "rmse_all"), ("clean_every_level", "rmse_clean_every_level"),
                      ("kink_every_level", "rmse_kink_every_level")):
        error = [metrics[n][key] for n in sizes]
        orders[name] = {"4_to_8_intervals": float(np.log2(error[0] / error[1])),
                        "8_to_16_intervals": float(np.log2(error[1] / error[2]))}
    for key in ("rmse_all", "rmse_clean_every_level", "rmse_kink_every_level"):
        error = [metrics[n][key] for n in sizes]
        if not error[2] < error[1] < error[0]:
            raise AssertionError(f"interpolation {key} did not decrease under grid refinement")
    if not np.all(np.isfinite([metrics[17][key] for key in ("rmse_all", "mae", "max_abs")])):
        raise AssertionError("non-finite interpolation metric")

    # Above eps = 1/G the thresholds k/G are about as dense in log eps as the
    # 17-point grid; count the surviving holdout points there that sit in a
    # cell containing one of them.
    clip_live = (eps_q > 1.0 / G) & ~flags[17]["extinct"]
    kinks_in_range = clip_kink_epsilons(G, float(np.exp(low[2])), float(np.exp(high[2])))

    # Size of each clip kink in the stationary minority mass for one example
    # that survives at every eps in range (alpha=1, rho=2), measured by
    # one-sided differences in log eps, with a control point between kinks.
    def minority_at(eps):
        return 1.0 - float(stationary_k2_batch(np.array([2.0]), G, 1.0, float(eps))[0])

    def slope_jump(eps, step=1e-6):
        left = (minority_at(eps) - minority_at(eps * np.exp(-step))) / step
        right = (minority_at(eps * np.exp(step)) - minority_at(eps)) / step
        return float(right - left)

    kink_example = {
        "alpha": 1.0,
        "reward_ratio": 2.0,
        "epsilon": kinks_in_range.tolist(),
        "p_minority": [minority_at(eps) for eps in kinks_in_range],
        "slope_jump_dp2_dlog_eps": [slope_jump(eps) for eps in kinks_in_range],
        "control_epsilon": 0.09,
        "control_slope_jump": slope_jump(0.09),
    }

    rows = []
    for i in range(n_holdout):
        rows.append([i, alpha_q[i], ratio_q[i], eps_q[i], margin[i], exact[i],
                     predictions[5][i], predictions[9][i], predictions[17][i],
                     abs(predictions[5][i] - exact[i]), abs(predictions[9][i] - exact[i]),
                     abs(predictions[17][i] - exact[i]), labels[5][i], labels[9][i], labels[17][i],
                     int(clean_every[i]), int(kink_every[i])])
    _write_csv(
        "fig3_interpolation_holdout.csv",
        ["point", "alpha", "reward_ratio", "epsilon", "extinction_margin", "exact_p_minority",
         "interp_n5", "interp_n9", "interp_n17", "abs_error_n5", "abs_error_n9", "abs_error_n17",
         "cell_n5", "cell_n9", "cell_n17", "clean_cell_every_level", "kink_cell_every_level"],
        rows,
    )
    _write_csv(
        "fig3_interpolation_convergence.csv",
        ["points_per_axis", "spacing", "rmse_all", "rmse_clean_every_level", "rmse_kink_every_level",
         "mae", "max_abs"],
        [[n, 1.0 / (n - 1), metrics[n]["rmse_all"], metrics[n]["rmse_clean_every_level"],
          metrics[n]["rmse_kink_every_level"], metrics[n]["mae"], metrics[n]["max_abs"]] for n in sizes],
    )

    # Panels a-b colour each point by what its 9^3-grid cell contains.
    pred9 = predictions[9]
    error9 = np.abs(pred9 - exact)
    category9 = _plot_category(labels[9])

    # Panel a: a direct off-grid parity check.
    for key, label, color, marker in FIG3_CATEGORIES:
        selected = category9 == key
        axes[0].scatter(exact[selected], pred9[selected], s=9, color=color, marker=marker, alpha=0.7,
                        edgecolors="none", label=f"{label} ({int(np.count_nonzero(selected))})",
                        rasterized=True)
    limit = max(float(np.max(exact)), float(np.max(pred9))) * 1.03
    axes[0].plot([0, limit], [0, limit], "--", color="0.25", lw=1.0)
    axes[0].set_xlim(-0.006, limit)
    axes[0].set_ylim(-0.006, limit)
    axes[0].set_xlabel("direct finite-$G$ solve")
    axes[0].set_ylabel("9$^3$-grid interpolation")
    axes[0].set_title(f"Off-grid parity ($n$={n_holdout})")
    axes[0].legend(loc="upper left", fontsize=5.8, title="kink inside the 9$^3$ cell", title_fontsize=5.8,
                   markerscale=1.4)
    panel_label(axes[0], "a")

    # Panel b: large errors come from cells that contain a kink.  Points in
    # cells where the minority is extinct throughout have zero error and
    # cannot be drawn on a log axis; their number is stated instead.
    drawn = error9 > 0.0
    for key, label, color, marker in FIG3_CATEGORIES:
        selected = (category9 == key) & drawn
        axes[1].scatter(np.maximum(np.abs(margin[selected]), 1e-4), error9[selected], s=8, alpha=0.5,
                        color=color, marker=marker, edgecolors="none", rasterized=True,
                        label=label.split(",")[0])
    axes[1].legend(loc="center left", fontsize=5.8, title="kink inside the 9$^3$ cell", title_fontsize=5.8,
                   markerscale=1.4)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$|m|=|\alpha\ln\min(G,1/\epsilon)-\ln\rho|$")
    axes[1].set_ylabel("absolute interpolation error")
    axes[1].set_title("Large errors sit in kinked cells")
    axes[1].text(0.03, 0.03, f"{int(np.count_nonzero(~drawn))} points in extinct cells:\nerror exactly 0 (not shown)",
                 transform=axes[1].transAxes, fontsize=5.8, color=INK, va="bottom")
    panel_label(axes[1], "b")

    # Panel c: empirical convergence of the fixed populations.
    resolution = np.array(sizes) - 1
    curves = (("all points", "rmse_all", "all", "0.2", "o", n_holdout),
              ("kink-free cells", "rmse_clean_every_level", "clean_every_level", PALETTE[0], "s",
               int(np.count_nonzero(clean_every))),
              ("kinked cells", "rmse_kink_every_level", "kink_every_level", PALETTE[3], "^",
               int(np.count_nonzero(kink_every))))
    for label, key, name, color, marker, count in curves:
        error = np.array([metrics[n][key] for n in sizes])
        order = orders[name]
        axes[2].loglog(resolution, error, "-", marker=marker, ms=4, color=color,
                       label=f"{label} ({count}): {order['4_to_8_intervals']:.2f}, "
                             f"{order['8_to_16_intervals']:.2f}")
    reference = np.array([4.0, 16.0])
    first_clean = metrics[5]["rmse_clean_every_level"]
    first_kink = metrics[5]["rmse_kink_every_level"]
    axes[2].loglog(reference, 0.55 * first_clean * (reference / 4.0) ** -2.0, ":", color="0.45", lw=1.0)
    axes[2].loglog(reference, 1.6 * first_kink * (reference / 4.0) ** -1.0, ":", color="0.45", lw=1.0)
    axes[2].text(11.0, 0.55 * first_clean * (11.0 / 4.0) ** -2.0 * 0.62, "slope $-2$", fontsize=6.0,
                 color="0.35", ha="center", va="top")
    axes[2].text(11.0, 1.6 * first_kink * (11.0 / 4.0) ** -1.0 * 1.22, "slope $-1$", fontsize=6.0,
                 color="0.35", ha="center", va="bottom")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(resolution)
    axes[2].xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    axes[2].set_xlabel("grid intervals per axis")
    axes[2].set_ylabel("holdout RMSE")
    axes[2].set_title("Uniform hypergrid refinement")
    axes[2].legend(loc="lower left", fontsize=5.8, title=r"observed order $4\to8$, $8\to16$",
                   title_fontsize=5.8)
    panel_label(axes[2], "c")

    savefig(fig, os.path.join(FIG_DIR, "fig3_interpolation_audit.png"))
    out = {
        "fine_grid_shape": list(fine_grid.shape),
        "holdout_points": n_holdout,
        "max_grid_node_error": node_error,
        "clip_kinks_in_range": kinks_in_range.tolist(),
        "clip_kink_example": kink_example,
        "metrics": {str(key): value for key, value in metrics.items()},
        "observed_orders": orders,
        "clean_every_level_points": int(np.count_nonzero(clean_every)),
        "kink_every_level_points": int(np.count_nonzero(kink_every)),
        "extinct_holdout_points": int(np.count_nonzero(extinct_points)),
        "clip_active_live_points_n17": int(np.count_nonzero(clip_live)),
        "clip_active_live_points_in_clip_kink_cells_n17": int(np.count_nonzero(clip_live & flags[17]["clip"])),
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
    print(f"  {r1['grid_points']:,} cells; max relative balance residual "
          f"{r1['max_relative_balance_residual']:.2e}")
    print(f"  margin-free scalar solver: max |dp1| {r1['max_abs_batch_vs_scalar_solver']:.2e}; "
          f"classification mismatches {r1['boundary_classification_mismatches']} of "
          f"{r1['classification_points_compared']:,} (+{r1['points_on_boundary']} on m=0, "
          f"{r1['points_on_boundary_collapsed']} collapsed)")
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
    clean, kink = r3["observed_orders"]["clean_every_level"], r3["observed_orders"]["kink_every_level"]
    print(f"  node exactness {r3['max_grid_node_error']:.2e}; "
          f"17^3 holdout RMSE {r3['metrics']['17']['rmse_all']:.2e}")
    print(f"  observed order, kink-free cells {clean['4_to_8_intervals']:.2f}, {clean['8_to_16_intervals']:.2f}; "
          f"kinked cells {kink['4_to_8_intervals']:.2f}, {kink['8_to_16_intervals']:.2f}")
    print(f"  ({time.perf_counter() - start:.1f}s)")

    SUMMARY["runtime_seconds"] = float(time.perf_counter() - total_start)
    SUMMARY["configuration"] = {
        "epsilon_default": EPS_DEFAULT,
        "random_seeds": [4043],
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
