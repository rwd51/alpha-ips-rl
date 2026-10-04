"""
Figures 2-6 and 8 for case 3 (figure 1 and 7 come from landscape.py).

  fig2_training_traces.png    p_i over training (one seed, log scale) and the seed
                              spread of max p, entropy, items kept (replay vs the
                              mean-field ODE) and held-out CTR, G = 64
  fig3_alpha_G_sweep.png      entropy, items kept and max p versus alpha for
                              G = 16, 64, 256 (replay vs theory)
  fig4_theory_vs_replay.png   per-item replay policy vs mean-field ODE, and the
                              survival maps (extinction alpha per item) for G = 16, 64
  fig5_tradeoff.png           held-out CTR vs diversity, and the seed spread of CTR
  fig6_alpha0_collapse.png    which item ordinary training (alpha = 0) locks onto,
                              and how that depends on the step size
  fig8_variants_replication.png  click vs deterministic reward, plain vs GRPO
                              baseline, and the Men's / Women's campaigns

Usage (from the repository root, after analyze.py):
  python3 next_phase/case3/make_figures.py
"""

from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import obd_ips as M  # noqa: E402
import run_replay as R  # noqa: E402
from src.plotstyle import apply_style, savefig, panel_label  # noqa: E402

A_COL = {0.0: "#D55E00", 0.25: "#E69F00", 0.5: "#CC79A7", 1.0: "#0072B2", 2.0: "#009E73", 3.0: "#56B4E9"}
A_MARK = {0.0: "X", 0.25: "v", 0.5: "s", 1.0: "o", 2.0: "^", 3.0: "D"}
G_COL = {16: "#6BAED6", 64: "#2171B5", 256: "#08306B"}
G_MARK = {16: "s", 64: "o", 256: "^"}
G_LS = {16: (0, (5, 2)), 64: "-", 256: (0, (1, 1.2))}


def run(tag):
    return dict(np.load(M.RES / "runs" / f"{tag}.npz", allow_pickle=True))


def theory(tag):
    f = M.RES / "theory_runs" / f"{tag}.npz"
    return dict(np.load(f, allow_pickle=True)) if f.exists() else None


def T(a, G, **kw):
    return R.cfg(alpha=a, G=G, **kw)["tag"]


def stats_A():
    tr, te = R.split_logs("all", "A")
    return M.item_stats(tr["item"], tr["click"], 80), M.item_stats(te["item"], te["click"], 80)


def band(ax, x, y, color, label, ls="-", marker=None):
    med = np.median(y, axis=1)
    lo, hi = np.percentile(y, 25, axis=1), np.percentile(y, 75, axis=1)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(x, med, color=color, ls=ls, label=label, marker=marker, markevery=0.2, ms=4)


# --------------------------------------------------------------------------

def fig2():
    st_tr, st_te = stats_A()
    rank = np.argsort(np.argsort(-st_tr["ctr"])) + 1
    best = int(np.argmax(st_tr["ctr"]))
    alphas = [0.0, 0.5, 1.0, 2.0]
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.2), constrained_layout=True)
    for k, a in enumerate(alphas):
        r = run(T(a, 64))
        ax = axes[0, k]
        t = r["rec_t"]
        tr = r["trace"][:, 0, :]                     # seed 0, (n_rec, K)
        m = t > 0
        for i in np.argsort(rank)[::-1]:             # draw best items last (on top)
            if rank[i] == 1:
                ax.plot(t[m], tr[m, i], color="#D55E00", lw=1.6, zorder=5, label=f"best item of training half ({i})")
            elif rank[i] <= 5:
                ax.plot(t[m], tr[m, i], color="#0072B2", lw=1.0, zorder=4,
                        label="ranks 2-5" if rank[i] == 2 else None)
            else:
                ax.plot(t[m], tr[m, i], color="0.72", lw=0.5, zorder=2,
                        label="other 75 items" if rank[i] == 6 else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.5)
        ax.axhline(M.TAU_KEPT, color="0.3", lw=0.7, ls=(0, (4, 3)))
        if k == 0:
            win = int(tr[-1].argmax())
            ax.text(0.03, 0.04, f"seed 0 locks onto item {win}\n(training-CTR rank {rank[win]})",
                    transform=ax.transAxes, fontsize=7.5)
            ax.set_ylabel("item probability p_i (log)")
            ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.6), fontsize=6.5)
        ax.text(0.98, 0.04 if k else 0.2, "kept threshold 1e-3", transform=ax.transAxes, ha="right", fontsize=6.5,
                color="0.3") if k == 3 else None
        ax.set_xlabel("training step")
        ax.set_title(f"alpha = {a:g}" + ("  (ordinary training)" if a == 0 else "  (paper's IPS)" if a == 1 else ""))
        panel_label(ax, "abcd"[k])
    # bottom row: seed spread
    ax_m, ax_h, ax_k, ax_v = axes[1]
    for a in alphas:
        r = run(T(a, 64))
        t = r["rec_t"].astype(float)
        m = t > 0
        lab = f"alpha = {a:g}"
        band(ax_m, t[m], r["rec_maxp"][m], A_COL[a], lab, marker=A_MARK[a])
        band(ax_h, t[m], r["rec_H"][m], A_COL[a], lab, marker=A_MARK[a])
        band(ax_k, t[m], r["rec_kept"][m], A_COL[a], lab + " replay", marker=A_MARK[a])
        th = theory(T(a, 64))
        if th is not None and a > 0:
            tk = th["mf_t"]
            kk = M.n_kept(th["mf_p"])
            mm = tk > 0
            ax_k.plot(tk[mm], kk[mm], color=A_COL[a], ls=(0, (2, 1.5)), lw=1.2)
            ax_k.plot([t[-1] * 1.35], [int(np.sum(th["p_inf"] > M.TAU_KEPT))], marker=A_MARK[a], color=A_COL[a],
                      mfc="white", ms=6, clip_on=False)
        v = r["rec_V"][1][m] * 100                   # held-out CTR (%), (n_rec, S)
        ax_v.fill_between(t[m], v.mean(1) - v.std(1), v.mean(1) + v.std(1), color=A_COL[a], alpha=0.18, lw=0)
        ax_v.plot(t[m], v.mean(1), color=A_COL[a], marker=A_MARK[a], markevery=0.2, ms=4, label=lab)
    ax_m.axhline(0.95, color="0.3", lw=0.7, ls=(0, (4, 3)))
    ax_m.set_ylabel("max_i p_i (median, IQR)")
    ax_m.set_title("Collapse: largest item probability")
    ax_h.set_ylabel("normalized entropy (median, IQR)")
    ax_h.set_title("Diversity of the policy")
    ax_k.set_ylabel("items with p_i > 1e-3")
    ax_k.set_title("Items kept: replay vs mean field")
    ax_k.plot([], [], color="0.3", ls=(0, (2, 1.5)), label="mean-field ODE, same budget")
    ax_k.plot([], [], color="0.3", marker="o", mfc="white", ls="none", label="stationary mean field (t -> inf)")
    ax_v.axhline(100 * st_te["ctr"].mean(), color="0.45", lw=0.8, ls=(0, (4, 3)))
    ax_v.text(1.2e3, 100 * st_te["ctr"].mean() + 0.01, "uniform policy", fontsize=7, color="0.3")
    ax_v.axhline(100 * st_te["ctr"][best], color="0.45", lw=0.8, ls=(0, (1, 1.5)))
    ax_v.text(1.2e3, 100 * st_te["ctr"][best] + 0.01, f"training-best item {best}", fontsize=7, color="0.3")
    ax_v.set_ylabel("held-out CTR, days 4-7 (%)")
    ax_v.set_title("Result on unseen traffic (mean, sd)")
    for k, ax in enumerate(axes[1]):
        ax.set_xscale("log")
        ax.set_xlabel("training step")
        panel_label(ax, "efgh"[k])
    ax_m.legend(loc="center right", fontsize=7)
    ax_k.legend(loc="lower left", fontsize=6.5)
    fig.suptitle("Training by replay on real ZOZOTOWN clicks (K = 80 items, group size G = 64)", fontsize=11,
                 fontweight="bold")
    savefig(fig, M.FIG / "fig2_training_traces.png")


# --------------------------------------------------------------------------

def load_rows():
    return json.loads((M.RES / "analysis.json").read_text())


def pick(rows, **kw):
    return [x for x in rows if all(x.get(k) == v for k, v in kw.items())]


def fig3(an):
    rows = an["runs"]
    ext = list(csv.DictReader(open(M.RES / "extinction_alpha.csv")))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
    base = dict(campaign="all", split="A", mode="click", variant="plain", J=0.1)
    for G in (16, 64, 256):
        rs = sorted(pick(rows, G=G, **base), key=lambda x: x["alpha"])
        a = np.array([x["alpha"] for x in rs])
        lab = f"G = {G}"
        # (a) entropy
        axes[0].errorbar(a, [x["H_mean"] for x in rs], yerr=[x["H_sd"] for x in rs], fmt=G_MARK[G], color=G_COL[G],
                         ms=5, capsize=2, lw=0, elinewidth=1, label=f"{lab} replay")
        ap = a[a > 0]
        rp = [x for x in rs if x["alpha"] > 0]
        axes[0].plot(ap, [x["H_mf_T"] for x in rp], color=G_COL[G], ls=G_LS[G], lw=1.1)
        # (b) items kept
        km = np.array([x["kept_median"] for x in rs])
        axes[1].errorbar(a, km, yerr=[km - [x["kept_min"] for x in rs], [x["kept_max"] for x in rs] - km],
                         fmt=G_MARK[G], color=G_COL[G], ms=5, capsize=2, lw=0, elinewidth=1, label=f"{lab} replay")
        axes[1].plot(ap, [x["kept_mf_T"] for x in rp], color=G_COL[G], ls=G_LS[G], lw=1.1)
        ag = np.linspace(0.02, 3.2, 161)
        aext = np.array([float(e[f"alpha_ext_G{G}"]) for e in ext])
        axes[1].plot(ag, [np.sum(aext < x) for x in ag], color=G_COL[G], lw=2.2, alpha=0.35)
        never = sum(e[f"never_kept_G{G}"] == "True" for e in ext)
        axes[1].annotate(f"max {80 - never}", xy=(3.2, 80 - never), xytext=(3.27, 80 - never), fontsize=7,
                         color=G_COL[G], va="center", annotation_clip=False)
        # (c) max p
        axes[2].plot(a, [x["maxp_median"] for x in rs], marker=G_MARK[G], color=G_COL[G], ls=G_LS[G], ms=5, label=lab)
    axes[0].plot([], [], color="0.4", ls="-", lw=1.1, label="mean-field ODE, same budget")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("normalized entropy (mean, sd)")
    axes[0].set_title("Diversity grows with alpha and G")
    axes[0].legend(loc="lower right", fontsize=7)
    axes[1].plot([], [], color="0.4", lw=2.2, alpha=0.35, label="stationary mean field (eventual)")
    axes[1].plot([], [], color="0.4", ls="-", lw=1.1, label="mean-field ODE, same budget")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("items kept (p_i > 1e-3; median, range)")
    axes[1].set_title("The group size caps how many items survive")
    axes[1].legend(loc="lower right", fontsize=6.8)
    axes[1].set_xlim(-0.1, 3.5)
    axes[2].axhline(0.95, color="0.3", lw=0.7, ls=(0, (4, 3)))
    axes[2].text(3.0, 0.75, "collapse threshold 0.95", ha="right", fontsize=7, color="0.3")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("alpha")
    axes[2].set_ylabel("largest item probability (median)")
    axes[2].set_title("Only alpha = 0 collapses")
    axes[2].legend(loc="lower left", fontsize=7)
    for k, ax in enumerate(axes):
        panel_label(ax, "abc"[k])
    savefig(fig, M.FIG / "fig3_alpha_G_sweep.png")


# --------------------------------------------------------------------------

def fig4(an):
    st_tr, _ = stats_A()
    ext = list(csv.DictReader(open(M.RES / "extinction_alpha.csv")))
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), constrained_layout=True)
    ax = axes[0]
    lo = 1e-6
    for G in (16, 64):
        for a in (0.5, 1.0, 2.0):
            r, th = run(T(a, G)), theory(T(a, G))
            pm = np.maximum(r["p_avg"].mean(0), lo)
            pt = np.maximum(th["mf_pavg"], lo)
            ax.plot(pt, pm, A_MARK[a], color=G_COL[G], ms=4, alpha=0.85, mfc="none" if G == 16 else G_COL[G],
                    label=f"alpha = {a:g}, G = {G}")
    ax.plot([lo, 0.5], [lo, 0.5], color="0.35", lw=0.8)
    ax.axhline(M.TAU_KEPT, color="0.5", lw=0.6, ls=(0, (4, 3)))
    ax.axvline(M.TAU_KEPT, color="0.5", lw=0.6, ls=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, 0.5)
    ax.set_ylim(lo, 0.5)
    ax.set_xlabel("mean-field ODE prediction of p_i (same budget)")
    ax.set_ylabel("replay: p_i averaged over 32 seeds")
    tvr = [x for x in an["theory_vs_replay"] if x["campaign"] == "all" and x["split"] == "A" and x["variant"] == "plain"
           and x["J"] == 0.1]
    med = np.median([x["l1_replay_vs_mf_T"] for x in tvr])
    ax.set_title(f"Theory tracks the real-data replay (median l1 = {med:.3f})")
    ax.legend(loc="upper left", fontsize=6.3, ncol=1)
    panel_label(ax, "a")
    rank = np.array([int(e["rank"]) for e in ext])
    order = np.argsort(rank)
    for k, G in enumerate((16, 64)):
        ax = axes[1 + k]
        aext = np.array([float(e[f"alpha_ext_G{G}"]) for e in ext])[order]
        never = np.array([e[f"never_kept_G{G}"] == "True" for e in ext])[order]
        k2 = np.array([float(e[f"alpha_K2_formula_G{G}"]) for e in ext])[order]
        x = np.arange(1, len(order) + 1)
        top = 3.35
        ycurve = np.where(np.isfinite(aext), aext, top)
        ax.fill_between(x, ycurve, top, step="mid", color=G_COL[G], alpha=0.13, lw=0,
                        label="predicted kept (stationary mean field)")
        ax.step(x, ycurve, where="mid", color=G_COL[G], lw=1.4, label="extinction alpha, general-K theory")
        ax.plot(x, k2, color="0.25", lw=0.9, ls=(0, (3, 2)), label="two-outcome formula ln(c_best/c_j)/ln G")
        ax.plot(x[never], np.full(never.sum(), top - 0.07), "|", color="#D55E00", ms=6, mew=1.2,
                label="never kept at any alpha")
        items = [e for e in ext]
        for a in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            r = run(T(a, G))
            pm = r["p_avg"].mean(0)
            items_sorted = np.array([int(e["item"]) for e in items])[order]
            kept = pm[items_sorted] > M.TAU_KEPT
            th = theory(T(a, G))
            dead_eventually = th["p_inf"][items_sorted] == 0
            ax.plot(x[kept & ~dead_eventually], np.full((kept & ~dead_eventually).sum(), a), "o", color="0.15", ms=2.6)
            ax.plot(x[kept & dead_eventually], np.full((kept & dead_eventually).sum(), a), "o", color="#E69F00",
                    ms=2.6)
            ax.plot(x[~kept], np.full((~kept).sum(), a), "o", mfc="white", mec="0.45", ms=2.6, mew=0.6)
        ax.plot([], [], "o", color="0.15", ms=3, label="replay: kept, theory: kept")
        ax.plot([], [], "o", color="#E69F00", ms=3, label="replay: still > 1e-3, theory: dying")
        ax.plot([], [], "o", mfc="white", mec="0.45", ms=3, label="replay: dropped below 1e-3")
        ax.set_ylim(0, top)
        ax.set_xlim(0, len(order) + 1)
        ax.set_xlabel("item rank by training-half CTR")
        ax.set_ylabel("alpha")
        ax.set_title(f"Who survives, G = {G}")
        if k == 1:
            ax.legend(loc="upper left", fontsize=6.0, framealpha=0.9, frameon=True)
        panel_label(ax, "bc"[k])
    savefig(fig, M.FIG / "fig4_theory_vs_replay.png")


# --------------------------------------------------------------------------

def fig5(an):
    rows = an["runs"]
    refs = an["reference_policies"]["all_A"]
    st_tr, st_te = stats_A()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    base = dict(campaign="all", split="A", mode="click", variant="plain", J=0.1)
    rng = np.random.default_rng(3)
    for G in (16, 64, 256):
        rs = sorted(pick(rows, G=G, **base), key=lambda x: x["alpha"])
        H = [x["H_mean"] for x in rs]
        V = [100 * x["ctr_test_mean"] for x in rs]
        ax.errorbar(H, V, xerr=[x["H_sd"] for x in rs], yerr=[100 * x["ctr_test_sd"] for x in rs], fmt=G_MARK[G] + "-",
                    color=G_COL[G], ms=5, lw=0.9, elinewidth=0.7, capsize=0, label=f"G = {G} (alpha = 0 ... 3)")
        for x in rs:
            if G == 64 and x["alpha"] in (0.25, 0.5, 1.0, 2.0):
                ax.annotate(f"a={x['alpha']:g}", (x["H_mean"], 100 * x["ctr_test_mean"]), xytext=(-6, -13),
                            textcoords="offset points", fontsize=7, color="0.2")
    r0 = run(T(0.0, 64))
    v0 = 100 * (r0["p_avg"] @ st_te["ctr"])
    h0 = M.norm_entropy(r0["p_avg"])
    ax.plot(h0 + rng.uniform(0.004, 0.03, len(v0)), v0, ".", color=A_COL[0.0], ms=3.5, alpha=0.4,
            label="alpha = 0, G = 64: each of 100 seeds (x jittered)")
    ref_style = {"uniform (the logging policy)": ("P", "uniform policy"),
                 "best item of the training half (perfect greedy)": ("*", "training-best item"),
                 "best item of the held-out half (hindsight oracle, not achievable)": ("*", "hindsight best (oracle)"),
                 "deployed Thompson sampling, week-average allocation": ("h", "deployed Thompson sampling")}
    for k, (mk, lab) in ref_style.items():
        if k in refs:
            fc = "white" if "oracle" in lab else "0.2"
            ax.plot(refs[k]["H"], 100 * refs[k]["ctr_test"], mk, ms=10 if mk == "*" else 7, mfc=fc, mec="0.2", label=lab)
    ax.set_xlabel("normalized entropy of the learned policy (diversity)")
    ax.set_ylabel("held-out CTR, days 4-7 (%)")
    ax.set_title("What diversity costs on unseen traffic")
    ax.legend(loc="upper right", fontsize=6.8, frameon=True, framealpha=0.95)
    panel_label(ax, "a")
    ax = axes[1]
    alphas = [0.0, 0.25, 0.5, 1.0, 2.0]
    for k, a in enumerate(alphas):
        r = run(T(a, 64))
        v = 100 * (r["p_avg"] @ st_te["ctr"])
        ax.plot(k + rng.uniform(-0.18, 0.18, len(v)), v, A_MARK.get(a, "o"), color=A_COL[a], ms=3.5, alpha=0.6)
        ax.plot([k - 0.3, k + 0.3], [np.median(v)] * 2, color="0.1", lw=1.6)
        ax.text(k, np.max(v) + 0.012, f"sd {np.std(v):.3f}", ha="center", fontsize=7)
    best = int(np.argmax(st_tr["ctr"]))
    ax.axhline(100 * st_te["ctr"][best], color="0.4", lw=0.8, ls=(0, (1, 1.5)))
    ax.text(len(alphas) - 0.5, 100 * st_te["ctr"][best] + 0.006, f"training-best item {best}", ha="right", fontsize=7)
    ax.axhline(100 * st_te["ctr"].max(), color="0.4", lw=0.8, ls=(0, (4, 3)))
    ax.text(len(alphas) - 0.5, 100 * st_te["ctr"].max() + 0.006, "hindsight best item", ha="right", fontsize=7)
    ax.axhline(100 * st_te["ctr"].mean(), color="0.4", lw=0.8, ls=(0, (6, 2)))
    ax.text(len(alphas) - 0.5, 100 * st_te["ctr"].mean() + 0.006, "uniform policy", ha="right", fontsize=7)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"alpha = {a:g}" for a in alphas])
    ax.set_ylabel("held-out CTR per seed (%)")
    ax.set_title("Seed-to-seed spread of the result (G = 64)")
    panel_label(ax, "b")
    savefig(fig, M.FIG / "fig5_tradeoff.png")


# --------------------------------------------------------------------------

def fig6(an):
    st_tr, st_te = stats_A()
    rank_tr = np.argsort(np.argsort(-st_tr["ctr"])) + 1
    a0 = [d for d in an["alpha0"] if d["campaign"] == "all" and d["split"] == "A" and d["mode"] == "click"
          and d["variant"] == "plain"]
    rows = an["runs"]
    fig, axs = plt.subplots(2, 2, figsize=(10.4, 7.6), constrained_layout=True)
    # (a) where the 100 seeds collapse
    ax = axs[0, 0]
    nb = 8
    width = 0.27
    for k, G in enumerate((16, 64, 256)):
        d = [x for x in a0 if x["G"] == G and x["J"] == 0.1][0]
        frac = np.zeros(nb + 1)
        for it, n in d["collapse_items"].items():
            frac[min(rank_tr[int(it)], nb + 1) - 1] += n / d["seeds"]
        ax.bar(np.arange(1, nb + 2) + (k - 1) * width, frac, width=width * 0.9, color=G_COL[G], label=f"G = {G}")
    ax.set_xticks(np.arange(1, nb + 2))
    ax.set_xticklabels([str(i) for i in range(1, nb + 1)] + [f">{nb}"])
    ax.set_xlabel("training-CTR rank of the item a seed collapsed onto")
    ax.set_ylabel("fraction of 100 seeds")
    best_te = int(np.argmax(st_te["ctr"]))
    ax.set_title("Ordinary training (J = 0.1) picks a random top item")
    ax.text(0.98, 0.62, f"rank 2 = item 41: CTR {100 * st_tr['ctr'][41]:.2f}% -> {100 * st_te['ctr'][41]:.2f}% on days 4-7\n"
            f"hindsight-best item {best_te} has training rank {rank_tr[best_te]}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.legend(loc="upper right", fontsize=7)
    panel_label(ax, "a")
    # (b) wrong-item rate vs step size
    ax = axs[0, 1]
    for G in (16, 64):
        ds = sorted([x for x in a0 if x["G"] == G], key=lambda x: x["J"])
        ax.plot([x["J"] for x in ds], [x["frac_not_train_best"] for x in ds], marker=G_MARK[G], color=G_COL[G],
                label=f"G = {G}")
        for x in ds:
            ax.annotate(f"{x['draws_to_collapse_median'] / 1e3:.0f}k", (x["J"], x["frac_not_train_best"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=6.5, color="0.3")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("step size J (logit jump per click)")
    ax.set_ylabel("fraction of seeds not on the replay-best item")
    ax.set_title("Smaller steps reduce, but do not remove, wrong picks")
    ax.text(0.03, 0.95, "labels: median sampled items until collapse", transform=ax.transAxes, fontsize=6.8,
            va="top", color="0.3")
    ax.legend(loc="lower right", fontsize=7)
    panel_label(ax, "b")
    # (c) held-out CTR of alpha = 0 vs step size, against IPS references
    ax = axs[1, 0]
    for G in (16, 64):
        ds = sorted([x for x in rows if x["campaign"] == "all" and x["split"] == "A" and x["mode"] == "click"
                     and x["variant"] == "plain" and x["alpha"] == 0 and x["G"] == G], key=lambda x: x["J"])
        J = [x["J"] for x in ds]
        ax.plot(J, [100 * x["ctr_test_mean"] for x in ds], marker=G_MARK[G], color=G_COL[G],
                label=f"alpha = 0, G = {G}: mean")
        ax.plot(J, [100 * x["ctr_test_p10"] for x in ds], marker=G_MARK[G], color=G_COL[G], ls=(0, (3, 2)),
                mfc="white", label=f"alpha = 0, G = {G}: worst 10% of seeds")
    for a, ls in ((0.25, "-"), (0.5, (0, (1, 1.5)))):
        x = [x for x in rows if x["tag"] == T(a, 64)][0]
        ax.axhline(100 * x["ctr_test_mean"], color=A_COL[a], lw=1.0, ls=ls)
        ax.text(0.0255, 100 * x["ctr_test_mean"] + 0.004, f"alpha = {a:g}, G = 64 (mean ~ worst 10%)",
                fontsize=6.8, color=A_COL[a], ha="left")
    ax.set_xscale("log")
    ax.set_xlabel("step size J of ordinary training")
    ax.set_ylabel("held-out CTR, days 4-7 (%)")
    ax.set_title("Slow ordinary training: higher mean, still a bad tail")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.30), fontsize=6.5)
    panel_label(ax, "c")
    # (d) winner's curse
    ax = axs[1, 1]
    allitems = {}
    for d in a0:
        if d["J"] == 0.1:
            for it, n in d["collapse_items"].items():
                allitems[int(it)] = allitems.get(int(it), 0) + n
    ax.plot(100 * st_tr["ctr"], 100 * st_te["ctr"], "o", color="0.8", ms=3, label="all 80 items")
    for it, n in allitems.items():
        ax.plot(100 * st_tr["ctr"][it], 100 * st_te["ctr"][it], "o", color=A_COL[0.0], ms=3 + 1.2 * np.sqrt(n),
                mfc="none", mew=1.2)
        if n >= 3 or it == best_te:
            ax.annotate(f"{it}", (100 * st_tr["ctr"][it], 100 * st_te["ctr"][it]), xytext=(6, -3),
                        textcoords="offset points", fontsize=7)
    ax.plot([], [], "o", color=A_COL[0.0], mfc="none", label="items chosen by alpha = 0 (size ~ #seeds)")
    lim = [0, 1.0]
    ax.plot(lim, lim, color="0.4", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("CTR in the training half (%)")
    ax.set_ylabel("CTR in the held-out half (%)")
    ax.set_title("Winner's curse: the chosen items regress")
    ax.legend(loc="upper left", fontsize=7)
    panel_label(ax, "d")
    savefig(fig, M.FIG / "fig6_alpha0_collapse.png")


# --------------------------------------------------------------------------

def fig8(an):
    rows = an["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    alphas = [0.0, 0.5, 1.0, 2.0]
    ax = axes[0]
    for G in (16, 64):
        for mode, ls, mfc in (("click", "-", None), ("ctr", (0, (3, 2)), "white")):
            rs = sorted(pick(rows, campaign="all", split="A", mode=mode, variant="plain", G=G, J=0.1),
                        key=lambda x: x["alpha"])
            rs = [x for x in rs if x["alpha"] in alphas]
            ax.plot([x["alpha"] for x in rs], [x["H_mean"] for x in rs], marker=G_MARK[G], color=G_COL[G], ls=ls,
                    mfc=mfc or G_COL[G], label=f"G = {G}, {'logged clicks' if mode == 'click' else 'deterministic CTR'}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("normalized entropy")
    ax.set_title("Click noise vs exact CTR rewards")
    ax.legend(loc="lower right", fontsize=6.5)
    panel_label(ax, "a")
    ax = axes[1]
    for G in (16, 64):
        for var, ls, mfc in (("plain", "-", None), ("grpo", (0, (3, 2)), "white")):
            rs = sorted(pick(rows, campaign="all", split="A", mode="click", variant=var, G=G, J=0.1),
                        key=lambda x: x["alpha"])
            rs = [x for x in rs if x["alpha"] in alphas]
            ax.plot([x["alpha"] for x in rs], [x["kept_median"] for x in rs], marker=G_MARK[G], color=G_COL[G], ls=ls,
                    mfc=mfc or G_COL[G], label=f"G = {G}, {'plain update' if var == 'plain' else 'GRPO mean baseline'}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("items kept (p_i > 1e-3, median)")
    ax.set_title("Plain update vs GRPO group-mean baseline")
    ax.legend(loc="lower right", fontsize=6.5)
    panel_label(ax, "b")
    ax = axes[2]
    cm = {"all": ("#0072B2", "o", 80), "men": ("#D55E00", "s", 34), "women": ("#009E73", "^", 46)}
    for camp, (col, mk, K) in cm.items():
        for G, ls in ((16, (0, (3, 2))), (64, "-")):
            rs = sorted(pick(rows, campaign=camp, split="A", mode="click", variant="plain", G=G, J=0.1),
                        key=lambda x: x["alpha"])
            rs = [x for x in rs if x["alpha"] in alphas]
            ax.plot([x["alpha"] for x in rs], [x["kept_median"] / K for x in rs], marker=mk, color=col, ls=ls,
                    mfc=col if G == 64 else "white", label=f"{camp} (K = {K}), G = {G}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("fraction of items kept")
    ax.set_title("Same picture in the Men's / Women's logs")
    ax.legend(loc="lower right", fontsize=6.2, ncol=1)
    panel_label(ax, "c")
    savefig(fig, M.FIG / "fig8_variants_replication.png")


def main():
    import sys
    apply_style()
    an = load_rows()
    only = set(sys.argv[1:])
    for name, fn in (("fig2", lambda: fig2()), ("fig3", lambda: fig3(an)), ("fig4", lambda: fig4(an)),
                     ("fig5", lambda: fig5(an)), ("fig6", lambda: fig6(an)), ("fig8", lambda: fig8(an))):
        if only and name not in only:
            continue
        try:
            fn()
            print("wrote", name)
        except (FileNotFoundError, IndexError, KeyError) as e:   # a suite has not been run yet
            plt.close("all")
            print(f"skipped {name}: missing input ({type(e).__name__}: {e})")
    print("figures written to", M.FIG)


if __name__ == "__main__":
    main()
