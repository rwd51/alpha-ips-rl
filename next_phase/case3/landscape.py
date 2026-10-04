"""
Step 3a -- the real reward landscape: per-item click-through rates (CTR) of the
Open Bandit Dataset's uniform-random logs, with 95% Wilson intervals, how many
items are "comparably good", and how noisy the ranking is. Also summarizes the
deployed Bernoulli Thompson Sampling (BTS) policy as a real-system reference.

Outputs:
  results/landscape_<campaign>.csv        per-item table
  results/landscape_summary.json          per-campaign summary + BTS reference
  figures/fig1_landscape.png              (a) sorted CTRs with CIs, (b) split-half
                                          scatter, (c) items within x% of the best
  figures/fig7_bts_reference.png          the deployed BTS allocation over time

Usage (from the repository root):  python3 next_phase/case3/landscape.py
"""

from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import obd_ips as M  # noqa: E402
from src.plotstyle import apply_style, savefig, panel_label  # noqa: E402

CAMPAIGNS = ("all", "men", "women")
CAMP_LABEL = {"all": "All (K = 80)", "men": "Men's (K = 34)", "women": "Women's (K = 46)"}
CAMP_COLOR = {"all": "#0072B2", "men": "#D55E00", "women": "#009E73"}
CAMP_MARK = {"all": "o", "men": "s", "women": "^"}


def posterior_p_best(k, n, draws=20000, seed=0):
    """P(item i has the highest true CTR) under independent Beta(1+k, 1+n-k)
    posteriors (uniform priors), by Monte Carlo."""
    rng = np.random.default_rng(seed)
    samp = rng.beta(1 + k[None, :], 1 + (n - k)[None, :], size=(draws, len(k)))
    return np.bincount(samp.argmax(axis=1), minlength=len(k)) / draws


def summarize(campaign):
    log = M.load_log(campaign)
    K = log["K"]
    st = M.item_stats(log["item"], log["click"], K)
    A, B = M.time_split(log, 0.5)
    sa = M.item_stats(A["item"], A["click"], K)
    sb = M.item_stats(B["item"], B["click"], K)
    pb = posterior_p_best(st["k"], st["n"])
    pb_a = posterior_p_best(sa["k"], sa["n"], seed=1)
    ctr = st["ctr"]
    best = int(np.argmax(ctr))
    order = np.argsort(-ctr)
    rank = np.empty(K, int)
    rank[order] = np.arange(1, K + 1)
    # two-proportion z statistic of each item against the best item
    se = np.sqrt(ctr * (1 - ctr) / st["n"] + ctr[best] * (1 - ctr[best]) / st["n"][best])
    zstat = np.where(np.arange(K) == best, 0.0, (ctr[best] - ctr) / se)
    pos = log["pos"]
    day = ((log["ts_us"] - log["ts_us"].min()) // 86_400_000_000).astype(int)
    rk = lambda x: np.argsort(np.argsort(x))  # noqa: E731
    summ = {
        "campaign": campaign, "rows": int(len(log["item"])), "K": K,
        "clicks": int(st["k"].sum()), "ctr_overall": float(log["click"].mean()),
        "impressions_per_item_min_max": [int(st["n"].min()), int(st["n"].max())],
        "clicks_per_item_min_mean_max": [int(st["k"].min()), float(st["k"].mean()), int(st["k"].max())],
        "best_item": best, "best_ctr": float(ctr[best]),
        "best_ctr_ci": [float(st["lo"][best]), float(st["hi"][best])],
        "min_ctr": float(ctr.min()), "ratio_best_to_worst": float(ctr[best] / ctr.min()),
        "ratio_best_to_median": float(ctr[best] / np.median(ctr)),
        "items_within_10pct": int(np.sum(ctr >= 0.9 * ctr[best])),
        "items_within_20pct": int(np.sum(ctr >= 0.8 * ctr[best])),
        "items_within_30pct": int(np.sum(ctr >= 0.7 * ctr[best])),
        "items_within_50pct": int(np.sum(ctr >= 0.5 * ctr[best])),
        "items_not_sig_worse_than_best_5pct_one_sided": int(np.sum(zstat < 1.645)),
        "posterior_P_best_of_top_item": float(pb[best]),
        "posterior_items_needed_for_90pct_P_best": int(np.searchsorted(np.cumsum(np.sort(pb)[::-1]), 0.9) + 1),
        "split_half_pearson": float(np.corrcoef(sa["ctr"], sb["ctr"])[0, 1]),
        "split_half_spearman": float(np.corrcoef(rk(sa["ctr"]), rk(sb["ctr"]))[0, 1]),
        "best_first_half": int(np.argmax(sa["ctr"])), "best_second_half": int(np.argmax(sb["ctr"])),
        "rank_in_second_half_of_first_half_best": int(np.sum(sb["ctr"] > sb["ctr"][np.argmax(sa["ctr"])]) + 1),
        "P_best_of_first_half_top_item_given_first_half": float(pb_a[np.argmax(sa["ctr"])]),
        "ctr_by_position": {int(q): float(log["click"][pos == q].mean()) for q in (1, 2, 3)},
        "ctr_by_day": [float(log["click"][day == d].mean()) for d in range(day.max() + 1)],
    }
    rows = []
    for i in order:
        rows.append({"item": int(i), "rank": int(rank[i]), "n": int(st["n"][i]), "clicks": int(st["k"][i]),
                     "ctr": float(ctr[i]), "ci_lo": float(st["lo"][i]), "ci_hi": float(st["hi"][i]),
                     "ctr_first_half": float(sa["ctr"][i]), "ctr_second_half": float(sb["ctr"][i]),
                     "P_best": float(pb[i])})
    with open(M.RES / f"landscape_{campaign}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return summ, st, sa, sb, pb


def bts_reference(st_all, st_test):
    """The deployed Bernoulli Thompson Sampling policy (campaign 'all')."""
    b = np.load(M.DATA / "obd_bts_all_hourly.npz")
    n, c = b["n"], b["clicks"]
    K = n.shape[1]
    share = n.sum(0) / n.sum()
    sh = n / n.sum(1, keepdims=True)
    H_hour = M.norm_entropy(sh)
    v_rand_full, se_full = M.policy_ctr(share, st_all)
    v_rand_test, se_test = M.policy_ctr(share, st_test)
    tot_ctr = c.sum() / n.sum()
    return {
        "rows": int(n.sum()), "hours": int(n.shape[0]), "items": int(K),
        "on_policy_ctr": float(tot_ctr),
        "on_policy_ctr_ci95": [float(x) for x in M.wilson(c.sum(), n.sum())],
        "share_norm_entropy": float(M.norm_entropy(share)),
        "share_effective_items_expH": float(np.exp(M.norm_entropy(share) * np.log(K))),
        "share_max": float(share.max()), "share_argmax_item": int(share.argmax()),
        "items_with_share_gt_1e-3": int(np.sum(share > 1e-3)),
        "hourly_norm_entropy_min_median_max": [float(H_hour.min()), float(np.median(H_hour)), float(H_hour.max())],
        "hourly_max_share_min_median_max": [float(sh.max(1).min()), float(np.median(sh.max(1))), float(sh.max(1).max())],
        "distinct_hourly_top_items": int(len(np.unique(sh.argmax(1)))),
        "replay_ctr_of_time_avg_share_on_random_log_full": [float(v_rand_full), float(se_full)],
        "replay_ctr_of_time_avg_share_on_random_log_second_half": [float(v_rand_test), float(se_test)],
        "rank_by_random_ctr_of_bts_top_item": int(np.sum(st_all["ctr"] > st_all["ctr"][share.argmax()]) + 1),
        "share": share.tolist(), "hourly_H": H_hour.tolist(), "hourly_maxshare": sh.max(1).tolist(),
    }


def fig_landscape(data):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    # (a) sorted CTR with 95% Wilson CIs, campaign all
    summ, st, sa, sb, pb = data["all"]
    ctr = st["ctr"]
    o = np.argsort(-ctr)
    x = np.arange(1, len(o) + 1)
    ax = axes[0]
    ax.errorbar(x, 100 * ctr[o], yerr=[100 * (ctr[o] - st["lo"][o]), 100 * (st["hi"][o] - ctr[o])],
                fmt="o", ms=3, color="#0072B2", ecolor="#9BB8D3", elinewidth=0.8, capsize=0, label="CTR, 95% CI")
    ax.axhline(100 * ctr.max(), color="0.25", lw=0.8)
    ax.axhline(80 * ctr.max(), color="0.25", lw=0.8, ls=(0, (4, 3)))
    ax.text(len(o), 100 * ctr.max() + 0.02, "best", ha="right", va="bottom", fontsize=7.5)
    ax.text(len(o), 80 * ctr.max() + 0.02, "80% of best", ha="right", va="bottom", fontsize=7.5)
    n20 = summ["items_within_20pct"]
    ax.annotate(f"{n20} items within 20% of best\n{summ['items_not_sig_worse_than_best_5pct_one_sided']} not significantly worse",
                xy=(n20, 100 * ctr[o][n20 - 1]), xytext=(22, 0.86), fontsize=7.5,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
    ax.set_xlabel("item rank (by CTR, 1.37M uniform-random impressions)")
    ax.set_ylabel("click-through rate (%)")
    ax.set_title("Real reward landscape (ZOZOTOWN, All)")
    ax.set_ylim(0, 1.05)
    panel_label(ax, "a")
    # (b) split-half scatter
    ax = axes[1]
    ax.plot(100 * sa["ctr"], 100 * sb["ctr"], "o", ms=3.5, color="#0072B2", alpha=0.8)
    lim = [0, 1.0]
    ax.plot(lim, lim, color="0.4", lw=0.8)
    ia, ib = int(np.argmax(sa["ctr"])), int(np.argmax(sb["ctr"]))
    ax.plot(100 * sa["ctr"][ia], 100 * sb["ctr"][ia], "s", ms=7, mfc="none", mec="#D55E00", mew=1.4,
            label=f"best of days 1-3.5 (item {ia})")
    ax.plot(100 * sa["ctr"][ib], 100 * sb["ctr"][ib], "D", ms=7, mfc="none", mec="#009E73", mew=1.4,
            label=f"best of days 4-7 (item {ib})")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("CTR, first half of the week (%)")
    ax.set_ylabel("CTR, second half of the week (%)")
    ax.set_title(f"Ranking is noisy (Pearson r = {summ['split_half_pearson']:.2f})")
    ax.legend(loc="upper left")
    panel_label(ax, "b")
    # (c) items within x% of the best, three campaigns
    ax = axes[2]
    xs = np.linspace(0, 1, 201)
    for camp in CAMPAIGNS:
        s2, st2, *_ = data[camp]
        c2 = st2["ctr"]
        frac = np.array([np.sum(c2 >= (1 - t) * c2.max()) for t in xs])
        ax.plot(100 * xs, frac, color=CAMP_COLOR[camp], label=CAMP_LABEL[camp],
                marker=CAMP_MARK[camp], markevery=25, ms=4)
    ax.axvline(20, color="0.4", lw=0.8, ls=(0, (4, 3)))
    ax.set_xlabel("allowed gap to the best item's CTR (%)")
    ax.set_ylabel("number of items within the gap")
    ax.set_title("Many comparably good items")
    ax.legend(loc="upper left")
    panel_label(ax, "c")
    savefig(fig, M.FIG / "fig1_landscape.png")


def fig_bts(bts, st_all):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), constrained_layout=True)
    ax = axes[0]
    hrs = np.arange(len(bts["hourly_H"]))
    ax.plot(hrs / 24, bts["hourly_H"], color="#0072B2", lw=1.2, label="normalized entropy of hourly allocation")
    ax.plot(hrs / 24, bts["hourly_maxshare"], color="#D55E00", lw=1.2, ls=(0, (4, 2)), label="largest item share in the hour")
    ax.set_ylim(0, 1)
    ax.set_xlabel("day of the week-long A/B test")
    ax.set_ylabel("value")
    ax.set_title("Deployed Thompson sampling never collapsed")
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.41))
    panel_label(ax, "a")
    ax = axes[1]
    share = np.array(bts["share"])
    ax.plot(100 * st_all["ctr"], 100 * share, "o", ms=3.5, color="#0072B2")
    ax.axhline(100 / len(share), color="0.4", lw=0.8, ls=(0, (4, 3)))
    ax.text(0.80, 100 / len(share) + 0.25, "uniform share 1/80", fontsize=7.5, ha="right")
    ax.set_xlabel("item CTR in the uniform-random log (%)")
    ax.set_ylabel("share of BTS impressions (%)")
    ax.set_title(f"BTS allocation over 7 days (H = {bts['share_norm_entropy']:.2f})")
    panel_label(ax, "b")
    savefig(fig, M.FIG / "fig7_bts_reference.png")


def main():
    M.RES.mkdir(parents=True, exist_ok=True)
    M.FIG.mkdir(parents=True, exist_ok=True)
    data = {c: summarize(c) for c in CAMPAIGNS}
    out = {c: data[c][0] for c in CAMPAIGNS}
    log = M.load_log("all")
    _, B = M.time_split(log, 0.5)
    st_test = M.item_stats(B["item"], B["click"], log["K"])
    bts = bts_reference(data["all"][1], st_test)
    out["bts_all"] = {k: v for k, v in bts.items() if k not in ("share", "hourly_H", "hourly_maxshare")}
    (M.RES / "landscape_summary.json").write_text(json.dumps(out, indent=1))
    fig_landscape(data)
    fig_bts(bts, data["all"][1])
    for c in CAMPAIGNS:
        s = out[c]
        print(f"{c}: K={s['K']} rows={s['rows']:,} CTR={100 * s['ctr_overall']:.3f}% best item {s['best_item']} "
              f"{100 * s['best_ctr']:.3f}% [{100 * s['best_ctr_ci'][0]:.3f}, {100 * s['best_ctr_ci'][1]:.3f}]; "
              f"within 20%: {s['items_within_20pct']}, not sig. worse: {s['items_not_sig_worse_than_best_5pct_one_sided']}, "
              f"P(best) top item {s['posterior_P_best_of_top_item']:.2f}, split-half r {s['split_half_pearson']:.2f}, "
              f"first-half best ranks #{s['rank_in_second_half_of_first_half_best']} in second half")
    print("BTS:", json.dumps(out["bts_all"], indent=None)[:900])


if __name__ == "__main__":
    main()
