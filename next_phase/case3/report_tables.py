"""
Markdown tables for the write-up (next_phase/03_real_rl_data.md), generated
from results/*.json|csv so that no number is copied by hand.

Usage (from the repository root, after analyze.py):
  python3 next_phase/case3/report_tables.py      -> results/report_tables.md
"""

from __future__ import annotations

import csv
import json

import numpy as np

import obd_ips as M

R = M.RES


def pct(x, d=3):
    return f"{100 * x:.{d}f}"


def main():
    an = json.loads((R / "analysis.json").read_text())
    land = json.loads((R / "landscape_summary.json").read_text())
    th = json.loads((R / "theory_summary.json").read_text())
    ext = list(csv.DictReader(open(R / "extinction_alpha.csv")))
    runs = an["runs"]
    out = []

    def tbl(header, rows):
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in rows:
            out.append("| " + " | ".join(str(x) for x in r) + " |")
        out.append("")

    # T1 landscape
    out.append("### T1. Reward landscape (uniform-random logs, all 7 days)\n")
    rows = []
    for c in ("all", "men", "women"):
        s = land[c]
        rows.append([c, f"{s['rows']:,}", s["K"], f"{s['clicks']:,}", pct(s["ctr_overall"]),
                     f"{pct(s['best_ctr'])} [{pct(s['best_ctr_ci'][0])}, {pct(s['best_ctr_ci'][1])}]",
                     f"{s['ratio_best_to_worst']:.1f}", s["items_within_10pct"], s["items_within_20pct"],
                     s["items_within_50pct"], s["items_not_sig_worse_than_best_5pct_one_sided"],
                     f"{s['posterior_P_best_of_top_item']:.2f}", f"{s['split_half_pearson']:.2f}",
                     f"#{s['rank_in_second_half_of_first_half_best']}"])
    tbl(["campaign", "impressions", "K", "clicks", "CTR %", "best item CTR % [95% CI]", "best/worst",
         "within 10%", "within 20%", "within 50%", "not sig. worse than best", "P(top item is truly best)",
         "split-half r", "rank of days-1-3.5 best in days 4-7"], rows)

    # T2 main, split in two tables for readability
    base = [x for x in runs if x["campaign"] == "all" and x["split"] == "A" and x["mode"] == "click"
            and x["variant"] == "plain" and x["J"] == 0.1]
    base = sorted(base, key=lambda x: (x["G"], x["alpha"]))
    out.append("### T2a. Collapse and diversity (All, K = 80; logged-click rewards; plain update; J = 0.1)\n")
    rows = []
    for x in base:
        rows.append([f"{x['alpha']:g}", x["G"], f"{x['seeds']} x {x['T']:,}", f"{100 * x['frac_collapsed']:.0f}%",
                     f"{x['maxp_median']:.3f}", f"{x['H_mean']:.3f}", f"{x['eff_items_mean']:.1f}",
                     f"{x['kept_median']:.0f} [{x['kept_min']}-{x['kept_max']}]", f"{x['kept_1e-4_median']:.0f}",
                     f"{x['l1_target_mean']:.2f}", f"{x['l1_ctrprop_mean']:.2f}"])
    tbl(["alpha", "G", "seeds x steps", "collapsed seeds (max p > 0.95)", "max p (median)", "entropy H (mean)",
         "effective items e^(H ln K)", "kept p > 1e-3 (median [range])", "kept p > 1e-4", "l1 to ideal c^(1/alpha)",
         "l1 to p ~ c"], rows)
    out.append("### T2b. Click-through rate of the learned policies (same runs)\n")
    rows = []
    for x in base:
        rows.append([f"{x['alpha']:g}", x["G"], pct(x["ctr_train_mean"]),
                     f"{pct(x['ctr_test_mean'])} ± {pct(x['ctr_test_sd'])}", pct(x["ctr_test_p10"]),
                     pct(x["ctr_test_stat_se"]), f"{x['recall_test_top5']:.2f}", f"{x['mass_test_top10']:.2f}"])
    tbl(["alpha", "G", "CTR % in the replay world (training half)", "held-out CTR % (mean ± sd over seeds)",
         "held-out CTR % 10th percentile", "s.e. of the held-out CTR estimate %", "held-out top-5 items kept",
         "mass on held-out top-10"], rows)

    # T3 alpha = 0
    out.append("### T3. Ordinary training (alpha = 0): where the collapse lands\n")
    rows = []
    for d in sorted(an["alpha0"], key=lambda d: (d["campaign"], d["split"], d["mode"], d["variant"], d["G"], d["J"])):
        r = [x for x in runs if x["tag"] == d["tag"]][0]
        items = ", ".join(f"{i}:{n}" for i, n in sorted(d["collapse_items"].items(), key=lambda kv: -kv[1]))
        rows.append([d["campaign"], d["split"], d["mode"], d["variant"], d["G"], f"{d['J']:g}", d["seeds"], items,
                     f"{100 * d['frac_not_train_best']:.0f}%", f"{100 * d['frac_not_test_best']:.0f}%",
                     f"{100 * d['frac_outside_test_top5']:.0f}%", f"{d['steps_to_collapse_median']:.0f}",
                     f"{d['draws_to_collapse_median']:.0f}",
                     f"{pct(r['ctr_test_mean'])} ± {pct(r['ctr_test_sd'])}", pct(r["ctr_test_p10"])])
    tbl(["campaign", "split", "reward", "update", "G", "J", "seeds", "item:seeds", "not replay-best",
         "not held-out-best", "outside held-out top 5", "median steps to collapse", "median draws",
         "held-out CTR % (mean ± sd)", "10th pct"], rows)

    # T4 theory vs replay
    out.append("### T4. Theory vs replay (items kept = p_i > 1e-3)\n")
    rows = []
    for x in sorted(an["theory_vs_replay"], key=lambda x: (x["campaign"], x["split"], x["variant"], x["G"], x["alpha"], -x["J"])):
        rows.append([x["campaign"], x["split"], x["variant"], f"{x['alpha']:g}", x["G"], f"{x['J']:g}", f"{x['T']:,}",
                     f"{x['kept_replay_median_seed']:.0f} / {x['kept_replay_seedmean']}", x["kept_mf_T"],
                     x["kept_mf_inf"], x["support_mf_inf"], f"{100 * x['agree_replay_vs_mf_T']:.0f}%",
                     f"{x['l1_replay_vs_mf_T']:.3f}", f"{x['l1_noise_floor']:.3f}", f"{x['l1_replay_vs_mf_inf']:.3f}",
                     f"{x['mass_on_doomed_replay']:.3f} / {x['mass_on_doomed_mf_T']:.3f}"])
    tbl(["campaign", "split", "update", "alpha", "G", "J", "steps", "replay kept (median seed / seed-mean)",
         "mean-field ODE kept (same budget)", "stationary kept", "stationary support (p > 0)",
         "per-item agreement (replay vs ODE)", "l1 replay vs ODE", "l1 noise floor", "l1 replay vs stationary",
         "mass on doomed items (replay / ODE)"], rows)

    # T5 extinction
    out.append("### T5. Extinction exponents from the real CTRs (training half)\n")
    nk = th["extinction"]["never_kept_count"]
    kc = th["extinction"]["kept_count_at_alpha"]
    rows = []
    for G in ("16", "64", "256"):
        rows.append([G, 80 - nk[G]] + [kc[G][a] for a in ("0.25", "0.5", "0.75", "1", "1.5", "2", "3")])
    tbl(["G", "max items keepable at any alpha", "eventual survivors a=0.25", "0.5", "0.75", "1", "1.5", "2", "3"], rows)
    rows = []
    for e in ext:
        if int(e["rank"]) in (2, 5, 10, 20, 30, 40, 60, 80):
            f = lambda v: "never" if v == "inf" or float(v) == np.inf else f"{float(v):.2f}"  # noqa: E731
            rows.append([e["rank"], e["item"], pct(float(e["ctr_train"])), f"{float(e['alpha_K2_formula_G16']):.2f}",
                         f(e["alpha_ext_G16"]), f"{float(e['alpha_K2_formula_G64']):.2f}", f(e["alpha_ext_G64"]),
                         f(e["alpha_ext_G256"])])
    tbl(["CTR rank", "item", "CTR % (train half)", "two-outcome formula G=16", "general-K G=16",
         "two-outcome formula G=64", "general-K G=64", "general-K G=256"], rows)
    chk = th["extinction"]["checks"]
    out.append(f"Grid interpolation vs `extinction_alpha` bisection: max |difference| = "
               f"{max(c['abs_diff'] for c in chk):.1e} over {len(chk)} checked items.\n")

    # T6 references
    out.append("### T6. Reference policies (All, split A)\n")
    rows = []
    for name, v in an["reference_policies"]["all_A"].items():
        rows.append([name, f"{pct(v['ctr_train'])} ± {pct(v['ctr_train_se'])}",
                     f"{pct(v['ctr_test'])} ± {pct(v['ctr_test_se'])}", f"{v['H']:.3f}", v["kept"]])
    tbl(["policy", "CTR % training half (± s.e.)", "CTR % held-out half (± s.e.)", "entropy H", "kept"], rows)

    # T7 variants
    out.append("### T7. Variants (All, K = 80): deterministic-CTR rewards, GRPO baseline, reversed split\n")
    rows = []
    for x in sorted([x for x in runs if x["campaign"] == "all" and x["J"] == 0.1 and x["alpha"] in (0, 0.5, 1, 2)
                     and x["G"] in (16, 64)],
                    key=lambda x: (x["G"], x["alpha"], x["split"], x["mode"], x["variant"])):
        rows.append([x["split"], x["mode"], x["variant"], f"{x['alpha']:g}", x["G"], f"{100 * x['frac_collapsed']:.0f}%",
                     f"{x['H_mean']:.3f}", f"{x['kept_median']:.0f}", pct(x["ctr_train_mean"]),
                     f"{pct(x['ctr_test_mean'])} ± {pct(x['ctr_test_sd'])}", pct(x["ctr_test_p10"]),
                     f"{x.get('l1_seedmean_mf_T', float('nan')):.3f}"])
    tbl(["split", "reward", "update", "alpha", "G", "collapsed", "H", "kept", "CTR train %", "held-out CTR %",
         "10th pct", "l1 to mean field (same budget)"], rows)

    # T8 replication
    out.append("### T8. Replication on the Men's (K = 34) and Women's (K = 46) campaigns\n")
    rows = []
    for x in sorted([x for x in runs if x["campaign"] in ("men", "women")],
                    key=lambda x: (x["campaign"], x["G"], x["alpha"])):
        a0 = [d for d in an["alpha0"] if d["tag"] == x["tag"]]
        wrong = f"{100 * a0[0]['frac_not_train_best']:.0f}% / {100 * a0[0]['frac_not_test_best']:.0f}%" if a0 else ""
        rows.append([x["campaign"], f"{x['alpha']:g}", x["G"], f"{100 * x['frac_collapsed']:.0f}%", f"{x['H_mean']:.3f}",
                     f"{x['kept_median']:.0f}", x.get("kept_mf_T", ""), pct(x["ctr_train_mean"]),
                     f"{pct(x['ctr_test_mean'])} ± {pct(x['ctr_test_sd'])}", pct(x["ctr_test_p10"]), wrong,
                     f"{x.get('l1_seedmean_mf_T', float('nan')):.3f}"])
    tbl(["campaign", "alpha", "G", "collapsed", "H", "kept", "mean-field kept", "CTR train %", "held-out CTR %",
         "10th pct", "alpha=0: not replay-best / not held-out-best", "l1 to mean field"], rows)
    for camp in ("men_A", "women_A", "all_B"):
        if camp in an["reference_policies"]:
            ref = an["reference_policies"][camp]
            u = ref["uniform (the logging policy)"]
            b = ref["best item of the training half (perfect greedy)"]
            o = ref["best item of the held-out half (hindsight oracle, not achievable)"]
            out.append(f"- {camp}: uniform {pct(u['ctr_test'])}%, training-best item {pct(b['ctr_test'])}%, "
                       f"hindsight best {pct(o['ctr_test'])}% (held-out CTR)")
    out.append("")
    (R / "report_tables.md").write_text("\n".join(out))
    print("\n".join(out)[:3000])


if __name__ == "__main__":
    main()
