"""
Step 3c-3e -- metrics of every replay run, the theory comparison and the
CTR-vs-diversity trade-off, written as tables.

The learned policy of a run is the time-averaged policy over the last 25% of
training (p_avg; for alpha = 0 every seed has long collapsed by then). Per run:

  collapse      fraction of seeds with max_i p_i > 0.95 (final policy)
  max p         median over seeds
  H             normalized entropy  -sum p ln p / ln K   (mean over seeds)
  kept          #items with p_i > 1e-3 (median over seeds; also 1e-4)
  l1_*          distance to the ideal target ctr^(1/alpha), to the
                CTR-proportional target, to the stationary mean field and to
                the mean-field ODE at the same budget
  CTR_train     expected CTR sum_i p_i c_i under the training half (the
                replay world's truth), mean over seeds
  CTR_test      the same under the held-out half (days 4-7): the honest metric;
                mean, sd and 10th percentile over seeds, and the statistical
                s.e. sqrt(sum p_i^2 c_i (1-c_i) / n_i) of the CTR estimates
  coverage      recall of the held-out half's top-5 items (p_i > 1e-3) and the
                probability mass on its top-10 items
  alpha = 0     which item each seed collapsed onto and how often it is not the
                best item of the replay world / of the held-out half

Outputs: results/summary_runs.csv (one row per run), results/summary_alpha0.json,
         results/theory_vs_replay.csv, results/reference_policies.json,
         results/analysis.json (everything the figures and the write-up quote)

Usage (from the repository root):  python3 next_phase/case3/analyze.py
"""

from __future__ import annotations

import csv
import json

import numpy as np

import obd_ips as M
import run_replay as R

TH = M.RES / "theory_runs"


def logs_stats(campaign, split):
    tr, te = R.split_logs(campaign, split)
    K = tr["K"]
    return M.item_stats(tr["item"], tr["click"], K), M.item_stats(te["item"], te["click"], K)


def load_run(tag):
    f = M.RES / "runs" / f"{tag}.npz"
    return dict(np.load(f, allow_pickle=True)) if f.exists() else None


def load_theory(tag):
    f = TH / f"{tag}.npz"
    return dict(np.load(f, allow_pickle=True)) if f.exists() else None


def theory_tag(c):
    """Mean field is shared by click / ctr rewards (identical expected update)."""
    return c["tag"].replace("_ctr_", "_click_")


def run_metrics(c, r, st_tr, st_te, th):
    a, G = c["alpha"], c["G"]
    K = len(st_tr["ctr"])
    pa, pf = r["p_avg"], r["p_final"]
    S = pa.shape[0]
    target = M.ideal_policy(st_tr["ctr"], a)
    pctr = st_tr["ctr"] / st_tr["ctr"].sum()
    v_tr = pa @ st_tr["ctr"]
    v_te = pa @ st_te["ctr"]
    se_te = np.sqrt((pa ** 2) @ (st_te["ctr"] * (1 - st_te["ctr"]) / st_te["n"]))
    H = M.norm_entropy(pa)
    kept = M.n_kept(pa)
    top5_te = np.argsort(-st_te["ctr"])[:5]
    top10_te = np.argsort(-st_te["ctr"])[:10]
    row = {
        "tag": c["tag"], "campaign": c["campaign"], "split": c["split"], "mode": c["mode"],
        "variant": c["variant"], "alpha": a, "G": G, "J": c["J"], "h": float(r["h"]), "T": int(r["T"]),
        "seeds": S, "seconds": float(r["seconds"]),
        "frac_collapsed": float(np.mean(pf.max(1) > 0.95)),
        "maxp_median": float(np.median(pa.max(1))),
        "H_mean": float(H.mean()), "H_sd": float(H.std()),
        "eff_items_mean": float(np.mean(np.exp(H * np.log(K)))),
        "kept_median": float(np.median(kept)), "kept_min": int(kept.min()), "kept_max": int(kept.max()),
        "kept_1e-4_median": float(np.median(M.n_kept(pa, 1e-4))),
        "l1_target_mean": float(M.l1(pa, target).mean()),
        "l1_ctrprop_mean": float(M.l1(pa, pctr).mean()),
        "ctr_train_mean": float(v_tr.mean()), "ctr_train_sd": float(v_tr.std()),
        "ctr_test_mean": float(v_te.mean()), "ctr_test_sd": float(v_te.std()),
        "ctr_test_p10": float(np.percentile(v_te, 10)), "ctr_test_p90": float(np.percentile(v_te, 90)),
        "ctr_test_stat_se": float(se_te.mean()),
        "online_ctr": float(r["online_ctr"]),
        # coverage of the items that turned out best on unseen traffic
        "recall_test_top5": float(np.mean((pa[:, top5_te] > M.TAU_KEPT).mean(1))),
        "mass_test_top10": float(np.mean(pa[:, top10_te].sum(1))),
    }
    if th is not None:
        pm = pa.mean(0)
        row.update({
            "l1_mf_inf_mean": float(M.l1(pa, th["p_inf"]).mean()),
            "l1_mf_T_mean": float(M.l1(pa, th["mf_pavg"]).mean()),
            "l1_seedmean_mf_T": float(M.l1(pm, th["mf_pavg"])),
            "l1_seedmean_mf_inf": float(M.l1(pm, th["p_inf"])),
            "kept_mf_T": int(M.n_kept(th["mf_pavg"])),
            "kept_mf_inf": int(M.n_kept(th["p_inf"])),
            "support_mf_inf": int(np.sum(th["p_inf"] > 0)),
            "H_mf_T": float(M.norm_entropy(th["mf_pavg"])),
            "H_mf_inf": float(M.norm_entropy(th["p_inf"])),
            "ctr_train_mf_T": float(th["mf_pavg"] @ st_tr["ctr"]),
            "ctr_test_mf_inf": float(th["p_inf"] @ st_te["ctr"]),
        })
    return row


def alpha0_details(c, r, st_tr, st_te):
    pa = r["p_avg"]
    win = pa.argmax(1)
    best_tr = int(np.argmax(st_tr["ctr"]))
    best_te = int(np.argmax(st_te["ctr"]))
    rank_tr = np.argsort(np.argsort(-st_tr["ctr"])) + 1
    rank_te = np.argsort(np.argsort(-st_te["ctr"])) + 1
    items, counts = np.unique(win, return_counts=True)
    # steps until max p first exceeds 0.95, per seed (recording grid resolution)
    t = r["rec_t"]
    hit = r["rec_maxp"] > 0.95
    first = np.where(hit.any(0), t[np.argmax(hit, axis=0)], np.nan)
    return {
        "tag": c["tag"], "G": c["G"], "J": c["J"], "split": c["split"], "campaign": c["campaign"],
        "mode": c["mode"], "variant": c["variant"], "seeds": int(len(win)),
        "collapse_items": {int(i): int(n) for i, n in zip(items, counts)},
        "collapse_items_train_rank": {int(i): int(rank_tr[i]) for i in items},
        "collapse_items_test_rank": {int(i): int(rank_te[i]) for i in items},
        "best_train_item": best_tr, "best_test_item": best_te,
        "frac_not_train_best": float(np.mean(win != best_tr)),
        "frac_not_test_best": float(np.mean(win != best_te)),
        "frac_outside_test_top5": float(np.mean(rank_te[win] > 5)),
        "mean_train_rank_of_choice": float(np.mean(rank_tr[win])),
        "mean_test_rank_of_choice": float(np.mean(rank_te[win])),
        "steps_to_collapse_median": float(np.nanmedian(first)),
        "steps_to_collapse_p90": float(np.nanpercentile(first, 90)),
        "draws_to_collapse_median": float(np.nanmedian(first) * c["G"]),
    }


def reference_policies(st_tr, st_te, K):
    best_tr = int(np.argmax(st_tr["ctr"]))
    top5 = np.argsort(-st_tr["ctr"])[:5]
    pols = {
        "uniform (the logging policy)": np.full(K, 1.0 / K),
        "best item of the training half (perfect greedy)": np.eye(K)[best_tr],
        "best item of the held-out half (hindsight oracle, not achievable)": np.eye(K)[int(np.argmax(st_te["ctr"]))],
        "uniform over the training half's top 5": np.bincount(top5, minlength=K) / 5.0,
        "CTR-proportional p ~ c (ideal alpha = 1)": st_tr["ctr"] / st_tr["ctr"].sum(),
        "p ~ c^2 (ideal alpha = 0.5)": M.ideal_policy(st_tr["ctr"], 0.5),
        "p ~ sqrt(c) (ideal alpha = 2)": M.ideal_policy(st_tr["ctr"], 2.0),
    }
    out = {}
    for name, p in pols.items():
        vt, set_ = M.policy_ctr(p, st_tr)
        ve, see = M.policy_ctr(p, st_te)
        out[name] = {"ctr_train": float(vt), "ctr_train_se": float(set_), "ctr_test": float(ve),
                     "ctr_test_se": float(see), "H": float(M.norm_entropy(p)), "kept": int(M.n_kept(p))}
    try:
        b = np.load(M.DATA / "obd_bts_all_hourly.npz")
        if K == b["n"].shape[1]:
            share = b["n"].sum(0) / b["n"].sum()
            ve, see = M.policy_ctr(share, st_te)
            vt, set_ = M.policy_ctr(share, st_tr)
            out["deployed Thompson sampling, week-average allocation"] = {
                "ctr_train": float(vt), "ctr_train_se": float(set_), "ctr_test": float(ve), "ctr_test_se": float(see),
                "H": float(M.norm_entropy(share)), "kept": int(M.n_kept(share)),
                "on_policy_ctr_measured": float(b["clicks"].sum() / b["n"].sum())}
    except FileNotFoundError:
        pass
    return out


def main():
    cfgs = [c for s in R.SUITES for c in R.suite(s)]
    rows, a0, tvr = [], [], []
    refs = {}
    for c in cfgs:
        r = load_run(c["tag"])
        if r is None:
            continue
        st_tr, st_te = logs_stats(c["campaign"], c["split"])
        key = f"{c['campaign']}_{c['split']}"
        if key not in refs:
            refs[key] = reference_policies(st_tr, st_te, len(st_tr["ctr"]))
        th = load_theory(theory_tag(c)) if c["variant"] in ("plain", "grpo") else None
        row = run_metrics(c, r, st_tr, st_te, th)
        rows.append(row)
        if c["alpha"] == 0:
            a0.append(alpha0_details(c, r, st_tr, st_te))
        if th is not None and c["alpha"] > 0 and c["mode"] == "click":
            pa = r["p_avg"]
            pm = pa.mean(0)
            half = pa.shape[0] // 2
            noise = float(M.l1(pa[:half].mean(0), pa[half:].mean(0)) / 2)
            obs = pm > M.TAU_KEPT
            pT = th["mf_pavg"] > M.TAU_KEPT
            pinf = th["p_inf"] > M.TAU_KEPT
            doomed = th["p_inf"] == 0
            tvr.append({
                "tag": c["tag"], "campaign": c["campaign"], "split": c["split"], "variant": c["variant"],
                "alpha": c["alpha"], "G": c["G"], "J": c["J"], "T": c["T"],
                "kept_replay_seedmean": int(obs.sum()), "kept_replay_median_seed": float(np.median(M.n_kept(pa))),
                "kept_mf_T": int(pT.sum()), "kept_mf_inf": int(pinf.sum()), "support_mf_inf": int((~doomed).sum()),
                "agree_replay_vs_mf_T": float(np.mean(obs == pT)),
                "agree_replay_vs_mf_inf": float(np.mean(obs == pinf)),
                "l1_replay_vs_mf_T": float(M.l1(pm, th["mf_pavg"])),
                "l1_replay_vs_mf_inf": float(M.l1(pm, th["p_inf"])),
                "l1_mf_T_vs_mf_inf": float(M.l1(th["mf_pavg"], th["p_inf"])),
                "l1_noise_floor": noise,
                "mass_on_doomed_replay": float(pm[doomed].sum()),
                "mass_on_doomed_mf_T": float(th["mf_pavg"][doomed].sum()),
                "spearman_replay_vs_mf_T": float(np.corrcoef(np.argsort(np.argsort(pm)),
                                                             np.argsort(np.argsort(th["mf_pavg"])))[0, 1]),
                "rk4_doubling_err": float(th["rk4_doubling_err"]),
            })
    for name, data in (("summary_runs.csv", rows), ("theory_vs_replay.csv", tvr)):
        if data:
            with open(M.RES / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    (M.RES / "summary_alpha0.json").write_text(json.dumps(a0, indent=1))
    (M.RES / "reference_policies.json").write_text(json.dumps(refs, indent=1))
    (M.RES / "analysis.json").write_text(json.dumps({"runs": rows, "alpha0": a0, "theory_vs_replay": tvr,
                                                     "reference_policies": refs}, indent=1))
    # console digest
    print(f"{len(rows)} runs analysed")
    hdr = "tag                                   collapse  maxp    H     kept  CTRtrain%  CTRtest% (sd, se)   l1(mfT)"
    print(hdr)
    for x in rows:
        print(f"{x['tag'][:38]:38s} {x['frac_collapsed']:5.2f}  {x['maxp_median']:6.3f} {x['H_mean']:5.3f} {x['kept_median']:5.0f}"
              f"   {100 * x['ctr_train_mean']:6.3f}   {100 * x['ctr_test_mean']:6.3f} ({100 * x['ctr_test_sd']:.3f}, {100 * x['ctr_test_stat_se']:.3f})"
              f"   {x.get('l1_seedmean_mf_T', float('nan')):.3f}")
    for d in a0:
        print(f"alpha=0 {d['tag']}: not train-best {d['frac_not_train_best']:.2f}, not test-best {d['frac_not_test_best']:.2f}, "
              f"items {d['collapse_items']}, steps to collapse (median) {d['steps_to_collapse_median']:.0f}")


if __name__ == "__main__":
    main()
