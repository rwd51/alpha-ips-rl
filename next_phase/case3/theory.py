"""
Step 3d -- the project's finite-group theory, evaluated with the REAL CTRs.

For every replay configuration with alpha > 0 (and alpha = 0 for reference):
  * the stationary mean field  p_inf  (src.finite_group_general.meanfield_stationary
    for the plain update; src.meanfield_rule.stationary with the group-mean-baseline
    weight for the grpo update), with r = the training half's per-item CTRs;
  * the mean-field ODE  dz_i/dt = p_i (c_i w_G(p_i) - S(p))  integrated by RK4
    (src.integrators.rk4_step) from the uniform policy over EXACTLY the replay's
    training budget (time T*h), with step-doubling error control, and its time
    average over the last 25% of the budget (the same window as the replay).

For the main campaign / split also the per-item extinction exponent
alpha_ext(j; G) for G in {16, 64, 256}: item j survives iff
c_j * G^alpha > S*(alpha). S*(alpha) is solved on an alpha grid and each item's
crossing found by linear interpolation of h_j(alpha) = ln c_j + alpha ln G -
ln S*(alpha); a few items are re-solved with src.finite_group_general.
extinction_alpha (bisection) to check the interpolation. Items that no alpha
can keep are flagged with src.finite_group_general.kept_at_large_alpha.

Outputs: results/theory_runs/<tag>.npz (git-ignored), results/theory_summary.json,
         results/extinction_alpha.csv

Usage (from the repository root, after run_replay.py):
  python3 next_phase/case3/theory.py --workers 7
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import obd_ips as M  # noqa: E402
import run_replay as R  # noqa: E402
from src.finite_group_general import kept_at_large_alpha  # noqa: E402

TH = M.RES / "theory_runs"
ALPHA_GRID = {16: np.round(np.arange(0.02, 4.0001, 0.02), 4),
              64: np.round(np.arange(0.02, 4.0001, 0.02), 4),
              256: np.round(np.arange(0.05, 3.0001, 0.05), 4)}


def train_ctr(campaign, split):
    tr, _ = R.split_logs(campaign, split)
    return M.item_stats(tr["item"], tr["click"], tr["K"])["ctr"]


def mf_job(c):
    """Stationary mean field + mean-field ODE over the replay budget for one config."""
    t0 = time.time()
    ctr = train_ctr(c["campaign"], c["split"])
    h = M.step_size(c["alpha"], c["G"], c["J"])
    out = {"tag": c["tag"], "alpha": c["alpha"], "G": c["G"], "variant": c["variant"]}
    if c["alpha"] > 0:
        p_inf, S_inf = M.predicted_policy(ctr, c["alpha"], c["G"], c["variant"])
    else:
        p_inf, S_inf = M.ideal_policy(ctr, 0.0), float(ctr.max())
    # step doubling: accept when the time-averaged policy changes by < 1e-4 in l1
    n, prev, err = 2000, None, np.inf
    while True:
        ts, ps, pavg = M.mf_trajectory(ctr, c["G"], c["alpha"], h, c["T"], variant=c["variant"], n_rk4=n)
        if prev is not None:
            err = float(np.abs(pavg - prev).sum())
            if err < 1e-4 or n >= 64000:
                break
        prev, n = pavg, 2 * n
    out.update(p_inf=p_inf, S_inf=S_inf, mf_t=ts, mf_p=ps, mf_pavg=pavg, n_rk4=n, rk4_doubling_err=err,
               seconds=time.time() - t0)
    TH.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TH / f"{c['tag']}.npz", **out)
    return c["tag"], n, err, time.time() - t0


def S_star_job(args):
    G, a, ctr = args
    _, S = M.predicted_policy(ctr, a, G)
    return G, a, S


def extinction_table(workers):
    ctr = train_ctr("all", "A")
    K = len(ctr)
    jobs = [(G, float(a), ctr) for G, grid in ALPHA_GRID.items() for a in grid]
    S = {}
    t0 = time.time()
    with Pool(workers) as pool:
        for G, a, s in pool.imap_unordered(S_star_job, jobs, chunksize=4):
            S[(G, a)] = s
    print(f"  S*(alpha) grids: {len(jobs)} mean-field solves in {time.time() - t0:.0f}s", flush=True)
    rows, alpha_ext, never = [], {}, {}
    for G, grid in ALPHA_GRID.items():
        lnS = np.log([S[(G, float(a))] for a in grid])
        ae = np.full(K, np.inf)
        for j in range(K):
            hj = np.log(ctr[j]) + grid * np.log(min(G, 1 / M.EPS)) - lnS
            pos = np.nonzero(hj > 0)[0]
            if len(pos) == 0:
                continue                      # not kept anywhere on the grid
            k = pos[0]
            if k == 0:
                ae[j] = grid[0]               # kept already at the smallest alpha on the grid
            else:                             # linear interpolation of the sign change
                ae[j] = grid[k - 1] + (grid[k] - grid[k - 1]) * (-hj[k - 1]) / (hj[k] - hj[k - 1])
        alpha_ext[G] = ae
        never[G] = np.array([not kept_at_large_alpha(ctr, j, G) for j in range(K)])
    # verification against the project's bisection-based extinction_alpha
    checks = []
    order = np.argsort(-ctr)
    for G in (16, 64, 256):
        cand = [j for j in order if np.isfinite(alpha_ext[G][j]) and alpha_ext[G][j] > ALPHA_GRID[G][0]]
        picks = [cand[len(cand) // 4], cand[len(cand) // 2], cand[-1]] if G != 256 else [cand[len(cand) // 2], cand[-1]]
        for j in picks:
            a0 = alpha_ext[G][j]
            t1 = time.time()
            try:
                a_exact, it = M.extinction_alpha(ctr, int(j), G, M.EPS, alpha_lo=max(1e-3, a0 - 0.1),
                                                 alpha_hi=a0 + 0.1, tol=1e-6)
            except ValueError:            # no sign change in the local bracket: widen it
                a_exact, it = M.extinction_alpha(ctr, int(j), G, M.EPS, alpha_lo=1e-3, alpha_hi=6.0, tol=1e-6)
            checks.append({"G": G, "item": int(j), "grid_interp": float(a0), "bisection": float(a_exact),
                           "abs_diff": float(abs(a0 - a_exact)), "bisection_iters": int(it),
                           "seconds": round(time.time() - t1, 1)})
            print(f"  check G={G} item {j}: interp {a0:.5f} vs extinction_alpha {a_exact:.5f}", flush=True)
    rank = np.empty(K, int)
    rank[order] = np.arange(1, K + 1)
    cbest = ctr.max()
    for j in order:
        row = {"item": int(j), "rank": int(rank[j]), "ctr_train": float(ctr[j]),
               "alpha_K2_formula_G16": float(np.log(cbest / ctr[j]) / np.log(16)),
               "alpha_K2_formula_G64": float(np.log(cbest / ctr[j]) / np.log(64)),
               "alpha_K2_formula_G256": float(np.log(cbest / ctr[j]) / np.log(256))}
        for G in (16, 64, 256):
            row[f"alpha_ext_G{G}"] = float(alpha_ext[G][j])
            row[f"never_kept_G{G}"] = bool(never[G][j])
        rows.append(row)
    with open(M.RES / "extinction_alpha.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(TH / "S_star_grid.npz", **{f"G{G}_alpha": ALPHA_GRID[G] for G in ALPHA_GRID},
                        **{f"G{G}_S": np.array([S[(G, float(a))] for a in ALPHA_GRID[G]]) for G in ALPHA_GRID})
    return {"checks": checks,
            "never_kept_count": {str(G): int(never[G].sum()) for G in never},
            "kept_count_at_alpha": {str(G): {f"{a:g}": int(np.sum(alpha_ext[G] < a)) for a in (0.25, 0.5, 0.75, 1, 1.5, 2, 3)}
                                    for G in alpha_ext}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--part", default="all", choices=["all", "mf", "extinction"])
    args = ap.parse_args()
    TH.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary = {}
    if args.part in ("all", "mf"):
        cfgs = [c for s in R.SUITES for c in R.suite(s)]
        # the mean field is identical for click and deterministic-CTR rewards; the
        # alpha = 0 step-size sweep shares the J = 0.1 ODE up to time rescaling
        cfgs = [c for c in cfgs if c["mode"] == "click" and not (c["alpha"] == 0 and c["J"] != R.J_MAIN)]
        todo = [c for c in cfgs if not (TH / f"{c['tag']}.npz").exists()]
        todo.sort(key=lambda c: -c["G"])
        print(f"mean-field ODE + stationary point for {len(todo)} configurations", flush=True)
        with Pool(args.workers) as pool:
            for tag, n, err, sec in pool.imap_unordered(mf_job, todo):
                print(f"  {tag}: RK4 steps {n}, doubling error {err:.1e}, {sec:.0f}s", flush=True)
    if args.part in ("all", "extinction"):
        summary["extinction"] = extinction_table(args.workers)
    summary["wall_s"] = round(time.time() - t0, 1)
    path = M.RES / "theory_summary.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(summary)
    path.write_text(json.dumps(old, indent=1))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
