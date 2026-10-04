"""
Step 3b -- alpha-IPS training by replay on the real Open Bandit Dataset logs.

Every configuration trains S independent seeds of a softmax policy over the K
items. Each step draws a group of G items; each drawn item's reward is a real
logged click of that item, resampled from the TRAINING half of the uniform-
random log (first 3.5 days). The second half (last 3.5 days) is never touched
during training and is used only to evaluate the learned policies.

Suites (each finishes in < 15 min on 6-7 cores):
  main         campaign "all" (K = 80), click rewards, plain update,
               alpha in {0, .25, .5, .75, 1, 1.5, 2, 3} x G in {16, 64},
               plus G = 256 for alpha in {0, .5, 1, 2}
  variants     deterministic-CTR rewards ("real reward landscape") and the GRPO
               group-mean-baseline update, alpha in {0, .5, 1, 2} x G in {16, 64}
  robustness   the alpha = 0 step-size sweep, the reversed time split
               (train on days 4-7, test on days 1-3.5), and alpha = .25, G = 16
               with a 2x smaller step (mean-field accuracy check)
  replication  campaigns "men" (K = 34) and "women" (K = 46)

Step size: h = J * G^(1 - alpha) with J = 0.1, i.e. one click on an item drawn
once in the group moves that item's logit by J for every alpha and G
(obd_ips.step_size). eps = 1e-3 (never binds: 1/G > eps for G <= 256).

Output: results/runs/<tag>.npz (one per configuration; git-ignored) and
results/runs/manifest_<suite>.json (timings).

Usage (from the repository root):
  python3 next_phase/case3/run_replay.py --suite main --workers 7
  python3 next_phase/case3/run_replay.py --suite variants --workers 7
  python3 next_phase/case3/run_replay.py --suite robustness --workers 7
  python3 next_phase/case3/run_replay.py --suite replication --workers 7
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import obd_ips as M  # noqa: E402

RUNS = M.RES / "runs"
SUITES = ("main", "variants", "robustness", "replication")
J_MAIN = 0.1
ALPHAS_MAIN = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
ALPHAS_CORE = [0.0, 0.5, 1.0, 2.0]


def cfg(campaign="all", split="A", mode="click", variant="plain", alpha=1.0, G=64,
        J=J_MAIN, T=None, S=None, seed=None):
    """One configuration. alpha = 0 runs use 100 seeds and 100k steps (collapse
    is absorbing); alpha > 0 runs use 32 seeds and 1M steps (400k at G = 256)."""
    if S is None:
        S = 100 if alpha == 0 else 32
    if T is None:
        T = 100_000 if alpha == 0 else (400_000 if G >= 256 else 1_000_000)
    if seed is None:   # fixed, distinct seed per configuration
        seed = 20260 + int(1000 * alpha) + 7 * G + {"A": 0, "B": 1}[split] * 100_003 \
            + {"click": 0, "ctr": 1}[mode] * 200_003 + {"plain": 0, "grpo": 1}[variant] * 300_007 \
            + {"all": 0, "men": 1, "women": 2}[campaign] * 400_009 + int(round(1000 * J))
    tag = f"{campaign}_{split}_{mode}_{variant}_a{alpha:g}_G{G}_J{J:g}"
    return dict(tag=tag, campaign=campaign, split=split, mode=mode, variant=variant,
                alpha=float(alpha), G=int(G), J=float(J), T=int(T), S=int(S), seed=int(seed))


def suite(name):
    out = []
    if name == "main":
        for G in (16, 64):
            for a in ALPHAS_MAIN:
                out.append(cfg(alpha=a, G=G))
        for a in ALPHAS_CORE:
            out.append(cfg(alpha=a, G=256))
    elif name == "variants":
        for G in (16, 64):
            for a in ALPHAS_CORE:
                out.append(cfg(mode="ctr", alpha=a, G=G))
                out.append(cfg(variant="grpo", alpha=a, G=G))
    elif name == "robustness":
        for G, Js in ((64, (0.025, 0.05, 0.2, 0.4)), (16, (0.025, 0.05, 0.2))):
            for J in Js:   # alpha = 0 step-size sweep: smaller steps need more of them
                out.append(cfg(alpha=0.0, G=G, J=J, T=int(100_000 * max(1.0, 0.1 / J)),
                               S=50 if J < 0.05 else 100))   # 50 seeds keep the 400k-step runs < 15 min
        for a in ALPHAS_CORE:
            out.append(cfg(split="B", alpha=a, G=64))
        # the one configuration where the replay deviates from the mean field
        # (alpha = .25, G = 16): does the gap shrink with a 2x smaller step?
        out.append(cfg(alpha=0.25, G=16, J=0.05, T=2_000_000, S=16))
    elif name == "replication":
        for camp in ("men", "women"):
            for G in (16, 64):
                for a in ALPHAS_CORE:
                    out.append(cfg(campaign=camp, alpha=a, G=G))
    else:
        raise ValueError(name)
    return out


_CACHE = {}


def split_logs(campaign, split):
    """(train_log, test_log) for split A (train = first half in time) or B (reversed)."""
    key = (campaign, split)
    if key not in _CACHE:
        log = M.load_log(campaign)
        first, second = M.time_split(log, 0.5)
        _CACHE[key] = (first, second) if split == "A" else (second, first)
    return _CACHE[key]


def run_one(c):
    train_log, test_log = split_logs(c["campaign"], c["split"])
    K = train_log["K"]
    world = M.ReplayWorld(train_log["item"], train_log["click"], K, c["mode"])
    st_tr = M.item_stats(train_log["item"], train_log["click"], K)
    st_te = M.item_stats(test_log["item"], test_log["click"], K)
    h = M.step_size(c["alpha"], c["G"], c["J"])
    t0 = time.time()
    r = M.train(world, c["alpha"], c["G"], c["T"], S=c["S"], seed=c["seed"], h=h,
                variant=c["variant"], eval_ctrs=(st_tr["ctr"], st_te["ctr"]))
    sec = time.time() - t0
    out = {k: v for k, v in r.items() if v is not None}
    out.update({"seconds": sec, "J": c["J"], "campaign": c["campaign"], "split": c["split"],
                "tag": c["tag"]})
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUNS / f"{c['tag']}.npz", **out)
    return c["tag"], sec, float(np.mean(r["p_final"].max(axis=1) > 0.95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=list(SUITES))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="re-run configurations that already exist")
    args = ap.parse_args()
    cfgs = suite(args.suite)
    todo = [c for c in cfgs if args.force or not (RUNS / f"{c['tag']}.npz").exists()]
    # longest jobs first for better load balance
    todo.sort(key=lambda c: -c["T"] * c["S"] * c["G"] ** 0.7)
    print(f"suite {args.suite}: {len(cfgs)} configurations, {len(todo)} to run, {args.workers} workers", flush=True)
    t0 = time.time()
    timings = {}
    with Pool(args.workers) as pool:
        for tag, sec, collapsed in pool.imap_unordered(run_one, todo):
            timings[tag] = round(sec, 1)
            print(f"  [{time.time() - t0:6.0f}s] {tag}: {sec:.0f}s, frac(max p > 0.95) = {collapsed:.2f}", flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"manifest_{args.suite}.json").write_text(json.dumps(
        {"configs": cfgs, "timings_s": timings, "wall_s": round(time.time() - t0, 1)}, indent=1))
    print(f"suite {args.suite} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
