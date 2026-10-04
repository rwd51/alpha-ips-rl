#!/usr/bin/env bash
# Case 3 -- the whole pipeline, in order. Run from the repository root:
#     bash next_phase/case3/run_all.sh            (all stages)
#     bash next_phase/case3/run_all.sh theory     (one stage)
# Each stage skips work that already exists (delete results/runs/ to redo).
# WORKERS=<n> sets the number of parallel runs (default 4 = physical cores).
# Needs only python3 with numpy, scipy, matplotlib (no pandas, no torch).
set -euo pipefail
cd "$(dirname "$0")/../.."
W=${WORKERS:-4}   # 4 physical cores on this laptop: one worker per core keeps every run < 15 min
stage=${1:-all}
run() { echo "=== $* ($(date +%H:%M:%S))"; /usr/bin/time -f "    wall %es, max RSS %MkB" "$@"; }

if [[ $stage == all || $stage == data ]]; then
  run python3 next_phase/case3/prepare_data.py --bts           # ~45 MB + 204 MB streamed, ~3 min
fi
if [[ $stage == all || $stage == landscape ]]; then
  run python3 next_phase/case3/landscape.py                    # seconds
fi
for s in main variants robustness replication; do
  if [[ $stage == all || $stage == "$s" ]]; then
    run python3 next_phase/case3/run_replay.py --suite "$s" --workers "$W"   # 7-17 min each
  fi
done
if [[ $stage == all || $stage == theory ]]; then
  run python3 next_phase/case3/theory.py --workers "$W"        # ~10 min
fi
if [[ $stage == all || $stage == analyze ]]; then
  run python3 next_phase/case3/analyze.py
  run python3 next_phase/case3/make_figures.py
  run python3 next_phase/case3/report_tables.py
fi
