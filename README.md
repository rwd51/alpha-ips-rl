# Numerical Simulation of Learning Dynamics in Multimodal RL

CSE 402 group project. Base paper: Sinha, Elango & Liu (2026),
*"Expected Return Causes Outcome-Level Mode Collapse in Reinforcement
Learning and How to Fix It with Inverse Probability Scaling"*,
arXiv:2601.21669. See `derivation.md` for the math and `RESULTS_exp<N>.md`
for the write-up of each experiment's results.

## Directory structure (convention — follow this for every new experiment)

```
src/                        shared core library, used by ALL experiments
  dynamics.py                softmax, ODE right-hand side, Monte-Carlo sampler
  integrators.py              Euler / RK4 / event-time detection
  fitting.py                   least-squares line fit
  plotstyle.py                  shared Nature-journal-like matplotlib style
  (add new shared modules here, e.g. rootfinding.py, hypergrid.py — never
   duplicate logic inside an experiments/ script)

experiments/
  exp1_collapse.py            one file per experiment: exp<N>_<short_name>.py
  exp2_ips_alpha_sweep.py     (next)
  exp3_rootfinding.py         (planned)
  exp4_hypergrid.py           (planned)
  exp5_bias_variance.py       (planned)

results/
  exp1_collapse/               one subfolder per experiment, same name as its script
    figures/                    generated PNGs (not tracked in git yet)
    data/                        generated CSV/JSON (not tracked in git yet)
  exp2_ips_alpha_sweep/
    figures/
    data/
  ...

RESULTS_exp1.md               one write-up per experiment, at repo root
RESULTS_exp2.md               (same naming: RESULTS_exp<N>.md)
...
derivation.md                  shared math derivations (append new sections here)
```

Rule of thumb when adding experiment N: create `experiments/exp<N>_<name>.py`,
have it write to `results/exp<N>_<name>/{figures,data}/`, and add
`RESULTS_exp<N>.md` at the root summarizing the findings. Put any reusable
numerical routine in `src/`, not inside the experiment script.

## Requirements

Python 3.10+, `numpy`, `scipy`, `matplotlib`.

```
pip install numpy scipy matplotlib
```

## Running

```
cd rl_mode_collapse
python3 experiments/exp1_collapse.py
```

Each experiment script is self-contained and runnable the same way. No
GPU, neural networks, or external RL library is needed anywhere in this
project — state is always just a K-length logit vector.
