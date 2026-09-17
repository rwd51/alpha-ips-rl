# Numerical Simulation of Learning Dynamics in Multimodal RL

CSE 402 group project. Base paper: Sinha, Elango & Liu (2026),
*"Expected Return Causes Outcome-Level Mode Collapse in Reinforcement
Learning and How to Fix It with Inverse Probability Scaling"*,
arXiv:2601.21669. See `derivation.md` for the math and `RESULTS_exp<N>.md`
for the write-up of each experiment's results. `main.ipynb` at the repo root
runs any experiment and displays its figures (it is what we import as a Kaggle
notebook; no GPU is needed anywhere).

## Directory structure (convention — follow this for every new experiment)

```
src/                        shared core library, used by ALL experiments
  dynamics.py                softmax, ODE right-hand side, Monte-Carlo sampler,
                               Jacobian / linear rates at the stationary point
  integrators.py              Euler / RK4 / event-time detection / log-spaced recording
  fitting.py                   least-squares line fit
  rootfinding.py               bisection (scalar, and vectorized boundary search)
  newton.py                    Newton-Raphson, secant, safeguarded and multivariate Newton
  stationarity.py              the drift / balance / log forms of the stationarity equation
  finite_group.py              finite-G mean-field theory of the sampled update (K=2)
  finite_group_general.py      the same for any K (nested root-finding, extinction alphas)
  hypergrid.py                 tensor grids, multilinear interpolation, batched K=2 solver
  estimators.py                weight rules (clip / add-lambda / Richardson / alpha-matched
                                 offset), their exact bias-variance statistics, and the
                                 sampled update with a pluggable rule
  meanfield_rule.py            the finite-G mean field for an ARBITRARY weight rule
  optimize1d.py                golden-section search (and a scan to audit it)
  eigen.py                     power method, Hotelling deflation, QR algorithm
  ema.py                       exponential moving averages: variance factor, lag,
                                 and the second-order coupled dynamics
  plotstyle.py                  shared Nature-journal-like matplotlib style
  (add new shared modules here, e.g. hypergrid.py — never
   duplicate logic inside an experiments/ script)

experiments/
  exp1_collapse.py            one file per experiment: exp<N>_<short_name>.py
  exp2_ips_alpha_sweep.py     alpha sweep: diversity knob, rates, stiffness, MC
  exp3_rootfinding.py         stationary point by bisection / Newton, general-K mean field
  exp4_hypergrid.py           multidimensional finite-G atlas and interpolation audit
  exp5_bias_variance.py       estimator bias/variance and what it costs downstream

tests/
  test_hypergrid.py           Experiment 4
  test_estimators.py          Experiment 5

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
derivation_exp3.md             Experiment 3 root-finding and general-K mean field
derivation_exp4.md             Experiment 4 hypergrid and interpolation derivations
derivation_exp5.md             Experiment 5 estimator error analysis
main.ipynb                     Kaggle-ready notebook: runs any experiment, shows its figures
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
python3 experiments/exp1_collapse.py
```

Each experiment script is self-contained and runnable the same way, from the
repository root. No GPU, neural networks, or external RL library is needed
anywhere in this project — state is always just a K-length logit vector.

Tests (also from the repository root):

```
python3 -m unittest discover -s tests -t . -v
```

## Important: keep your local CSVs

`results/*/data/*.csv` is gitignored (not pushed) for now, but **everyone
should keep their own generated CSVs locally, not delete them** — we'll
need those trajectories later for large-scale runs.

## Note: `rhs_sampled` fix (made with Experiment 2)

`src/dynamics.py::rhs_sampled` used to give an outcome that did not appear
in the group (`p_hat = 0`) a spurious weight `r * eps^(1-alpha)` instead of 0.
That is negligible at alpha=0 (~1e-3) but enormous for alpha > 1 (1000·r at
alpha=2). It is fixed now. Experiment 1's committed figures predate the fix; a
full re-run with the fix leaves every Exp. 1 conclusion unchanged (details in
`RESULTS_exp2.md`). Use the fixed version for any new Monte-Carlo experiment.
