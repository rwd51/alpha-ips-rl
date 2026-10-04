"""
Check 07 -- is an EMA "unbiased in steady state" for the learning dynamics?

The update uses the term  p^_t * max(pbar_t, eps)^-alpha,  with
pbar_t = (1-beta) pbar_{t-1} + beta p^_t.  The current group enters BOTH factors,
so even for a static policy the dynamics-level distortion
      D_EMA = E[p^_t max(pbar_t, eps)^-alpha] / p^(1-alpha) - 1
is not zero at finite beta.  Second-order delta method (static p, sigma^2 = p(1-p)/G):
      D_EMA ~ alpha beta (1-p)/(G p) * [ (alpha+1) / (2 (2-beta)) - 1 ]
(beta = 1 gives the plain rule's alpha(alpha-1)(1-p)/(2Gp); beta -> 0 gives 0).
At alpha = 1 the plain clip has the exact D = -(1-p)^G, so for Gp >~ 3 an EMA with
finite beta is MORE biased, at the level the dynamics see, than the paper's rule.

Monte Carlo: 200k independent chains per cell, burn-in 30/beta steps, then
averages over 400 further steps (chains are independent; SE from chain means).

Run:  python3 -u next_phase/case2_checks/c07_ema_bias.py      (~1-2 min)
"""
import numpy as np

rng = np.random.default_rng(3)
EPS = 1e-3


def formula(p, G, alpha, beta):
    return alpha * beta * (1 - p) / (G * p) * ((alpha + 1) / (2 * (2 - beta)) - 1)


print(" alpha  G    p     beta | D_EMA (MC +- SE)        | delta-method | plain clip D")
for alpha in [1.0, 2.0]:
    for G, p in [(16, 0.3), (16, 0.5), (64, 0.3)]:
        n = np.arange(G + 1)
        from scipy.stats import binom
        # plain rule, exact size-biased distortion: E[p^ omega(n)]/p^(1-alpha) - 1
        om = np.where(n > 0, np.maximum(n / G, EPS) ** (-alpha), 0.0)
        D_plain = float(binom.pmf(n, G, p) @ ((n / G) * om)) / p ** (1 - alpha) - 1
        for beta in [0.02, 0.1, 0.3]:
            N = 200000
            pbar = np.full(N, p)
            for t in range(int(30 / beta)):
                pbar = (1 - beta) * pbar + beta * rng.binomial(G, p, N) / G
            acc = np.zeros(N)
            T = 400
            for t in range(T):
                ph = rng.binomial(G, p, N) / G
                pbar = (1 - beta) * pbar + beta * ph
                acc += ph * np.maximum(pbar, EPS) ** (-alpha)
            vals = acc / T / p ** (1 - alpha) - 1
            print(f"  {alpha:3.1f}  {G:3d}  {p:4.2f}  {beta:5.2f} | {vals.mean():+.5f} +- {vals.std()/np.sqrt(N):.5f}"
                  f"     | {formula(p, G, alpha, beta):+.5f}     | {D_plain:+.6f}")
