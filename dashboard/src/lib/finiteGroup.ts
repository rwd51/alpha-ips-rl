/**
 * Finite-group theory of Experiments 2-5: exact binomial sums, the weight
 * rules of src/estimators.py, and the two-outcome mean field of
 * src/finite_group.py.
 */

const LF = new Float64Array(2100);
for (let n = 1; n < LF.length; n++) LF[n] = LF[n - 1] + Math.log(n);

/** Binomial(n, p) probabilities for k = 0..n, computed in log space. */
export function binomPmf(n: number, p: number): Float64Array {
  const out = new Float64Array(n + 1);
  if (p <= 0) {
    out[0] = 1;
    return out;
  }
  if (p >= 1) {
    out[n] = 1;
    return out;
  }
  const lp = Math.log(p);
  const lq = Math.log1p(-p);
  for (let k = 0; k <= n; k++) out[k] = Math.exp(LF[n] - LF[k] - LF[n - k] + k * lp + (n - k) * lq);
  return out;
}

export type RuleKind = "clip" | "lap1" | "lap05" | "offset";
export type WeightRule = (n: number) => number;

/**
 * Weight rules omega(n) for an outcome seen n times in a group of G:
 *   clip   max(n/G, eps)^-a                     (the paper, Eq. 9)
 *   lap*   ((n + lam)/(G + 2 lam))^-a           (Laplace lam=1, Jeffreys lam=1/2, K=2)
 *   offset max((n - c)/(G - c), eps)^-a, c=(1-a)/2  (Experiment 5's alpha-matched offset)
 */
export function makeRule(kind: RuleKind, G: number, a: number, eps: number): WeightRule {
  if (kind === "lap1" || kind === "lap05") {
    const lam = kind === "lap1" ? 1 : 0.5;
    return (n) => Math.pow((n + lam) / (G + 2 * lam), -a);
  }
  if (kind === "offset") {
    const c = (1 - a) / 2;
    return (n) => Math.pow(Math.max((n - c) / (G - c), eps), -a);
  }
  return (n) => Math.pow(Math.max(n / G, eps), -a);
}

/** Size-biased effective weight w_G(p) = E[omega(1 + B)], B ~ Binomial(G-1, p). */
export function effWeight(p: number, G: number, om: WeightRule): number {
  const pm = binomPmf(G - 1, p);
  let s = 0;
  for (let b = 0; b < G; b++) s += pm[b] * om(1 + b);
  return s;
}

/**
 * Two-outcome finite-group stationary p_1 for reward ratio rho = r1/r2:
 * root of rho w(p) = w(1-p) by bisection, or 1 when the weaker outcome dies,
 * which happens exactly when omega(1)/omega(G) <= rho (Exp 5, Eq. 5.8).
 */
export function meanfieldK2(rho: number, G: number, om: WeightRule): number {
  if (om(G) * rho - om(1) >= 0) return 1;
  let lo = 0.5;
  let hi = 1 - 1e-12;
  for (let i = 0; i < 52; i++) {
    const m = 0.5 * (lo + hi);
    if (rho * effWeight(m, G, om) - effWeight(1 - m, G, om) > 0) lo = m;
    else hi = m;
  }
  return 0.5 * (lo + hi);
}

/** Experiment 2's survival threshold alpha_c = ln(rho) / ln min(G, 1/eps). */
export function criticalAlpha(rho: number, G: number, eps: number): number {
  return Math.log(rho) / Math.log(Math.min(G, 1 / eps));
}
