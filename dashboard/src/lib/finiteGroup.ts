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
  // exact value at the mode, then the ratio recurrence outward (terms that underflow become 0)
  const lp = Math.log(p);
  const lq = Math.log1p(-p);
  const m = Math.min(n, Math.floor((n + 1) * p));
  out[m] = Math.exp(LF[n] - LF[m] - LF[n - m] + m * lp + (n - m) * lq);
  const odds = p / (1 - p);
  for (let k = m; k < n; k++) out[k + 1] = (out[k] * (n - k) * odds) / (k + 1);
  for (let k = m; k > 0; k--) out[k - 1] = (out[k] * k) / ((n - k + 1) * odds);
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

/** d/dp of w_G(p) = E[omega(1 + B)]: (G-1) E[omega(2 + B') - omega(1 + B')], B' ~ Binomial(G-2, p). */
export function effWeightPrime(p: number, G: number, om: WeightRule): number {
  if (G < 2) return 0;
  const pm = binomPmf(G - 2, p);
  let s = 0;
  for (let b = 0; b <= G - 2; b++) s += pm[b] * (om(2 + b) - om(1 + b));
  return (G - 1) * s;
}

export interface NestedResult {
  p: number[];
  S: number;
  outer: number;
  /** total inner iterations over every outer step: the "work" of Exp 3 Fig 3 */
  work: number;
}

/**
 * K-outcome finite-group stationary point by nested root finding (Exp 3,
 * src/finite_group_general.py): inner, invert the decreasing w_G(p_i) = S / r_i
 * for each outcome; outer, choose S so that sum_i p_i(S) = 1. Outcomes with
 * r_i omega(1) <= S are extinct. "newton" uses safeguarded Newton at both
 * levels (bisection when a step leaves its bracket), "bisection" bisects both.
 */
export function nestedStationary(
  r: number[],
  G: number,
  om: WeightRule,
  method: "newton" | "bisection",
  tol = 1e-13,
  trace?: number[],
): NestedResult {
  const tab = Float64Array.from({ length: G + 1 }, (_, n) => om(n));
  om = (n) => tab[n];
  const w0 = om(1);
  const w1 = om(G);
  let work = 0;
  const inner = (target: number): number => {
    // solve w(p) = target on [0, 1]; w decreases from w0 to w1
    if (target >= w0) return 0;
    if (target <= w1) return 1;
    let lo = 0;
    let hi = 1;
    let x = 0.5;
    for (let it = 0; it < 200; it++) {
      work++;
      const f = effWeight(x, G, om) - target;
      if (f > 0) lo = x;
      else hi = x;
      let nx = 0.5 * (lo + hi);
      if (method === "newton") {
        const d = effWeightPrime(x, G, om);
        const nt = d !== 0 ? x - f / d : NaN;
        if (nt > lo && nt < hi) nx = nt;
      }
      if (Math.abs(nx - x) < tol || hi - lo < tol) return nx;
      x = nx;
    }
    return x;
  };
  const probs = (S: number) => r.map((ri) => inner(S / ri));
  // T decreases in S; bracket between the smallest and largest possible levels
  let lo = Math.min(...r) * w1;
  let hi = Math.max(...r) * w0;
  let S = 0.5 * (lo + hi);
  let outer = 0;
  for (; outer < 200; outer++) {
    const p = probs(S);
    const t = p.reduce((a, b) => a + b, 0) - 1;
    trace?.push(Math.abs(t));
    if (t > 0) lo = S;
    else hi = S;
    let nS = 0.5 * (lo + hi);
    if (method === "newton") {
      // dT/dS = sum over interior outcomes of 1 / (r_i w'(p_i))
      let d = 0;
      for (let i = 0; i < r.length; i++) if (p[i] > 0 && p[i] < 1) d += 1 / (r[i] * effWeightPrime(p[i], G, om));
      const nt = d !== 0 ? S - t / d : NaN;
      if (nt > lo && nt < hi) nS = nt;
    }
    // stop on the residual: near saturation T is so steep that a tiny step in S
    // can still leave sum p far from 1, so a step-size test alone stops too early
    if (Math.abs(t) < 1e-12 || hi - lo <= 4e-16 * Math.abs(S)) {
      outer++;
      break;
    }
    S = nS;
  }
  const p = probs(S);
  const tot = p.reduce((a, b) => a + b, 0);
  return { p: p.map((x) => x / tot), S, outer, work };
}

/**
 * Smallest alpha at which outcome j keeps positive mass, by bisection on alpha
 * of r_j omega(1) - S(alpha). Returns Infinity if it is still extinct at aMax.
 */
export function outcomeAlphaC(r: number[], j: number, G: number, eps: number, aMax = 12): number {
  const alive = (a: number) => {
    const om = makeRule("clip", G, a, eps);
    const res = nestedStationary(r, G, om, "newton", 1e-11);
    return r[j] * om(1) > res.S * (1 + 1e-9);
  };
  if (!alive(aMax)) return Infinity;
  let lo = 1e-3;
  let hi = aMax;
  if (alive(lo)) return lo;
  // 22 halvings of a log bracket spanning 12000x leave a relative error of about 2e-6
  for (let i = 0; i < 22; i++) {
    const m = Math.sqrt(lo * hi);
    if (alive(m)) hi = m;
    else lo = m;
  }
  return Math.sqrt(lo * hi);
}
