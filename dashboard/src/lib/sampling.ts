/**
 * The Monte-Carlo agent: src/dynamics.py::rhs_sampled, with the
 * inverse-transform sampler of Experiment 1.
 */
import { softmax, type Vec } from "./model";

export type Rng = () => number;

/** Small, fast, seedable PRNG (mulberry32) so runs are reproducible. */
export function mulberry(seed: number): Rng {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal draw by Box-Muller. */
export function gaussR(R: Rng): number {
  let u = 0;
  while (u === 0) u = R();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * R());
}

/** Inverse-transform sampling: index = number of CDF entries below u. */
export function drawOutcome(cdf: Vec, u: number): number {
  let k = 0;
  while (k < cdf.length - 1 && u > cdf[k]) k++;
  return k;
}

export function cumulative(p: Vec): Vec {
  const cdf = new Array<number>(p.length);
  let c = 0;
  for (let i = 0; i < p.length; i++) {
    c += p[i];
    cdf[i] = c;
  }
  cdf[p.length - 1] = 1;
  return cdf;
}

/** Counts of G draws from the policy p. */
export function sampleCounts(p: Vec, G: number, R: Rng): Vec {
  const cdf = cumulative(p);
  const cnt = new Array<number>(p.length).fill(0);
  for (let g = 0; g < G; g++) cnt[drawOutcome(cdf, R())]++;
  return cnt;
}

/**
 * One sampled update from a group of G:
 *   w_i = r_i p^_i max(p^_i, eps)^-a   (outcomes absent from the group give 0)
 *   g_i = w_i - p_i sum_k w_k
 */
export function sampledUpdate(z: Vec, r: Vec, a: number, G: number, eps: number, R: Rng, counts?: Vec) {
  const p = softmax(z);
  const K = z.length;
  const cnt = counts ?? sampleCounts(p, G, R);
  const w = new Array<number>(K);
  let S = 0;
  for (let i = 0; i < K; i++) {
    const ph = cnt[i] / G;
    w[i] = ph > 0 ? r[i] * ph * Math.pow(Math.max(ph, eps), -a) : 0;
    S += w[i];
  }
  for (let i = 0; i < K; i++) w[i] -= p[i] * S;
  return { g: w, cnt };
}
