/**
 * The outcome-selection bandit of the project, ported from src/dynamics.py.
 *
 * Equation (*) of derivation.md:
 *   z_i' = r_i p_i^(1-a) - p_i * sum_k r_k p_k^(1-a),   p = softmax(z)
 * a = 0 is expected-return training (collapse), a = 1 is the paper's IPS.
 */
export type Vec = number[];
export type Deriv = (z: Vec) => Vec;

export function softmax(z: ArrayLike<number>): Vec {
  let m = -Infinity;
  for (let i = 0; i < z.length; i++) if (z[i] > m) m = z[i];
  const e = new Array<number>(z.length);
  let s = 0;
  for (let i = 0; i < z.length; i++) {
    e[i] = Math.exp(z[i] - m);
    s += e[i];
  }
  for (let i = 0; i < z.length; i++) e[i] /= s;
  return e;
}

/** Right-hand side of the idealised flow (rhs_alpha). */
export function rhs(z: Vec, r: Vec, a: number): Vec {
  const p = softmax(z);
  const K = z.length;
  const w = new Array<number>(K);
  let S = 0;
  for (let i = 0; i < K; i++) {
    w[i] = r[i] * Math.pow(Math.max(p[i], 1e-12), 1 - a);
    S += w[i];
  }
  for (let i = 0; i < K; i++) w[i] -= p[i] * S;
  return w;
}

/** Classical fourth-order Runge-Kutta step (src/integrators.py::rk4_step). */
export function rk4(z: Vec, h: number, f: Deriv): Vec {
  const n = z.length;
  const t = new Array<number>(n);
  const k1 = f(z);
  for (let i = 0; i < n; i++) t[i] = z[i] + 0.5 * h * k1[i];
  const k2 = f(t);
  for (let i = 0; i < n; i++) t[i] = z[i] + 0.5 * h * k2[i];
  const k3 = f(t);
  for (let i = 0; i < n; i++) t[i] = z[i] + h * k3[i];
  const k4 = f(t);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = z[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  return out;
}

/** Explicit Euler step (src/integrators.py::euler_step). */
export function eulerStep(z: Vec, h: number, f: Deriv): Vec {
  const k = f(z);
  return z.map((v, i) => v + h * k[i]);
}

/** Stationary distribution p* ~ r^(1/a); a = 0 splits the top reward evenly. */
export function pStar(r: Vec, a: number): Vec {
  const rm = Math.max(...r);
  if (a <= 0) {
    const n = r.filter((x) => x === rm).length;
    return r.map((x) => (x === rm ? 1 / n : 0));
  }
  const w = r.map((x) => Math.pow(x / rm, 1 / a));
  const s = w.reduce((x, y) => x + y, 0);
  return w.map((v) => v / s);
}

/** ||r||_{1/a} = (sum_k r_k^(1/a))^a, computed without overflow. */
export function rewardNorm(r: Vec, a: number): number {
  const rm = Math.max(...r);
  return rm * Math.pow(r.reduce((s, x) => s + Math.pow(x / rm, 1 / a), 0), a);
}

/** Shannon entropy divided by ln K: 1 = uniform, 0 = collapsed. */
export function entropyN(p: Vec): number {
  let H = 0;
  for (const v of p) if (v > 0) H -= v * Math.log(v);
  return H / Math.log(p.length);
}

/** Eigenvalues of a small symmetric matrix by Jacobi rotations, ascending. */
export function symEig(M: number[][]): Vec {
  const n = M.length;
  const A = M.map((row) => row.slice());
  for (let sweep = 0; sweep < 80; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    if (off < 1e-26) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const x = A[k][p];
          const y = A[k][q];
          A[k][p] = c * x - s * y;
          A[k][q] = s * x + c * y;
        }
        for (let k = 0; k < n; k++) {
          const x = A[p][k];
          const y = A[q][k];
          A[p][k] = c * x - s * y;
          A[q][k] = s * x + c * y;
        }
      }
    }
  }
  return A.map((row, i) => row[i]).sort((x, y) => x - y);
}

/**
 * Experiment 2's linear rates: nonzero eigenvalues of
 * -J* = a ||r||_{1/a} (diag p* - p* p*^T), ascending. null at a = 0.
 */
export function linearRates(r: Vec, a: number): Vec | null {
  if (a <= 0) return null;
  const ps = pStar(r, a);
  const K = r.length;
  const c = a * rewardNorm(r, a);
  const M: number[][] = [];
  for (let i = 0; i < K; i++) {
    M.push([]);
    for (let j = 0; j < K; j++) M[i].push((i === j ? ps[i] : 0) - ps[i] * ps[j]);
  }
  return symEig(M)
    .slice(1)
    .map((v) => Math.max(v, 0) * c);
}

/** Two-outcome linear rate 2 a ||r|| p* q*. */
export function lambdaK2(r1: number, r2: number, a: number): number {
  const ps = pStar([r1, r2], a);
  return 2 * a * rewardNorm([r1, r2], a) * ps[0] * ps[1];
}
