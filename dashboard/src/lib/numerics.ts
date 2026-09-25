/** Stability of explicit Runge-Kutta steps on the test equation y' = lambda y. */

/** RK4's real-axis stability limit: root of x^3 - 4x^2 + 12x - 24 = 0. */
export const RK4_LIMIT = 2.7852935634;

type Complex = [number, number];

function cmul(a: Complex, b: Complex): Complex {
  return [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
}

/**
 * |R_n(z)| where R_n(z) = sum_{k<=n} z^k / k!, the amplification factor of an
 * order-n Runge-Kutta step (n = 1 Euler, n = 4 classical RK4).
 */
export function ampPoly(z: Complex, n: number): number {
  let t: Complex = [1, 0];
  let s: Complex = [1, 0];
  for (let k = 1; k <= n; k++) {
    t = cmul(t, z);
    t = [t[0] / k, t[1] / k];
    s = [s[0] + t[0], s[1] + t[1]];
  }
  return Math.hypot(s[0], s[1]);
}
