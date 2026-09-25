/** Number formatting for readouts. */

export function fx(v: number, d = 2): string {
  return Number.isFinite(v) ? v.toFixed(d) : "—";
}

/** d significant figures, switching to scientific notation for very large or small values. */
export function fsig(v: number, d = 3): string {
  if (!Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) {
    const e = Math.floor(Math.log10(a));
    return (v / Math.pow(10, e)).toFixed(d - 1) + "e" + e;
  }
  return v.toPrecision(d);
}
