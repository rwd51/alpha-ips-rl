/** Palette: Okabe-Ito (the project's src/plotstyle.py), adapted for a dark stage. */

export const C = {
  ground: "#0a1018",
  stage: "#0c141f",
  surface: "#0f1823",
  rule: "#22314a",
  rule2: "#1a2638",
  ink: "#e6edf5",
  ink2: "#b4c2d2",
  muted: "#7d8fa6",
  sky: "#56b4e9",
  orange: "#e69f00",
  green: "#3cc49a",
  pink: "#d98bbe",
  yellow: "#f0e442",
  verm: "#ef7a3a",
  blue: "#8fa8ff",
} as const;

export const SERIES: string[] = [C.sky, C.orange, C.green, C.pink, C.yellow, C.verm, C.blue];

export type RGB = [number, number, number];

export function hexRgb(h: string): RGB {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function rgba(c: string | RGB, a = 1): string {
  const v = typeof c === "string" ? hexRgb(c) : c;
  return `rgba(${v[0] | 0},${v[1] | 0},${v[2] | 0},${a})`;
}

/** Single-hue sequential ramps (dark to light). */
export const RAMP_SKY: RGB[] = [
  [12, 28, 44],
  [31, 78, 110],
  [86, 180, 233],
  [214, 240, 255],
];
export const RAMP_WARM: RGB[] = [
  [40, 26, 14],
  [120, 70, 10],
  [230, 159, 0],
  [255, 228, 170],
];

export function ramp(t: number, stops: RGB[] = RAMP_SKY): RGB {
  t = Math.min(1, Math.max(0, t));
  const n = stops.length - 1;
  const x = t * n;
  const i = Math.min(n - 1, Math.floor(x));
  const f = x - i;
  const a = stops[i];
  const b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
