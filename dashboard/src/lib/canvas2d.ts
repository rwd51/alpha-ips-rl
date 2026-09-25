/** Helpers for the 2-D canvas instruments. */
import { C } from "./color";

export const MONO = '"IBM Plex Mono", ui-monospace, Menlo, monospace';

export interface Surface2D {
  g: CanvasRenderingContext2D;
  w: number;
  h: number;
  /** device pixels per CSS pixel */
  s: number;
}

/** Match the canvas backing store to its CSS size (capped at 2x). */
export function fit2D(cv: HTMLCanvasElement): Surface2D {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const r = cv.getBoundingClientRect();
  const w = Math.max(10, Math.round(r.width * dpr));
  const h = Math.max(10, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) {
    cv.width = w;
    cv.height = h;
  }
  const g = cv.getContext("2d");
  if (!g) throw new Error("2-D canvas not available");
  return { g, w, h, s: dpr };
}

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}
export type Tick = [number, string];

/** Grid, tick labels, frame and axis titles for a chart box. */
export function axes(
  g: CanvasRenderingContext2D,
  s: number,
  B: Box,
  X: (v: number) => number,
  Y: (v: number) => number,
  xt: Tick[],
  yt: Tick[],
  xl = "",
  yl = "",
): void {
  g.save();
  g.lineWidth = s;
  g.font = `${10.5 * s}px ${MONO}`;
  for (const [v, t] of xt) {
    const x = X(v);
    g.strokeStyle = C.rule2;
    g.beginPath();
    g.moveTo(x, B.y);
    g.lineTo(x, B.y + B.h);
    g.stroke();
    g.fillStyle = C.muted;
    g.textAlign = "center";
    g.textBaseline = "top";
    g.fillText(t, x, B.y + B.h + 5 * s);
  }
  for (const [v, t] of yt) {
    const y = Y(v);
    g.strokeStyle = C.rule2;
    g.beginPath();
    g.moveTo(B.x, y);
    g.lineTo(B.x + B.w, y);
    g.stroke();
    g.fillStyle = C.muted;
    g.textAlign = "right";
    g.textBaseline = "middle";
    g.fillText(t, B.x - 6 * s, y);
  }
  g.strokeStyle = C.rule;
  g.strokeRect(B.x, B.y, B.w, B.h);
  g.fillStyle = C.ink2;
  g.font = `${11 * s}px ${MONO}`;
  if (xl) {
    g.textAlign = "center";
    g.textBaseline = "top";
    g.fillText(xl, B.x + B.w / 2, B.y + B.h + 20 * s);
  }
  if (yl) {
    g.save();
    g.translate(B.x - 38 * s, B.y + B.h / 2);
    g.rotate(-Math.PI / 2);
    g.textAlign = "center";
    g.textBaseline = "bottom";
    g.fillText(yl, 0, 0);
    g.restore();
  }
  g.restore();
}

export function label(
  g: CanvasRenderingContext2D,
  s: number,
  text: string,
  x: number,
  y: number,
  color: string = C.ink2,
  align: CanvasTextAlign = "left",
  size = 11,
  base: CanvasTextBaseline = "middle",
): void {
  g.save();
  g.fillStyle = color;
  g.font = `${size * s}px ${MONO}`;
  g.textAlign = align;
  g.textBaseline = base;
  g.fillText(text, x, y);
  g.restore();
}

/** Draw a polyline through points produced by fn(k) for k = 0..n. */
export function polyline(g: CanvasRenderingContext2D, n: number, fn: (k: number) => [number, number]): void {
  g.beginPath();
  for (let k = 0; k <= n; k++) {
    const [x, y] = fn(k);
    if (k) g.lineTo(x, y);
    else g.moveTo(x, y);
  }
  g.stroke();
}
