/** Can a grid replace the solver? Piecewise-linear interpolation of the finite-group stationary mass, audited on held-out points (Exp 4 Fig 3). */
import { useMemo, useState } from "react";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { Readouts, Segmented, Slider } from "../components/controls";
import { Instrument, Notes, Stage } from "../components/Instrument";
import { makeRule, meanfieldK2 } from "../lib/finiteGroup";
import { mulberry } from "../lib/sampling";
import { lsq } from "../lib/numerics";
import { C, rgba } from "../lib/color";
import { axes, fit2D, label, polyline, type Box } from "../lib/canvas2d";
import { fsig } from "../lib/format";

type Axis = "eps" | "alpha" | "rho";
const G = 16;
// Exp 4 Fig 3 grid ranges (log-spaced), at G = 16
const RANGE: Record<Axis, [number, number]> = { alpha: [0.2, 3], rho: [1.1, 16], eps: [1e-3, 0.4] };
const NAME: Record<Axis, string> = { alpha: "exponent α", rho: "reward ratio ρ", eps: "clip ε" };

interface Params {
  axis: Axis;
  alpha: number;
  rho: number;
  eps: number;
  N: number;
}

/** Minority mass p2 of the K = 2 finite-group stationary point. */
function p2(alpha: number, rho: number, eps: number): number {
  return 1 - meanfieldK2(rho, G, makeRule("clip", G, alpha, eps));
}

export interface Audit {
  lo: number;
  hi: number;
  truthX: number[];
  truthY: number[];
  nodesX: number[];
  nodesY: number[];
  kinks: { x: number; kind: "clip" | "extinction" }[];
  hold: { x: number; err: number; cls: "smooth" | "kinked" | "extinct" }[];
  rmseSmooth: number;
  rmseKinked: number;
  nSmooth: number;
  nKinked: number;
  /** RMSE per refinement level, each over the cells of that level */
  levels: { N: number; smooth: number; kinked: number }[];
  orderSmooth: number;
  orderKinked: number;
}

const LEVELS = [5, 9, 17, 33, 65];

/** Everything the panel shows, for one slice through the atlas. */
export function audit(p: Params): Audit {
  const [a0, a1] = RANGE[p.axis];
  const lo = Math.log(a0);
  const hi = Math.log(a1);
  const f = (lx: number) => {
    const v = Math.exp(lx);
    return p.axis === "alpha" ? p2(v, p.rho, p.eps) : p.axis === "rho" ? p2(p.alpha, v, p.eps) : p2(p.alpha, p.rho, v);
  };
  // kink locations along this axis (derivation_exp4.md section 3)
  const kinks: Audit["kinks"] = [];
  const inRange = (x: number) => x > a0 && x < a1;
  if (p.axis === "eps") {
    for (let k = 1; k <= G; k++) if (inRange(k / G)) kinks.push({ x: Math.log(k / G), kind: "clip" });
    const e = Math.pow(p.rho, -1 / p.alpha); // m = 0 once the clip binds (eps > 1/G)
    if (e > 1 / G && inRange(e)) kinks.push({ x: Math.log(e), kind: "extinction" });
  } else if (p.axis === "alpha") {
    const ac = Math.log(p.rho) / Math.log(Math.min(G, 1 / p.eps));
    if (inRange(ac)) kinks.push({ x: Math.log(ac), kind: "extinction" });
  } else {
    const rc = Math.pow(Math.min(G, 1 / p.eps), p.alpha);
    if (inRange(rc)) kinks.push({ x: Math.log(rc), kind: "extinction" });
  }
  const grid = (n: number) => {
    const xs = Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1));
    return { xs, ys: xs.map(f) };
  };
  const interp = (gx: number[], gy: number[], x: number) => {
    const h = gx[1] - gx[0];
    const i = Math.min(gx.length - 2, Math.max(0, Math.floor((x - gx[0]) / h)));
    const t = (x - gx[i]) / h;
    return { v: gy[i] * (1 - t) + gy[i + 1] * t, i };
  };
  const kinkIn = (gx: number[], i: number) => kinks.some((k) => k.x > gx[i] && k.x < gx[i + 1]);
  const R = mulberry(4242);
  const hx = Array.from({ length: 400 }, () => lo + (hi - lo) * R());
  const ht = hx.map(f);
  const rms = (v: number[]) => (v.length ? Math.sqrt(v.reduce((a, b) => a + b * b, 0) / v.length) : NaN);
  /** Errors at the held-out points for an N-point grid, each point classed by the cell it falls in. */
  const classify = (N: number) => {
    const gr = grid(N);
    return hx.map((x, n) => {
      const c = interp(gr.xs, gr.ys, x);
      const kinked = kinkIn(gr.xs, c.i);
      const extinct = !kinked && gr.ys[c.i] === 0 && gr.ys[c.i + 1] === 0;
      const cls: "smooth" | "kinked" | "extinct" = kinked ? "kinked" : extinct ? "extinct" : "smooth";
      return { x, err: Math.abs(c.v - ht[n]), cls };
    });
  };
  const levels = LEVELS.map((N) => {
    const h = classify(N);
    return {
      N,
      smooth: rms(h.filter((q) => q.cls === "smooth").map((q) => q.err)),
      kinked: rms(h.filter((q) => q.cls === "kinked").map((q) => q.err)),
    };
  });
  // order = least-squares slope of log RMSE against log spacing h = 1/(N-1), over levels with a nonzero
  // error; at least three levels, since one kink's position inside its cell makes a single pair noisy
  const order = (key: "smooth" | "kinked") => {
    const pts = levels.filter((l) => Number.isFinite(l[key]) && l[key] > 1e-13);
    if (pts.length < 3) return NaN;
    return lsq(pts.map((l) => Math.log(1 / (l.N - 1))), pts.map((l) => Math.log(l[key]))).slope;
  };
  const coarse = grid(p.N);
  const hold = classify(p.N);
  const TN = 480;
  const truthX = Array.from({ length: TN + 1 }, (_, i) => lo + ((hi - lo) * i) / TN);
  return {
    lo,
    hi,
    truthX,
    truthY: truthX.map(f),
    nodesX: coarse.xs,
    nodesY: coarse.ys,
    kinks,
    hold,
    rmseSmooth: rms(hold.filter((q) => q.cls === "smooth").map((q) => q.err)),
    rmseKinked: rms(hold.filter((q) => q.cls === "kinked").map((q) => q.err)),
    nSmooth: hold.filter((q) => q.cls === "smooth").length,
    nKinked: hold.filter((q) => q.cls === "kinked").length,
    levels,
    orderSmooth: order("smooth"),
    orderKinked: order("kinked"),
  };
}

function createEngine(): Engine<Params> {
  let host!: EngineHost;
  let g!: CanvasRenderingContext2D;
  let W = 10;
  let H = 10;
  let s = 1;
  let P: Params = { axis: "alpha", alpha: 1, rho: 4, eps: 1e-3, N: 9 };
  let A = audit(P);
  let prevY: number[] | null = null;
  let morph = 1;
  let probe = 0;

  return {
    init(h) {
      host = h;
    },
    setParams(p) {
      const sameAxis = p.axis === P.axis && p.alpha === P.alpha && p.rho === P.rho && p.eps === P.eps;
      // when only the resolution changes, morph the interpolant from its old shape
      prevY = sameAxis && p.N !== P.N ? A.truthX.map((x) => lin(A, x)) : null;
      P = p;
      A = audit(p);
      morph = prevY ? 0 : 1;
    },
    resize() {
      ({ g, w: W, h: H, s } = fit2D(host.canvases[0]));
    },
    step(dt) {
      probe += dt * 0.12;
      if (morph < 1) morph = Math.min(1, morph + dt * 2);
    },
    draw() {
      g.fillStyle = C.stage;
      g.fillRect(0, 0, W, H);
      const { lo, hi } = A;
      const T: Box = { x: 60 * s, y: 84 * s, w: W - 80 * s, h: H * 0.5 - 84 * s };
      const X = (lx: number) => T.x + ((lx - lo) / (hi - lo)) * T.w;
      const ymax = Math.max(0.05, ...A.truthY) * 1.12;
      const Y = (v: number) => T.y + T.h - (v / ymax) * T.h;
      const [r0, r1] = RANGE[P.axis];
      const ticks = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.1, 2, 3, 4, 8, 16].filter((v) => v >= r0 * 0.999 && v <= r1 * 1.001);
      axes(
        g,
        s,
        T,
        X,
        Y,
        ticks.map((v) => [Math.log(v), String(v)] as [number, string]),
        [[0, "0"], [0.1, "0.1"], [0.2, "0.2"], [0.3, "0.3"], [0.4, "0.4"], [0.5, "0.5"]].filter(([v]) => (v as number) <= ymax) as [number, string][],
        "",
        "minority mass p₂",
      );
      // kinked cells
      for (let i = 0; i + 1 < A.nodesX.length; i++) {
        if (A.kinks.some((k) => k.x > A.nodesX[i] && k.x < A.nodesX[i + 1])) {
          g.fillStyle = rgba(C.verm, 0.1);
          g.fillRect(X(A.nodesX[i]), T.y, X(A.nodesX[i + 1]) - X(A.nodesX[i]), T.h);
        }
      }
      for (const k of A.kinks) {
        g.strokeStyle = rgba(k.kind === "clip" ? C.yellow : C.verm, 0.7);
        g.lineWidth = s;
        g.setLineDash([3 * s, 3 * s]);
        polyline(g, 1, (i) => [X(k.x), T.y + i * T.h]);
      }
      g.setLineDash([]);
      g.save();
      g.beginPath();
      g.rect(T.x, T.y, T.w, T.h);
      g.clip();
      g.strokeStyle = C.sky;
      g.lineWidth = 2.4 * s;
      polyline(g, A.truthX.length - 1, (i) => [X(A.truthX[i]), Y(A.truthY[i])]);
      g.strokeStyle = C.orange;
      g.lineWidth = 1.6 * s;
      polyline(g, A.truthX.length - 1, (i) => {
        const v = lin(A, A.truthX[i]);
        const w = prevY ? prevY[i] + (v - prevY[i]) * morph : v;
        return [X(A.truthX[i]), Y(w)];
      });
      g.fillStyle = C.orange;
      A.nodesX.forEach((x, i) => {
        g.beginPath();
        g.arc(X(x), Y(A.nodesY[i]), 3.4 * s, 0, 2 * Math.PI);
        g.fill();
      });
      // a probe sweeping the axis: solver value against the grid's guess
      const ph = (Math.sin(probe * 2 * Math.PI - Math.PI / 2) + 1) / 2;
      const px = lo + (hi - lo) * (0.02 + 0.96 * ph);
      const ti = Math.round(((px - lo) / (hi - lo)) * (A.truthX.length - 1));
      const tv = A.truthY[ti];
      const iv = lin(A, A.truthX[ti]);
      g.strokeStyle = rgba(C.ink, 0.3);
      g.lineWidth = s;
      polyline(g, 1, (i) => [X(px), T.y + i * T.h]);
      g.restore();
      label(g, s, "solver (blue) · grid interpolant (orange) · dashed: clip kink ε = k/G (yellow), extinction m = 0 (red)", T.x + T.w, T.y - 10 * s, C.muted, "right", 10);
      label(g, s, NAME[P.axis] + " (log)", T.x + T.w / 2, T.y + T.h + 22 * s, C.ink2, "center", 11, "top");
      // error panel
      const E: Box = { x: T.x, y: T.y + T.h + 58 * s, w: T.w * 0.66, h: H - (T.y + T.h + 58 * s) - 40 * s };
      const XE = (lx: number) => E.x + ((lx - lo) / (hi - lo)) * E.w;
      const YE = (e: number) => E.y + ((-1 - Math.log10(Math.max(e, 1e-7))) / 6) * E.h;
      axes(g, s, E, XE, YE, [], [[0.1, "0.1"], [1e-3, "10⁻³"], [1e-5, "10⁻⁵"], [1e-7, "≤10⁻⁷"]], "", "|error|");
      label(g, s, "error at 400 held-out points, by the cell each one falls in", E.x, E.y - 10 * s, C.muted, "left", 10);
      for (const h of A.hold) {
        g.fillStyle = h.cls === "kinked" ? rgba(C.verm, 0.9) : h.cls === "smooth" ? rgba(C.green, 0.9) : rgba(C.muted, 0.5);
        g.beginPath();
        g.arc(XE(h.x), YE(h.err), 2.1 * s, 0, 2 * Math.PI);
        g.fill();
      }
      // convergence inset: RMSE against grid points per axis, log-log, with slope-1 and slope-2 guides
      const Q: Box = { x: E.x + E.w + 64 * s, y: E.y, w: T.x + T.w - (E.x + E.w + 64 * s), h: E.h };
      const XQ = (N: number) => Q.x + ((Math.log2(N - 1) - 2) / 4) * Q.w;
      const YQ = (e: number) => Q.y + ((-1 - Math.log10(Math.max(e, 1e-7))) / 6) * Q.h;
      axes(g, s, Q, XQ, YQ, [[5, "5"], [9, "9"], [17, "17"], [33, "33"], [65, "65"]], [[0.1, "0.1"], [1e-4, "10⁻⁴"], [1e-7, "10⁻⁷"]], "points / axis");
      label(g, s, "RMSE vs grid size", Q.x, Q.y - 10 * s, C.muted, "left", 10);
      g.save();
      g.beginPath();
      g.rect(Q.x, Q.y, Q.w, Q.h);
      g.clip();
      for (const [key, col] of [["smooth", C.green], ["kinked", C.verm]] as const) {
        const pts = A.levels.filter((l) => Number.isFinite(l[key]) && l[key] > 1e-13);
        if (!pts.length) continue;
        g.strokeStyle = col;
        g.lineWidth = 1.6 * s;
        polyline(g, pts.length - 1, (i) => [XQ(pts[i].N), YQ(pts[i][key])]);
        g.fillStyle = col;
        for (const l of pts) {
          g.beginPath();
          g.arc(XQ(l.N), YQ(l[key]), (l.N === P.N ? 4.2 : 2.6) * s, 0, 2 * Math.PI);
          g.fill();
        }
        // guide through the first point with the theoretical slope
        const p0 = pts[0];
        const slope = key === "smooth" ? 2 : 1;
        g.strokeStyle = rgba(col, 0.35);
        g.setLineDash([3 * s, 3 * s]);
        polyline(g, 1, (i) => {
          const N = i ? 65 : p0.N;
          return [XQ(N), YQ(p0[key] * Math.pow((p0.N - 1) / (N - 1), slope))];
        });
        g.setLineDash([]);
      }
      g.restore();
      host.emit({
        hud: `${NAME[P.axis]} = ${fsig(Math.exp(px), 3)}\nsolver p₂ = ${tv.toFixed(4)}\ngrid   p₂ = ${iv.toFixed(4)}`,
      });
    },
    dispose() {},
  };
}

/** The coarse-grid interpolant at log-coordinate x. */
function lin(A: Audit, x: number): number {
  const gx = A.nodesX;
  const h = gx[1] - gx[0];
  const i = Math.min(gx.length - 2, Math.max(0, Math.floor((x - gx[0]) / h)));
  const t = (x - gx[i]) / h;
  return A.nodesY[i] * (1 - t) + A.nodesY[i + 1] * t;
}

export function InterpolationAudit() {
  const [axis, setAxis] = useState<Axis>("alpha");
  const [alpha, setAlpha] = useState(1);
  const [lgRho, setLgRho] = useState(Math.log(4));
  const [lgEps, setLgEps] = useState(Math.log(1e-3));
  const [N, setN] = useState(9);
  const rho = Math.exp(lgRho);
  const eps = Math.exp(lgEps);
  const params = useMemo(() => ({ axis, alpha, rho, eps, N }), [axis, alpha, rho, eps, N]);
  const handle = useEngine(createEngine, params);
  const A = useMemo(() => audit(params), [params]);
  const hud = typeof handle.readouts.hud === "string" ? handle.readouts.hud : "";

  return (
    <Instrument
      id="interp"
      chips={["Exp 4 Fig 3"]}
      ghostChips={["interpolation audit"]}
      title="Can a grid replace the solver?"
      claim="Interpolating a precomputed grid is second-order accurate where the answer is smooth, and drops to first order in any cell that contains a kink."
      stage={<Stage handle={handle} aspect="16 / 11" hud={hud} playOnTop label="Solver curve and grid interpolant along one axis of the atlas, with the error at held-out points colored by cell type." />}
    >
      <Segmented
        name="interp-axis"
        legend="Slice along"
        options={[
          { value: "eps", label: "Clip ε" },
          { value: "alpha", label: "Exponent α" },
          { value: "rho", label: "Ratio ρ" },
        ]}
        value={axis}
        onChange={setAxis}
      />
      {axis !== "alpha" ? <Slider id="interp-a" label="Exponent α" min={0.2} max={3} step={0.01} value={alpha} onChange={setAlpha} format={(v) => v.toFixed(2)} /> : null}
      {axis !== "rho" ? <Slider id="interp-r" label="Reward ratio ρ" min={Math.log(1.1)} max={Math.log(16)} step={0.01} value={lgRho} onChange={setLgRho} format={(v) => Math.exp(v).toFixed(2)} /> : null}
      {axis !== "eps" ? <Slider id="interp-e" label="Clip ε" min={Math.log(1e-3)} max={Math.log(0.4)} step={0.01} value={lgEps} onChange={setLgEps} format={(v) => fsig(Math.exp(v), 2)} /> : null}
      <Segmented
        name="interp-n"
        legend="Grid points per axis"
        options={[
          { value: "5", label: "5" },
          { value: "9", label: "9" },
          { value: "17", label: "17 (Exp 4)" },
          { value: "33", label: "33" },
        ]}
        value={String(N) as "5" | "9" | "17" | "33"}
        onChange={(v) => setN(parseInt(v, 10))}
      />
      <Readouts
        items={[
          [<span style={{ color: C.green }}>RMSE, kink-free cells</span>, `${fsig(A.rmseSmooth, 3)} (${A.nSmooth} pts)`],
          [<span style={{ color: C.verm }}>RMSE, kinked cells</span>, `${fsig(A.rmseKinked, 3)} (${A.nKinked} pts)`],
          ["order, kink-free cells (fit over 5–65)", Number.isFinite(A.orderSmooth) ? A.orderSmooth.toFixed(2) : "— (too few kink-free cells)"],
          ["order, kinked cells (fit over 5–65)", Number.isFinite(A.orderKinked) ? A.orderKinked.toFixed(2) : "— (no kink here)"],
        ]}
      />
      <Notes
        tryThis={[
          "Default slice (along α, ρ = 4, ε = 10⁻³, G = 16): the red line is the extinction point α_c = ln 4 / ln 16 = 0.5. To its left the minority is exactly 0 and the grid is exact (grey). The one red-shaded cell holds the large errors. The fits read about 2.0 for kink-free cells and 1.0 for the kinked one.",
          "Step the grid 5 → 9 → 17 → 33 and watch the inset: the green line falls along the slope-2 guide, the red one along slope 1.",
          "Switch to the ε slice: yellow lines mark every clip threshold ε = k/16. Above 1/16 they sit as close together as the grid nodes, so almost every cell is kinked, and the kink-free fit has too few cells to report. This is Exp 4's practical conclusion: decide survival from the formula, not from the grid.",
                ]}
        math={
          <>
            Linear interpolation between nodes h apart has error (h²/8)·|f″| where f is smooth: order 2. Inside a cell that contains a kink, f″ does not
            exist, and the error is (h/4)·(jump in slope): order 1. The minority mass has kinks where the margin m = α ln min(G, 1/ε) − ln ρ crosses 0, and at
            every ε = k/G, where the clip starts to bind on groups that saw the outcome k times. This slice is one-dimensional. Exp 4's 17³ audit on 800
            points gives the same picture: orders 2.19 and 1.86 in kink-free cells, 1.01 and 1.21 in kinked cells, and kinked-cell RMSE 16× larger.
          </>
        }
        analogy="Reading a mountain's height off a contour map: between contours you can interpolate, but on a cliff edge the map's resolution, not the terrain, decides how wrong you are."
      />
    </Instrument>
  );
}
