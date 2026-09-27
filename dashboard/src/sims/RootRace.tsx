/** Bisection, secant and Newton racing to the same root of the K = 2 stationarity condition (Exp 3 Fig 1a). */
import { useMemo, useState } from "react";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { useLinkedAlpha } from "../engine/useLinkedAlpha";
import { Readouts, Slider } from "../components/controls";
import { Instrument, Notes, Stage } from "../components/Instrument";
import { lsq } from "../lib/numerics";
import { C, rgba } from "../lib/color";
import { axes, fit2D, label, polyline } from "../lib/canvas2d";

interface Params {
  alpha: number;
  rho: number;
  u0: number;
}

const logSig = (u: number) => (u > 0 ? -Math.log1p(Math.exp(-u)) : u - Math.log1p(Math.exp(u)));
const sig = (u: number) => 1 / (1 + Math.exp(-u));

// the three formulations of src/stationarity.py for r = (rho, 1)
function drift(u: number, rho: number, a: number) {
  const p = sig(u);
  const q = sig(-u);
  return 2 * (rho * Math.pow(p, 1 - a) * q - p * Math.pow(q, 1 - a));
}
function driftP(u: number, rho: number, a: number) {
  const p = sig(u);
  const q = sig(-u);
  return 2 * (rho * Math.pow(p, 1 - a) * q * ((1 - a) * q - p) - p * Math.pow(q, 1 - a) * (q - (1 - a) * p));
}
const balance = (u: number, rho: number, a: number) => rho * Math.exp(-a * logSig(u)) - Math.exp(-a * logSig(-u));
const balanceP = (u: number, rho: number, a: number) => -a * (rho * Math.exp(-a * logSig(u)) * sig(-u) + Math.exp(-a * logSig(-u)) * sig(u));

export interface Run {
  name: string;
  color: string;
  iters: number[];
  ok: boolean;
}

/** Every method's iterates, with the repository's stopping rules (|step| < 1e-15; bisection half-width < 1e-15). */
export function runAll(a: number, rho: number, u0: number): { runs: Run[]; ustar: number } {
  const ustar = Math.log(rho) / a;
  const tol = 1e-15;
  const bis: number[] = [];
  {
    let lo = -20;
    let hi = 20;
    const flo = drift(lo, rho, a);
    for (let it = 1; it <= 200; it++) {
      const mid = 0.5 * (lo + hi);
      bis.push(mid);
      const fm = drift(mid, rho, a);
      if (fm === 0 || 0.5 * (hi - lo) < tol) break;
      if (Math.sign(fm) === Math.sign(flo)) lo = mid;
      else hi = mid;
    }
  }
  const newton = (f: (u: number) => number, df: (u: number) => number) => {
    const tr = [u0];
    let x = u0;
    for (let it = 1; it <= 100; it++) {
      const fx = f(x);
      const d = df(x);
      if (fx === 0) return { tr, ok: true };
      if (d === 0 || !Number.isFinite(fx) || !Number.isFinite(d)) return { tr, ok: false };
      const st = fx / d;
      x -= st;
      tr.push(x);
      if (!Number.isFinite(x) || Math.abs(x) > 700) return { tr, ok: false };
      if (Math.abs(st) < tol) return { tr, ok: true };
    }
    return { tr, ok: false };
  };
  const sec = (() => {
    const tr = [u0, u0 + 0.5];
    let xp = u0;
    let x = u0 + 0.5;
    let fp = drift(xp, rho, a);
    let fx = drift(x, rho, a);
    for (let it = 1; it <= 100; it++) {
      if (fx === 0) return { tr, ok: true };
      const den = fx - fp;
      if (den === 0 || !Number.isFinite(den)) return { tr, ok: Math.abs(x - ustar) < 1e-9 };
      const st = (fx * (x - xp)) / den;
      xp = x;
      fp = fx;
      x -= st;
      tr.push(x);
      if (!Number.isFinite(x) || Math.abs(x) > 700) return { tr, ok: false };
      if (Math.abs(st) < tol) return { tr, ok: true };
      fx = drift(x, rho, a);
    }
    return { tr, ok: false };
  })();
  const nd = newton((u) => drift(u, rho, a), (u) => driftP(u, rho, a));
  const nb = newton((u) => balance(u, rho, a), (u) => balanceP(u, rho, a));
  const nl = newton((u) => Math.log(rho) - a * u, () => -a);
  return {
    ustar,
    runs: [
      { name: "bisection", color: C.muted, iters: bis, ok: true },
      { name: "secant", color: C.pink, iters: sec.tr, ok: sec.ok },
      { name: "Newton · drift", color: C.orange, iters: nd.tr, ok: nd.ok },
      { name: "Newton · balance", color: C.sky, iters: nb.tr, ok: nb.ok },
      { name: "Newton · log", color: C.green, iters: nl.tr, ok: nl.ok },
    ],
  };
}

/** Empirical order: least-squares slope of ln e_{n+1} against ln e_n over pairs inside [1e-14, 1e-1]. */
export function empiricalOrder(err: number[]): number {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i + 1 < err.length; i++) {
    const a = err[i];
    const b = err[i + 1];
    if (a > 1e-14 && a < 0.1 && b > 1e-14 && b < 0.1) {
      xs.push(Math.log(a));
      ys.push(Math.log(b));
    }
  }
  return xs.length >= 2 ? lsq(xs, ys).slope : NaN;
}

function summary(run: Run, ustar: number): string {
  // iterates after the start u0, as counted in RESULTS_exp3.md (the secant's second start u0 + 1/2 included)
  const n = run.name === "bisection" ? run.iters.length : run.iters.length - 1;
  if (!run.ok) return `diverged after ${n} steps`;
  const e = run.iters.map((u) => Math.abs(u - ustar));
  if (run.name === "bisection") {
    const sel = e.filter((v) => v > 1e-14 && v < 0.1);
    const ratio = sel.length > 2 ? Math.pow(sel[sel.length - 1] / sel[0], 1 / (sel.length - 1)) : NaN;
    return `${n} iterations · linear, ratio ${ratio.toFixed(3)}`;
  }
  if (n <= 1) return `${n} iteration · exact`;
  const q = empiricalOrder(e);
  return `${n} iterations · order ${Number.isFinite(q) ? q.toFixed(2) : "—"}`;
}

function createEngine(): Engine<Params> {
  let host!: EngineHost;
  let g!: CanvasRenderingContext2D;
  let W = 10;
  let H = 10;
  let s = 1;
  let res = runAll(1, 4, 0);
  let shown = 60;
  let hold = 0;
  let first = true;

  return {
    init(h) {
      host = h;
    },
    setParams(p) {
      res = runAll(p.alpha, p.rho, p.u0);
      if (first) first = false;
      else shown = 0;
      hold = 0;
    },
    resize() {
      ({ g, w: W, h: H, s } = fit2D(host.canvases[0]));
    },
    step(dt) {
      const maxLen = Math.max(...res.runs.map((r) => r.iters.length));
      if (shown >= maxLen) {
        hold += dt;
        if (hold > 3) {
          shown = 0;
          hold = 0;
        }
      } else shown = Math.min(maxLen, shown + dt * 4);
    },
    draw() {
      g.fillStyle = C.stage;
      g.fillRect(0, 0, W, H);
      const { runs, ustar } = res;
      const B = { x: 112 * s, y: 28 * s, w: W - 132 * s, h: H * 0.54 - 28 * s };
      const NX = 60;
      const X = (n: number) => B.x + (n / NX) * B.w;
      const Y = (e: number) => B.y + ((2 - Math.log10(Math.max(e, 1e-16))) / 18) * B.h;
      axes(
        g,
        s,
        B,
        X,
        Y,
        [[0, "0"], [10, "10"], [20, "20"], [30, "30"], [40, "40"], [50, "50"], [60, "60"]],
        [[100, "10²"], [1, "1"], [1e-4, "10⁻⁴"], [1e-8, "10⁻⁸"], [1e-12, "10⁻¹²"], [1e-16, "10⁻¹⁶"]],
        "iteration n",
        "error |uₙ − u*|",
      );
      g.save();
      g.beginPath();
      g.rect(B.x, B.y, B.w, B.h);
      g.clip();
      const k = Math.floor(shown);
      for (const run of runs) {
        const off = run.name === "bisection" ? 1 : 0;
        const e = run.iters.map((u) => Math.abs(u - ustar));
        const n = Math.min(e.length - 1, k);
        if (n < 0) continue;
        g.strokeStyle = run.color;
        g.lineWidth = 1.8 * s;
        polyline(g, n, (i) => [X(i + off), Y(e[i])]);
        g.fillStyle = run.color;
        for (let i = 0; i <= n; i++) {
          g.beginPath();
          g.arc(X(i + off), Y(e[i]), (run.name === "bisection" ? 1.8 : 3) * s, 0, 2 * Math.PI);
          g.fill();
        }
      }
      g.restore();
      label(g, s, "error of each method, log scale · machine precision is about 10⁻¹⁶", B.x, 14 * s, C.muted, "left", 10);
      // lower strip: where each method's current iterate sits on the u axis
      const S = { x: B.x, y: B.y + B.h + 56 * s, w: B.w, h: H - (B.y + B.h + 56 * s) - 30 * s };
      const span = 6;
      const XU = (u: number) => S.x + ((u - (ustar - span)) / (2 * span)) * S.w;
      const rowH = S.h / runs.length;
      g.strokeStyle = rgba(C.ink, 0.35);
      g.lineWidth = s;
      polyline(g, 1, (i) => [XU(ustar), S.y - 6 * s + i * (S.h + 6 * s)]);
      label(g, s, `u* = ln ρ / α = ${ustar.toFixed(4)} · strip spans u* ± 6`, XU(ustar) + 6 * s, S.y + S.h + 14 * s, C.ink2, "left", 10);
      runs.forEach((run, ri) => {
        const yc = S.y + (ri + 0.5) * rowH;
        g.strokeStyle = C.rule2;
        polyline(g, 1, (i) => [S.x + i * S.w, yc]);
        label(g, s, run.name, S.x - 8 * s, yc, run.color, "right", 9.5);
        const n = Math.min(run.iters.length - 1, k);
        if (n < 0) return;
        const u = run.iters[n];
        const x = Math.max(S.x, Math.min(S.x + S.w, XU(u)));
        const out = XU(u) < S.x || XU(u) > S.x + S.w;
        g.fillStyle = run.color;
        g.beginPath();
        if (out) {
          const dir = XU(u) < S.x ? -1 : 1;
          g.moveTo(x + dir * 7 * s, yc);
          g.lineTo(x - dir * 3 * s, yc - 5 * s);
          g.lineTo(x - dir * 3 * s, yc + 5 * s);
        } else g.arc(x, yc, 4.5 * s, 0, 2 * Math.PI);
        g.fill();
      });
      host.emit({});
    },
    dispose() {},
  };
}

export function RootRace() {
  const [alpha, setAlpha] = useLinkedAlpha("roots", 1, 0.2, 4);
  const [rho, setRho] = useState(4);
  const [u0, setU0] = useState(0);
  const params = useMemo(() => ({ alpha, rho, u0 }), [alpha, rho, u0]);
  const handle = useEngine(createEngine, params);
  const { runs, ustar } = useMemo(() => runAll(alpha, rho, u0), [alpha, rho, u0]);

  return (
    <Instrument
      id="roots"
      chips={["Exp 3 Fig 1a"]}
      ghostChips={["bisection · secant · Newton"]}
      title="The root-finder race"
      claim="Five solvers, one root. The error plot shows each method's order of convergence: a straight slope for bisection, steeper and steeper bends for secant and Newton."
      stage={<Stage handle={handle} aspect="16 / 11" label="Error of each root-finding method against iteration on a log scale, with a strip showing each method's current iterate." />}
    >
      <Slider id="roots-a" label="Exponent α" min={0.2} max={4} step={0.01} value={alpha} onChange={setAlpha} format={(v) => v.toFixed(2)} />
      <Slider id="roots-rho" label="Reward ratio ρ" min={1.25} max={10} step={0.05} value={rho} onChange={setRho} format={(v) => v.toFixed(2)} />
      <Slider id="roots-u0" label="Start u₀ (open methods)" min={-10} max={10} step={0.1} value={u0} onChange={setU0} format={(v) => v.toFixed(1)} />
      <Readouts items={runs.map((r) => [<span style={{ color: r.color }}>{r.name}</span>, summary(r, ustar)])} />
      <Notes
        tryThis={[
          "Defaults (α = 1, ρ = 4, u₀ = 0) are Exp 3's setting: bisection 56, secant 9 at order 1.61, Newton on the drift 5 at order 2.00, balance 6, log 1.",
          "Slide u₀ to 4 at α = 1: Newton on the drift runs off to infinity (arrow at the edge), while Newton on the balance form still lands. Same root, different equation.",
          "Raise α above about 1.11: the drift form converges from every start again (Exp 3 found the cutoff at α = 1.113).",
        ]}
        math={
          <>
            All five solve ż₁ − ż₂ = 0 for u = z₁ − z₂. Bisection halves a bracket, so its error falls by 1/2 per step: order 1. Secant replaces the derivative
            with the slope through the last two points: order (1 + √5)/2 ≈ 1.618. Newton squares the error each step: order 2. The log form ln ρ − αu is a
            straight line, so Newton solves it in one step. The order is measured as the least-squares slope of ln eₙ₊₁ against ln eₙ.
          </>
        }
        analogy="Finding a street address: bisection asks “left or right half of town?” each time; Newton reads the slope of the house numbers and jumps straight there, which works brilliantly unless the numbering goes flat far from the target."
      />
    </Instrument>
  );
}
