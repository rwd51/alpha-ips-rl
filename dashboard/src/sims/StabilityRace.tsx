/** Euler vs RK4 on the alpha-dynamics, next to both stability regions (Exp 1 Fig 3, Exp 2 Fig 3c). */
import { useMemo, useState } from "react";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { useLinkedAlpha } from "../engine/useLinkedAlpha";
import { Readouts, Slider, VerdictChip } from "../components/controls";
import { Instrument, Notes, Stage } from "../components/Instrument";
import { eulerStep, lambdaK2, pStar, rhs, rk4, softmax, type Deriv, type Vec } from "../lib/model";
import { ampPoly, RK4_LIMIT } from "../lib/numerics";
import { C, rgba } from "../lib/color";
import { axes, fit2D, label } from "../lib/canvas2d";
import { fsig } from "../lib/format";

interface Params {
  alpha: number;
  h: number;
}
const R2 = [4, 1];
const U0 = Math.log(1.5); // p1 = 0.6

function uDot(u: number, a: number): number {
  const d = rhs([u, 0], R2, a);
  return d[0] - d[1];
}
export function localRate(a: number): number {
  return -(uDot(U0 + 1e-5, a) - uDot(U0 - 1e-5, a)) / 2e-5;
}

function createEngine(): Engine<Params> {
  let host!: EngineHost;
  let g!: CanvasRenderingContext2D;
  let W = 10;
  let H = 10;
  let s = 1;
  let a = 2;
  let h = 0.3;
  let lam = 8;
  let lamLoc = 10;
  let T = 6;
  let ref: [number, number][] = [];
  let eu: [number, number][] = [];
  let rk: [number, number][] = [];
  let anim = 99;
  let raster: HTMLCanvasElement | null = null;
  let rasterKey = "";

  function run(step: (z: Vec, h: number, f: Deriv) => Vec, hh: number, cap: number): [number, number][] {
    const f: Deriv = (z) => rhs(z, R2, a);
    const out: [number, number][] = [[0, 0.6]];
    let z = [U0, 0];
    let t = 0;
    for (let n = 0; n < cap && t < T - 1e-12; n++) {
      z = step(z, hh, f);
      t += hh;
      const m = Math.max(z[0], z[1]);
      z = [z[0] - m, z[1] - m];
      const p1 = softmax(z)[0];
      if (!Number.isFinite(p1)) break;
      out.push([t, p1]);
    }
    return out;
  }
  function buildRaster(w: number, hh: number) {
    const key = `${w}x${hh}`;
    if (rasterKey === key && raster) return;
    rasterKey = key;
    raster = document.createElement("canvas");
    raster.width = w;
    raster.height = hh;
    const rg = raster.getContext("2d")!;
    const img = rg.createImageData(w, hh);
    const inE = new Uint8Array(w * hh);
    const inR = new Uint8Array(w * hh);
    for (let j = 0; j < hh; j++)
      for (let i = 0; i < w; i++) {
        const x = -4.2 + (5.4 * i) / (w - 1);
        const y = 3.2 - (6.4 * j) / (hh - 1);
        inE[j * w + i] = Math.hypot(1 + x, y) < 1 ? 1 : 0;
        inR[j * w + i] = ampPoly([x, y], 4) < 1 ? 1 : 0;
      }
    for (let j = 0; j < hh; j++)
      for (let i = 0; i < w; i++) {
        const k = j * w + i;
        const eb = (i + 1 < w && inE[k] !== inE[k + 1]) || (j + 1 < hh && inE[k] !== inE[k + w]);
        const rb = (i + 1 < w && inR[k] !== inR[k + 1]) || (j + 1 < hh && inR[k] !== inR[k + w]);
        let c = [12, 20, 31];
        if (inR[k]) c = [18, 52, 74];
        if (inE[k]) c = [70, 52, 20];
        if (rb) c = [86, 180, 233];
        if (eb) c = [239, 122, 58];
        img.data.set([c[0], c[1], c[2], 255], k * 4);
      }
    rg.putImageData(img, 0, 0);
  }

  return {
    init(hst) {
      host = hst;
    },
    setParams(p) {
      a = p.alpha;
      h = p.h;
      lam = lambdaK2(4, 1, a);
      lamLoc = localRate(a);
      T = Math.min(60, Math.max(6, 12 / lam));
      ref = run(rk4, Math.min(0.01, 0.1 / Math.max(lam, lamLoc)), 8000);
      eu = run(eulerStep, h, 8000);
      rk = run(rk4, h, 8000);
      anim = anim === 99 ? 99 : 0;
    },
    resize() {
      ({ g, w: W, h: H, s } = fit2D(host.canvases[0]));
      rasterKey = "";
    },
    step(dt) {
      anim += dt;
      if (anim > 5.5) anim = 0;
    },
    draw() {
      g.fillStyle = C.stage;
      g.fillRect(0, 0, W, H);
      const split = W * 0.6;
      const B = { x: 50 * s, y: 22 * s, w: split - 70 * s, h: H - 64 * s };
      const X = (t: number) => B.x + (t / T) * B.w;
      const Y = (p: number) => B.y + B.h - ((p + 0.03) / 1.06) * B.h;
      axes(g, s, B, X, Y, [0, T / 4, T / 2, (3 * T) / 4, T].map((v) => [v, fsig(v, 2)]), [[0, "0"], [0.5, "0.5"], [1, "1"]], "time t (start p₁ = 0.6)", "p₁(t)");
      const ps = pStar(R2, a)[0];
      g.save();
      g.beginPath();
      g.rect(B.x, B.y, B.w, B.h);
      g.clip();
      g.strokeStyle = rgba(C.ink, 0.3);
      g.setLineDash([3 * s, 4 * s]);
      g.lineWidth = s;
      g.beginPath();
      g.moveTo(B.x, Y(ps));
      g.lineTo(B.x + B.w, Y(ps));
      g.stroke();
      g.setLineDash([]);
      g.strokeStyle = rgba(C.ink2, 0.35);
      g.lineWidth = 5 * s;
      g.beginPath();
      ref.forEach(([t, p], k) => (k ? g.lineTo(X(t), Y(p)) : g.moveTo(X(t), Y(p))));
      g.stroke();
      const frac = Math.min(1, anim / 3.2);
      for (const [tr, col] of [
        [eu, C.verm],
        [rk, C.sky],
      ] as [[number, number][], string][]) {
        const n = Math.max(1, Math.round(frac * (tr.length - 1)));
        g.strokeStyle = col;
        g.lineWidth = 1.6 * s;
        g.beginPath();
        for (let k = 0; k <= n; k++) (k ? g.lineTo : g.moveTo).call(g, X(tr[k][0]), Y(tr[k][1]));
        g.stroke();
        if (tr.length < 400)
          for (let k = 0; k <= n; k++) {
            g.fillStyle = col;
            g.beginPath();
            g.arc(X(tr[k][0]), Y(tr[k][1]), 2.6 * s, 0, 2 * Math.PI);
            g.fill();
          }
      }
      g.restore();
      label(g, s, "Euler", B.x + 8 * s, B.y + 10 * s, C.verm, "left", 11);
      label(g, s, "RK4", B.x + 60 * s, B.y + 10 * s, C.sky, "left", 11);
      label(g, s, "exact", B.x + 100 * s, B.y + 10 * s, C.ink2, "left", 11);
      const B2 = { x: split + 30 * s, y: 22 * s, w: W - split - 46 * s, h: H - 64 * s };
      buildRaster(Math.max(2, Math.round(B2.w)), Math.max(2, Math.round(B2.h)));
      if (raster) g.drawImage(raster, B2.x, B2.y, B2.w, B2.h);
      const Xc = (x: number) => B2.x + ((x + 4.2) / 5.4) * B2.w;
      const Yc = (y: number) => B2.y + ((3.2 - y) / 6.4) * B2.h;
      axes(g, s, B2, Xc, Yc, [[-4, "−4"], [-2, "−2"], [0, "0"]], [[-2, "−2i"], [0, "0"], [2, "2i"]], "Re(z), z = −hλ");
      const zx = -h * lam;
      const zl = -h * lamLoc;
      const cx = (x: number) => Math.max(-4.15, x);
      g.strokeStyle = C.ink;
      g.lineWidth = 1.5 * s;
      g.beginPath();
      g.arc(Xc(cx(zl)), Yc(0), 5 * s, 0, 2 * Math.PI);
      g.stroke();
      g.fillStyle = C.yellow;
      g.beginPath();
      g.arc(Xc(cx(zx)), Yc(0), 5.5 * s, 0, 2 * Math.PI);
      g.fill();
      label(g, s, "RK4 region", B2.x + 6 * s, B2.y + 10 * s, C.sky, "left", 10);
      label(g, s, "Euler disc", B2.x + 6 * s, B2.y + 24 * s, C.verm, "left", 10);
      label(g, s, "● −hλ at p*   ○ at the start", B2.x + 6 * s, B2.y + B2.h - 10 * s, C.ink2, "left", 9.5);
      if (zx < -4.15) label(g, s, "← off the map", Xc(-4.1) + 8 * s, Yc(0) - 12 * s, C.yellow, "left", 10);
    },
    dispose() {},
  };
}

export function StabilityRace() {
  const [alpha, setAlpha] = useLinkedAlpha("stab", 2, 0.5, 3);
  const [lgH, setLgH] = useState(-0.523);
  const h = Math.pow(10, lgH);
  const params = useMemo(() => ({ alpha, h }), [alpha, h]);
  const handle = useEngine(createEngine, params);
  const lam = lambdaK2(4, 1, alpha);
  const hl = h * lam;

  return (
    <Instrument
      id="stab"
      chips={["Exp 1 Fig 3", "Exp 2 Fig 3c"]}
      title="Euler vs RK4: the stability race"
      claim="Same equation, same step size. Past h·λ = 2 Euler overshoots into chaos; RK4 holds on until 2.785."
      stage={<Stage handle={handle} aspect="16 / 9" label="Left: Euler and RK4 trajectories against a reference. Right: the stability regions of both methods in the complex plane with the current step marked." />}
    >
      <Slider id="stab-a" label="Exponent α" min={0.5} max={3} step={0.01} value={alpha} onChange={setAlpha} format={(v) => v.toFixed(2)} />
      <Slider id="stab-h" label="Step size h" min={-2} max={0} step={0.005} value={lgH} onChange={setLgH} format={(v) => Math.pow(10, v).toFixed(3)} />
      <div className="row">
        <VerdictChip v={{ ok: hl < 2, text: hl < 2 ? "Euler stable (hλ < 2)" : "Euler unstable (hλ ≥ 2)" }} />
        <VerdictChip v={{ ok: hl < RK4_LIMIT, text: hl < RK4_LIMIT ? "RK4 stable (hλ < 2.785)" : "RK4 unstable (hλ ≥ 2.785)" }} />
      </div>
      <Readouts
        items={[
          ["rate λ at p*", lam.toFixed(3)],
          ["h·λ", `${hl.toFixed(3)}  (local at start ${(h * localRate(alpha)).toFixed(2)})`],
        ]}
      />
      <Notes
        tryThis={[
          "At α = 2 (λ = 8), slide h from 0.2 to 0.3: Euler's zig-zag stops damping and starts bouncing between 0 and 1, while RK4 still converges.",
          "Keep h and lower α: λ shrinks, the marker moves right into both stable regions.",
        ]}
        math="Near p* each step multiplies the error by R(−hλ). Euler: R(z) = 1 + z, stable while |1 + z| < 1. RK4: R(z) = 1 + z + z²/2 + z³/6 + z⁴/24, stable in the larger rounded region. On the negative real axis that means hλ < 2 and hλ < 2.785."
        analogy="Walking downhill in fixed-length strides with your eyes closed: short strides reach the valley floor, strides longer than the valley overshoot up the far side, and each overshoot is bigger than the last."
      />
    </Instrument>
  );
}
