/** EMA over steps turns settling into a damped oscillator x'' + k x' + k lambda x = 0 (Exp 5 Fig 5c). */
import { useMemo, useState } from "react";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { useLinkedAlpha } from "../engine/useLinkedAlpha";
import { Readouts, Slider, VerdictChip } from "../components/controls";
import { Instrument, Notes, Stage } from "../components/Instrument";
import { lambdaK2 } from "../lib/model";
import { C, rgba } from "../lib/color";
import { axes, fit2D, label } from "../lib/canvas2d";
import { fsig } from "../lib/format";

interface Params {
  alpha: number;
  kappa: number;
}
const X0 = 0.25;

/** Roots of s^2 + k s + k lambda = 0 and the slow decay rate. */
export function emaRoots(kap: number, lam: number) {
  const d = kap * kap - 4 * kap * lam;
  if (d < 0) return { re: [-kap / 2, -kap / 2], im: [Math.sqrt(-d) / 2, -Math.sqrt(-d) / 2], rate: kap / 2, d };
  const q = Math.sqrt(d);
  return { re: [(-kap + q) / 2, (-kap - q) / 2], im: [0, 0], rate: (kap - q) / 2, d };
}

function createEngine(): Engine<Params> {
  let host!: EngineHost;
  let g!: CanvasRenderingContext2D;
  let W = 10;
  let H = 10;
  let s = 1;
  let kap = 1;
  let lam = 1.6;
  let roots = emaRoots(1, 1.6);
  let Tw = 10;
  let tc = 1.2;

  const x = (t: number) => {
    const d = roots.d;
    if (Math.abs(d) < 1e-9) return (X0 + (kap / 2) * X0 * t) * Math.exp((-kap * t) / 2);
    if (d < 0) {
      const w = Math.sqrt(-d) / 2;
      return Math.exp((-kap * t) / 2) * (X0 * Math.cos(w * t) + ((kap * X0) / 2 / w) * Math.sin(w * t));
    }
    const [s1, s2] = roots.re;
    return ((-s2 * X0) / (s1 - s2)) * Math.exp(s1 * t) + ((s1 * X0) / (s1 - s2)) * Math.exp(s2 * t);
  };

  return {
    init(h) {
      host = h;
    },
    setParams(p) {
      kap = p.kappa;
      lam = lambdaK2(4, 1, p.alpha);
      roots = emaRoots(kap, lam);
      Tw = Math.min(60, Math.max(2, 7 / roots.rate));
    },
    resize() {
      ({ g, w: W, h: H, s } = fit2D(host.canvases[0]));
    },
    step(dt) {
      tc += dt / 4.5;
      if (tc > 1.25) tc = 0;
    },
    draw() {
      g.fillStyle = C.stage;
      g.fillRect(0, 0, W, H);
      const t = Math.min(1, tc) * Tw;
      const xv = x(t);
      // spring, damper and mass
      const sx = 20 * s;
      const sw = W * 0.2;
      const cy = H * 0.42;
      g.fillStyle = C.rule;
      g.fillRect(sx, cy - 50 * s, 6 * s, 100 * s);
      const rest = sx + sw * 0.62;
      const mx = rest + (xv / X0) * sw * 0.3;
      const bw = 34 * s;
      g.strokeStyle = C.ink2;
      g.lineWidth = 1.6 * s;
      g.beginPath();
      g.moveTo(sx + 6 * s, cy - 18 * s);
      const coils = 9;
      for (let k = 1; k <= coils * 2; k++) {
        const xx = sx + 6 * s + ((mx - bw / 2 - sx - 6 * s) * k) / (coils * 2);
        g.lineTo(xx, cy - 18 * s + (k === coils * 2 ? 0 : (k % 2 ? -8 : 8) * s));
      }
      g.stroke();
      const dmx = sx + (mx - sx) * 0.5;
      g.strokeStyle = C.orange;
      g.beginPath();
      g.moveTo(sx + 6 * s, cy + 18 * s);
      g.lineTo(dmx, cy + 18 * s);
      g.stroke();
      g.strokeRect(dmx - 12 * s, cy + 10 * s, 24 * s, 16 * s);
      g.beginPath();
      g.moveTo(dmx + 4 * s, cy + 18 * s);
      g.lineTo(mx - bw / 2, cy + 18 * s);
      g.stroke();
      g.fillStyle = C.sky;
      g.fillRect(mx - bw / 2, cy - bw / 2 - 6 * s, bw, bw + 12 * s);
      g.strokeStyle = rgba(C.ink, 0.3);
      g.setLineDash([3 * s, 3 * s]);
      g.beginPath();
      g.moveTo(rest, cy - 60 * s);
      g.lineTo(rest, cy + 60 * s);
      g.stroke();
      g.setLineDash([]);
      label(g, s, "spring = reward landscape (λ)", sx, cy - 72 * s, C.ink2, "left", 10);
      label(g, s, "damper = EMA (κ)", sx, cy + 44 * s, C.orange, "left", 10);
      label(g, s, "p* (rest)", rest, cy + 72 * s, C.muted, "center", 10);
      // displacement over time
      const split = W * 0.28;
      const split2 = W * 0.68;
      const B = { x: split + 34 * s, y: 22 * s, w: split2 - split - 50 * s, h: H - 62 * s };
      const X = (tt: number) => B.x + (tt / Tw) * B.w;
      const Y = (v: number) => B.y + B.h / 2 - (v / (X0 * 1.15)) * (B.h / 2);
      axes(g, s, B, X, Y, [[0, "0"], [Tw / 2, fsig(Tw / 2, 2)], [Tw, fsig(Tw, 2)]], [[-X0, "−0.25"], [0, "0"], [X0, "0.25"]], "time t", "p₁ − p₁*");
      g.save();
      g.beginPath();
      g.rect(B.x, B.y, B.w, B.h);
      g.clip();
      g.strokeStyle = rgba(C.ink2, 0.6);
      g.setLineDash([5 * s, 4 * s]);
      g.lineWidth = 1.2 * s;
      g.beginPath();
      for (let k = 0; k <= 200; k++) {
        const tt = (Tw * k) / 200;
        if (k) g.lineTo(X(tt), Y(X0 * Math.exp(-lam * tt)));
        else g.moveTo(X(tt), Y(X0 * Math.exp(-lam * tt)));
      }
      g.stroke();
      g.setLineDash([]);
      g.strokeStyle = C.sky;
      g.lineWidth = 2.2 * s;
      g.beginPath();
      const kmax = Math.round(400 * Math.min(1, tc));
      for (let k = 0; k <= kmax; k++) {
        const tt = (Tw * k) / 400;
        if (k) g.lineTo(X(tt), Y(x(tt)));
        else g.moveTo(X(tt), Y(x(tt)));
      }
      g.stroke();
      g.fillStyle = C.sky;
      g.beginPath();
      g.arc(X(t), Y(xv), 4 * s, 0, 2 * Math.PI);
      g.fill();
      g.restore();
      label(g, s, "with EMA", B.x + 8 * s, B.y + 10 * s, C.sky, "left", 10);
      label(g, s, "no EMA: e^(−λt)", B.x + 80 * s, B.y + 10 * s, C.ink2, "left", 10);
      // s-plane root locus
      const B3 = { x: split2 + 30 * s, y: 22 * s, w: W - split2 - 44 * s, h: H - 62 * s };
      const xr = [-3.2 * lam, 0.5 * lam];
      const yr = 1.4 * lam;
      const X3 = (v: number) => B3.x + ((v - xr[0]) / (xr[1] - xr[0])) * B3.w;
      const Y3 = (v: number) => B3.y + B3.h / 2 - (v / yr) * (B3.h / 2);
      axes(g, s, B3, X3, Y3, [[-2 * lam, "−2λ"], [-lam, "−λ"], [0, "0"]], [[0, "0"]], "Re s", "Im s");
      g.save();
      g.beginPath();
      g.rect(B3.x, B3.y, B3.w, B3.h);
      g.clip();
      g.strokeStyle = rgba(C.orange, 0.7);
      g.lineWidth = 1.4 * s;
      g.setLineDash([4 * s, 4 * s]);
      g.beginPath();
      g.ellipse(X3(-lam), Y3(0), (lam * B3.w) / (xr[1] - xr[0]), (lam * B3.h) / 2 / yr, 0, 0, 2 * Math.PI);
      g.stroke();
      g.setLineDash([]);
      g.beginPath();
      g.moveTo(X3(xr[0]), Y3(0));
      g.lineTo(X3(-lam), Y3(0));
      g.stroke();
      for (let k = 0; k < 2; k++) {
        g.fillStyle = C.yellow;
        g.beginPath();
        g.arc(X3(Math.max(xr[0] + 0.02 * lam, roots.re[k])), Y3(roots.im[k]), 5 * s, 0, 2 * Math.PI);
        g.fill();
      }
      g.restore();
      label(g, s, "roots move on a circle", B3.x + 6 * s, B3.y + 10 * s, C.orange, "left", 10);
      label(g, s, "of radius λ about −λ", B3.x + 6 * s, B3.y + 23 * s, C.orange, "left", 10);
    },
    dispose() {},
  };
}

export function EmaOscillator() {
  const [alpha, setAlpha] = useLinkedAlpha("ema", 1, 0.25, 3);
  const [lgK, setLgK] = useState(0);
  const kappa = Math.pow(10, lgK);
  const params = useMemo(() => ({ alpha, kappa }), [alpha, kappa]);
  const handle = useEngine(createEngine, params);
  const lam = lambdaK2(4, 1, alpha);
  const rt = emaRoots(kappa, lam);
  const crit = Math.abs(kappa / (4 * lam) - 1) < 0.04;

  return (
    <Instrument
      id="ema"
      chips={["Exp 5 Fig 5c"]}
      title="The EMA oscillator"
      claim="Averaging p̂ over steps turns first-order settling into a damped spring. The EMA is the damper and the reward landscape is the spring."
      stage={
        <Stage
          handle={handle}
          aspect="16 / 9"
          hud={`κ = ${fsig(kappa, 3)}   κ/4λ = ${fsig(kappa / (4 * lam), 3)}`}
          label="A mass on a spring with a damper, its displacement over time, and the two roots of the characteristic equation moving on a circle in the complex plane."
        />
      }
    >
      <Slider id="ema-a" label="Exponent α" min={0.25} max={3} step={0.01} value={alpha} onChange={setAlpha} format={(v) => v.toFixed(2)} />
      <Slider id="ema-k" label="EMA rate κ = β/h" min={-1.3} max={3} step={0.01} value={lgK} onChange={setLgK} format={(v) => fsig(Math.pow(10, v), 3)} />
      <Readouts
        items={[
          ["stiffness λ", lam.toFixed(3)],
          ["decay rate", `${fsig(rt.rate, 3)}  (no EMA: ${lam.toFixed(3)})`],
          ["regime", <VerdictChip v={{ ok: crit ? true : null, text: crit ? "critical: fastest, rate 2λ" : rt.d < 0 ? "underdamped: rings" : "overdamped: creeps" }} />],
        ]}
      />
      <Notes
        tryThis={[
          "Small κ: the policy rings around p* before settling (underdamped).",
          "Slide κ to 4λ: the two roots meet at −2λ and settling is fastest, exactly twice the no-EMA rate.",
          "Large κ: one root returns to −λ. The EMA forgets fast and the dynamics act first-order again.",
        ]}
        math="x″ + κx′ + κλx = 0, roots s = (−κ ± √(κ² − 4κλ))/2. While the roots are complex they trace a circle of radius λ centred at −λ."
        analogy="A car's suspension. Too little damping bounces, too much creeps, critical damping settles fastest, which is momentum's speed-up in closed form."
      />
    </Instrument>
  );
}
