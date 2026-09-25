/**
 * Hero: equation (*) on the three-outcome simplex. Particles follow the flow
 * toward p*, and a dashed curve traces p*(alpha) from a corner to the centre.
 */
import { useMemo, useState } from "react";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { useLinkedAlpha } from "../engine/useLinkedAlpha";
import { Segmented, Slider } from "../components/controls";
import { Stage } from "../components/Instrument";
import { entropyN, pStar, rhs } from "../lib/model";
import { mulberry } from "../lib/sampling";
import { C, ramp, rgba } from "../lib/color";
import { fit2D, label } from "../lib/canvas2d";

interface Params {
  alpha: number;
  rewards: string;
}

function createEngine(): Engine<Params> {
  let host!: EngineHost;
  let gT!: CanvasRenderingContext2D;
  let gO!: CanvasRenderingContext2D;
  let W = 10;
  let H = 10;
  let s = 1;
  let V: [number, number][] = [[0, 0], [0, 0], [0, 0]];
  let r = [4, 2, 1];
  let a = 1;
  let N = 1300;
  let P = new Float64Array(0);
  let age = new Float64Array(0);
  let life = new Float64Array(0);
  const R = mulberry(11);
  let vref = 1;
  let path: number[][] = [];
  let pst = [1 / 3, 1 / 3, 1 / 3];
  let pulse = 0;

  const spawn = (i: number) => {
    let t = 0;
    for (let k = 0; k < 3; k++) {
      const e = -Math.log(1 - R() * 0.999999);
      P[i * 3 + k] = e;
      t += e;
    }
    for (let k = 0; k < 3; k++) P[i * 3 + k] = Math.max(1e-4, P[i * 3 + k] / t);
    age[i] = 0;
    life[i] = 2.5 + R() * 4.5;
  };
  // velocity in probability space: p_i' = p_i (z_i' - sum_k p_k z_k')
  const vel = (p0: number, p1: number, p2: number, out: number[]) => {
    const zd = rhs([Math.log(p0), Math.log(p1), Math.log(p2)], r, a);
    const m = p0 * zd[0] + p1 * zd[1] + p2 * zd[2];
    out[0] = p0 * (zd[0] - m);
    out[1] = p1 * (zd[1] - m);
    out[2] = p2 * (zd[2] - m);
    return Math.hypot(out[0], out[1], out[2]);
  };
  const xy = (p0: number, p1: number, p2: number): [number, number] => [
    p0 * V[0][0] + p1 * V[1][0] + p2 * V[2][0],
    p0 * V[0][1] + p1 * V[1][1] + p2 * V[2][1],
  ];

  function recompute() {
    pst = pStar(r, a);
    // typical field speed on a fixed probe set, so particle speed is comparable across alpha
    const tmp = [0, 0, 0];
    const sp: number[] = [];
    for (let i = 1; i < 12; i++)
      for (let j = 1; i + j < 12; j++) {
        const p0 = i / 12;
        const p1 = j / 12;
        sp.push(vel(p0, p1, 1 - p0 - p1, tmp));
      }
    sp.sort((x, y) => x - y);
    vref = sp[Math.floor(sp.length * 0.6)];
    path = [];
    for (let k = 0; k <= 140; k++) path.push(pStar(r, Math.pow(10, -1.7 + (k * 3.4) / 140)));
  }

  function advance(dt: number) {
    const tmp = [0, 0, 0];
    gT.fillStyle = "rgba(12,20,31,0.075)";
    gT.fillRect(0, 0, W, H);
    gT.lineWidth = 1.25 * s;
    gT.lineCap = "round";
    const frozen = !(vref > 1e-12);
    for (let i = 0; i < N; i++) {
      const o = i * 3;
      let p0 = P[o];
      let p1 = P[o + 1];
      let p2 = P[o + 2];
      const sp = frozen ? 0 : vel(p0, p1, p2, tmp);
      const [x0, y0] = xy(p0, p1, p2);
      if (sp > 0) {
        const L = dt * 0.32 * Math.tanh((1.3 * sp) / vref);
        p0 = Math.max(p0 + (tmp[0] / sp) * L, 1e-5);
        p1 = Math.max(p1 + (tmp[1] / sp) * L, 1e-5);
        p2 = Math.max(p2 + (tmp[2] / sp) * L, 1e-5);
        const t = p0 + p1 + p2;
        P[o] = p0 / t;
        P[o + 1] = p1 / t;
        P[o + 2] = p2 / t;
      }
      const [x1, y1] = xy(P[o], P[o + 1], P[o + 2]);
      gT.strokeStyle = rgba(ramp(0.25 + 0.75 * Math.tanh(sp / (vref || 1))), 0.85);
      gT.beginPath();
      gT.moveTo(x0, y0);
      gT.lineTo(x1 + 0.01, y1);
      gT.stroke();
      age[i] += dt;
      if (age[i] > life[i]) spawn(i);
    }
  }

  return {
    init(h) {
      host = h;
      gT = host.canvases[0].getContext("2d")!;
      gO = host.canvases[1].getContext("2d")!;
      N = host.reducedMotion ? 500 : 1300;
      P = new Float64Array(N * 3);
      age = new Float64Array(N);
      life = new Float64Array(N);
      for (let i = 0; i < N; i++) spawn(i);
    },
    setParams(p) {
      a = p.alpha;
      r = p.rewards.split(",").map(Number);
      recompute();
    },
    resize() {
      const A = fit2D(host.canvases[0]);
      const B = fit2D(host.canvases[1]);
      gT = A.g;
      gO = B.g;
      W = A.w;
      H = A.h;
      s = A.s;
      const pad = 46 * s;
      const side = Math.min(W - 2 * pad, ((H - 2 * pad) * 2) / Math.sqrt(3));
      const h = (side * Math.sqrt(3)) / 2;
      const cx = W / 2;
      const top = (H - h) / 2 + 4 * s;
      V = [
        [cx, top],
        [cx - side / 2, top + h],
        [cx + side / 2, top + h],
      ];
      gT.fillStyle = C.stage;
      gT.fillRect(0, 0, W, H);
      for (let k = 0; k < 70; k++) advance(1 / 60); // paint trails so the first frame is complete
    },
    step(dt) {
      pulse += dt;
      advance(dt);
    },
    draw() {
      gO.clearRect(0, 0, W, H);
      gO.strokeStyle = C.rule;
      gO.lineWidth = 1.2 * s;
      gO.beginPath();
      gO.moveTo(...V[0]);
      gO.lineTo(...V[1]);
      gO.lineTo(...V[2]);
      gO.closePath();
      gO.stroke();
      const cxy = xy(1 / 3, 1 / 3, 1 / 3);
      gO.fillStyle = C.muted;
      gO.beginPath();
      gO.arc(cxy[0], cxy[1], 2.2 * s, 0, 2 * Math.PI);
      gO.fill();
      gO.setLineDash([4 * s, 5 * s]);
      gO.strokeStyle = rgba(C.orange, 0.8);
      gO.lineWidth = 1.4 * s;
      gO.beginPath();
      path.forEach((p, k) => {
        const q = xy(p[0], p[1], p[2]);
        if (k) gO.lineTo(q[0], q[1]);
        else gO.moveTo(q[0], q[1]);
      });
      gO.stroke();
      gO.setLineDash([]);
      const labs: [string, number, number, CanvasTextAlign][] = [
        ["O1", 0, -16, "center"],
        ["O2", -8, 20, "left"],
        ["O3", 8, 20, "right"],
      ];
      labs.forEach(([t, dx, dy, al], k) => label(gO, s, `${t} · r=${r[k]}`, V[k][0] + dx * s, V[k][1] + dy * s, C.ink2, al, 12));
      label(gO, s, "uniform", cxy[0] + 7 * s, cxy[1] + 9 * s, C.muted, "left", 10);
      const q = xy(pst[0], pst[1], pst[2]);
      const unstable = a === 0 && r.filter((x) => x === Math.max(...r)).length > 1;
      const glow = 10 + 3 * Math.sin(pulse * 3);
      const gr = gO.createRadialGradient(q[0], q[1], 0, q[0], q[1], glow * 2.2 * s);
      gr.addColorStop(0, rgba(C.orange, 0.55));
      gr.addColorStop(1, rgba(C.orange, 0));
      gO.fillStyle = gr;
      gO.beginPath();
      gO.arc(q[0], q[1], glow * 2.2 * s, 0, 2 * Math.PI);
      gO.fill();
      gO.lineWidth = 2 * s;
      gO.strokeStyle = C.orange;
      gO.fillStyle = unstable ? C.stage : "#fff3d6";
      gO.beginPath();
      gO.arc(q[0], q[1], 5 * s, 0, 2 * Math.PI);
      gO.fill();
      gO.stroke();
      label(gO, s, "p*", q[0] + 9 * s, q[1] - 10 * s, C.orange, "left", 12);
    },
    dispose() {},
  };
}

const PRESETS = [
  { value: "4,2,1", label: "4 · 2 · 1" },
  { value: "3,3,1", label: "3 · 3 · 1" },
  { value: "9,3,1", label: "9 · 3 · 1" },
  { value: "1,1,1", label: "1 · 1 · 1" },
];

export function Hero() {
  const [alpha, setAlpha] = useLinkedAlpha("hero", 1, 0, 3);
  const [rewards, setRewards] = useState("4,2,1");
  const params = useMemo(() => ({ alpha, rewards }), [alpha, rewards]);
  const handle = useEngine(createEngine, params, 2);
  const r = rewards.split(",").map(Number);
  const ps = pStar(r, alpha);
  let hud = `p* = (${ps.map((v) => v.toFixed(3)).join(", ")})\nH/ln 3 = ${entropyN(ps).toFixed(3)}   T = α = ${alpha.toFixed(2)}`;
  if (alpha === 0 && r[0] === r[1] && r[1] === r[2]) hud += "\nall rewards equal at α = 0: the flow is flat";
  else if (alpha === 0 && r.filter((x) => x === Math.max(...r)).length > 1) hud += "\nα = 0: the even split of the tie is unstable";

  return (
    <section className="hero" id="hero">
      <div>
        <div className="eyebrow">CSE 402 · alpha-ips-rl · Experiments 1–5, live</div>
        <h1>
          One exponent decides which answers <em>survive</em>.
        </h1>
        <p className="lede">
          Training on expected reward piles every sample onto one outcome. Dividing each reward by its probability to
          the power α turns that collapse into a dial: the policy settles where every outcome earns the same scaled
          reward. Every instrument on this page runs the project's own equations live in your browser.
        </p>
        <div className="formula" aria-label="The two equations behind every instrument">
          <div>
            ż<sub>i</sub> = r<sub>i</sub>·p<sub>i</sub>
            <sup>1−α</sup> − p<sub>i</sub>·Σ<sub>k</sub> r<sub>k</sub>·p<sub>k</sub>
            <sup>1−α</sup>
          </div>
          <span>how the logits move (derivation.md, Eq. *)</span>
          <div style={{ marginTop: 6 }}>
            p*<sub>i</sub> ∝ r<sub>i</sub>
            <sup>1/α</sup> = exp(ln r<sub>i</sub> / α)
          </div>
          <span>where they settle: a Boltzmann law with temperature α</span>
        </div>
        <div className="panel" style={{ marginTop: 18 }}>
          <Slider id="hero-a" label="Exponent α" min={0} max={3} step={0.01} value={alpha} onChange={setAlpha} format={(v) => v.toFixed(2)} />
          <Segmented name="hero-r" legend="Rewards of the three outcomes" options={PRESETS} value={rewards} onChange={setRewards} />
        </div>
      </div>
      <div>
        <Stage
          handle={handle}
          canvases={2}
          aspect="1.18 / 1"
          playOnTop
          hud={hud}
          label="Particles flowing across a triangle of three-outcome policies toward the stationary point, which moves along a dashed path from a corner to the centre as alpha grows."
        />
        <p className="hero-caption">
          Each point in the triangle is a policy over three outcomes. Particles follow equation (*). The bright dot is
          p*, and the dashed curve is the path p* takes as α goes from 0 (a corner: collapse) to ∞ (the centre: uniform).
        </p>
      </div>
    </section>
  );
}
