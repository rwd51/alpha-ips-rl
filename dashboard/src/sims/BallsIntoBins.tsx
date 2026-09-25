/** Sampled training with equal rewards: G balls per step, plus 32 parallel runs (Exp 1 Fig 2, Exp 2 Fig 4a). */
import { useMemo, useState } from "react";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { useLinkedAlpha } from "../engine/useLinkedAlpha";
import { Button, Segmented, Slider, VerdictChip } from "../components/controls";
import { Instrument, Notes, Stage } from "../components/Instrument";
import { entropyN, softmax } from "../lib/model";
import { cumulative, drawOutcome, mulberry, sampleCounts, sampledUpdate } from "../lib/sampling";
import { C, SERIES, rgba } from "../lib/color";
import { axes, fit2D, label } from "../lib/canvas2d";

type Mode = "watch" | "fast";
interface Params {
  alpha: number;
  G: number;
  K: number;
  h: number;
  mode: Mode;
  reset: number;
}
interface Ball {
  k: number;
  t0: number;
  slot: number;
}
const M = 32;
const HL = 420;

function createEngine(): Engine<Params> {
  let host!: EngineHost;
  let g!: CanvasRenderingContext2D;
  let W = 10;
  let H = 10;
  let s = 1;
  let a = 0;
  let G = 16;
  let K = 3;
  let h = 0.3;
  let mode: Mode = "watch";
  let resetSeen = -1;
  const R = mulberry(5);
  let z: number[] = [];
  let zs: number[][] = [];
  let ent = new Float32Array(0);
  let len = 0;
  let balls: Ball[] = [];
  let phase = 0;
  let stepsDone = 0;
  let last: number[] | null = null;

  function reset() {
    z = new Array(K).fill(0);
    zs = Array.from({ length: M }, () => new Array(K).fill(0));
    ent = new Float32Array(M * HL).fill(1);
    len = 0;
    stepsDone = 0;
    balls = [];
    phase = 0;
    last = null;
  }
  function norm(v: number[]) {
    const mx = Math.max(...v);
    for (let k = 0; k < v.length; k++) v[k] = Math.max(v[k] - mx, -60);
  }
  function stepMulti(n: number) {
    const ones = new Array(K).fill(1);
    for (let rep = 0; rep < n; rep++) {
      for (let m = 0; m < M; m++) {
        const u = sampledUpdate(zs[m], ones, a, G, 1e-3, R);
        for (let k = 0; k < K; k++) zs[m][k] += h * u.g[k];
        norm(zs[m]);
      }
      if (len < HL) len++;
      for (let m = 0; m < M; m++) {
        ent.copyWithin(m * HL, m * HL + 1, m * HL + HL);
        ent[m * HL + HL - 1] = entropyN(softmax(zs[m]));
      }
      stepsDone++;
    }
  }
  function mainUpdate(counts: number[]) {
    const u = sampledUpdate(z, new Array(K).fill(1), a, G, 1e-3, R, counts);
    for (let k = 0; k < K; k++) z[k] += h * u.g[k];
    norm(z);
  }
  function report() {
    const lam = a * Math.pow(K, a - 1);
    let col = 0;
    for (let m = 0; m < M; m++) if (Math.max(...softmax(zs[m])) > 0.95) col++;
    if (h * lam > 2) host.emit({ v: { ok: false, text: `step too large: h·λ = ${(h * lam).toFixed(2)} > 2` } });
    else host.emit({ v: { ok: col === 0 ? true : col === M ? false : null, text: `${col} of ${M} runs collapsed` } });
    host.emit({ hud: `step ${stepsDone}` });
  }

  return {
    init(hst) {
      host = hst;
    },
    setParams(p) {
      const kChanged = p.K !== K;
      a = p.alpha;
      G = p.G;
      K = p.K;
      h = p.h;
      if (p.mode !== mode) {
        balls = [];
        phase = 0;
      }
      mode = p.mode;
      if (resetSeen === -1 || p.reset !== resetSeen || kChanged) {
        const first = resetSeen === -1;
        resetSeen = p.reset;
        reset();
        if (first) stepMulti(60);
      }
      report();
    },
    resize() {
      ({ g, w: W, h: H, s } = fit2D(host.canvases[0]));
    },
    step(dt) {
      if (mode === "fast") {
        stepMulti(6);
        for (let k = 0; k < 6; k++) {
          const c = sampleCounts(softmax(z), G, R);
          mainUpdate(c);
          last = c;
        }
      } else {
        stepMulti(1);
        phase += dt;
        if (balls.length === 0 && phase > 0.15) {
          const cdf = cumulative(softmax(z));
          const cnt = new Array(K).fill(0);
          for (let i = 0; i < G; i++) {
            const k = drawOutcome(cdf, R());
            balls.push({ k, t0: phase + i * Math.min(0.045, 0.7 / G), slot: cnt[k]++ });
          }
        }
        if (balls.length && phase > balls[balls.length - 1].t0 + 0.55) {
          const c = new Array(K).fill(0);
          balls.forEach((b) => c[b.k]++);
          mainUpdate(c);
          last = c;
          balls = [];
          phase = 0;
        }
      }
      report();
    },
    draw() {
      g.fillStyle = C.stage;
      g.fillRect(0, 0, W, H);
      const p = softmax(z);
      const top = 30 * s;
      const binTop = H * 0.34;
      const binBot = H * 0.5;
      const l = 40 * s;
      const rgt = W - 18 * s;
      const cw = (rgt - l) / K;
      label(g, s, `one run: G = ${G} samples per step`, rgt, 14 * s, C.muted, "right", 10);
      const hx = (l + rgt) / 2;
      g.fillStyle = C.rule;
      g.beginPath();
      g.moveTo(hx - 26 * s, top);
      g.lineTo(hx + 26 * s, top);
      g.lineTo(hx + 6 * s, top + 16 * s);
      g.lineTo(hx - 6 * s, top + 16 * s);
      g.closePath();
      g.fill();
      for (let k = 0; k < K; k++) {
        const x0 = l + k * cw + cw * 0.12;
        const w = cw * 0.76;
        const ph = (binBot - top - 22 * s) * p[k];
        g.fillStyle = rgba(SERIES[k], 0.14);
        g.fillRect(x0, binBot - ph, w, ph);
        g.strokeStyle = rgba(SERIES[k], 0.7);
        g.lineWidth = 1.5 * s;
        g.beginPath();
        g.moveTo(x0, binBot - ph);
        g.lineTo(x0 + w, binBot - ph);
        g.stroke();
        g.strokeStyle = C.rule;
        g.lineWidth = s;
        g.beginPath();
        g.moveTo(x0, binTop);
        g.lineTo(x0, binBot);
        g.lineTo(x0 + w, binBot);
        g.lineTo(x0 + w, binTop);
        g.stroke();
        label(g, s, `O${k + 1}  p=${p[k].toFixed(2)}`, x0 + w / 2, binBot + 11 * s, C.ink2, "center", 10);
      }
      const br = Math.max(2.2 * s, Math.min(6 * s, (cw * 0.76) / 9));
      const perRow = Math.max(1, Math.floor((cw * 0.76) / (2.3 * br)));
      const slotXY = (k: number, i: number): [number, number] => [
        l + k * cw + cw * 0.12 + br * 1.3 + (i % perRow) * 2.3 * br,
        binBot - br * 1.3 - Math.floor(i / perRow) * 2.3 * br,
      ];
      const drawBall = (k: number, x: number, y: number) => {
        g.fillStyle = SERIES[k];
        g.beginPath();
        g.arc(x, y, br, 0, 2 * Math.PI);
        g.fill();
      };
      if (mode === "watch") {
        for (const b of balls) {
          const tt = (phase - b.t0) / 0.4;
          if (tt < 0) continue;
          const [tx, ty] = slotXY(b.k, b.slot);
          if (tt >= 1) drawBall(b.k, tx, ty);
          else drawBall(b.k, hx + (tx - hx) * tt, top + 16 * s + (ty - top - 16 * s) * tt * tt);
        }
      } else if (last) {
        for (let k = 0; k < K; k++) for (let i = 0; i < last[k]; i++) drawBall(k, ...slotXY(k, i));
      }
      const B = { x: 40 * s, y: H * 0.6, w: W - 58 * s, h: H * 0.4 - 36 * s };
      const X = (i: number) => B.x + (i / (HL - 1)) * B.w;
      const Y = (v: number) => B.y + B.h - v * B.h;
      axes(g, s, B, X, Y, [], [[0, "0"], [0.5, "0.5"], [1, "1"]], `${M} parallel runs · last ${HL} steps`);
      label(g, s, "entropy H/ln K of each run", B.x, B.y - 10 * s, C.muted, "left", 10);
      g.save();
      g.beginPath();
      g.rect(B.x, B.y, B.w, B.h);
      g.clip();
      for (let m = 0; m < M; m++) {
        const pm = softmax(zs[m]);
        const mx = Math.max(...pm);
        g.strokeStyle = rgba(mx > 0.95 ? SERIES[pm.indexOf(mx)] : C.sky, 0.55);
        g.lineWidth = 1.1 * s;
        g.beginPath();
        for (let i = HL - len; i < HL; i++) {
          if (i === HL - len) g.moveTo(X(i), Y(ent[m * HL + i]));
          else g.lineTo(X(i), Y(ent[m * HL + i]));
        }
        g.stroke();
      }
      g.restore();
    },
    dispose() {},
  };
}

export function BallsIntoBins() {
  const [alpha, setAlpha] = useLinkedAlpha("balls", 0, 0, 1.5);
  const [lgG, setLgG] = useState(4);
  const [K, setK] = useState(3);
  const [h, setH] = useState(0.3);
  const [mode, setMode] = useState<Mode>("watch");
  const [reset, setReset] = useState(0);
  const G = Math.pow(2, lgG);
  const params = useMemo(() => ({ alpha, G, K, h, mode, reset }), [alpha, G, K, h, mode, reset]);
  const handle = useEngine(createEngine, params);
  const hud = typeof handle.readouts.hud === "string" ? handle.readouts.hud : "";

  return (
    <Instrument
      id="balls"
      chips={["Exp 1 Fig 2", "Exp 2 Fig 4a"]}
      title="Balls into bins"
      claim="Every step draws G samples from the policy and updates it from the counts. With equal rewards, noise alone decides everything at α = 0."
      stage={<Stage handle={handle} aspect="16 / 12" hud={hud} label="Samples falling into bins, the policy bars, and entropy traces of many parallel training runs." />}
    >
      <Slider id="balls-a" label="Exponent α" min={0} max={1.5} step={0.01} value={alpha} onChange={setAlpha} format={(v) => v.toFixed(2)} />
      <Slider id="balls-g" label="Group size G" min={1} max={6} step={1} value={lgG} onChange={setLgG} format={(v) => String(Math.pow(2, v))} />
      <Slider id="balls-k" label="Outcomes K" min={2} max={5} step={1} value={K} onChange={setK} />
      <Slider id="balls-h" label="Learning rate h" min={0.05} max={0.6} step={0.01} value={h} onChange={setH} format={(v) => v.toFixed(2)} />
      <Segmented
        name="balls-m"
        legend="Speed"
        options={[
          { value: "watch", label: "Watch each group" },
          { value: "fast", label: "Fast-forward" },
        ]}
        value={mode}
        onChange={setMode}
      />
      <div className="row">
        <Button onClick={() => setReset((n) => n + 1)}>Restart all runs</Button>
        <VerdictChip v={handle.readouts.v} />
      </div>
      <Notes
        tryThis={[
          "α = 0, fast-forward: the 32 entropy traces fall to 0 one by one, each onto a different random winner.",
          "Raise α to 0.25: the same noise, but the traces now stay near 1. Any α > 0 adds a pull back toward p*.",
          <>Push α and h up together: past h·α·K<sup>α−1</sup> ≈ 2 the updates overshoot (the warning chip).</>,
        ]}
        math={
          <>
            Each outcome's weight is r·p̂·max(p̂, ε)<sup>−α</sup>. With all rewards equal the average update at α = 0 is exactly zero, so p does a
            random walk that sticks at the edges: an outcome near p = 0 gets pushes scaled by p and cannot come back.
          </>
        }
        analogy="Genetic drift in a population of fixed size (Wright–Fisher): neutral alleles still fix, at random. α > 0 acts like negative frequency-dependent selection, which keeps rare types alive."
      />
    </Instrument>
  );
}
