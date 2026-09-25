/** Ideal free distribution with interference: foragers settle at n_i ~ r_i^(1/alpha), the same law as p*. */
import { useMemo, useState } from "react";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { useLinkedAlpha } from "../engine/useLinkedAlpha";
import { Button, Readouts, Slider, text } from "../components/controls";
import { Instrument, Notes, Stage } from "../components/Instrument";
import { pStar } from "../lib/model";
import { mulberry } from "../lib/sampling";
import { C, rgba } from "../lib/color";
import { fit2D, label } from "../lib/canvas2d";

interface Params {
  alpha: number;
  scatter: number;
}
interface Agent {
  k: number;
  x: number;
  y: number;
  tx: number;
  ty: number;
  mv: boolean;
}
const RW = [8, 4, 2, 1];
const K = 4;
const N = 240;

function createEngine(): Engine<Params> {
  let host!: EngineHost;
  let g!: CanvasRenderingContext2D;
  let W = 10;
  let H = 10;
  let s = 1;
  let a = 1;
  let scatterSeen = -1;
  const R = mulberry(41);
  let ag: Agent[] = [];
  const cnt = new Array<number>(K).fill(0);
  let acc = 0;
  let clock = 0;
  let moves: number[] = [];
  let P: { x: number; y: number; rad: number }[] = [];

  const intake = (k: number, n: number) => RW[k] / Math.pow(Math.max(n, 1), a);
  const targetIn = (k: number): [number, number] => {
    const ang = R() * 2 * Math.PI;
    const rr = Math.sqrt(R()) * P[k].rad * 0.85;
    return [P[k].x + rr * Math.cos(ang), P[k].y + rr * Math.sin(ang)];
  };
  function scatter() {
    cnt.fill(0);
    ag = [];
    for (let i = 0; i < N; i++) {
      const k = Math.floor(R() * K);
      cnt[k]++;
      const [x, y] = P.length ? targetIn(k) : [0, 0];
      ag.push({ k, x, y, tx: x, ty: y, mv: false });
    }
  }
  function tick(now: number) {
    for (let j = 0; j < Math.ceil(N * 0.05); j++) {
      const A = ag[Math.floor(R() * N)];
      const cur = intake(A.k, cnt[A.k]);
      let best = A.k;
      let bv = cur;
      for (let k = 0; k < K; k++)
        if (k !== A.k) {
          const v = intake(k, cnt[k] + 1);
          if (v > bv * 1.02) {
            bv = v;
            best = k;
          }
        }
      if (best !== A.k) {
        cnt[A.k]--;
        cnt[best]++;
        A.k = best;
        [A.tx, A.ty] = targetIn(best);
        A.mv = true;
        moves.push(now);
      }
    }
  }

  return {
    init(h) {
      host = h;
    },
    setParams(p) {
      a = p.alpha;
      if (p.scatter !== scatterSeen) {
        const first = scatterSeen === -1;
        scatterSeen = p.scatter;
        if (!first && P.length) scatter();
      }
    },
    resize() {
      ({ g, w: W, h: H, s } = fit2D(host.canvases[0]));
      const firstLayout = P.length === 0;
      P = RW.map((rw, k) => ({ x: W * (0.14 + 0.24 * k), y: H * 0.38, rad: Math.sqrt(rw / 8) * Math.min(W * 0.1, H * 0.25) }));
      if (firstLayout) {
        scatter();
        for (let k = 0; k < 160; k++) tick(0);
        ag.forEach((A) => {
          A.x = A.tx;
          A.y = A.ty;
          A.mv = false;
        });
        moves = [];
      } else {
        ag.forEach((A) => {
          [A.tx, A.ty] = targetIn(A.k);
          A.x = A.tx;
          A.y = A.ty;
        });
      }
    },
    step(dt) {
      clock += dt;
      acc += dt;
      while (acc > 0.08) {
        acc -= 0.08;
        tick(clock);
      }
      for (const A of ag) {
        const dx = A.tx - A.x;
        const dy = A.ty - A.y;
        const d = Math.hypot(dx, dy);
        const v = Math.min(d, dt * 260 * s);
        if (d > 0.5) {
          A.x += (dx / d) * v;
          A.y += (dy / d) * v;
        } else A.mv = false;
        if (!A.mv && R() < 0.02) [A.tx, A.ty] = targetIn(A.k);
      }
      moves = moves.filter((t) => clock - t < 1);
    },
    draw() {
      g.fillStyle = C.stage;
      g.fillRect(0, 0, W, H);
      const ps = pStar(RW, a);
      P.forEach((pt, k) => {
        const gr = g.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, pt.rad);
        gr.addColorStop(0, rgba(C.green, 0.28));
        gr.addColorStop(1, rgba(C.green, 0.05));
        g.fillStyle = gr;
        g.beginPath();
        g.arc(pt.x, pt.y, pt.rad, 0, 2 * Math.PI);
        g.fill();
        g.strokeStyle = rgba(C.green, 0.5);
        g.lineWidth = s;
        g.stroke();
        label(g, s, `food ${RW[k]}`, pt.x, pt.y - pt.rad - 12 * s, C.ink2, "center", 11);
      });
      for (const A of ag) {
        g.fillStyle = A.mv ? C.orange : rgba(C.ink, 0.8);
        g.beginPath();
        g.arc(A.x, A.y, 2.3 * s, 0, 2 * Math.PI);
        g.fill();
      }
      const by = H * 0.74;
      const bh = H * 0.18;
      P.forEach((pt, k) => {
        const w = Math.min(60 * s, W * 0.12);
        const x0 = pt.x - w / 2;
        const share = cnt[k] / N;
        g.fillStyle = C.rule2;
        g.fillRect(x0, by, w, bh);
        g.fillStyle = rgba(C.sky, 0.85);
        g.fillRect(x0, by + bh - share * bh, w, share * bh);
        g.strokeStyle = C.yellow;
        g.lineWidth = 2 * s;
        g.beginPath();
        g.moveTo(x0 - 4 * s, by + bh - ps[k] * bh);
        g.lineTo(x0 + w + 4 * s, by + bh - ps[k] * bh);
        g.stroke();
        label(g, s, `${cnt[k]} (${(share * 100).toFixed(0)}%)`, pt.x, by + bh + 12 * s, C.ink, "center", 10.5);
      });
      label(g, s, "foragers per patch (bars) vs predicted share N·r^(1/α)/Σ (yellow)", 16 * s, H - 10 * s, C.muted, "left", 10);
      let l1 = 0;
      for (let k = 0; k < K; k++) l1 += Math.abs(cnt[k] / N - ps[k]);
      host.emit({ l1: `ℓ₁ = ${l1.toFixed(3)}`, moves: String(moves.length) });
    },
    dispose() {},
  };
}

export function Foragers() {
  const [alpha, setAlpha] = useLinkedAlpha("ifd", 1, 0, 3);
  const [scatter, setScatter] = useState(0);
  const params = useMemo(() => ({ alpha, scatter }), [alpha, scatter]);
  const handle = useEngine(createEngine, params);

  return (
    <Instrument
      id="ifd"
      chips={[]}
      ghostChips={["analogy · ecology"]}
      title="Foragers and patches"
      claim="Each forager moves to the patch where it would eat the most. Crowding cuts everyone's share, and α sets how sharply."
      stage={
        <Stage
          handle={handle}
          aspect="16 / 10"
          hud={`intake per forager = food / crowd^α,  α = ${alpha.toFixed(2)}`}
          label="Foragers moving between four food patches of different richness, with their counts compared against the predicted share."
        />
      }
    >
      <Slider id="ifd-a" label="Crowding α" min={0} max={3} step={0.01} value={alpha} onChange={setAlpha} format={(v) => v.toFixed(2)} />
      <div className="row">
        <Button onClick={() => setScatter((n) => n + 1)}>Scatter foragers</Button>
      </div>
      <Readouts
        items={[
          ["share vs predicted", text(handle.readouts.l1)],
          ["moves last second", text(handle.readouts.moves)],
        ]}
      />
      <div className="tbl">
        <table className="map">
          <thead>
            <tr>
              <th>Ecology</th>
              <th>This project</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>patch</td><td>outcome</td></tr>
            <tr><td>food supply of a patch</td><td>reward r<sub>i</sub></td></tr>
            <tr><td>share of foragers there</td><td>probability p<sub>i</sub></td></tr>
            <tr><td>intake per forager, r / n<sup>α</sup></td><td>scaled reward r / p<sup>α</sup></td></tr>
            <tr><td>interference exponent</td><td>α</td></tr>
          </tbody>
        </table>
      </div>
      <Notes analogy="This is the ideal free distribution with interference (Sutherland, 1983). IPS training reaches its equilibrium the way the foragers do: no outcome can gain by getting more probability.">
        <p>
          With no crowding penalty (α = 0) every forager piles into the richest patch: collapse. With α = 1 they match food supply one-for-one.
          Everyone stops moving when intake r<sub>i</sub>/n<sub>i</sub>
          <sup>α</sup> is equal everywhere, which gives n<sub>i</sub> ∝ r<sub>i</sub>
          <sup>1/α</sup>, the same law as p*.
        </p>
      </Notes>
    </Instrument>
  );
}
