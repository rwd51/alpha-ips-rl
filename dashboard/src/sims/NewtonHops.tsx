/** Newton-Raphson on the three residual forms of Exp 3 (src/stationarity.py), animated as tangent hops. */
import { useMemo, useState } from "react";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { useLinkedAlpha } from "../engine/useLinkedAlpha";
import { Readouts, Segmented, Slider, text } from "../components/controls";
import { Instrument, Notes, Stage } from "../components/Instrument";
import { C, ramp, rgba } from "../lib/color";
import { axes, fit2D, label } from "../lib/canvas2d";
import { fsig } from "../lib/format";

type Form = "drift" | "balance" | "log";
interface Params {
  alpha: number;
  u0: number;
  form: Form;
}
const r1 = 4;
const r2 = 1;
const UL = 16;
const sig = (u: number) => 1 / (1 + Math.exp(-u));
const logsig = (u: number) => (u >= 0 ? -Math.log1p(Math.exp(-u)) : u - Math.log1p(Math.exp(u)));

function residual(form: Form, a: number) {
  const F = (u: number) => {
    const p = sig(u);
    const q = 1 - p;
    if (form === "drift") return 2 * (r1 * Math.pow(p, 1 - a) * q - r2 * p * Math.pow(q, 1 - a));
    if (form === "balance") return r1 * Math.exp(-a * logsig(u)) - r2 * Math.exp(-a * logsig(-u));
    return Math.log(r1 / r2) - a * u;
  };
  const dF = (u: number) => {
    const p = sig(u);
    const q = 1 - p;
    if (form === "drift") return 2 * (r1 * Math.pow(p, 1 - a) * q * ((1 - a) * q - p) - r2 * p * Math.pow(q, 1 - a) * (q - (1 - a) * p));
    if (form === "balance") return -a * (r1 * Math.exp(-a * logsig(u)) * q + r2 * Math.exp(-a * logsig(-u)) * p);
    return -a;
  };
  return { F, dF };
}

function newton(F: (u: number) => number, dF: (u: number) => number, start: number, a: number, record: boolean) {
  let u = start;
  const out = record ? [u] : [];
  for (let n = 1; n <= 60; n++) {
    const f0 = F(u);
    const d = dF(u);
    if (!Number.isFinite(f0) || !Number.isFinite(d) || d === 0) return { ok: false, n, u, out };
    const step = f0 / d;
    u -= step;
    if (record) out.push(u);
    if (!Number.isFinite(u) || Math.abs(u) > 700) return { ok: false, n, u, out };
    if (Math.abs(step) < 1e-10) return { ok: Math.abs(u - Math.log(r1 / r2) / a) < 1e-6, n, u, out };
  }
  return { ok: false, n: 60, u, out };
}

function createEngine(): Engine<Params> {
  let host!: EngineHost;
  let g!: CanvasRenderingContext2D;
  let W = 10;
  let H = 10;
  let s = 1;
  let P: Params = { alpha: 0.5, u0: 7, form: "drift" };
  let its: number[] = [];
  let anim = 99;
  let basin: number[] = [];
  let fns = residual("drift", 0.5);

  return {
    init(h) {
      host = h;
    },
    setParams(p) {
      P = p;
      fns = residual(p.form, p.alpha);
      const res = newton(fns.F, fns.dF, p.u0, p.alpha, true);
      its = res.out;
      anim = anim === 99 ? 99 : 0;
      basin = [];
      let good = 0;
      for (let k = 0; k < 300; k++) {
        const b = newton(fns.F, fns.dF, -15 + (30 * k) / 299, p.alpha, false);
        basin.push(b.ok ? b.n : -1);
        if (b.ok) good++;
      }
      host.emit({
        res: res.ok ? `converged in ${res.n} steps to u = ${res.u.toFixed(5)}` : `diverged after ${res.n} steps (u → ${fsig(res.u, 3)})`,
        frac: `${Math.round(good / 3)}% of u₀ in [−15, 15]`,
      });
    },
    resize() {
      ({ g, w: W, h: H, s } = fit2D(host.canvases[0]));
    },
    step(dt) {
      anim += dt;
      if (anim > its.length * 0.75 + 2) anim = 0;
    },
    draw() {
      const { F, dF } = fns;
      const a = P.alpha;
      g.fillStyle = C.stage;
      g.fillRect(0, 0, W, H);
      const B = { x: 54 * s, y: 26 * s, w: W - 72 * s, h: H * 0.62 - 26 * s };
      const f0 = P.form === "drift" ? 0.05 : 1;
      const tr = (v: number) => Math.sign(v) * Math.log10(1 + Math.abs(v) / f0);
      let ymax = 0.5;
      for (let k = 0; k <= 200; k++) {
        const v = tr(F(-UL + (2 * UL * k) / 200));
        if (Number.isFinite(v)) ymax = Math.max(ymax, Math.abs(v));
      }
      ymax *= 1.1;
      const X = (u: number) => B.x + ((u + UL) / (2 * UL)) * B.w;
      const Y = (v: number) => B.y + B.h / 2 - (tr(v) / ymax) * (B.h / 2);
      axes(g, s, B, X, Y, [-15, -10, -5, 0, 5, 10, 15].map((v) => [v, String(v).replace("-", "−")]), [[0, "0"]], "", `${P.form} residual (symlog)`);
      g.save();
      g.beginPath();
      g.rect(B.x, B.y, B.w, B.h);
      g.clip();
      g.strokeStyle = rgba(C.ink, 0.35);
      g.lineWidth = s;
      g.beginPath();
      g.moveTo(B.x, Y(0));
      g.lineTo(B.x + B.w, Y(0));
      g.stroke();
      g.strokeStyle = C.sky;
      g.lineWidth = 2.2 * s;
      g.beginPath();
      for (let k = 0; k <= 400; k++) {
        const u = -UL + (2 * UL * k) / 400;
        if (k) g.lineTo(X(u), Y(F(u)));
        else g.moveTo(X(u), Y(F(u)));
      }
      g.stroke();
      const us = Math.log(r1 / r2) / a;
      g.fillStyle = C.green;
      g.beginPath();
      g.arc(X(us), Y(0), 5 * s, 0, 2 * Math.PI);
      g.fill();
      const shown = Math.min(its.length - 1, Math.floor(anim / 0.75));
      const partial = Math.min(1, (anim - shown * 0.75) / 0.5);
      for (let k = 0; k <= shown && k < its.length - 1; k++) {
        const uk = its[k];
        const fk = F(uk);
        const dk = dF(uk);
        const un = its[k + 1];
        const frac = k === shown ? partial : 1;
        const uEnd = uk + (un - uk) * frac;
        g.strokeStyle = C.orange;
        g.lineWidth = 1.6 * s;
        g.beginPath();
        for (let j = 0; j <= 40; j++) {
          const u = uk + ((uEnd - uk) * j) / 40;
          const v = fk + dk * (u - uk);
          if (j) g.lineTo(X(u), Y(v));
          else g.moveTo(X(u), Y(v));
        }
        g.stroke();
        g.fillStyle = C.orange;
        g.beginPath();
        g.arc(X(uk), Y(fk), 3.5 * s, 0, 2 * Math.PI);
        g.fill();
        if (frac >= 1 && Math.abs(un) < UL) {
          g.setLineDash([3 * s, 3 * s]);
          g.strokeStyle = rgba(C.orange, 0.6);
          g.lineWidth = s;
          g.beginPath();
          g.moveTo(X(un), Y(0));
          g.lineTo(X(un), Y(F(un)));
          g.stroke();
          g.setLineDash([]);
        }
      }
      g.restore();
      const lastU = its[Math.min(its.length - 1, shown + 1)];
      if (Math.abs(lastU) > UL && shown >= its.length - 2)
        label(
          g,
          s,
          lastU > 0 ? `flung off → u = ${fsig(lastU, 3)}` : `← flung off: u = ${fsig(lastU, 3)}`,
          lastU > 0 ? B.x + B.w - 8 * s : B.x + 8 * s,
          B.y + 14 * s,
          C.verm,
          lastU > 0 ? "right" : "left",
          11,
        );
      label(g, s, `root u* = ${us.toFixed(3)}`, X(us) + 8 * s, Y(0) + 14 * s, C.green, "left", 10.5);
      const Bb = { x: B.x, y: H * 0.74, w: B.w, h: H * 0.1 };
      const maxIt = Math.max(8, ...basin.filter((v) => v > 0));
      basin.forEach((v, k) => {
        g.fillStyle = v < 0 ? "#141c28" : rgba(ramp(1 - 0.85 * Math.min(1, (v - 1) / (maxIt - 1))));
        g.fillRect(Bb.x + (k / 300) * Bb.w, Bb.y, Bb.w / 300 + 0.6, Bb.h);
      });
      g.strokeStyle = C.rule;
      g.lineWidth = s;
      g.strokeRect(Bb.x, Bb.y, Bb.w, Bb.h);
      const xm = Bb.x + ((P.u0 + 15) / 30) * Bb.w;
      g.fillStyle = C.ink;
      g.beginPath();
      g.moveTo(xm, Bb.y - 2 * s);
      g.lineTo(xm - 5 * s, Bb.y - 10 * s);
      g.lineTo(xm + 5 * s, Bb.y - 10 * s);
      g.closePath();
      g.fill();
      label(g, s, "Newton basin over u₀ ∈ [−15, 15]: bright = few iterations, dark = diverged", Bb.x, Bb.y + Bb.h + 14 * s, C.muted, "left", 10);
      label(g, s, "−15", Bb.x, Bb.y + Bb.h + 28 * s, C.muted, "left", 10);
      label(g, s, "15", Bb.x + Bb.w, Bb.y + Bb.h + 28 * s, C.muted, "right", 10);
    },
    dispose() {},
  };
}

export function NewtonHops() {
  const [alpha, setAlpha] = useLinkedAlpha("newton", 0.5, 0.2, 4);
  const [u0, setU0] = useState(7);
  const [form, setForm] = useState<Form>("drift");
  const params = useMemo(() => ({ alpha, u0, form }), [alpha, u0, form]);
  const handle = useEngine(createEngine, params);

  return (
    <Instrument
      id="newton"
      chips={["Exp 3 Fig 1"]}
      title="Newton's tangent hops"
      claim="Solving ż = 0 directly. The same root, written three ways, gives three very different Newton journeys."
      stage={
        <Stage
          handle={handle}
          aspect="16 / 11"
          hud={`u* = ln 4 / α = ${(Math.log(4) / alpha).toFixed(4)}`}
          label="A residual curve with Newton tangent lines hopping toward the root, above a strip coloured by iterations needed from each start."
        />
      }
    >
      <Slider id="newt-a" label="Exponent α" min={0.2} max={4} step={0.01} value={alpha} onChange={setAlpha} format={(v) => v.toFixed(2)} />
      <Slider id="newt-u" label="Start u₀" min={-15} max={15} step={0.1} value={u0} onChange={setU0} format={(v) => v.toFixed(1)} />
      <Segmented
        name="newt-f"
        legend="Residual form"
        options={[
          { value: "drift", label: "Drift" },
          { value: "balance", label: "Balance" },
          { value: "log", label: "Log" },
        ]}
        value={form}
        onChange={setForm}
      />
      <Readouts
        items={[
          ["Newton result", text(handle.readouts.res)],
          ["starts that converge", text(handle.readouts.frac)],
          ["bisection on [−40, 40]", "47 iterations"],
        ]}
      />
      <Notes
        tryThis={[
          "Drift form, α = 0.5, start at 7: the curve is flat out there, the tangent's zero lands far away, and Newton is flung off.",
          "Same start, balance form: steady hops of about 1/α logits, then quadratic snap-in.",
          "Log form: one hop, from anywhere. It is a straight line.",
        ]}
        math={
          <>
            Two outcomes, rewards (4, 1), unknown u = z₁ − z₂, root u* = ln 4/α. Drift g(u) = u̇; balance b = g/(2pq) = r₁p<sup>−α</sup> − r₂q<sup>−α</sup>;
            log l = ln(r₁/r₂) − αu.
          </>
        }
        analogy="Sliding down tangent lines. Near the root it snaps in fast; on a nearly flat plateau the tangent points almost sideways and throws you off the map."
      />
    </Instrument>
  );
}
