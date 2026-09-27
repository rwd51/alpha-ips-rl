/** What the learning dynamics actually see from each weight rule: the size-biased distortion D(p) = w_G(p) p^alpha - 1 (Exp 5 Fig 4). */
import { useMemo, useState } from "react";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { useLinkedAlpha } from "../engine/useLinkedAlpha";
import { Readouts, Slider } from "../components/controls";
import { Instrument, Notes, Stage } from "../components/Instrument";
import { binomPmf, effWeight, makeRule } from "../lib/finiteGroup";
import { lsq } from "../lib/numerics";
import { C, rgba } from "../lib/color";
import { axes, fit2D, label, polyline, type Box } from "../lib/canvas2d";
import { fsig } from "../lib/format";

const EPS = 1e-3;
const GS = [4, 8, 16, 32, 64, 128, 256];
const PS = Array.from({ length: 181 }, (_, k) => Math.pow(10, -3 + (3 * k) / 180) * (k === 180 ? 0.999 : 1));

interface Params {
  alpha: number;
  G: number;
  p0: number;
}

type RuleId = "clip" | "rich" | "offset" | "lap1";
const RULES: { id: RuleId; name: string; color: string }[] = [
  { id: "clip", name: "clip (paper)", color: C.sky },
  { id: "rich", name: "guarded Richardson", color: C.orange },
  { id: "offset", name: "α-matched offset", color: C.green },
  { id: "lap1", name: "Laplace λ=1", color: C.pink },
];

/**
 * Guarded split-half Richardson weights W(a, b) for half-counts a, b in 0..M (src/estimators.py::RichardsonRule):
 * 2 f((a+b)/G) - [f(a/M) + f(b/M)]/2, falling back to f((a+b)/G) unless both halves saw the outcome and the value is positive.
 */
function richardsonTable(G: number, a: number): Float64Array[] {
  const M = G / 2;
  const f = (x: number) => Math.pow(Math.max(x, EPS), -a);
  return Array.from({ length: M + 1 }, (_, i) =>
    Float64Array.from({ length: M + 1 }, (_, j) => {
      const plain = f((i + j) / G);
      const out = 2 * plain - 0.5 * (f(i / M) + f(j / M));
      return Math.min(i, j) >= 1 && out > 0 ? out : plain;
    }),
  );
}

/** Size-biased weight of the split rule (Exp 5, Eq. 5.7): the designated sample's half has one guaranteed hit. */
function richEff(p: number, G: number, W: Float64Array[]): number {
  const M = G / 2;
  const A = binomPmf(M - 1, p);
  const B = binomPmf(M, p);
  let s = 0;
  for (let i = 0; i < M; i++) {
    let row = 0;
    for (let j = 0; j <= M; j++) row += B[j] * W[i + 1][j];
    s += A[i] * row;
  }
  return s;
}

/** D(p) = w_G(p) p^alpha - 1 for one rule. */
export function distortion(rule: RuleId, a: number, G: number): (p: number) => number {
  if (rule === "rich") {
    const W = richardsonTable(G, a);
    return (p) => richEff(p, G, W) * Math.pow(p, a) - 1;
  }
  const om = makeRule(rule, G, a, EPS);
  const tab = Float64Array.from({ length: G + 1 }, (_, n) => om(n));
  return (p) => effWeight(p, G, (n) => tab[n]) * Math.pow(p, a) - 1;
}

function createEngine(): Engine<Params> {
  let host!: EngineHost;
  let g!: CanvasRenderingContext2D;
  let W = 10;
  let H = 10;
  let s = 1;
  let P: Params = { alpha: 1, G: 16, p0: 0.1 };
  let curves: number[][] = [];
  let vsG: number[][] = [];
  let probe = 0;

  return {
    init(h) {
      host = h;
    },
    setParams(p) {
      const needG = p.alpha !== P.alpha || p.p0 !== P.p0 || vsG.length === 0;
      P = p;
      curves = RULES.map((r) => {
        const D = distortion(r.id, p.alpha, p.G);
        return PS.map(D);
      });
      if (needG) vsG = RULES.map((r) => GS.map((G) => distortion(r.id, p.alpha, G)(p.p0)));
    },
    resize() {
      ({ g, w: W, h: H, s } = fit2D(host.canvases[0]));
    },
    step(dt) {
      probe += dt * 0.1;
    },
    draw() {
      g.fillStyle = C.stage;
      g.fillRect(0, 0, W, H);
      const { alpha: a, G, p0 } = P;
      const L: Box = { x: 62 * s, y: 34 * s, w: W * 0.6 - 62 * s, h: H - 108 * s };
      const X = (p: number) => L.x + ((Math.log10(p) + 3) / 3) * L.w;
      const ylo = -10;
      const yhi = 1;
      const Y = (v: number) => L.y + ((yhi - Math.log10(Math.max(Math.abs(v), 1e-10))) / (yhi - ylo)) * L.h;
      axes(
        g,
        s,
        L,
        X,
        Y,
        [[1e-3, "10⁻³"], [1e-2, "0.01"], [0.1, "0.1"], [1, "1"]],
        [[1, "1"], [1e-2, "10⁻²"], [1e-4, "10⁻⁴"], [1e-6, "10⁻⁶"], [1e-8, "10⁻⁸"], [1e-10, "10⁻¹⁰"]],
        "outcome probability p (log)",
        "|D(p)| = |w_G(p)·pᵅ − 1|",
      );
      g.save();
      g.beginPath();
      g.rect(L.x, L.y, L.w, L.h);
      g.clip();
      if (Math.abs(a - 1) < 1e-9) {
        g.strokeStyle = rgba(C.ink, 0.55);
        g.lineWidth = 5 * s;
        polyline(g, PS.length - 1, (k) => [X(PS[k]), Y(Math.pow(1 - PS[k], G))]);
      }
      // clip last, so it stays visible over the offset rule, which is the same rule at alpha = 1
      [1, 2, 3, 0].forEach((i) => {
        const r = RULES[i];
        g.strokeStyle = r.color;
        g.lineWidth = (r.id === "clip" ? 2.4 : 1.7) * s;
        polyline(g, PS.length - 1, (k) => [X(PS[k]), Y(curves[i][k])]);
      });
      g.strokeStyle = rgba(C.ink, 0.25);
      g.lineWidth = s;
      g.setLineDash([4 * s, 4 * s]);
      polyline(g, 1, (k) => [X(p0), L.y + k * L.h]);
      g.setLineDash([]);
      g.restore();
      label(g, s, `each rule at G = ${G}, α = ${a.toFixed(2)}${Math.abs(a - 1) < 1e-9 ? " · wide grey: (1 − p)^G" : ""}`, L.x, 16 * s, C.muted, "left", 10);
      // right: |D| at p0 against G
      const R: Box = { x: L.x + L.w + 70 * s, y: L.y, w: W - (L.x + L.w + 70 * s) - 18 * s, h: L.h };
      const XR = (Gv: number) => R.x + ((Math.log2(Gv) - 2) / 6) * R.w;
      axes(g, s, R, XR, Y, GS.map((v) => [v, String(v)] as [number, string]), [], "group size G (log)");
      g.save();
      g.beginPath();
      g.rect(R.x, R.y, R.w, R.h);
      g.clip();
      RULES.forEach((r, i) => {
        g.strokeStyle = r.color;
        g.lineWidth = 1.7 * s;
        polyline(g, GS.length - 1, (k) => [XR(GS[k]), Y(vsG[i][k])]);
        g.fillStyle = r.color;
        GS.forEach((Gv, k) => {
          g.beginPath();
          g.arc(XR(Gv), Y(vsG[i][k]), (Gv === G ? 4.4 : 2.6) * s, 0, 2 * Math.PI);
          g.fill();
        });
      });
      g.restore();
      label(g, s, `|D| at p = ${p0}`, R.x, 16 * s, C.muted, "left", 10);
      // legend under the left chart, two rows
      RULES.forEach((r, i) => {
        const lx = L.x + (i % 2) * (L.w / 2);
        const ly = H - 36 * s + Math.floor(i / 2) * 15 * s;
        g.fillStyle = r.color;
        g.fillRect(lx, ly - 1.5 * s, 14 * s, 3 * s);
        label(g, s, r.name + (r.id === "offset" && Math.abs(a - 1) < 1e-9 ? " (= clip at α = 1)" : ""), lx + 19 * s, ly, C.ink2, "left", 9.5);
      });
      // a probe sweeping p, marking where each rule sits
      const ph = (Math.sin(probe * 2 * Math.PI - Math.PI / 2) + 1) / 2;
      const k = Math.round(ph * (PS.length - 1));
      g.strokeStyle = rgba(C.ink, 0.2);
      g.lineWidth = s;
      polyline(g, 1, (i) => [X(PS[k]), L.y + i * L.h]);
      RULES.forEach((r, i) => {
        g.fillStyle = r.color;
        g.beginPath();
        g.arc(X(PS[k]), Y(curves[i][k]), 3.6 * s, 0, 2 * Math.PI);
        g.fill();
      });
      host.emit({});
    },
    dispose() {},
  };
}

/** Order in 1/G of |D| at p0: minus the least-squares slope of log|D| on log G over G = 64..256. */
function orderInG(alpha: number, p0: number, rule: RuleId): number {
  const gs = [64, 128, 256];
  const d = gs.map((G) => Math.abs(distortion(rule, alpha, G)(p0)));
  if (d.some((v) => !(v > 1e-14))) return Infinity;
  return -lsq(gs.map(Math.log), d.map(Math.log)).slope;
}

export function DynamicsDistortion() {
  const [alpha, setAlpha] = useLinkedAlpha("distort", 1, 0.25, 3);
  const [lgG, setLgG] = useState(4);
  const [p0, setP0] = useState(0.1);
  const G = Math.pow(2, lgG);
  const params = useMemo(() => ({ alpha, G, p0 }), [alpha, G, p0]);
  const handle = useEngine(createEngine, params);
  const orders = useMemo(() => RULES.map((r) => orderInG(alpha, p0, r.id)), [alpha, p0]);

  return (
    <Instrument
      id="distort"
      chips={["Exp 5 Fig 4", "Exp 5 Fig 3c"]}
      ghostChips={["size-biased weight"]}
      title="What the dynamics actually see"
      claim="Training does not feel the estimator's own bias. It feels a size-biased version, and at α = 1 the paper's clip is almost exact there, which is why the better estimator trains worse."
      stage={<Stage handle={handle} aspect="16 / 10" label="Log-log chart of each weight rule's dynamics distortion against outcome probability, and against group size at a fixed probability." />}
    >
      <Slider id="dist-a" label="Exponent α" min={0.25} max={3} step={0.01} value={alpha} onChange={setAlpha} format={(v) => v.toFixed(2)} />
      <Slider id="dist-g" label="Group size G" min={2} max={8} step={1} value={lgG} onChange={setLgG} format={(v) => String(Math.pow(2, v))} />
      <Slider id="dist-p" label="Probe p (right chart)" min={0.02} max={0.6} step={0.01} value={p0} onChange={setP0} format={(v) => v.toFixed(2)} />
      <Readouts
        items={RULES.map((r, i) => [
          <span style={{ color: r.color }}>{r.name}</span>,
          orders[i] === Infinity ? "below round-off by G = 64" : orders[i] > 3.5 ? `faster than any power (${orders[i].toFixed(1)})` : `order ${orders[i].toFixed(2)} in 1/G`,
        ])}
      />
      <Notes
        tryThis={[
          "At α = 1 the blue clip curve sits on the wide grey line (1 − p)^G (the α-offset rule is the clip itself here, c = 0): the paper's rule is off only by the chance that the group misses the outcome entirely, which falls exponentially in Gp. The orange Richardson curve is a straight line of slope −1, an O(1/(Gp)) error the clip does not have.",
          "Move α away from 1: the clip picks up its own O(1/G) term with coefficient α(α − 1)/2, and the green α-matched offset becomes the best rule (order 2).",
          "Right chart at G = 64: clip below Richardson. That is Exp 5's training result: ℓ₁ distance 0.0016 for the clip against 0.0294 for guarded Richardson.",
        ]}
        math={
          <>
            In the update, an outcome's weight is multiplied by its own count, so the dynamics average ω over groups that are guaranteed to contain it:
            w<sub>G</sub>(p) = E[ω(1 + B)], B ~ Binomial(G − 1, p). The extra hit shifts p̂ up by about (1 − p)/G, and that partly cancels the convexity bias
            of p̂<sup>−α</sup>. What is left: D = α(α − 1)(1 − p)/(2Gp) for the clip, −α(1 − p)/(Gp) for guarded Richardson, and O(1/G²) for the offset
            c = (1 − α)/2. At α = 1 the clip is exact up to the miss term: D = −(1 − p)<sup>G</sup>. Richardson fixes the estimator's bias (order 1 → 2.10)
            but brings in a first-order dynamics error.
          </>
        }
        analogy="Grading a test by surveying only the students who showed up: the sample is biased toward attendance, and a correction designed for a random sample can make the estimate worse, not better."
      />
    </Instrument>
  );
}
