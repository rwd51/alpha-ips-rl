/** The effective inverse-probability weight flattens at omega(1)/omega(G) (Exp 2-3, Exp 5 Fig 3). */
import { useMemo, useState } from "react";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { useLinkedAlpha } from "../engine/useLinkedAlpha";
import { Readouts, Segmented, Slider, VerdictChip } from "../components/controls";
import { Instrument, Notes, Stage } from "../components/Instrument";
import { effWeight, makeRule, type RuleKind } from "../lib/finiteGroup";
import { mulberry } from "../lib/sampling";
import { C, rgba } from "../lib/color";
import { axes, fit2D, label, polyline } from "../lib/canvas2d";
import { fsig } from "../lib/format";

interface Params {
  G: number;
  alpha: number;
  eps: number;
  rho: number;
  rule: RuleKind;
}
const PS: number[] = [];
for (let k = 0; k <= 170; k++) PS.push(Math.pow(10, -4 + (4 * k) / 170));

export function dynamicRange(rule: RuleKind, G: number, a: number, eps: number): number {
  const om = makeRule(rule, G, a, eps);
  return om(1) / om(G);
}

function createEngine(): Engine<Params> {
  let host!: EngineHost;
  let g!: CanvasRenderingContext2D;
  let W = 10;
  let H = 10;
  let s = 1;
  let P: Params = { G: 16, alpha: 1, eps: 1e-3, rho: 4, rule: "clip" };
  let curve: number[] = [];
  let clipCurve: number[] = [];
  let DR = 1;
  let probeT = 2.2;
  let hits: boolean[] = [];
  let hitT = 0;
  const R = mulberry(9);

  return {
    init(h) {
      host = h;
    },
    setParams(p) {
      P = p;
      const om = makeRule(p.rule, p.G, p.alpha, p.eps);
      const omc = makeRule("clip", p.G, p.alpha, p.eps);
      const n1 = om(p.G);
      const nc = omc(p.G);
      curve = PS.map((q) => effWeight(q, p.G, om) / n1);
      clipCurve = PS.map((q) => effWeight(q, p.G, omc) / nc);
      DR = om(1) / om(p.G);
    },
    resize() {
      ({ g, w: W, h: H, s } = fit2D(host.canvases[0]));
    },
    step(dt) {
      probeT += dt;
      hitT += dt;
    },
    draw() {
      const { G, alpha: a, rho, rule } = P;
      g.fillStyle = C.stage;
      g.fillRect(0, 0, W, H);
      const B = { x: 58 * s, y: 66 * s, w: W - 76 * s, h: H - 110 * s };
      const ly0 = Math.log10(0.7);
      const ly1 = Math.log10(3e4);
      const X = (p: number) => B.x + ((Math.log10(p) + 4) / 4) * B.w;
      const Y = (v: number) => B.y + B.h - ((Math.log10(Math.max(v, 1e-9)) - ly0) / (ly1 - ly0)) * B.h;
      axes(
        g,
        s,
        B,
        X,
        Y,
        [[1e-4, "10⁻⁴"], [1e-3, "10⁻³"], [1e-2, "0.01"], [0.1, "0.1"], [1, "1"]],
        [[1, "1"], [10, "10"], [100, "100"], [1e3, "10³"], [1e4, "10⁴"]],
        "outcome probability p (log)",
        "boost w(p) / w(1)",
      );
      g.save();
      g.beginPath();
      g.rect(B.x, B.y, B.w, B.h);
      g.clip();
      g.strokeStyle = C.ink2;
      g.lineWidth = 1.4 * s;
      g.setLineDash([5 * s, 4 * s]);
      polyline(g, PS.length - 1, (k) => [X(PS[k]), Y(Math.pow(PS[k], -a))]);
      g.setLineDash([]);
      g.strokeStyle = rgba(C.sky, 0.5);
      g.lineWidth = s;
      g.setLineDash([2 * s, 3 * s]);
      polyline(g, 1, (k) => [B.x + k * B.w, Y(DR)]);
      g.strokeStyle = rgba(C.orange, 0.8);
      g.setLineDash([7 * s, 4 * s]);
      polyline(g, 1, (k) => [B.x + k * B.w, Y(rho)]);
      g.setLineDash([]);
      if (rule !== "clip") {
        g.strokeStyle = rgba(C.muted, 0.8);
        g.lineWidth = 1.2 * s;
        polyline(g, PS.length - 1, (k) => [X(PS[k]), Y(clipCurve[k])]);
      }
      g.strokeStyle = C.sky;
      g.lineWidth = 2.6 * s;
      polyline(g, PS.length - 1, (k) => [X(PS[k]), Y(curve[k])]);
      g.restore();
      label(g, s, `ceiling ω(1)/ω(G) = ${DR.toFixed(1)}`, B.x + 8 * s, Y(DR) - 9 * s, C.sky, "left", 10.5);
      label(g, s, `reward ratio ρ = ${rho.toFixed(1)}`, B.x + 8 * s, Y(rho) + (Math.abs(Y(rho) - Y(DR)) < 20 * s ? 12 : -9) * s, C.orange, "left", 10.5);
      label(g, s, "ideal p⁻ᵅ", X(2e-3), Y(Math.pow(2e-3, -a)) - 12 * s, C.ink2, "left", 10);
      // a probe sweeping log p, with one sampled group drawn at it
      const ph = (Math.sin(probeT * 0.45 - Math.PI / 2) + 1) / 2;
      const pp = Math.pow(10, -3.6 + ph * 3.55);
      const idx = Math.max(0, Math.min(170, Math.round(((Math.log10(pp) + 4) / 4) * 170)));
      const xp = X(pp);
      const wv = curve[idx];
      g.strokeStyle = rgba(C.ink, 0.25);
      g.lineWidth = s;
      polyline(g, 1, (k) => [xp, B.y + k * B.h]);
      g.fillStyle = C.sky;
      g.beginPath();
      g.arc(xp, Y(wv), 5 * s, 0, 2 * Math.PI);
      g.fill();
      g.strokeStyle = C.ink2;
      g.lineWidth = 1.5 * s;
      g.beginPath();
      g.arc(xp, Y(Math.pow(pp, -a)), 4.5 * s, 0, 2 * Math.PI);
      g.stroke();
      if (hitT > 0.35 || hits.length !== G) {
        hitT = 0;
        hits = Array.from({ length: G }, () => R() < pp);
      }
      const nh = hits.filter(Boolean).length;
      const cols = Math.min(G, 32);
      const rows = Math.ceil(G / cols);
      const cs = Math.min(10 * s, (B.w * 0.42) / cols);
      const gx = B.x + B.w - cols * cs - 8 * s;
      const gy = B.y + 10 * s;
      g.fillStyle = rgba(C.ground, 0.85);
      g.fillRect(gx - 8 * s, gy - 6 * s, cols * cs + 14 * s, rows * cs + 40 * s);
      for (let i = 0; i < G; i++) {
        g.fillStyle = hits[i] ? C.orange : C.rule;
        g.fillRect(gx + (i % cols) * cs, gy + Math.floor(i / cols) * cs, cs - 2 * s, cs - 2 * s);
      }
      label(g, s, `one group: ${nh} hit${nh === 1 ? "" : "s"} → p̂ = ${nh}/${G}`, gx, gy + rows * cs + 12 * s, C.ink2, "left", 10);
      label(g, s, `expected hits Gp = ${fsig(G * pp, 2)}`, gx, gy + rows * cs + 26 * s, C.muted, "left", 10);
      host.emit({ hud: `p = ${fsig(pp, 3)}\nideal boost ${fsig(Math.pow(pp, -a), 3)}\nactual boost ${fsig(wv, 3)}` });
    },
    dispose() {},
  };
}

const RULES: { value: RuleKind; label: string }[] = [
  { value: "clip", label: "Clip (paper)" },
  { value: "lap1", label: "Laplace λ=1" },
  { value: "lap05", label: "Jeffreys λ=½" },
  { value: "offset", label: "α-offset" },
];

export function WeightCeiling() {
  const [lgG, setLgG] = useState(4);
  const [alpha, setAlpha] = useLinkedAlpha("ceiling", 1, 0.1, 3);
  const [lgE, setLgE] = useState(-3);
  const [rho, setRho] = useState(4);
  const [rule, setRule] = useState<RuleKind>("clip");
  const G = Math.pow(2, lgG);
  const eps = Math.pow(10, lgE);
  const params = useMemo(() => ({ G, alpha, eps, rho, rule }), [G, alpha, eps, rho, rule]);
  const handle = useEngine(createEngine, params);
  const DR = dynamicRange(rule, G, alpha, eps);
  const hud = typeof handle.readouts.hud === "string" ? handle.readouts.hud : "";

  return (
    <Instrument
      id="ceiling"
      chips={["Exp 2", "Exp 3 Fig 3a", "Exp 5 Fig 3"]}
      title="The weight ceiling"
      claim="A group of G can never report a probability below 1/G, so the inverse-probability boost flattens at a ceiling."
      stage={<Stage handle={handle} aspect="16 / 11" hud={hud} label="Log-log chart of the effective weight against outcome probability, flat at a ceiling for rare outcomes, with a probe showing one sampled group." />}
    >
      <Slider id="ceil-g" label="Group size G" min={1} max={8} step={1} value={lgG} onChange={setLgG} format={(v) => String(Math.pow(2, v))} />
      <Slider id="ceil-a" label="Exponent α" min={0.1} max={3} step={0.01} value={alpha} onChange={setAlpha} format={(v) => v.toFixed(2)} />
      <Slider id="ceil-e" label="Clip ε" min={-4} max={-0.3} step={0.01} value={lgE} onChange={setLgE} format={(v) => fsig(Math.pow(10, v), 2)} />
      <Slider id="ceil-rho" label="Reward ratio ρ" min={1} max={10} step={0.05} value={rho} onChange={setRho} format={(v) => v.toFixed(2)} />
      <Segmented name="ceil-rule" legend="Weight rule (Exp 5)" options={RULES} value={rule} onChange={setRule} />
      <Readouts
        items={[
          ["dynamic range ω(1)/ω(G)", DR.toFixed(2)],
          ["weak outcome, ratio ρ", <VerdictChip v={{ ok: DR > rho, text: DR > rho ? `survives: ${DR.toFixed(1)} > ${rho.toFixed(1)}` : `dies: ${DR.toFixed(1)} ≤ ${rho.toFixed(1)}` }} />],
        ]}
      />
      <Notes
        tryThis={[
          "Watch the probe: for a rare outcome, the group shows zero or one hit, so the boost is stuck at the ceiling while the ideal keeps climbing.",
          <>Raise ε above 1/G: now the clip, not the group, sets the ceiling at (1/ε)<sup>α</sup>.</>,
          <>Switch to Laplace smoothing: the ceiling drops to ((G+1)/2)<sup>α</sup>, roughly halving the reward ratio that can survive.</>,
        ]}
        math={
          <>
            The dynamics only see w<sub>G</sub>(p) = E[ω(1+B)], B ~ Binomial(G−1, p). A weaker outcome survives exactly when ω(1)/ω(G) &gt; ρ; for the
            clip that is min(G, 1/ε)<sup>α</sup> &gt; ρ (Exp 5, Eq. 5.8).
          </>
        }
        analogy="An instrument's resolution limit: a detector that counts in steps of 1/G cannot tell “rare” from “very rare”, so it cannot correct for how rare."
      />
    </Instrument>
  );
}
