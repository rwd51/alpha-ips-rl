/** Exact distribution of the weight estimate for one outcome, plus live sampling (Exp 5 Figs 1-2). */
import { useMemo, useState } from "react";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { useLinkedAlpha } from "../engine/useLinkedAlpha";
import { Readouts, Segmented, Slider } from "../components/controls";
import { Instrument, Notes, Stage } from "../components/Instrument";
import { binomPmf, makeRule, type RuleKind } from "../lib/finiteGroup";
import { mulberry } from "../lib/sampling";
import { C, rgba } from "../lib/color";
import { axes, fit2D, label, type Tick } from "../lib/canvas2d";
import { fsig } from "../lib/format";

type Rule = "clip" | "lap1" | "offset";
interface Params {
  p: number;
  G: number;
  alpha: number;
  eps: number;
  rule: Rule;
}
const EPS: number[] = [];
for (let k = 0; k <= 100; k++) EPS.push(Math.pow(10, -4 + (3.95 * k) / 100));

/** Exact relative moments of omega(n) p^alpha, n ~ Binomial(G, p) (Exp 5, Section 0). */
export function estimatorStats(p: number, G: number, a: number, eps: number, rule: RuleKind) {
  const pmf = binomPmf(G, p);
  const om = makeRule(rule, G, a, eps);
  const pa = Math.pow(p, a);
  const vals: number[] = [];
  for (let n = 0; n <= G; n++) vals.push(om(n) * pa);
  let mean = 0;
  for (let n = 0; n <= G; n++) mean += pmf[n] * vals[n];
  let v = 0;
  let m2 = 0;
  for (let n = 0; n <= G; n++) {
    v += pmf[n] * (vals[n] - mean) ** 2;
    m2 += pmf[n] * (vals[n] - 1) ** 2;
  }
  const share = m2 > 0 ? (pmf[0] * (vals[0] - 1) ** 2) / m2 : 0;
  return { pmf, vals, mean, sd: Math.sqrt(v), rmse: Math.sqrt(m2), share };
}
function clipMse(pmf: Float64Array, G: number, a: number, e: number, p: number): number {
  const oc = makeRule("clip", G, a, e);
  const pa = Math.pow(p, a);
  let t = 0;
  for (let n = 0; n <= G; n++) t += pmf[n] * (oc(n) * pa - 1) ** 2;
  return t;
}

function createEngine(): Engine<Params> {
  let host!: EngineHost;
  let g!: CanvasRenderingContext2D;
  let W = 10;
  let H = 10;
  let s = 1;
  let P: Params = { p: 0.1, G: 16, alpha: 1, eps: 1e-3, rule: "clip" };
  let st = estimatorStats(0.1, 16, 1, 1e-3, "clip");
  let mse: number[] = [];
  let emp = new Float64Array(17);
  let draws = 0;
  const R = mulberry(31);

  function sample() {
    for (let k = 0; k < 250; k++) {
      let n = 0;
      for (let i = 0; i < P.G; i++) if (R() < P.p) n++;
      emp[n]++;
      draws++;
    }
  }
  const star = (x: number, y: number, r: number, col: string) => {
    g.fillStyle = col;
    g.beginPath();
    for (let k = 0; k < 10; k++) {
      const ang = -Math.PI / 2 + (k * Math.PI) / 5;
      const rr = k % 2 ? r * 0.45 : r;
      g.lineTo(x + rr * Math.cos(ang), y + rr * Math.sin(ang));
    }
    g.closePath();
    g.fill();
  };

  return {
    init(h) {
      host = h;
    },
    setParams(p) {
      P = p;
      st = estimatorStats(p.p, p.G, p.alpha, p.eps, p.rule);
      mse = EPS.map((e) => clipMse(st.pmf, p.G, p.alpha, e, p.p));
      emp = new Float64Array(p.G + 1);
      draws = 0;
      for (let k = 0; k < 40; k++) sample();
    },
    resize() {
      ({ g, w: W, h: H, s } = fit2D(host.canvases[0]));
    },
    step() {
      sample();
    },
    draw() {
      const { pmf, vals, mean, sd } = st;
      const { G, alpha: a, eps, p, rule } = P;
      g.fillStyle = C.stage;
      g.fillRect(0, 0, W, H);
      const split = W * 0.6;
      const B = { x: 56 * s, y: 40 * s, w: split - 72 * s, h: H - 82 * s };
      const X = (v: number) => B.x + ((Math.log10(Math.min(Math.max(v, 1e-2), 1e5)) + 2) / 7) * B.w;
      const Y = (q: number) => B.y + B.h - ((Math.log10(Math.max(q, 1e-8)) + 8) / 8) * B.h;
      axes(
        g,
        s,
        B,
        X,
        Y,
        [[0.01, "0.01"], [1, "1"], [100, "100"], [1e4, "10⁴"]],
        [[1e-8, "10⁻⁸"], [1e-6, "10⁻⁶"], [1e-4, "10⁻⁴"], [1e-2, "0.01"], [1, "1"]],
        "weight estimate ÷ true p⁻ᵅ (log)",
        "probability (log)",
      );
      g.save();
      g.beginPath();
      g.rect(B.x, B.y, B.w, B.h);
      g.clip();
      const lo = Math.max(mean - sd, 1e-2);
      g.fillStyle = rgba(C.orange, 0.12);
      g.fillRect(X(lo), B.y, X(mean + sd) - X(lo), B.h);
      for (let n = 0; n <= G; n++) {
        if (pmf[n] < 1e-8) continue;
        const x = X(vals[n]);
        g.strokeStyle = n === 0 ? C.verm : rgba(C.sky, 0.8);
        g.lineWidth = (n === 0 ? 2.4 : 1.6) * s;
        g.beginPath();
        g.moveTo(x, B.y + B.h);
        g.lineTo(x, Y(pmf[n]));
        g.stroke();
        g.fillStyle = n === 0 ? C.verm : C.sky;
        g.beginPath();
        g.arc(x, Y(pmf[n]), 2.6 * s, 0, 2 * Math.PI);
        g.fill();
      }
      if (draws > 0)
        for (let n = 0; n <= G; n++) {
          if (emp[n] === 0) continue;
          g.strokeStyle = C.ink;
          g.lineWidth = 1.3 * s;
          g.beginPath();
          g.arc(X(vals[n]), Y(emp[n] / draws), 4.2 * s, 0, 2 * Math.PI);
          g.stroke();
        }
      g.strokeStyle = C.green;
      g.lineWidth = 2 * s;
      g.beginPath();
      g.moveTo(X(1), B.y);
      g.lineTo(X(1), B.y + B.h);
      g.stroke();
      g.strokeStyle = C.orange;
      g.setLineDash([5 * s, 4 * s]);
      g.beginPath();
      g.moveTo(X(mean), B.y);
      g.lineTo(X(mean), B.y + B.h);
      g.stroke();
      g.setLineDash([]);
      g.restore();
      label(g, s, "truth", X(1) + 5 * s, B.y + 10 * s, C.green, "left", 10);
      label(g, s, "mean ± sd", X(mean) + 5 * s, B.y + 24 * s, C.orange, "left", 10);
      if (pmf[0] >= 1e-8) label(g, s, "empty group", X(vals[0]) - 5 * s, Y(pmf[0]) - 12 * s, C.verm, "right", 10);
      label(g, s, `stems: exact · rings: ${draws.toLocaleString()} live draws · empty seen ${emp[0]}×`, B.x, B.y + B.h + 34 * s, C.muted, "left", 9.5);
      // total error against the clip
      const B2 = { x: split + 44 * s, y: 40 * s, w: W - split - 60 * s, h: H - 82 * s };
      const my = mse.map((v) => Math.log10(Math.max(v, 1e-6)));
      const ymin = Math.floor(Math.min(...my));
      const ymax = Math.ceil(Math.max(...my));
      const X2 = (e: number) => B2.x + ((Math.log10(e) + 4) / 3.95) * B2.w;
      const Y2 = (v: number) => B2.y + B2.h - ((Math.log10(Math.max(v, 1e-6)) - ymin) / Math.max(1, ymax - ymin)) * B2.h;
      const yt: Tick[] = [];
      for (let e = ymin; e <= ymax; e += Math.max(1, Math.ceil((ymax - ymin) / 4))) yt.push([Math.pow(10, e), e === 0 ? "1" : `10^${e}`]);
      axes(g, s, B2, X2, Y2, [[1e-4, "10⁻⁴"], [1e-2, "0.01"], [1, "1"]], yt, "clip ε (log)", "relative MSE");
      g.save();
      g.beginPath();
      g.rect(B2.x, B2.y, B2.w, B2.h);
      g.clip();
      g.strokeStyle = C.sky;
      g.lineWidth = 2 * s;
      g.beginPath();
      EPS.forEach((e, k) => (k ? g.lineTo(X2(e), Y2(mse[k])) : g.moveTo(X2(e), Y2(mse[k]))));
      g.stroke();
      star(X2(Math.min(0.89, p)), Y2(clipMse(pmf, G, a, p, p)), 7 * s, C.yellow);
      const ec = Math.min(0.89, Math.max(1e-4, eps));
      g.fillStyle = C.ink;
      g.beginPath();
      g.arc(X2(ec), Y2(mse[Math.round(((Math.log10(ec) + 4) / 3.95) * 100)]), 4 * s, 0, 2 * Math.PI);
      g.fill();
      g.restore();
      label(g, s, "★ ε = p (exact optimum)", B2.x + 6 * s, B2.y + 10 * s, C.yellow, "left", 10);
      label(g, s, rule === "clip" ? "● your ε" : "curve is for the clip rule", B2.x + 6 * s, B2.y + 24 * s, C.ink2, "left", 10);
    },
    dispose() {},
  };
}

export function EstimatorLab() {
  const [lgP, setLgP] = useState(-1);
  const [lgG, setLgG] = useState(4);
  const [alpha, setAlpha] = useLinkedAlpha("est", 1, 0.25, 3);
  const [lgE, setLgE] = useState(-3);
  const [rule, setRule] = useState<Rule>("clip");
  const p = Math.pow(10, lgP);
  const G = Math.pow(2, lgG);
  const eps = Math.pow(10, lgE);
  const params = useMemo(() => ({ p, G, alpha, eps, rule }), [p, G, alpha, eps, rule]);
  const handle = useEngine(createEngine, params);
  const st = useMemo(() => estimatorStats(p, G, alpha, eps, rule), [p, G, alpha, eps, rule]);

  return (
    <Instrument
      id="est"
      chips={["Exp 5 Fig 1", "Exp 5 Fig 2"]}
      title="Estimator lab"
      claim="The full distribution of the weight estimate for one outcome, computed exactly and then sampled live."
      stage={
        <Stage
          handle={handle}
          aspect="16 / 10"
          hud={`G·p = ${fsig(G * p, 3)} expected hits`}
          label="Left: stems of the probability of each weight value with live sampled dots. Right: total error against the clip, with its minimum at epsilon equals p."
        />
      }
    >
      <Slider id="est-p" label="True p" min={-3} max={-0.05} step={0.01} value={lgP} onChange={setLgP} format={(v) => Math.pow(10, v).toFixed(3)} />
      <Slider id="est-g" label="Group size G" min={1} max={7} step={1} value={lgG} onChange={setLgG} format={(v) => String(Math.pow(2, v))} />
      <Slider id="est-a" label="Exponent α" min={0.25} max={3} step={0.01} value={alpha} onChange={setAlpha} format={(v) => v.toFixed(2)} />
      <Slider id="est-e" label="Clip ε" min={-4} max={-0.05} step={0.01} value={lgE} onChange={setLgE} format={(v) => fsig(Math.pow(10, v), 2)} />
      <Segmented
        name="est-rule"
        legend="Weight rule"
        options={[
          { value: "clip", label: "Clip (paper)" },
          { value: "lap1", label: "Laplace λ=1" },
          { value: "offset", label: "α-offset" },
        ]}
        value={rule}
        onChange={setRule}
      />
      <Readouts
        items={[
          ["relative bias", fsig(st.mean - 1, 3)],
          ["relative sd", fsig(st.sd, 3)],
          ["relative RMSE", fsig(st.rmse, 3)],
          ["P(empty group)", `${fsig(st.pmf[0], 3)}  (${Math.round(st.share * 100)}% of the error)`],
        ]}
      />
      <Notes
        tryThis={[
          <>
            At p = 0.1, G = 16 the rare empty group sits far right: probability (1−p)<sup>G</sup> ≈ 0.19, weight 1/ε = 1000. It carries most of the
            error. Watch the live dots take a while to find it.
          </>,
          "Slide ε: total error falls, then rises, bottoming out exactly at ε = p (the star), a closed-form optimum from Exp 5.",
        ]}
        math="Every moment is an exact binomial sum over n = 0…G. The delta-method asymptote b ≈ α(α+1)(1−p)/(2Gp) only holds once Gp ≳ 20 expected hits."
        analogy="The course's total-error curve for numerical differentiation: truncation error falls with the step, round-off rises, and the best step is in between. Here the clip plays the step."
      />
    </Instrument>
  );
}
