/** Nested root finding for the K-outcome finite-group stationary point, and each outcome's own survival threshold (Exp 3 Fig 3). */
import { useMemo, useState } from "react";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { useLinkedAlpha } from "../engine/useLinkedAlpha";
import { Readouts, Slider } from "../components/controls";
import { Instrument, Notes, Stage } from "../components/Instrument";
import { criticalAlpha, makeRule, nestedStationary, outcomeAlphaC } from "../lib/finiteGroup";
import { pStar } from "../lib/model";
import { C, SERIES, rgba } from "../lib/color";
import { axes, fit2D, label, polyline, type Box } from "../lib/canvas2d";
import { fsig } from "../lib/format";

const R5 = [5, 4, 3, 2, 1];
const EPS = 1e-3;
const G_LIST = [2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 64, 128, 256];

interface Params {
  alpha: number;
  G: number;
  ac: number[];
}

interface Solve {
  p: number[];
  newton: { outer: number; work: number; tr: number[] };
  bisect: { outer: number; work: number; tr: number[] };
}

export function solveBoth(alpha: number, G: number): Solve {
  const om = makeRule("clip", G, alpha, EPS);
  const tn: number[] = [];
  const tb: number[] = [];
  const n = nestedStationary(R5, G, om, "newton", 1e-13, tn);
  const b = nestedStationary(R5, G, om, "bisection", 1e-13, tb);
  return { p: n.p, newton: { outer: n.outer, work: n.work, tr: tn }, bisect: { outer: b.outer, work: b.work, tr: tb } };
}

function createEngine(): Engine<Params> {
  let host!: EngineHost;
  let g!: CanvasRenderingContext2D;
  let W = 10;
  let H = 10;
  let s = 1;
  let P: Params = { alpha: 1, G: 16, ac: [0, 0, 0, 0, 0] };
  let sol = solveBoth(1, 16);
  let shown = 99;
  let hold = 0;
  let first = true;

  return {
    init(h) {
      host = h;
    },
    setParams(p) {
      P = p;
      sol = solveBoth(p.alpha, p.G);
      if (first) first = false;
      else shown = 0;
      hold = 0;
    },
    resize() {
      ({ g, w: W, h: H, s } = fit2D(host.canvases[0]));
    },
    step(dt) {
      const n = Math.max(sol.newton.tr.length, sol.bisect.tr.length);
      if (shown >= n) {
        hold += dt;
        if (hold > 3) {
          shown = 0;
          hold = 0;
        }
      } else shown = Math.min(n, shown + dt * 6);
    },
    draw() {
      g.fillStyle = C.stage;
      g.fillRect(0, 0, W, H);
      const { alpha: a, G, ac } = P;
      const ideal = pStar(R5, a);
      // top left: the stationary distribution against the ideal p* for this alpha
      const A: Box = { x: 50 * s, y: 30 * s, w: W * 0.42 - 50 * s, h: H * 0.5 - 50 * s };
      const bw = A.w / 5;
      const pmax = Math.max(0.7, ...sol.p, ...ideal) * 1.05;
      const YA = (v: number) => A.y + A.h - (v / pmax) * A.h;
      axes(g, s, A, () => A.x, YA, [], [[0, "0"], [0.25, "0.25"], [0.5, "0.5"]]);
      for (let i = 0; i < 5; i++) {
        const x = A.x + i * bw + bw * 0.18;
        const w = bw * 0.64;
        const alive = sol.p[i] > 1e-9;
        g.fillStyle = rgba(SERIES[i], alive ? 0.9 : 0.25);
        g.fillRect(x, YA(sol.p[i]), w, A.y + A.h - YA(sol.p[i]));
        g.strokeStyle = "#fff";
        g.lineWidth = 1.4 * s;
        g.setLineDash([4 * s, 3 * s]);
        polyline(g, 1, (k) => [x - 3 * s + k * (w + 6 * s), YA(ideal[i])]);
        g.setLineDash([]);
        label(g, s, `O${i + 1}`, x + w / 2, A.y + A.h + 11 * s, C.ink2, "center", 10);
        label(g, s, alive ? sol.p[i].toFixed(3) : "0", x + w / 2, YA(sol.p[i]) - 8 * s, alive ? C.ink : C.verm, "center", 9.5);
      }
      label(g, s, `p at G = ${G} (bars) · ideal p* (dashes)`, A.x, 14 * s, C.muted, "left", 10);
      // top right: each outcome's own threshold, K = 5 mean field against the pairwise formula
      const Bx = A.x + A.w + 70 * s;
      const B: Box = { x: Bx, y: A.y, w: W - Bx - 22 * s, h: A.h };
      const la0 = Math.log10(0.03);
      const la1 = Math.log10(6);
      const XB = (v: number) => B.x + ((Math.log10(Math.min(Math.max(v, 0.03), 6)) - la0) / (la1 - la0)) * B.w;
      const rowH = B.h / 4;
      axes(g, s, B, XB, () => B.y, [[0.05, "0.05"], [0.1, "0.1"], [0.5, "0.5"], [1, "1"], [4, "4"]], [], "threshold α_c (log)");
      for (let j = 1; j < 5; j++) {
        const yc = B.y + (j - 0.5) * rowH;
        const pw = criticalAlpha(R5[0] / R5[j], G, EPS);
        label(g, s, `O${j + 1}`, B.x - 8 * s, yc, SERIES[j], "right", 10);
        g.strokeStyle = rgba(C.muted, 0.9);
        g.lineWidth = 2 * s;
        g.beginPath();
        g.arc(XB(pw), yc + 6 * s, 3.5 * s, 0, 2 * Math.PI);
        g.stroke();
        if (Number.isFinite(ac[j])) {
          g.fillStyle = SERIES[j];
          g.beginPath();
          g.arc(XB(ac[j]), yc - 5 * s, 4.5 * s, 0, 2 * Math.PI);
          g.fill();
        } else {
          label(g, s, "lost at every α", B.x + B.w - 6 * s, yc - 5 * s, C.verm, "right", 9.5);
        }
      }
      g.strokeStyle = rgba(C.ink, 0.5);
      g.lineWidth = 1.2 * s;
      polyline(g, 1, (k) => [XB(a), B.y + k * B.h]);
      label(g, s, `α = ${a.toFixed(2)}`, XB(a) + 5 * s, B.y + 9 * s, C.ink, "left", 9.5);
      label(g, s, "● K=5 mean field   ○ pairwise ln(r₁/rⱼ)/ln G", B.x, 14 * s, C.muted, "left", 10);
      // bottom: outer residual |sum p - 1| per outer iteration, both nestings
      const Cb: Box = { x: 50 * s, y: A.y + A.h + 64 * s, w: W - 72 * s, h: H - (A.y + A.h + 64 * s) - 40 * s };
      const nMax = Math.max(sol.bisect.tr.length, 12);
      const XC = (n: number) => Cb.x + (n / nMax) * Cb.w;
      const YC = (v: number) => Cb.y + ((1 - Math.log10(Math.max(v, 1e-16))) / 17) * Cb.h;
      axes(
        g,
        s,
        Cb,
        XC,
        YC,
        [0, 10, 20, 30, 40, 50].filter((v) => v <= nMax).map((v) => [v, String(v)] as [number, string]),
        [[1, "1"], [1e-5, "10⁻⁵"], [1e-10, "10⁻¹⁰"], [1e-15, "10⁻¹⁵"]],
        "outer iteration",
      );
      label(g, s, "outer residual |Σ pᵢ(S) − 1| · each outer step runs an inner solve per outcome", Cb.x, Cb.y - 12 * s, C.muted, "left", 10);
      const k = Math.floor(shown);
      const series: [number[], string][] = [
        [sol.bisect.tr, C.muted],
        [sol.newton.tr, C.orange],
      ];
      g.save();
      g.beginPath();
      g.rect(Cb.x, Cb.y, Cb.w, Cb.h);
      g.clip();
      for (const [tr, col] of series) {
        const n = Math.min(tr.length - 1, k);
        if (n < 0) continue;
        g.strokeStyle = col;
        g.lineWidth = 1.8 * s;
        polyline(g, n, (i) => [XC(i + 1), YC(tr[i])]);
        g.fillStyle = col;
        for (let i = 0; i <= n; i++) {
          g.beginPath();
          g.arc(XC(i + 1), YC(tr[i]), 2.6 * s, 0, 2 * Math.PI);
          g.fill();
        }
      }
      g.restore();
      host.emit({});
    },
    dispose() {},
  };
}

export function NestedSolver() {
  const [alpha, setAlpha] = useLinkedAlpha("nested", 1, 0.05, 4);
  const [gi, setGi] = useState(G_LIST.indexOf(16));
  const G = G_LIST[gi];
  // each outcome's own threshold does not depend on alpha; recompute only when G changes
  const ac = useMemo(() => R5.map((_, j) => (j === 0 ? 0 : outcomeAlphaC(R5, j, G, EPS))), [G]);
  const params = useMemo(() => ({ alpha, G, ac }), [alpha, G, ac]);
  const handle = useEngine(createEngine, params);
  const sol = useMemo(() => solveBoth(alpha, G), [alpha, G]);
  const kept = sol.p.filter((x) => x > 1e-9).length;
  const weak = ac[4];

  return (
    <Instrument
      id="nested"
      chips={["Exp 3 Fig 3", "Exp 4 Fig 2"]}
      ghostChips={["nested root finding"]}
      title="The nested solver, five outcomes"
      claim="Two root-finders inside each other find the finite-group stationary point. With five outcomes, the weakest needs a larger α than the two-outcome formula says."
      stage={<Stage handle={handle} aspect="16 / 11" label="Stationary distribution bars, each outcome's survival threshold, and the outer residual of the nested solver per iteration." />}
    >
      <Slider id="nest-a" label="Exponent α" min={0.05} max={4} step={0.01} value={alpha} onChange={setAlpha} format={(v) => v.toFixed(2)} />
      <Slider id="nest-g" label="Group size G" min={0} max={G_LIST.length - 1} step={1} value={gi} onChange={setGi} format={(v) => String(G_LIST[v])} />
      <Readouts
        items={[
          ["outcomes kept", `${kept} of 5`],
          [<>weakest outcome needs α &gt;</>, Number.isFinite(weak) ? `${weak.toFixed(3)} (pairwise ${criticalAlpha(5, G, EPS).toFixed(3)})` : "no α works at this G"],
          [<span style={{ color: C.orange }}>Newton / Newton</span>, `${sol.newton.outer} outer · ${sol.newton.work} inner steps`],
          [<span style={{ color: C.muted }}>bisection / bisection</span>, `${sol.bisect.outer} outer · ${sol.bisect.work} inner steps`],
          ["speed-up in total work", `${fsig(sol.bisect.work / Math.max(sol.newton.work, 1), 3)}×`],
        ]}
      />
      <Notes
        tryThis={[
          "α = 1, G = 16: all five outcomes survive, and the weakest one's threshold is 0.886 (filled dot), not the pairwise 0.580 (ring). Exp 3's table: 0.0805, 0.203, 0.419, 0.886.",
          "Drop G to 5: the fifth outcome reads “lost at every α”. Exp 3 found it needs G ≥ 6 before any exponent can keep it.",
          "Step G through 2, 4, 8, 16 at α = 1: 2, 3, 4, then 5 outcomes kept, the same values as Exp 4's table (0.6667/0.3333 at G = 2, and so on).",
        ]}
        math={
          <>
            A survivor satisfies rᵢ·w<sub>G</sub>(pᵢ) = S, where w<sub>G</sub> is the size-biased weight. Inner problem: for a trial level S, invert the
            decreasing w<sub>G</sub>(pᵢ) = S/rᵢ for each outcome (pᵢ = 0 when rᵢ·ω(1) ≤ S). Outer problem: choose S so the pᵢ add up to 1. Both are monotone
            with a guaranteed bracket, so bisection always works. Newton is safeguarded: it takes a Newton step only when the step stays inside the bracket.
            Exp 3 measured 43–55 outer steps for bisection against 6–16 for Newton (8–20× less work). This browser's solver has different tolerances, so its
            counts differ, but the gap is the same size.
          </>
        }
        analogy="Pouring a fixed amount of water into five connected tanks with different floor heights: the outer loop finds the common water level S, and each inner loop reads off how deep one tank is at that level. A tank whose floor sits above the water gets nothing."
      />
    </Instrument>
  );
}
