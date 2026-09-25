/**
 * Map over (alpha, G): two outcomes solved live, five outcomes precomputed with
 * the project's nested root-finding solver (Exp 4 Figs 1-2).
 */
import { useMemo, useState } from "react";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { Readouts, Segmented, Slider, text } from "../components/controls";
import { Instrument, Notes, Stage } from "../components/Instrument";
import { makeRule, meanfieldK2 } from "../lib/finiteGroup";
import { C, ramp, rgba, type RGB } from "../lib/color";
import { axes, fit2D, label } from "../lib/canvas2d";
import { fsig } from "../lib/format";
import ATLAS from "../data/atlasK5.json";

type Mode = "two" | "five";
interface Params {
  mode: Mode;
  rho: number;
  eps: number;
}
interface AtlasData {
  alphas: number[];
  G: number[];
  p: number[][][];
}
const AT = ATLAS as AtlasData;
const GS = AT.G;
const NA = 44;
const A0 = Math.log10(0.06);
const A1 = Math.log10(4);
const ALPHAS_TWO = Array.from({ length: NA }, (_, i) => Math.pow(10, A0 + ((A1 - A0) * (i + 0.5)) / NA));
const SUP: RGB[] = [
  [36, 48, 68],
  [31, 78, 110],
  [45, 127, 176],
  [86, 180, 233],
  [196, 234, 255],
];
const LGS = GS.map(Math.log2);
const gEdges = (i: number): [number, number] => [
  i === 0 ? LGS[0] - (LGS[1] - LGS[0]) / 2 : (LGS[i - 1] + LGS[i]) / 2,
  i === GS.length - 1 ? LGS[i] + (LGS[i] - LGS[i - 1]) / 2 : (LGS[i] + LGS[i + 1]) / 2,
];

function createEngine(): Engine<Params> {
  let host!: EngineHost;
  let g!: CanvasRenderingContext2D;
  let W = 10;
  let H = 10;
  let s = 1;
  let mode: Mode = "two";
  let rho = 4;
  let eps = 1e-3;
  const two = GS.map(() => new Float64Array(NA).fill(-1));
  let row = 0;
  let hover: { gi: number; f: number } | null = null;
  let B = { x: 0, y: 0, w: 1, h: 1 };
  let first = true;
  const L0 = gEdges(0)[0];
  const L1 = gEdges(GS.length - 1)[1];
  const X = (al: number) => B.x + ((Math.log10(al) - A0) / (A1 - A0)) * B.w;
  const YL = (lg: number) => B.y + B.h - ((lg - L0) / (L1 - L0)) * B.h;

  function solveRow() {
    if (row >= GS.length) return;
    const G = GS[row];
    for (let i = 0; i < NA; i++) two[row][i] = 1 - meanfieldK2(rho, G, makeRule("clip", G, ALPHAS_TWO[i], eps));
    row++;
  }
  const onMove = (e: PointerEvent) => {
    const rc = host.canvases[0].getBoundingClientRect();
    const px = (e.clientX - rc.left) * s;
    const py = (e.clientY - rc.top) * s;
    if (px < B.x || px > B.x + B.w || py < B.y || py > B.y + B.h) {
      hover = null;
    } else {
      let gi = 0;
      for (let i = 0; i < GS.length; i++) {
        const [lo, hi] = gEdges(i);
        if (py <= YL(lo) && py >= YL(hi)) gi = i;
      }
      hover = { gi, f: (px - B.x) / B.w };
    }
    host.invalidate();
  };
  const onLeave = () => {
    hover = null;
    host.invalidate();
  };

  return {
    init(h) {
      host = h;
      host.canvases[0].addEventListener("pointermove", onMove);
      host.canvases[0].addEventListener("pointerleave", onLeave);
    },
    setParams(p) {
      const reset = p.rho !== rho || p.eps !== eps || first;
      mode = p.mode;
      rho = p.rho;
      eps = p.eps;
      if (reset) {
        row = 0;
        if (first) while (row < GS.length) solveRow(); // complete map for the first frame
        first = false;
      }
    },
    resize() {
      ({ g, w: W, h: H, s } = fit2D(host.canvases[0]));
      B = { x: 54 * s, y: 18 * s, w: W - 130 * s, h: H - 60 * s };
    },
    step() {
      if (mode === "two" && row < GS.length) solveRow();
    },
    draw() {
      g.fillStyle = C.stage;
      g.fillRect(0, 0, W, H);
      for (let gi = 0; gi < GS.length; gi++) {
        const [lo, hi] = gEdges(gi);
        const y0 = YL(hi);
        const y1 = YL(lo);
        if (mode === "two") {
          for (let i = 0; i < NA; i++) {
            const v = two[gi][i];
            const x0 = B.x + (i / NA) * B.w;
            const x1 = B.x + ((i + 1) / NA) * B.w;
            g.fillStyle = v < 0 ? C.surface : rgba(v <= 1e-9 ? [16, 24, 36] : ramp(0.2 + 0.8 * Math.sqrt(v / 0.5)));
            g.fillRect(x0, y0, x1 - x0 + 0.5, y1 - y0 + 0.5);
          }
        } else {
          const al = AT.alphas;
          for (let i = 0; i < al.length; i++) {
            const lg = al.map(Math.log10);
            const lo2 = i === 0 ? lg[0] - (lg[1] - lg[0]) / 2 : (lg[i - 1] + lg[i]) / 2;
            const hi2 = i === al.length - 1 ? lg[i] + (lg[i] - lg[i - 1]) / 2 : (lg[i] + lg[i + 1]) / 2;
            const x0 = X(Math.pow(10, lo2));
            const x1 = X(Math.pow(10, hi2));
            const k = AT.p[gi][i].filter((v) => v > 0).length;
            g.fillStyle = rgba(SUP[k - 1]);
            g.fillRect(x0, y0, x1 - x0 + 0.5, y1 - y0 + 0.5);
          }
        }
      }
      axes(
        g,
        s,
        B,
        X,
        YL,
        [[0.1, "0.1"], [0.2, "0.2"], [0.5, "0.5"], [1, "1"], [2, "2"], [4, "4"]],
        [2, 4, 8, 16, 32, 64, 128, 256].map((v) => [Math.log2(v), String(v)]),
        "exponent α (log)",
        "group size G",
      );
      const lx = B.x + B.w + 18 * s;
      if (mode === "two") {
        g.save();
        g.beginPath();
        g.rect(B.x, B.y, B.w, B.h);
        g.clip();
        g.strokeStyle = C.orange;
        g.lineWidth = 2 * s;
        g.setLineDash([6 * s, 4 * s]);
        g.beginPath();
        let started = false;
        for (let k = 0; k <= 160; k++) {
          const lg = 0.8 + (7.4 * k) / 160;
          const ac = Math.log(rho) / Math.log(Math.min(Math.pow(2, lg), 1 / eps));
          if (ac <= 0) continue;
          if (started) g.lineTo(X(ac), YL(lg));
          else {
            g.moveTo(X(ac), YL(lg));
            started = true;
          }
        }
        g.stroke();
        g.restore();
        g.setLineDash([]);
        label(g, s, "α = ln ρ / ln min(G, 1/ε)", B.x + 8 * s, B.y + 12 * s, C.orange, "left", 10.5);
        const lh = B.h * 0.6;
        for (let k = 0; k < 40; k++) {
          g.fillStyle = rgba(ramp(0.2 + 0.8 * Math.sqrt(k / 39)));
          g.fillRect(lx, B.y + lh - ((k + 1) * lh) / 40, 12 * s, lh / 40 + 1);
        }
        label(g, s, "0.5", lx + 16 * s, B.y + 4 * s, C.muted, "left", 10);
        label(g, s, "0", lx + 16 * s, B.y + lh, C.muted, "left", 10);
        label(g, s, "minority p₂", lx - 2 * s, B.y + lh + 16 * s, C.ink2, "left", 10);
        if (row < GS.length) label(g, s, `solving row ${row + 1} of ${GS.length}…`, B.x + B.w - 6 * s, B.y + B.h - 10 * s, C.ink, "right", 10);
      } else {
        for (let k = 0; k < 5; k++) {
          g.fillStyle = rgba(SUP[k]);
          g.fillRect(lx, B.y + (4 - k) * 22 * s, 14 * s, 18 * s);
          label(g, s, String(k + 1), lx + 20 * s, B.y + (4 - k) * 22 * s + 9 * s, C.ink2, "left", 11);
        }
        label(g, s, "outcomes", lx - 2 * s, B.y + 5 * 22 * s + 10 * s, C.ink2, "left", 10);
        label(g, s, "kept", lx - 2 * s, B.y + 5 * 22 * s + 24 * s, C.ink2, "left", 10);
      }
      if (hover) {
        const G = GS[hover.gi];
        let al: number;
        let pv: string;
        if (mode === "two") {
          const i = Math.min(NA - 1, Math.max(0, Math.floor(hover.f * NA)));
          al = ALPHAS_TWO[i];
          const p2 = two[hover.gi][i];
          pv = p2 < 0 ? "solving…" : `(${(1 - p2).toFixed(4)}, ${p2.toFixed(4)})`;
          const [lo, hi] = gEdges(hover.gi);
          g.strokeStyle = "#fff";
          g.lineWidth = 1.5 * s;
          g.strokeRect(B.x + (i / NA) * B.w, YL(hi), B.w / NA, YL(lo) - YL(hi));
        } else {
          const lxv = A0 + hover.f * (A1 - A0);
          let i = 0;
          let best = 1e9;
          AT.alphas.forEach((v, k) => {
            const d = Math.abs(Math.log10(v) - lxv);
            if (d < best) {
              best = d;
              i = k;
            }
          });
          al = AT.alphas[i];
          pv = "(" + AT.p[hover.gi][i].map((v) => (v / 1e4).toFixed(3)).join(", ") + ")";
        }
        host.emit({ cell: `α = ${al.toFixed(3)}, G = ${G}`, p: pv });
      }
    },
    dispose() {
      host.canvases[0].removeEventListener("pointermove", onMove);
      host.canvases[0].removeEventListener("pointerleave", onLeave);
    },
  };
}

export function PhaseAtlas() {
  const [mode, setMode] = useState<Mode>("two");
  const [rho, setRho] = useState(4);
  const [lgE, setLgE] = useState(-3);
  const eps = Math.pow(10, lgE);
  const params = useMemo(() => ({ mode, rho, eps }), [mode, rho, eps]);
  const handle = useEngine(createEngine, params);

  return (
    <Instrument
      id="atlas"
      chips={["Exp 4 Fig 1a", "Exp 4 Fig 2a"]}
      title="Phase atlas"
      claim="A map over α and G. Two outcomes are solved live in your browser; five outcomes come from the project's nested root-finding solver."
      stage={<Stage handle={handle} aspect="16 / 11" hint="hover a cell" noPlay label="Heat map over alpha and group size showing minority mass or number of surviving outcomes, with the analytic boundary." />}
    >
      <Segmented
        name="atlas-m"
        legend="Outcomes"
        options={[
          { value: "two", label: "Two (live solve)" },
          { value: "five", label: "Five: 5 4 3 2 1" },
        ]}
        value={mode}
        onChange={setMode}
      />
      <Slider id="atlas-rho" label="Reward ratio ρ" min={1.2} max={10} step={0.1} value={rho} onChange={setRho} format={(v) => v.toFixed(1)} disabled={mode === "five"} />
      <Slider id="atlas-e" label="Clip ε" min={-3} max={-0.3} step={0.01} value={lgE} onChange={setLgE} format={(v) => fsig(Math.pow(10, v), 2)} disabled={mode === "five"} />
      <Readouts
        items={[
          ["cell", handle.readouts.cell ? text(handle.readouts.cell) : "hover the map"],
          ["stationary p", text(handle.readouts.p)],
        ]}
      />
      <Notes
        tryThis={[
          "Two outcomes: the dark-to-bright edge follows the dashed curve α = ln ρ / ln min(G, 1/ε) exactly (Exp 4 checked 19 125 points, zero mismatches).",
          "Five outcomes: bands of 1 to 5 survivors. At α = 1, G = 2 keeps 2, G = 4 keeps 3, G = 8 keeps 4, G = 16 keeps all 5.",
        ]}
        math={
          <>
            Every cell is a stationary point of the finite-group mean field: r<sub>i</sub>·w<sub>G</sub>(p<sub>i</sub>) = S for survivors,
            p<sub>i</sub> = 0 for the rest. Two outcomes need one bisection per cell; five need the nested solver of Exp 3.
          </>
        }
      />
    </Instrument>
  );
}
