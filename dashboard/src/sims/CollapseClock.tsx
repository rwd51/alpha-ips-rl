/** Collapse time against reward gap and against group size, fitted by least squares (Exp 1 Figs 1c and 2c). */
import { useMemo, useState } from "react";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { Button, Readouts, Segmented, text } from "../components/controls";
import { Instrument, Notes, Stage } from "../components/Instrument";
import { lsq } from "../lib/numerics";
import { mulberry } from "../lib/sampling";
import { C, RAMP_WARM, SERIES, ramp, rgba } from "../lib/color";
import { axes, fit2D, label, polyline, type Box } from "../lib/canvas2d";
import { fsig } from "../lib/format";

type Mode = "gap" | "group";
interface Params {
  mode: Mode;
  thr: number;
  restart: number;
}

// Exp 1 Fig 1c: 12 gaps log-spaced over [0.02, 4], r = (1 + gap, 1), z0 = (0.05, -0.05), RK4 h = 0.05
const DELTAS = Array.from({ length: 12 }, (_, k) => Math.pow(10, Math.log10(0.02) + (k * (Math.log10(4) - Math.log10(0.02))) / 11));
// Exp 1 Fig 2c: exact tie, z0 = 0, h = 0.5, 15000 steps, collapse at max p >= 0.98
const GS = [4, 8, 16, 32, 64];
const SEEDS = 32;
const MC_H = 0.5;
const MC_STEPS = 15000;
const MC_THR = 0.98;
const U0 = -1;
const U1 = 4;
const NT = 260;
const TS = Array.from({ length: NT + 1 }, (_, k) => Math.pow(10, U0 + (k * (U1 - U0)) / NT));

const sig = (u: number) => 1 / (1 + Math.exp(-u));
/** u = z1 - z2 obeys u' = 2 gap p q at alpha = 0 (Exp 1, derivation.md). */
const drift = (u: number, d: number) => {
  const p = sig(u);
  return 2 * d * p * (1 - p);
};
function rk4u(u: number, h: number, d: number): number {
  const k1 = drift(u, d);
  const k2 = drift(u + 0.5 * h * k1, d);
  const k3 = drift(u + 0.5 * h * k2, d);
  const k4 = drift(u + h * k3, d);
  return u + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
}

/** Closed form of the collapse time: integrate du / (2 gap p q) = (2u + 2 sinh u) / (2 gap). */
export function collapseConstant(thr: number): number {
  const F = (u: number) => 2 * u + 2 * Math.sinh(u);
  return (F(Math.log(thr / (1 - thr))) - F(0.1)) / 2;
}

function createEngine(): Engine<Params> {
  let host!: EngineHost;
  let g!: CanvasRenderingContext2D;
  let W = 10;
  let H = 10;
  let s = 1;
  let P: Params = { mode: "gap", thr: 0.99, restart: 0 };
  // gap mode
  let traj: number[][] = [];
  let tcol: number[] = [];
  let clockU = U0;
  let hold = 0;
  // group mode
  let seed = 1;
  let us: Float64Array[] = [];
  let tc: Float64Array[] = [];
  let step = 0;
  let first = true;

  function buildGap() {
    const ut = Math.log(P.thr / (1 - P.thr));
    traj = [];
    tcol = [];
    for (const d of DELTAS) {
      // collapse time exactly as experiments/exp1_collapse.py: step, then test the event
      let u = 0.1;
      let n = 0;
      const maxSteps = Math.ceil((3 * 54 / d + 60) / 0.05);
      let t = NaN;
      while (n < maxSteps) {
        u = rk4u(u, 0.05, d);
        n++;
        if (u >= ut) {
          t = n * 0.05;
          break;
        }
      }
      tcol.push(t);
      // p1(t) on the display clock
      const row: number[] = [];
      let uu = 0.1;
      let tt = 0;
      for (const T of TS) {
        while (tt < T - 1e-12 && uu < 40) {
          const h = Math.min(0.05, T - tt);
          uu = rk4u(uu, h, d);
          tt += h;
        }
        row.push(sig(uu));
      }
      traj.push(row);
    }
  }
  function startMC() {
    seed++;
    us = GS.map(() => new Float64Array(SEEDS));
    tc = GS.map(() => new Float64Array(SEEDS).fill(NaN));
    step = 0;
    clockU = 1;
    hold = 0;
  }
  function restart() {
    hold = 0;
    if (P.mode === "gap") {
      buildGap();
      clockU = U0;
    } else startMC();
  }

  /** Advance every unfinished run by one sampled update: u += 2h (n1/G - p1). */
  function mcTo(targetStep: number, R: () => number, budget: number) {
    let draws = 0;
    while (step < targetStep && step < MC_STEPS && draws < budget) {
      step++;
      for (let gi = 0; gi < GS.length; gi++) {
        const G = GS[gi];
        const u = us[gi];
        const t = tc[gi];
        for (let k = 0; k < SEEDS; k++) {
          if (!Number.isNaN(t[k])) continue;
          const p = sig(u[k]);
          let n1 = 0;
          for (let j = 0; j < G; j++) if (R() < p) n1++;
          draws += G;
          u[k] += 2 * MC_H * (n1 / G - p);
          const q = sig(u[k]);
          if (Math.max(q, 1 - q) >= MC_THR) t[k] = step * MC_H;
        }
      }
    }
  }
  let rng = mulberry(1);

  function mcStats() {
    const med: number[] = [];
    const frac: number[] = [];
    GS.forEach((_, gi) => {
      const v = Array.from(tc[gi]).filter((x) => !Number.isNaN(x)).sort((a, b) => a - b);
      frac.push(v.length / SEEDS);
      med.push(v.length ? v[Math.floor((v.length - 1) / 2)] * 0.5 + v[Math.ceil((v.length - 1) / 2)] * 0.5 : NaN);
    });
    return { med, frac };
  }

  function drawFit(B: Box, xs: number[], ys: number[], xr: [number, number], yr: [number, number], xt: [number, string][], yt: [number, string][], xl: string, color: string) {
    const X = (v: number) => B.x + ((Math.log10(v) - xr[0]) / (xr[1] - xr[0])) * B.w;
    const Y = (v: number) => B.y + B.h - ((Math.log10(v) - yr[0]) / (yr[1] - yr[0])) * B.h;
    axes(g, s, B, X, Y, xt, yt, xl);
    const ok = xs.map((_, i) => Number.isFinite(ys[i]) && ys[i] > 0);
    const fx = xs.filter((_, i) => ok[i]).map(Math.log10);
    const fy = ys.filter((_, i) => ok[i]).map(Math.log10);
    let fit = null as null | ReturnType<typeof lsq>;
    if (fx.length >= 3) {
      fit = lsq(fx, fy);
      g.save();
      g.beginPath();
      g.rect(B.x, B.y, B.w, B.h);
      g.clip();
      g.strokeStyle = rgba(C.ink2, 0.8);
      g.lineWidth = 1.4 * s;
      g.setLineDash([6 * s, 4 * s]);
      const f = fit;
      polyline(g, 1, (k) => {
        const lx = xr[0] + k * (xr[1] - xr[0]);
        return [X(Math.pow(10, lx)), Y(Math.pow(10, f.slope * lx + f.intercept))];
      });
      g.setLineDash([]);
      g.restore();
    }
    xs.forEach((x, i) => {
      if (!ok[i]) return;
      g.fillStyle = color;
      g.beginPath();
      g.arc(X(x), Y(ys[i]), 4.2 * s, 0, 2 * Math.PI);
      g.fill();
    });
    return fit;
  }

  return {
    init(h) {
      host = h;
    },
    setParams(p) {
      const modeChanged = p.mode !== P.mode;
      const need = modeChanged || p.thr !== P.thr || p.restart !== P.restart || traj.length === 0;
      P = p;
      if (need) {
        restart();
        if (P.mode === "group") rng = mulberry(seed * 7919);
      }
      if (first) {
        // the resting frame shows a finished sweep with its fit
        first = false;
        clockU = U1;
        hold = 1;
      }
    },
    resize() {
      ({ g, w: W, h: H, s } = fit2D(host.canvases[0]));
    },
    step(dt) {
      if (P.mode === "gap") {
        if (clockU >= U1) {
          hold += dt;
          if (hold > 3.5) restart();
        } else clockU = Math.min(U1, clockU + dt * 0.8);
        return;
      }
      if (step >= MC_STEPS || tc.every((t) => t.every((x) => !Number.isNaN(x)))) {
        hold += dt;
        if (hold > 4) {
          restart();
          rng = mulberry(seed * 7919);
        }
        return;
      }
      clockU = Math.min(Math.log10(MC_STEPS * MC_H), clockU + dt * 0.55);
      mcTo(Math.ceil(Math.pow(10, clockU) / MC_H), rng, 450000);
    },
    draw() {
      g.fillStyle = C.stage;
      g.fillRect(0, 0, W, H);
      const gap = 34 * s;
      const L: Box = { x: 58 * s, y: 30 * s, w: (W - 58 * s - 20 * s - gap - 52 * s) * 0.56, h: H - 84 * s };
      const Rb: Box = { x: L.x + L.w + gap + 52 * s, y: L.y, w: W - (L.x + L.w + gap + 52 * s) - 20 * s, h: L.h };
      if (P.mode === "gap") {
        const tNow = Math.pow(10, clockU);
        const X = (t: number) => L.x + ((Math.log10(Math.max(t, 0.1)) - U0) / (U1 - U0)) * L.w;
        const Y = (v: number) => L.y + L.h - ((v - 0.5) / 0.5) * L.h;
        axes(
          g,
          s,
          L,
          X,
          Y,
          [[0.1, "0.1"], [1, "1"], [10, "10"], [100, "100"], [1e3, "10³"], [1e4, "10⁴"]],
          [[0.5, "0.5"], [0.75, "0.75"], [1, "1"]],
          "time t (log)",
        );
        g.save();
        g.beginPath();
        g.rect(L.x, L.y, L.w, L.h);
        g.clip();
        g.strokeStyle = rgba(C.orange, 0.7);
        g.setLineDash([5 * s, 4 * s]);
        g.lineWidth = s;
        polyline(g, 1, (k) => [L.x + k * L.w, Y(P.thr)]);
        g.setLineDash([]);
        const kNow = Math.min(NT, Math.floor(((clockU - U0) / (U1 - U0)) * NT));
        traj.forEach((row, i) => {
          const c = ramp(0.25 + (0.75 * i) / (DELTAS.length - 1), RAMP_WARM);
          g.strokeStyle = rgba(c, 0.95);
          g.lineWidth = 1.6 * s;
          polyline(g, kNow, (k) => [X(TS[k]), Y(row[k])]);
          if (tcol[i] <= tNow) {
            g.fillStyle = rgba(c, 1);
            g.beginPath();
            g.arc(X(tcol[i]), Y(P.thr), 3.6 * s, 0, 2 * Math.PI);
            g.fill();
          }
        });
        g.strokeStyle = rgba(C.ink, 0.22);
        polyline(g, 1, (k) => [X(tNow), L.y + k * L.h]);
        g.restore();
        label(g, s, "p₁(t) for 12 reward gaps Δ", L.x, 16 * s, C.muted, "left", 10);
        label(g, s, `threshold ${P.thr}`, L.x + L.w - 6 * s, Y(P.thr) + 11 * s, C.orange, "right", 10);
        const seen = DELTAS.map((_, i) => (tcol[i] <= tNow ? tcol[i] : NaN));
        const fit = drawFit(
          Rb,
          DELTAS,
          seen,
          [Math.log10(0.015), Math.log10(5)],
          [Math.log10(8), Math.log10(6000)],
          [[0.02, "0.02"], [0.1, "0.1"], [1, "1"], [4, "4"]],
          [[10, "10"], [100, "100"], [1e3, "10³"]],
          "reward gap Δ (log)",
          C.orange,
        );
        label(g, s, "collapse time (log)", Rb.x, 16 * s, C.muted, "left", 10);
        const Ct = collapseConstant(P.thr);
        host.emit({
          fit: fit ? `t ≈ ${fsig(Math.pow(10, fit.intercept), 4)} · Δ^${fit.slope.toFixed(3)}` : "waiting for 3 points",
          r2: fit ? fit.r2.toFixed(8) : "—",
          theory: `${fsig(Ct, 4)} · Δ^−1`,
          done: `${seen.filter(Number.isFinite).length} of 12`,
        });
      } else {
        const { med, frac } = mcStats();
        const t0 = 1;
        const t1 = Math.log10(MC_STEPS * MC_H);
        const X = (t: number) => L.x + ((Math.log10(Math.max(t, 10)) - t0) / (t1 - t0)) * L.w;
        const rowH = L.h / GS.length;
        axes(g, s, L, X, () => L.y, [[10, "10"], [100, "100"], [1e3, "10³"], [7500, "7500"]], [], "collapse time (log)");
        GS.forEach((G, gi) => {
          const yc = L.y + (gi + 0.5) * rowH;
          label(g, s, `G=${G}`, L.x - 8 * s, yc, C.ink2, "right", 10.5);
          const R = mulberry(gi * 31 + 5);
          tc[gi].forEach((t) => {
            const jit = (R() - 0.5) * rowH * 0.62;
            if (Number.isNaN(t)) return;
            g.fillStyle = rgba(SERIES[gi], 0.85);
            g.beginPath();
            g.arc(X(t), yc + jit, 2.8 * s, 0, 2 * Math.PI);
            g.fill();
          });
          if (Number.isFinite(med[gi])) {
            g.strokeStyle = C.ink;
            g.lineWidth = 2 * s;
            polyline(g, 1, (k) => [X(med[gi]), yc - rowH * 0.38 + k * rowH * 0.76]);
          }
        });
        const tNow = step * MC_H;
        g.strokeStyle = rgba(C.ink, 0.22);
        g.lineWidth = s;
        polyline(g, 1, (k) => [X(tNow), L.y + k * L.h]);
        label(g, s, `${SEEDS} runs per G · white bar = median · clock t = ${fsig(tNow, 3)}`, L.x, 16 * s, C.muted, "left", 10);
        const fit = drawFit(
          Rb,
          GS,
          med.map((m, i) => (frac[i] >= 0.5 ? m : NaN)),
          [Math.log10(3), Math.log10(90)],
          [Math.log10(50), Math.log10(1e4)],
          [[4, "4"], [8, "8"], [16, "16"], [32, "32"], [64, "64"]],
          [[100, "100"], [1e3, "10³"], [1e4, "10⁴"]],
          "group size G (log)",
          C.sky,
        );
        label(g, s, "median collapse time (log)", Rb.x, 16 * s, C.muted, "left", 10);
        const all = tc.reduce((a, t) => a + Array.from(t).filter((x) => !Number.isNaN(x)).length, 0);
        host.emit({
          fit: fit ? `t ∝ G^${fit.slope.toFixed(2)}` : "waiting for the medians",
          r2: fit ? fit.r2.toFixed(3) : "—",
          theory: "G¹ (drift time ∝ G)",
          done: `${all} of ${SEEDS * GS.length} collapsed`,
        });
      }
    },
    dispose() {},
  };
}

export function CollapseClock() {
  const [mode, setMode] = useState<Mode>("gap");
  const [thr, setThr] = useState(0.99);
  const [restart, setRestart] = useState(0);
  const params = useMemo(() => ({ mode, thr, restart }), [mode, thr, restart]);
  const handle = useEngine(createEngine, params);
  const gapMode = mode === "gap";

  return (
    <Instrument
      id="clock"
      chips={["Exp 1 Fig 1c", "Exp 1 Fig 2c"]}
      ghostChips={["least squares"]}
      title="The collapse clock"
      claim="How long collapse takes follows a straight line on log-log axes: 1/Δ in the reward gap, and roughly G in the group size."
      stage={<Stage handle={handle} aspect="16 / 9" label="Left: collapse runs on a log clock. Right: collapse time on log-log axes with a least-squares line." />}
    >
      <Segmented
        name="clock-mode"
        legend="What to vary"
        options={[
          { value: "gap", label: "Reward gap Δ (exact flow)" },
          { value: "group", label: "Group size G (sampled)" },
        ]}
        value={mode}
        onChange={setMode}
      />
      {gapMode ? (
        <Segmented
          name="clock-thr"
          legend="Collapse means max p reaches"
          options={[
            { value: "0.9", label: "0.90" },
            { value: "0.95", label: "0.95" },
            { value: "0.99", label: "0.99 (Exp 1)" },
          ]}
          value={String(thr) as "0.9" | "0.95" | "0.99"}
          onChange={(v) => setThr(parseFloat(v))}
        />
      ) : null}
      <Button onClick={() => setRestart((n) => n + 1)}>{gapMode ? "Replay sweep" : "New Monte Carlo seeds"}</Button>
      <Readouts
        items={[
          ["least-squares fit", text(handle.readouts.fit)],
          ["R²", text(handle.readouts.r2)],
          ["theory", text(handle.readouts.theory)],
          ["runs", text(handle.readouts.done)],
        ]}
      />
      <Notes
        tryThis={[
          "Reward gap mode: the 12 runs cross the threshold in order, and each crossing adds a point on the right. With three or more, the fit reads t ≈ 54 · Δ^−1.00, which is Exp 1's 53.94 · Δ^−1.00 (R² = 0.99999994).",
          "Change the threshold to 0.9: the slope stays at −1 and only the prefactor moves. A different definition of \"collapsed\" does not change the law.",
          "Group size mode: 32 sampled runs per G on an exact tie, same settings as Exp 1 Fig 2c (h = 0.5, threshold 0.98). Bigger groups take longer. The repo's 80-seed fit is G^1.07 (1.066 after Exp 2's sampler fix); with 32 seeds expect a value near 1, varying with the seed.",
        ]}
        math={
          <>
            With two outcomes the flow reduces to one equation for the logit gap u = z₁ − z₂: u′ = 2Δ·p₁p₂. Separating variables, t = [2u + 2 sinh u] / (2Δ)
            between the start and the threshold, so the collapse time is exactly C/Δ with C = 53.89 at threshold 0.99. The fitted 53.94 is RK4 with h = 0.05
            landing on the first step past the threshold. The group-size law has no closed form. A group of G shrinks the noise in p̂ by 1/G, and a random
            walk needs about G times longer to drift the same distance.
          </>
        }
        analogy="Neutral genetic drift: in a population of size N a neutral allele takes on the order of N generations to fix. The group size plays the role of population size."
      />
    </Instrument>
  );
}
