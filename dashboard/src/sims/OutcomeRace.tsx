/** Probabilities racing on a logarithmic clock; tied tiers show rich-get-richer at alpha = 0. */
import { useMemo, useState } from "react";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { useLinkedAlpha } from "../engine/useLinkedAlpha";
import { Button, Readouts, Segmented, Slider, Toggle, text } from "../components/controls";
import { Instrument, Notes, Stage } from "../components/Instrument";
import { entropyN, linearRates, pStar, rewardNorm, rhs, rk4, softmax } from "../lib/model";
import { C, SERIES, rgba } from "../lib/color";
import { axes, fit2D, label } from "../lib/canvas2d";
import { fsig } from "../lib/format";

type Preset = "tiers" | "graded" | "tied";
const PRESETS: Record<Preset, number[]> = {
  tiers: [5, 5, 5, 3, 3, 1, 1],
  graded: [5, 4, 3, 2, 1],
  tied: [1, 1, 1, 1, 1, 1],
};
interface Params {
  alpha: number;
  preset: Preset;
  tilt: boolean;
  restart: number;
}
const U0 = -2;
const U1 = 4;

function createEngine(): Engine<Params> {
  let host!: EngineHost;
  let g!: CanvasRenderingContext2D;
  let W = 10;
  let H = 10;
  let s = 1;
  let r = PRESETS.tiers;
  let a = 0;
  let tilt = true;
  let z: number[] = [];
  let t = 0;
  let u = U0;
  let hold = 0;
  let hist: [number, number[]][] = [];
  let ps: number[] = [];
  let first = true;

  const colorOf = (i: number) => {
    const vals = [...new Set(r)].sort((x, y) => y - x);
    return SERIES[vals.indexOf(r[i]) % SERIES.length];
  };
  const dashOf = (i: number) => {
    let rank = 0;
    for (let j = 0; j < i; j++) if (r[j] === r[i]) rank++;
    return [[], [6, 4], [2, 3]][rank % 3];
  };
  function restart() {
    z = r.map((_, i) => (tilt ? [0.2, 0.1][i] ?? 0 : 0));
    t = 0;
    u = U0;
    hold = 0;
    hist = [[Math.pow(10, U0), softmax(z)]];
    ps = pStar(r, a);
  }
  function integrateTo(tt: number) {
    const lb = Math.max(Math.max(...r), a > 0 ? a * rewardNorm(r, a) : 0);
    const hmax = Math.min(0.2, 0.2 / lb);
    const f = (zz: number[]) => rhs(zz, r, a);
    let n = 0;
    while (t < tt && n < 6000) {
      const h = Math.min(hmax, tt - t);
      z = rk4(z, h, f);
      const m = Math.max(...z);
      z = z.map((v) => Math.max(v - m, -700));
      t += h;
      n++;
    }
    return t >= tt;
  }

  return {
    init(h) {
      host = h;
    },
    setParams(p) {
      a = p.alpha;
      r = PRESETS[p.preset];
      tilt = p.tilt;
      restart();
      if (first) {
        // fast-forward so the resting frame shows a finished race
        first = false;
        while (u < U1) {
          u += 0.05;
          integrateTo(Math.pow(10, u));
          hist.push([t, softmax(z)]);
        }
        hold = 1.5;
      }
    },
    resize() {
      ({ g, w: W, h: H, s } = fit2D(host.canvases[0]));
    },
    step(dt) {
      if (u >= U1) {
        hold += dt;
        if (hold > 3) restart();
        return;
      }
      const target = Math.min(U1, u + dt * 1.15);
      const ok = integrateTo(Math.pow(10, target));
      u = ok ? target : Math.log10(Math.max(t, 1e-9));
      hist.push([t, softmax(z)]);
    },
    draw() {
      g.fillStyle = C.stage;
      g.fillRect(0, 0, W, H);
      const p = softmax(z);
      const K = r.length;
      const bt = 34 * s;
      const bb = H * 0.36;
      const bl = 56 * s;
      const bw = (W - 18 * s - bl) / K;
      for (let i = 0; i < K; i++) {
        const x = bl + i * bw + bw * 0.16;
        const w = bw * 0.68;
        const hgt = (bb - bt) * p[i];
        g.fillStyle = rgba(colorOf(i), 0.9);
        g.fillRect(x, bb - hgt, w, hgt);
        g.strokeStyle = "#fff";
        g.lineWidth = 1.5 * s;
        g.setLineDash([4 * s, 3 * s]);
        const yt = bb - (bb - bt) * ps[i];
        g.beginPath();
        g.moveTo(x - 3 * s, yt);
        g.lineTo(x + w + 3 * s, yt);
        g.stroke();
        g.setLineDash([]);
        label(g, s, `O${i + 1}`, x + w / 2, bb + 10 * s, C.ink2, "center", 10.5);
        label(g, s, `r=${r[i]}`, x + w / 2, bb + 23 * s, C.muted, "center", 9.5);
        label(g, s, p[i] < 0.0005 ? "0" : p[i].toFixed(3), x + w / 2, bb - hgt - 8 * s, C.ink, "center", 10);
      }
      label(g, s, "current p (bars) and p* (white dashes)", bl, 14 * s, C.muted, "left", 10);
      const B = { x: 56 * s, y: H * 0.47, w: W - 74 * s, h: H * 0.53 - 44 * s };
      const X = (tt: number) => B.x + ((Math.log10(Math.max(tt, 1e-2)) - U0) / (U1 - U0)) * B.w;
      const Y = (v: number) => B.y + B.h - v * B.h;
      axes(
        g,
        s,
        B,
        X,
        Y,
        [[1e-2, "0.01"], [1e-1, "0.1"], [1, "1"], [10, "10"], [100, "100"], [1e3, "10³"], [1e4, "10⁴"]],
        [[0, "0"], [0.5, "0.5"], [1, "1"]],
        "time t (log scale)",
      );
      g.save();
      g.beginPath();
      g.rect(B.x, B.y, B.w, B.h);
      g.clip();
      for (let i = 0; i < K; i++) {
        g.strokeStyle = rgba(colorOf(i), 0.35);
        g.lineWidth = s;
        g.setLineDash([2 * s, 4 * s]);
        g.beginPath();
        g.moveTo(B.x, Y(ps[i]));
        g.lineTo(B.x + B.w, Y(ps[i]));
        g.stroke();
      }
      for (let i = K - 1; i >= 0; i--) {
        g.strokeStyle = colorOf(i);
        g.lineWidth = 1.8 * s;
        g.setLineDash(dashOf(i).map((v) => v * s));
        g.beginPath();
        hist.forEach(([tt, pp], k) => {
          if (k) g.lineTo(X(tt), Y(pp[i]));
          else g.moveTo(X(tt), Y(pp[i]));
        });
        g.stroke();
      }
      g.setLineDash([]);
      const xn = X(Math.max(t, 1e-2));
      g.strokeStyle = rgba(C.ink, 0.25);
      g.beginPath();
      g.moveTo(xn, B.y);
      g.lineTo(xn, B.y + B.h);
      g.stroke();
      g.restore();
      host.emit({ t: t < 1e-2 ? "0" : fsig(t, 3), h: `${entropyN(p).toFixed(3)} → ${entropyN(ps).toFixed(3)}` });
    },
    dispose() {},
  };
}

export function OutcomeRace() {
  const [alpha, setAlpha] = useLinkedAlpha("race", 0, 0, 3);
  const [preset, setPreset] = useState<Preset>("tiers");
  const [tilt, setTilt] = useState(true);
  const [restart, setRestart] = useState(0);
  const params = useMemo(() => ({ alpha, preset, tilt, restart }), [alpha, preset, tilt, restart]);
  const handle = useEngine(createEngine, params);
  const lr = linearRates(PRESETS[preset], alpha);
  const rate = lr ? `${fsig(lr[0], 3)}  (settles by t ≈ ${fsig(10 / Math.max(lr[0], 1e-9), 2)})` : "none: losers fade like 1/t";

  return (
    <Instrument
      id="race"
      chips={["Exp 1", "Exp 2 Figs 1–2"]}
      ghostChips={["worked example"]}
      title="The outcome race"
      claim="Watch probabilities evolve on a logarithmic clock. Equal rewards get equal shares only when α > 0."
      stage={<Stage handle={handle} aspect="16 / 11" label="Bars of current outcome probabilities above a chart of probability against log time." />}
    >
      <Slider id="race-a" label="Exponent α" min={0} max={3} step={0.01} value={alpha} onChange={setAlpha} format={(v) => v.toFixed(2)} />
      <Segmented
        name="race-r"
        legend="Rewards"
        options={[
          { value: "tiers", label: "Tiers 5 5 5 3 3 1 1" },
          { value: "graded", label: "Graded 5 4 3 2 1" },
          { value: "tied", label: "All tied" },
        ]}
        value={preset}
        onChange={setPreset}
      />
      <div className="row">
        <Toggle id="race-tilt" label="Start O1 slightly ahead" checked={tilt} onChange={setTilt} />
        <Button onClick={() => setRestart((n) => n + 1)}>Restart</Button>
      </div>
      <Readouts
        items={[
          ["time t", text(handle.readouts.t)],
          ["temperature T = α", alpha.toFixed(2)],
          [<>settling rate λ<sub>min</sub></>, rate],
          ["entropy now → p*", text(handle.readouts.h)],
        ]}
      />
      <Notes
        tryThis={[
          "Tiers at α = 0: the weak tiers vanish first, then the three tied top outcomes drift apart and O1 slowly takes everything. A 0.1-logit head start is enough.",
          "Same tiers at α = 1: the head start is erased by t ≈ 5 and the tiers settle at 5/23, 3/23, 1/23 each.",
          "All tied at α = 0: nothing moves at all (Exp 1 Fig 1a).",
        ]}
        math={
          <>
            At α = 0, ż<sub>i</sub> = p<sub>i</sub>(r<sub>i</sub> − r̄): the sign comes from the advantage, the size from p<sub>i</sub>. Near p*
            the error decays like e<sup>−λt</sup>, with λ = α‖r‖<sub>1/α</sub>μ from the Jacobian (Exp 2); at α = 0 there is no such rate and
            losers fade like 1/t.
          </>
        }
        analogy="p* ∝ exp(ln r / α) is a Boltzmann distribution with energy E = −ln r and temperature T = α. α → 0 freezes into the ground state (collapse); large α melts toward uniform."
      />
    </Instrument>
  );
}
