/**
 * Four-outcome policies as points in a tetrahedron (3-D). Each particle is one training run.
 *
 * "Tie race" mode starts every run together on a logarithmic clock and colours it by which of
 * the tied top outcomes it started ahead on, to show rich-get-richer among equal rewards at
 * alpha = 0 (and the slow return to an even split for small alpha > 0).
 */
import { useMemo, useState } from "react";
import * as THREE from "three";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { useLinkedAlpha } from "../engine/useLinkedAlpha";
import { Readouts, Segmented, Slider, text } from "../components/controls";
import { Instrument, Notes, Stage } from "../components/Instrument";
import { entropyN, linearRates, pStar, rewardNorm, rhs, rk4, softmax, type Vec } from "../lib/model";
import { gaussR, mulberry, sampledUpdate } from "../lib/sampling";
import { C, SERIES, hexRgb, ramp } from "../lib/color";
import { dotTexture, line3, make3D, setLinePoints, textSprite, type P3, type Scene3D } from "../lib/three3d";
import { fsig } from "../lib/format";

type Mode = "ideal" | "sampled" | "tie";
interface Params {
  alpha: number;
  rewards: string;
  mode: Mode;
  G: number;
  scatter: number;
}

const RC = 1.55;
const V: P3[] = [[0, RC, 0]];
for (let k = 0; k < 3; k++) {
  const ang = Math.PI / 2 + (k * 2 * Math.PI) / 3;
  V.push([RC * ((2 * Math.SQRT2) / 3) * Math.cos(ang), -RC / 3, RC * ((2 * Math.SQRT2) / 3) * Math.sin(ang)]);
}
const pos = (p: ArrayLike<number>): P3 => [0, 1, 2].map((d) => p[0] * V[0][d] + p[1] * V[1][d] + p[2] * V[2][d] + p[3] * V[3][d]) as P3;
const RACE_U0 = -2;
const RACE_U1 = 4;
const SERIES_RGB = SERIES.map((c) => hexRgb(c).map((v) => v / 255));

/** Indices of the outcomes sharing the top reward. */
function topTie(r: Vec): number[] {
  const m = Math.max(...r);
  return r.map((x, i) => (x === m ? i : -1)).filter((i) => i >= 0);
}

function createEngine(): Engine<Params> {
  let T3!: Scene3D;
  let host!: EngineHost;
  let r = [4, 3, 2, 1];
  let a = 1;
  let mode: Mode = "ideal";
  let G = 16;
  let scatterSeen = -1;
  let N = 320;
  const TR = 8;
  let Z = new Float64Array(0);
  let age = new Float64Array(0);
  let life = new Float64Array(0);
  let hist = new Float32Array(0);
  let lead = new Int8Array(0);
  let tPart = new Float64Array(0);
  let pts!: THREE.Points;
  let trails!: THREE.LineSegments;
  let marker!: THREE.Mesh;
  let glow!: THREE.Sprite;
  let pathLine!: THREE.Line;
  let labels: THREE.Sprite[] = [];
  const R = mulberry(23);
  let frame = 0;
  let speedRef = 0.02;
  let raceU = RACE_U0;
  let hold = 0;
  let tie: number[] = [0];
  let lastKey = "";

  const resetHist = (i: number) => {
    const q = pos(softmax(Z.subarray(i * 4, i * 4 + 4)));
    for (let t = 0; t < TR; t++) hist.set(q, (i * TR + t) * 3);
  };
  const spawn = (i: number, spread = 1.9) => {
    for (let k = 0; k < 4; k++) Z[i * 4 + k] = spread * gaussR(R);
    age[i] = 0;
    life[i] = 4 + R() * 5;
    resetHist(i);
  };
  function startRace() {
    tie = topTie(r);
    for (let i = 0; i < N; i++) {
      spawn(i, 0.9);
      tPart[i] = 0;
      let best = tie[0];
      for (const k of tie) if (Z[i * 4 + k] > Z[i * 4 + best]) best = k;
      lead[i] = best;
    }
    raceU = RACE_U0;
    hold = 0;
  }

  function refreshScene() {
    const q = pos(pStar(r, a));
    marker.position.set(...q);
    glow.position.set(...q);
    const pts3: THREE.Vector3[] = [];
    for (let k = 0; k <= 120; k++) pts3.push(new THREE.Vector3(...pos(pStar(r, Math.pow(10, -1.7 + (k * 3.4) / 120)))));
    setLinePoints(pathLine, pts3);
    labels.forEach((sp) => {
      T3.scene.remove(sp);
      (sp.material as THREE.SpriteMaterial).map?.dispose();
      sp.material.dispose();
    });
    labels = V.map((v, k) => {
      const sp = textSprite(`O${k + 1} · r=${r[k]}`, mode === "tie" && tie.includes(k) && tie.length > 1 ? SERIES[k] : C.ink2, 0.15);
      sp.position.set(v[0] * 1.2, v[1] * 1.2 + (k === 0 ? 0.08 : -0.04), v[2] * 1.2);
      T3.scene.add(sp);
      return sp;
    });
  }

  /** Adaptive RK4 up to time `target`: each step moves the logits by at most 0.3. */
  function integrateTo(i: number, target: number, cap: number) {
    const f = (zz: Vec) => rhs(zz, r, a);
    let z = Array.from(Z.subarray(i * 4, i * 4 + 4));
    let t = tPart[i];
    for (let n = 0; n < 40 && t < target - 1e-12; n++) {
      const d = f(z);
      const mx = Math.max(Math.abs(d[0]), Math.abs(d[1]), Math.abs(d[2]), Math.abs(d[3]));
      if (mx < 1e-12) {
        t = target;
        break;
      }
      const h = Math.min(target - t, 0.3 / mx, cap);
      z = rk4(z, h, f);
      t += h;
    }
    const m = Math.max(...z);
    for (let j = 0; j < 4; j++) Z[i * 4 + j] = Math.max(z[j] - m, -60);
    tPart[i] = t;
  }

  function raceReadouts() {
    let wins = 0;
    const shares: number[] = [];
    for (let i = 0; i < N; i++) {
      const p = softmax(Z.subarray(i * 4, i * 4 + 4));
      let best = tie[0];
      let tot = 0;
      for (const k of tie) {
        tot += p[k];
        if (p[k] > p[best]) best = k;
      }
      const share = p[lead[i]] / Math.max(tot, 1e-300);
      // "clearly ahead": more than one percentage point above an even split
      if (best === lead[i] && share > 1 / tie.length + 0.01) wins++;
      shares.push(share);
    }
    shares.sort((x, y) => x - y);
    const t = Math.pow(10, Math.min(raceU, RACE_U1));
    host.emit({
      clock: t < 0.011 ? "0" : fsig(t, 3),
      wins: tie.length > 1 ? `${Math.round((100 * wins) / N)}% of runs` : "no tie at the top",
      share: tie.length > 1 ? `${(100 * shares[Math.floor(N / 2)]).toFixed(1)}% (even split: ${(100 / tie.length).toFixed(1)}%)` : "—",
    });
  }

  function advance(dt: number, spin: boolean) {
    if (spin) T3.spin(dt);
    const maxr = Math.max(...r);
    const lb = Math.max(maxr, a > 0 ? 0.5 * a * rewardNorm(r, a) : maxr);
    const lr = linearRates(r, a);
    const lmin = lr ? Math.max(lr[0], 0.02) : 0.3;
    const posA = pts.geometry.attributes.position.array as Float32Array;
    const colA = pts.geometry.attributes.color.array as Float32Array;
    const tp = trails.geometry.attributes.position.array as Float32Array;
    const tc = trails.geometry.attributes.color.array as Float32Array;
    frame++;
    let meanSp = 0;
    let target = 0;
    if (mode === "tie") {
      if (raceU >= RACE_U1) {
        hold += dt;
        if (hold > 3) startRace();
      } else raceU = Math.min(RACE_U1, raceU + dt * 0.7);
      target = Math.pow(10, raceU) - Math.pow(10, RACE_U0);
    }
    const cap = lr ? 1.5 / lr[lr.length - 1] : Infinity;
    const f = (zz: Vec) => rhs(zz, r, a);
    for (let i = 0; i < N; i++) {
      const before = pos(softmax(Z.subarray(i * 4, i * 4 + 4)));
      if (mode === "tie") {
        integrateTo(i, target, cap);
      } else {
        let z = Array.from(Z.subarray(i * 4, i * 4 + 4));
        if (mode === "ideal") {
          const h = Math.min(0.05, 0.25 / lb);
          const want = Math.min(40, 1.6 / lmin) * dt;
          const n = Math.min(8, Math.max(1, Math.ceil(want / h)));
          const he = Math.min(h, want / n);
          for (let k = 0; k < n; k++) z = rk4(z, he, f);
        } else {
          const h = Math.min(0.12, 0.4 / lb);
          for (let k = 0; k < 2; k++) {
            const u = sampledUpdate(z, r, a, G, 1e-3, R);
            for (let j = 0; j < 4; j++) z[j] += h * u.g[j];
          }
        }
        const m = Math.max(...z);
        for (let j = 0; j < 4; j++) Z[i * 4 + j] = Math.max(z[j] - m, -60);
        age[i] += dt;
        if (age[i] > life[i]) spawn(i);
      }
      const q = pos(softmax(Z.subarray(i * 4, i * 4 + 4)));
      const sp = Math.hypot(q[0] - before[0], q[1] - before[1], q[2] - before[2]) / Math.max(dt, 1e-3);
      meanSp += sp;
      posA.set(q, i * 3);
      const base: number[] = mode === "tie" ? SERIES_RGB[lead[i]] : ramp(0.3 + 0.7 * Math.tanh(sp / speedRef)).map((v) => v / 255);
      colA.set(base, i * 3);
      if (frame % 3 === 0) {
        hist.copyWithin(i * TR * 3 + 3, i * TR * 3, i * TR * 3 + (TR - 1) * 3);
        hist.set(q, i * TR * 3);
      }
      const trailRGB = mode === "tie" ? base : [0.34, 0.7, 0.91];
      for (let t = 0; t < TR - 1; t++) {
        const b = (i * (TR - 1) + t) * 6;
        const h0 = (i * TR + t) * 3;
        const h1 = h0 + 3;
        if (t === 0) tp.set(q, b);
        else tp.set(hist.subarray(h0, h0 + 3), b);
        tp.set(hist.subarray(h1, h1 + 3), b + 3);
        const f0 = 0.55 * (1 - t / (TR - 1));
        const f1 = 0.55 * (1 - (t + 1) / (TR - 1));
        tc.set([trailRGB[0] * f0, trailRGB[1] * f0, trailRGB[2] * f0, trailRGB[0] * f1, trailRGB[1] * f1, trailRGB[2] * f1], b);
      }
    }
    speedRef = 0.9 * speedRef + 0.1 * Math.max(1e-3, (meanSp / N) * 1.5);
    pts.geometry.attributes.position.needsUpdate = true;
    pts.geometry.attributes.color.needsUpdate = true;
    trails.geometry.attributes.position.needsUpdate = true;
    trails.geometry.attributes.color.needsUpdate = true;
    if (mode === "tie") raceReadouts();
  }

  return {
    init(h) {
      host = h;
      N = host.reducedMotion ? 160 : 320;
      Z = new Float64Array(N * 4);
      age = new Float64Array(N);
      life = new Float64Array(N);
      hist = new Float32Array(N * TR * 3);
      lead = new Int8Array(N);
      tPart = new Float64Array(N);
      T3 = make3D(host.canvases[0], host.invalidate, { radius: 5.5, phi: 1.2, theta: 0.6, target: [0, 0.3, 0], autoRotate: !host.reducedMotion });
      const edges: THREE.Vector3[] = [];
      for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) edges.push(new THREE.Vector3(...V[i]), new THREE.Vector3(...V[j]));
      T3.scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(edges), new THREE.LineBasicMaterial({ color: 0x34496a })));
      const pg = new THREE.BufferGeometry();
      pg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * 3), 3));
      pg.setAttribute("color", new THREE.BufferAttribute(new Float32Array(N * 3), 3));
      pts = new THREE.Points(
        pg,
        new THREE.PointsMaterial({ size: 0.085, map: dotTexture(), vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }),
      );
      pts.frustumCulled = false;
      T3.scene.add(pts);
      const tg = new THREE.BufferGeometry();
      tg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * (TR - 1) * 6), 3));
      tg.setAttribute("color", new THREE.BufferAttribute(new Float32Array(N * (TR - 1) * 6), 3));
      trails = new THREE.LineSegments(
        tg,
        new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false, blending: THREE.AdditiveBlending }),
      );
      trails.frustumCulled = false;
      T3.scene.add(trails);
      marker = new THREE.Mesh(new THREE.SphereGeometry(0.055, 20, 14), new THREE.MeshBasicMaterial({ color: 0xffe2a8 }));
      glow = new THREE.Sprite(
        new THREE.SpriteMaterial({ map: dotTexture(), color: 0xe69f00, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }),
      );
      glow.scale.set(0.5, 0.5, 1);
      T3.scene.add(marker, glow);
      pathLine = line3([[0, 0, 0], [0, 0, 0]], 0xe69f00, 0.75);
      T3.scene.add(pathLine);
      for (let i = 0; i < N; i++) spawn(i);
    },
    setParams(p) {
      a = p.alpha;
      r = p.rewards.split(",").map(Number);
      G = p.G;
      const scattered = scatterSeen !== -1 && p.scatter !== scatterSeen;
      scatterSeen = p.scatter;
      const key = `${p.mode}|${p.rewards}|${p.alpha}`;
      const modeChanged = p.mode !== mode;
      mode = p.mode;
      tie = topTie(r);
      if (mode === "tie" && (key !== lastKey || scattered)) startRace();
      else if (scattered) for (let i = 0; i < N; i++) spawn(i);
      else if (modeChanged) for (let i = 0; i < N; i++) resetHist(i);
      lastKey = key;
      refreshScene();
      if (frame === 0) for (let k = 0; k < 40; k++) advance(1 / 60, false);
    },
    resize() {
      T3.resize();
    },
    step(dt) {
      advance(dt, true);
    },
    draw() {
      T3.render();
    },
    dispose() {
      T3.dispose();
    },
  };
}

const REWARDS = [
  { value: "4,3,2,1", label: "4 3 2 1" },
  { value: "5,5,2,1", label: "5 5 2 1" },
  { value: "5,5,5,1", label: "5 5 5 1" },
  { value: "1,1,1,1", label: "1 1 1 1" },
];

export function Tetrahedron() {
  const [alpha, setAlpha] = useLinkedAlpha("tetra", 1, 0, 3);
  const [rewards, setRewards] = useState("4,3,2,1");
  const [mode, setMode] = useState<Mode>("ideal");
  const [lgG, setLgG] = useState(4);
  const [scatter, setScatter] = useState(0);
  const G = Math.pow(2, lgG);
  const params = useMemo(() => ({ alpha, rewards, mode, G, scatter }), [alpha, rewards, mode, G, scatter]);
  const handle = useEngine(createEngine, params);
  const r = rewards.split(",").map(Number);
  const ps = pStar(r, alpha);
  const tied = topTie(r);

  const chooseMode = (m: Mode) => {
    setMode(m);
    // the tie race needs at least two outcomes sharing the top reward
    if (m === "tie" && topTie(rewards.split(",").map(Number)).length < 2) setRewards("5,5,5,1");
  };
  const hud =
    mode === "tie"
      ? tied.length > 1
        ? `tie race · t = ${text(handle.readouts.clock)}\ncolour = which of ${tied.map((k) => `O${k + 1}`).join(", ")} started ahead`
        : "tie race needs tied top rewards:\nchoose 5 5 2 1 or 5 5 5 1"
      : "";

  return (
    <Instrument
      id="tetra"
      chips={["Exp 2"]}
      ghostChips={["new 3-D view"]}
      title="Four outcomes, one tetrahedron"
      claim="Every policy over four outcomes is a point inside a tetrahedron. Each particle below is one training run."
      stage={
        <Stage
          handle={handle}
          aspect="4 / 3"
          grab
          hint="drag to rotate"
          hud={hud}
          label="A rotating tetrahedron whose corners are the four outcomes, with hundreds of particles flowing to the stationary point."
        />
      }
    >
      <Slider id="tetra-a" label="Exponent α" min={0} max={3} step={0.01} value={alpha} onChange={setAlpha} format={(v) => v.toFixed(2)} />
      <Segmented name="tetra-r" legend="Rewards" options={REWARDS} value={rewards} onChange={setRewards} />
      <Segmented
        name="tetra-m"
        legend="Training"
        options={[
          { value: "ideal", label: "Ideal flow" },
          { value: "sampled", label: "Sampled groups" },
          { value: "tie", label: "Tie race" },
        ]}
        value={mode}
        onChange={chooseMode}
      />
      <Slider id="tetra-g" label="Group size G" min={1} max={6} step={1} value={lgG} onChange={setLgG} format={(v) => String(Math.pow(2, v))} disabled={mode !== "sampled"} />
      <div className="row">
        <button className="btn" type="button" onClick={() => setScatter((n) => n + 1)}>
          {mode === "tie" ? "Restart race" : "Scatter particles"}
        </button>
      </div>
      {mode === "tie" ? (
        <Readouts
          items={[
            ["clock t", text(handle.readouts.clock)],
            ["early leader clearly ahead", text(handle.readouts.wins)],
            ["its share of the tie (median)", text(handle.readouts.share)],
          ]}
        />
      ) : (
        <Readouts
          items={[
            ["stationary p*", ps.map((v) => v.toFixed(3)).join("  ")],
            ["entropy H/ln 4", entropyN(ps).toFixed(3)],
          ]}
        />
      )}
      <Notes
        tryThis={[
          "Tie race, rewards 5 5 5 1, α = 0: every run is coloured by whichever of O1, O2, O3 it started ahead on, and each colour ends in its own corner. Equal rewards, yet the likelier twin takes everything.",
          "Same race at α = 0.02: the colours pull apart until t ≈ 10 (the leader's share of the tie rises to about 0.63), then drift back to an even split by t ≈ 1000. A small α looks like collapse for a while, then wins.",
          "Sampled groups, 1 1 1 1, α = 0: pure noise drives runs to faces, then edges, then corners (genetic drift in 3-D).",
        ]}
        math={
          <>
            For two tied outcomes, d/dt ln(p₁/p₂) = (p₁ − p₂)(r − r̄) at α = 0: whichever is already likelier grows faster, as long as some worse outcome keeps
            r̄ below r, so the ordering can never flip. Any α &gt; 0 adds restoring terms of order α (from p<sup>1−α</sup> ≈ p(1 − α ln p)) that pull back
            toward the even split. The orange curve is p*(α).
          </>
        }
      />
    </Instrument>
  );
}
