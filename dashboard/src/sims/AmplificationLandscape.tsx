/** log10|R_n(z)| as 3-D terrain: the stable crater grows toward e^z as the method order rises. */
import { useMemo, useState } from "react";
import * as THREE from "three";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { useLinkedAlpha } from "../engine/useLinkedAlpha";
import { Readouts, Slider, VerdictChip } from "../components/controls";
import { Instrument, Notes, Stage } from "../components/Instrument";
import { lambdaK2 } from "../lib/model";
import { ampPoly } from "../lib/numerics";
import { C, RAMP_WARM, ramp } from "../lib/color";
import { line3, make3D, textSprite, type Scene3D } from "../lib/three3d";
import { fsig } from "../lib/format";

interface Params {
  n: number;
  alpha: number;
  h: number;
}
const NX = 96;
const NY = 80;
const XR = [-5, 1.5];
const YR = [-3.6, 3.6];
const SC = 0.7;
const HS = 0.55;
export const ORDER_NAMES = ["", "Euler", "RK2", "RK3", "RK4", "order 5", "order 6", "exact eᶻ"];
const wx = (x: number) => (x - (XR[0] + XR[1]) / 2) * SC;
const wz = (y: number) => -y * SC;
const lg = (v: number) => Math.max(-2.5, Math.min(2.0, Math.log10(Math.max(v, 1e-12))));
/** |R_n(z)| on the real axis, or |e^z| for the exact flow (n = 7). */
export function amplification(z: number, n: number): number {
  return n >= 7 ? Math.exp(z) : ampPoly([z, 0], n);
}

function createEngine(): Engine<Params> {
  let host!: EngineHost;
  let T3!: Scene3D;
  let mesh!: THREE.Mesh;
  let marker!: THREE.Mesh;
  let stem!: THREE.Line;
  const cnt = (NX + 1) * (NY + 1);
  const cur = new Float32Array(cnt);
  const tgt = new Float32Array(cnt);
  let first = true;
  let lastN = -1;

  function applyHeights() {
    const P = mesh.geometry.attributes.position.array as Float32Array;
    const Cc = mesh.geometry.attributes.color.array as Float32Array;
    for (let k = 0; k < cnt; k++) {
      const v = cur[k];
      P[k * 3 + 1] = v * HS;
      const c = v < 0 ? ramp(0.25 + 0.75 * Math.min(1, -v / 2.5)) : ramp(0.2 + 0.8 * Math.min(1, v / 2), RAMP_WARM);
      Cc[k * 3] = c[0] / 255;
      Cc[k * 3 + 1] = c[1] / 255;
      Cc[k * 3 + 2] = c[2] / 255;
    }
    mesh.geometry.attributes.position.needsUpdate = true;
    mesh.geometry.attributes.color.needsUpdate = true;
    mesh.geometry.computeVertexNormals();
  }

  return {
    init(hst) {
      host = hst;
      T3 = make3D(host.canvases[0], host.invalidate, { radius: 7.8, phi: 0.95, theta: 1.25, target: [0, -0.2, 0], autoRotate: !host.reducedMotion });
      const geo = new THREE.BufferGeometry();
      const P = new Float32Array(cnt * 3);
      for (let j = 0; j <= NY; j++)
        for (let i = 0; i <= NX; i++) {
          const k = j * (NX + 1) + i;
          P[k * 3] = wx(XR[0] + ((XR[1] - XR[0]) * i) / NX);
          P[k * 3 + 2] = wz(YR[0] + ((YR[1] - YR[0]) * j) / NY);
        }
      geo.setAttribute("position", new THREE.BufferAttribute(P, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(cnt * 3), 3));
      const idx: number[] = [];
      for (let j = 0; j < NY; j++)
        for (let i = 0; i < NX; i++) {
          const a0 = j * (NX + 1) + i;
          const c0 = a0 + NX + 1;
          idx.push(a0, c0, a0 + 1, a0 + 1, c0, c0 + 1);
        }
      geo.setIndex(idx);
      mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
      mesh.frustumCulled = false;
      T3.scene.add(mesh);
      const plane = new THREE.Mesh(
        new THREE.PlaneGeometry((XR[1] - XR[0]) * SC, (YR[1] - YR[0]) * SC),
        new THREE.MeshBasicMaterial({ color: 0x56b4e9, transparent: true, opacity: 0.14, side: THREE.DoubleSide, depthWrite: false }),
      );
      plane.rotation.x = -Math.PI / 2;
      T3.scene.add(plane);
      T3.scene.add(line3([[wx(XR[0]), 0.005, 0], [wx(XR[1]), 0.005, 0]], 0xb4c2d2, 0.5));
      [-4, -2, 0].forEach((v) => {
        const sp = textSprite(String(v).replace("-", "−"), C.muted, 0.2);
        sp.position.set(wx(v), 0.02, wz(YR[0]) + 0.35);
        T3.scene.add(sp);
      });
      const lr = textSprite("Re z", C.ink2, 0.22);
      lr.position.set(wx(XR[1]) + 0.35, 0.02, 0);
      const li = textSprite("Im z", C.ink2, 0.22);
      li.position.set(wx(0), 0.02, wz(YR[1]) - 0.3);
      const l1 = textSprite("|R| = 1", C.sky, 0.2);
      l1.position.set(wx(XR[0]) - 0.1, 0.1, wz(YR[1]));
      T3.scene.add(lr, li, l1);
      marker = new THREE.Mesh(new THREE.SphereGeometry(0.075, 20, 14), new THREE.MeshBasicMaterial({ color: 0x3cc49a }));
      stem = line3([[0, 0, 0], [0, 1, 0]], 0xf0e442);
      T3.scene.add(marker, stem);
    },
    setParams(p) {
      if (p.n !== lastN) {
        lastN = p.n;
        for (let j = 0; j <= NY; j++)
          for (let i = 0; i <= NX; i++) {
            const x = XR[0] + ((XR[1] - XR[0]) * i) / NX;
            const y = YR[0] + ((YR[1] - YR[0]) * j) / NY;
            tgt[j * (NX + 1) + i] = p.n >= 7 ? lg(Math.exp(x)) : lg(ampPoly([x, y], p.n));
          }
        if (first || host.reducedMotion) {
          cur.set(tgt);
          applyHeights();
          first = false;
        }
      }
      const z = -p.h * lambdaK2(4, 1, p.alpha);
      const amp = amplification(z, p.n);
      const xm = Math.max(XR[0], z);
      marker.position.set(wx(xm), lg(amp) * HS, 0);
      stem.geometry.setFromPoints([new THREE.Vector3(wx(xm), 0, 0), new THREE.Vector3(wx(xm), lg(amp) * HS, 0)]);
      (marker.material as THREE.MeshBasicMaterial).color.set(amp < 1 ? C.green : C.verm);
    },
    resize() {
      T3.resize();
    },
    step(dt) {
      T3.spin(dt);
      let moved = false;
      const k = Math.min(1, dt * 5);
      for (let i = 0; i < cnt; i++) {
        const d = tgt[i] - cur[i];
        if (Math.abs(d) > 1e-4) {
          cur[i] += d * k;
          moved = true;
        }
      }
      if (moved) applyHeights();
    },
    draw() {
      T3.render();
    },
    dispose() {
      T3.dispose();
    },
  };
}

export function AmplificationLandscape() {
  const [n, setN] = useState(4);
  const [alpha, setAlpha] = useLinkedAlpha("amp", 2, 0.5, 3);
  const [lgH, setLgH] = useState(-0.523);
  const h = Math.pow(10, lgH);
  const params = useMemo(() => ({ n, alpha, h }), [n, alpha, h]);
  const handle = useEngine(createEngine, params);
  const z = -h * lambdaK2(4, 1, alpha);
  const amp = amplification(z, n);
  const ok = amp < 1;

  return (
    <Instrument
      id="amp"
      chips={["Exp 1 Fig 3", "Exp 2 Fig 3"]}
      ghostChips={["3-D"]}
      title="Amplification landscape"
      claim="log₁₀|R(z)| over the complex plane. The crater below the glowing plane is where a method is stable, and it grows toward the exact flow eᶻ as the order rises."
      stage={
        <Stage
          handle={handle}
          aspect="4 / 3"
          grab
          hint="drag to rotate"
          hud={`${ORDER_NAMES[n]}   z = −hλ = ${z.toFixed(3)}\nblue crater: |R| < 1 (stable)`}
          label="A 3-D surface of the logarithm of the amplification factor over the complex plane, with a plane at zero and a marker for the current step."
        />
      }
    >
      <Slider id="amp-n" label="Method order" min={1} max={7} step={1} value={n} onChange={setN} format={(v) => ORDER_NAMES[v]} />
      <Slider id="amp-a" label="Exponent α" min={0.5} max={3} step={0.01} value={alpha} onChange={setAlpha} format={(v) => v.toFixed(2)} />
      <Slider id="amp-h" label="Step size h" min={-2} max={0} step={0.005} value={lgH} onChange={setLgH} format={(v) => Math.pow(10, v).toFixed(3)} />
      <Readouts
        items={[
          ["|R(−hλ)|", fsig(amp, 4) + (z < XR[0] ? "  (off the map)" : "")],
          ["this step", <VerdictChip v={{ ok, text: ok ? "stable: |R| < 1" : "unstable: |R| ≥ 1" }} />],
        ]}
      />
      <Notes
        tryThis={[
          "Step the order from 1 (Euler) to 4 (RK4): the stable crater widens, reaching 2.785 on the real axis.",
          "Go to “exact eᶻ”: the whole left half-plane is stable. The true flow never blows up; only the discretisation does.",
        ]}
        math={
          <>
            On the test equation y′ = λy, an order-n Runge–Kutta step multiplies by the Taylor polynomial R<sub>n</sub>(z) = Σ<sub>k≤n</sub> z<sup>k</sup>/k! of
            e<sup>z</sup>, with z = hλ. Euler is n = 1 and classical RK4 is n = 4, so a higher order is a better copy of the exponential.
          </>
        }
      />
    </Instrument>
  );
}
