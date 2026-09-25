/** 3-D surface of alpha_c(G, rho) cut by a plane at the chosen alpha (Exp 2 Fig 4c, Exp 4 Fig 1). */
import { useMemo, useState } from "react";
import * as THREE from "three";
import type { Engine, EngineHost } from "../engine/types";
import { useEngine } from "../engine/useEngine";
import { useLinkedAlpha } from "../engine/useLinkedAlpha";
import { Readouts, Slider, VerdictChip } from "../components/controls";
import { Instrument, Notes, Stage } from "../components/Instrument";
import { criticalAlpha } from "../lib/finiteGroup";
import { C, ramp } from "../lib/color";
import { line3, make3D, setLinePoints, textSprite, type Scene3D } from "../lib/three3d";
import { fsig } from "../lib/format";

interface Params {
  alpha: number;
  eps: number;
  lgSel: number;
  rhoSel: number;
}
const NX = 64;
const NZ = 48;
const XR = [-2.2, 2.2];
const ZR = [-1.7, 1.7];
const Y0 = -1.1;
const YS = 0.85;
const ACMAX = 3.3;
const xOf = (lg: number) => XR[0] + ((lg - 1) / 7) * (XR[1] - XR[0]);
const zOf = (rho: number) => ZR[0] + ((rho - 1) / 9) * (ZR[1] - ZR[0]);
const yOf = (ac: number) => Y0 + Math.min(ac, ACMAX) * YS;
const GREEN = [[20, 70, 60], [40, 150, 120], [60, 196, 154], [190, 245, 225]] as [number, number, number][];
const SLATE = [[24, 32, 46], [52, 66, 90], [90, 104, 128]] as [number, number, number][];

function createEngine(): Engine<Params> {
  let host!: EngineHost;
  let T3!: Scene3D;
  let mesh!: THREE.Mesh;
  let plane!: THREE.Mesh;
  let cut!: THREE.Line;
  let marker!: THREE.Mesh;
  let stem!: THREE.Line;

  function rebuild(p: Params) {
    const { alpha: a, eps, lgSel, rhoSel } = p;
    const posA = mesh.geometry.attributes.position.array as Float32Array;
    const colA = mesh.geometry.attributes.color.array as Float32Array;
    for (let j = 0; j <= NZ; j++)
      for (let i = 0; i <= NX; i++) {
        const lg = 1 + (7 * i) / NX;
        const rho = 1 + (9 * j) / NZ;
        const ac = criticalAlpha(rho, Math.pow(2, lg), eps);
        const o = (j * (NX + 1) + i) * 3;
        posA[o] = xOf(lg);
        posA[o + 1] = yOf(ac);
        posA[o + 2] = zOf(rho);
        const c = ac < a ? ramp(0.35 + 0.6 * Math.min(1, (a - ac) / Math.max(a, 0.3)), GREEN) : ramp(0.15 + 0.35 * Math.min(1, ac / ACMAX), SLATE);
        colA[o] = c[0] / 255;
        colA[o + 1] = c[1] / 255;
        colA[o + 2] = c[2] / 255;
      }
    mesh.geometry.attributes.position.needsUpdate = true;
    mesh.geometry.attributes.color.needsUpdate = true;
    mesh.geometry.computeVertexNormals();
    plane.position.y = yOf(a);
    const pts: THREE.Vector3[] = [];
    for (let k = 0; k <= 90; k++) {
      const rho = 1 + (9 * k) / 90;
      const Gs = Math.pow(rho, 1 / a);
      if (Gs >= 2 && Gs <= 256 && Gs <= 1 / eps) pts.push(new THREE.Vector3(xOf(Math.log2(Gs)), yOf(a) + 0.01, zOf(rho)));
    }
    cut.visible = pts.length > 1;
    if (pts.length > 1) setLinePoints(cut, pts);
    const ac = criticalAlpha(rhoSel, Math.pow(2, lgSel), eps);
    const ok = a > ac;
    marker.position.set(xOf(lgSel), yOf(a), zOf(rhoSel));
    (marker.material as THREE.MeshBasicMaterial).color.set(ok ? C.green : C.verm);
    stem.geometry.setFromPoints([new THREE.Vector3(xOf(lgSel), yOf(ac), zOf(rhoSel)), new THREE.Vector3(xOf(lgSel), yOf(a), zOf(rhoSel))]);
    (stem.material as THREE.LineBasicMaterial).color.set(ok ? C.green : C.verm);
  }

  return {
    init(h) {
      host = h;
      T3 = make3D(host.canvases[0], host.invalidate, { radius: 9.0, phi: 1.02, theta: -2.2, target: [0, 0.1, 0], autoRotate: !host.reducedMotion });
      const geo = new THREE.BufferGeometry();
      const idx: number[] = [];
      geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array((NX + 1) * (NZ + 1) * 3), 3));
      geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array((NX + 1) * (NZ + 1) * 3), 3));
      for (let j = 0; j < NZ; j++)
        for (let i = 0; i < NX; i++) {
          const a0 = j * (NX + 1) + i;
          const c0 = a0 + NX + 1;
          idx.push(a0, c0, a0 + 1, a0 + 1, c0, c0 + 1);
        }
      geo.setIndex(idx);
      mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
      mesh.frustumCulled = false;
      T3.scene.add(mesh);
      plane = new THREE.Mesh(
        new THREE.PlaneGeometry(XR[1] - XR[0] + 0.3, ZR[1] - ZR[0] + 0.3),
        new THREE.MeshBasicMaterial({ color: 0x56b4e9, transparent: true, opacity: 0.17, side: THREE.DoubleSide, depthWrite: false }),
      );
      plane.rotation.x = -Math.PI / 2;
      T3.scene.add(plane);
      cut = line3([[0, 0, 0], [0, 0, 0]], 0xf0e442);
      marker = new THREE.Mesh(new THREE.SphereGeometry(0.07, 20, 14), new THREE.MeshBasicMaterial({ color: 0x3cc49a }));
      stem = line3([[0, 0, 0], [0, 1, 0]], 0x3cc49a);
      T3.scene.add(cut, marker, stem);
      const base = [
        [XR[0], Y0, ZR[0]], [XR[1], Y0, ZR[0]],
        [XR[0], Y0, ZR[0]], [XR[0], Y0, ZR[1]],
        [XR[0], Y0, ZR[0]], [XR[0], Y0 + ACMAX * YS, ZR[0]],
      ].map((p) => new THREE.Vector3(p[0], p[1], p[2]));
      T3.scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(base), new THREE.LineBasicMaterial({ color: 0x3a4d6b })));
      [2, 4, 16, 64, 256].forEach((G) => {
        const sp = textSprite(`G=${G}`, C.muted, 0.16);
        sp.position.set(xOf(Math.log2(G)), Y0 - 0.12, ZR[0] - 0.28);
        T3.scene.add(sp);
      });
      [1, 2, 5, 10].forEach((rho) => {
        const sp = textSprite(`ρ=${rho}`, C.muted, 0.16);
        sp.position.set(XR[0] - 0.42, Y0 - 0.1, zOf(rho));
        T3.scene.add(sp);
      });
      [1, 2, 3].forEach((v) => {
        const sp = textSprite(`αc = ${v}`, C.muted, 0.16);
        sp.position.set(XR[0] - 0.4, yOf(v), ZR[0] - 0.1);
        T3.scene.add(sp);
      });
    },
    setParams(p) {
      rebuild(p);
    },
    resize() {
      T3.resize();
    },
    step(dt) {
      T3.spin(dt);
    },
    draw() {
      T3.render();
    },
    dispose() {
      T3.dispose();
    },
  };
}

export function SurvivalSurface() {
  const [alpha, setAlpha] = useLinkedAlpha("surface", 1, 0.05, 3);
  const [lgE, setLgE] = useState(-3);
  const [lgSel, setLgSel] = useState(4);
  const [rhoSel, setRhoSel] = useState(4);
  const eps = Math.pow(10, lgE);
  const params = useMemo(() => ({ alpha, eps, lgSel, rhoSel }), [alpha, eps, lgSel, rhoSel]);
  const handle = useEngine(createEngine, params);
  const G = Math.pow(2, lgSel);
  const ac = criticalAlpha(rhoSel, G, eps);
  const ok = alpha > ac;

  return (
    <Instrument
      id="surface"
      chips={["Exp 2 Fig 4c", "Exp 4 Fig 1"]}
      ghostChips={["3-D"]}
      title="Survival surface"
      claim="The height is the smallest α that keeps a weaker outcome alive. Everything under the glowing plane survives."
      stage={
        <Stage
          handle={handle}
          aspect="4 / 3"
          grab
          hint="drag to rotate"
          hud={`α = ${alpha.toFixed(2)}   ε = ${fsig(eps, 2)}\ngreen: survives   slate: dies`}
          label="A 3-D surface of the critical exponent over group size and reward ratio, cut by a horizontal plane at the chosen alpha."
        />
      }
    >
      <Slider id="surf-a" label="Exponent α" min={0.05} max={3} step={0.01} value={alpha} onChange={setAlpha} format={(v) => v.toFixed(2)} />
      <Slider id="surf-e" label="Clip ε" min={-3} max={-0.3} step={0.01} value={lgE} onChange={setLgE} format={(v) => fsig(Math.pow(10, v), 2)} />
      <Slider id="surf-g" label="Your G" min={1} max={8} step={0.05} value={lgSel} onChange={setLgSel} format={(v) => String(Math.round(Math.pow(2, v)))} />
      <Slider id="surf-rho" label="Your ratio ρ" min={1} max={10} step={0.05} value={rhoSel} onChange={setRhoSel} format={(v) => v.toFixed(2)} />
      <Readouts
        items={[
          [<>needed α<sub>c</sub></>, ac.toFixed(3)],
          ["your point", <VerdictChip v={{ ok, text: ok ? "weak outcome survives" : "weak outcome dies" }} />],
          ["energy gap vs T·S", `ln ρ = ${Math.log(rhoSel).toFixed(2)} ${ok ? "<" : "≥"} α·ln G = ${(alpha * Math.log(Math.min(G, 1 / eps))).toFixed(2)}`],
        ]}
      />
      <Notes
        tryThis={[
          "Lower α: the plane sinks and the surviving region (green) shrinks toward large groups and small ratios.",
          "Raise ε: past G = 1/ε the surface flattens into a plateau, because extra samples no longer raise the ceiling.",
        ]}
        math={
          <>
            α<sub>c</sub> = ln ρ / ln min(G, 1/ε). The glowing curve where the plane cuts the surface is G = ρ<sup>1/α</sup>: the smallest group
            that protects ratio ρ.
          </>
        }
        analogy="Rewrite survival as ln ρ < α·ln G. With energy gap ΔE = ln ρ, temperature T = α and entropy S = ln G (G equally likely draws), a weaker outcome survives when ΔE < T·S, the same form as a free-energy balance."
      />
    </Instrument>
  );
}
