/** three.js helpers shared by the 3-D instruments: scene, drag-to-orbit camera, labels. */
import * as THREE from "three";

// Colours are given in sRGB and drawn as-is (no linear/sRGB conversion).
THREE.ColorManagement.enabled = false;

export interface Orbit {
  theta: number;
  phi: number;
  radius: number;
  target: THREE.Vector3;
  auto: boolean;
  drag: boolean;
}

export interface Scene3D {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  orbit: Orbit;
  resize(): void;
  spin(dt: number): void;
  render(): void;
  dispose(): void;
}

export interface SceneOptions {
  fov?: number;
  theta?: number;
  phi?: number;
  radius?: number;
  target?: [number, number, number];
  autoRotate?: boolean;
}

export function make3D(canvas: HTMLCanvasElement, onChange: () => void, o: SceneOptions = {}): Scene3D {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.setClearColor(0x0c141f, 1);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(o.fov ?? 38, 1, 0.05, 200);
  const orbit: Orbit = {
    theta: o.theta ?? 0.9,
    phi: o.phi ?? 1.05,
    radius: o.radius ?? 6,
    target: new THREE.Vector3(...(o.target ?? [0, 0, 0])),
    auto: o.autoRotate ?? true,
    drag: false,
  };
  const place = () => {
    const { theta, phi, radius, target } = orbit;
    camera.position.set(
      target.x + radius * Math.sin(phi) * Math.cos(theta),
      target.y + radius * Math.cos(phi),
      target.z + radius * Math.sin(phi) * Math.sin(theta),
    );
    camera.lookAt(target);
  };
  place();

  let lx = 0;
  let ly = 0;
  const down = (e: PointerEvent) => {
    orbit.drag = true;
    orbit.auto = false;
    lx = e.clientX;
    ly = e.clientY;
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch {
      /* capture is optional */
    }
  };
  const move = (e: PointerEvent) => {
    if (!orbit.drag) return;
    orbit.theta += (e.clientX - lx) * 0.009;
    orbit.phi = Math.min(2.9, Math.max(0.2, orbit.phi - (e.clientY - ly) * 0.009));
    lx = e.clientX;
    ly = e.clientY;
    place();
    onChange();
  };
  const up = () => {
    orbit.drag = false;
  };
  canvas.addEventListener("pointerdown", down);
  canvas.addEventListener("pointermove", move);
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);

  // legacy-style lighting: physically based units need a factor of pi
  scene.add(new THREE.AmbientLight(0xffffff, 0.55 * Math.PI));
  const sun = new THREE.DirectionalLight(0xffffff, 0.7 * Math.PI);
  sun.position.set(3, 6, 4);
  scene.add(sun);

  return {
    renderer,
    scene,
    camera,
    orbit,
    resize() {
      const r = canvas.getBoundingClientRect();
      renderer.setSize(Math.max(10, r.width), Math.max(10, r.height), false);
      camera.aspect = r.width / Math.max(1, r.height);
      camera.updateProjectionMatrix();
    },
    spin(dt: number) {
      if (orbit.auto) {
        orbit.theta += dt * 0.12;
        place();
      }
    },
    render() {
      renderer.render(scene, camera);
    },
    dispose() {
      canvas.removeEventListener("pointerdown", down);
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up);
      canvas.removeEventListener("pointercancel", up);
      scene.traverse((obj) => {
        const m = obj as THREE.Mesh;
        if (m.geometry) m.geometry.dispose();
        const mat = m.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
        else if (mat) mat.dispose();
      });
      renderer.dispose();
    },
  };
}

/** A text label that always faces the camera. */
export function textSprite(text: string, color = "#b4c2d2", h = 0.26): THREE.Sprite {
  const c = document.createElement("canvas");
  const g = c.getContext("2d")!;
  const px = 40;
  const font = `500 ${px}px "IBM Plex Mono", ui-monospace, monospace`;
  g.font = font;
  c.width = Math.ceil(g.measureText(text).width) + 16;
  c.height = px + 16;
  g.font = font;
  g.fillStyle = color;
  g.textBaseline = "middle";
  g.fillText(text, 8, c.height / 2);
  const tex = new THREE.CanvasTexture(c);
  tex.minFilter = THREE.LinearFilter;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  sp.scale.set((h * c.width) / c.height, h, 1);
  sp.renderOrder = 10;
  return sp;
}

/** Soft round sprite used for particles and glows. */
export function dotTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, "rgba(255,255,255,1)");
  gr.addColorStop(0.35, "rgba(255,255,255,0.85)");
  gr.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = gr;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

export type P3 = [number, number, number];

export function line3(points: P3[], color: number | string, opacity = 1): THREE.Line {
  const geo = new THREE.BufferGeometry().setFromPoints(points.map((p) => new THREE.Vector3(...p)));
  const ln = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }));
  ln.frustumCulled = false;
  return ln;
}

/** Replace a line's points. setFromPoints cannot grow an existing buffer, so swap the geometry. */
export function setLinePoints(ln: THREE.Line, points: THREE.Vector3[]): void {
  ln.geometry.dispose();
  ln.geometry = new THREE.BufferGeometry().setFromPoints(points);
}
