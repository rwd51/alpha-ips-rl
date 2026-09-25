import { useCallback, useEffect, useRef, useState } from "react";
import { useLab } from "./LabContext";
import type { Engine, EngineHost, Readouts } from "./types";

export const REDUCED_MOTION =
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

function shallowEqual(a: object, b: object): boolean {
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => (a as Record<string, unknown>)[k] === (b as Record<string, unknown>)[k]);
}

/**
 * Mounts an Engine on the stage's canvases and drives it:
 *  - animates only while the stage is on screen (IntersectionObserver)
 *  - resizes with the stage (ResizeObserver)
 *  - honours the instrument's Play/Pause button and the global "Pause all"
 *  - forwards parameter changes and batches readouts back into React state
 */
export function useEngine<P extends object>(create: () => Engine<P>, params: P, canvasCount = 1) {
  const canvases = useRef<(HTMLCanvasElement | null)[]>([]);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const engineRef = useRef<Engine<P> | null>(null);
  const paramsRef = useRef(params);
  const lastParams = useRef<P | null>(null);
  const [readouts, setReadouts] = useState<Readouts>({});
  const [playing, setPlaying] = useState(!REDUCED_MOTION);
  const { paused } = useLab();
  const live = useRef({ playing, paused, dirty: true, visible: false });
  live.current.playing = playing;
  live.current.paused = paused;
  paramsRef.current = params;

  useEffect(() => {
    const eng = create();
    engineRef.current = eng;
    let pending: Readouts = {};
    let timer = 0;
    const flush = () => {
      const batch = pending;
      pending = {};
      timer = 0;
      setReadouts((prev) => ({ ...prev, ...batch }));
    };
    const host: EngineHost = {
      canvases: canvases.current.slice(0, canvasCount).filter((c): c is HTMLCanvasElement => c !== null),
      emit(r) {
        Object.assign(pending, r);
        if (!timer) timer = window.setTimeout(flush, 90);
      },
      invalidate() {
        live.current.dirty = true;
      },
      reducedMotion: !!REDUCED_MOTION,
    };
    eng.init(host);
    eng.setParams(paramsRef.current);
    lastParams.current = paramsRef.current;
    eng.resize();
    eng.draw();

    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          live.current.visible = e.isIntersecting;
          if (e.isIntersecting) live.current.dirty = true;
        }
      },
      { rootMargin: "120px" },
    );
    const ro = new ResizeObserver(() => {
      eng.resize();
      live.current.dirty = true;
    });
    if (stageRef.current) {
      io.observe(stageRef.current);
      ro.observe(stageRef.current);
    }

    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const L = live.current;
      if (L.visible) {
        const run = L.playing && !L.paused;
        if (run || L.dirty) {
          try {
            if (run) eng.step(dt);
            eng.draw();
          } catch (err) {
            console.error(err);
            setPlaying(false);
          }
          L.dirty = false;
        }
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      io.disconnect();
      ro.disconnect();
      window.clearTimeout(timer);
      eng.dispose();
      engineRef.current = null;
    };
    // the engine is created once per mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const eng = engineRef.current;
    if (!eng) return;
    if (lastParams.current && shallowEqual(lastParams.current, params)) return;
    lastParams.current = params;
    eng.setParams(params);
    live.current.dirty = true;
  });

  const setCanvas = useCallback(
    (i: number) => (el: HTMLCanvasElement | null) => {
      canvases.current[i] = el;
    },
    [],
  );
  const togglePlay = useCallback(() => setPlaying((p) => !p), []);

  return { stageRef, setCanvas, readouts, playing, togglePlay };
}

export type EngineHandle = ReturnType<typeof useEngine>;
