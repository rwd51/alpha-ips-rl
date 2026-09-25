/**
 * Every instrument is an imperative Engine (canvas drawing, simulation state)
 * wrapped by a React component that owns the controls.
 *
 * The component passes plain parameters; the engine reports readouts back.
 */
export interface Verdict {
  ok: boolean | null;
  text: string;
}

export type Readouts = Record<string, string | Verdict>;

export interface EngineHost {
  /** The stage's canvases, in the order the component rendered them. */
  canvases: HTMLCanvasElement[];
  /** Report readouts to the React panel (batched, ~10 updates per second). */
  emit(r: Readouts): void;
  /** Request a redraw even while paused (e.g. after a drag). */
  invalidate(): void;
  /** True when the viewer asked the OS to reduce motion. */
  reducedMotion: boolean;
}

export interface Engine<P> {
  init(host: EngineHost): void;
  /** Called on mount and whenever a parameter changes (shallow comparison). */
  setParams(p: P): void;
  resize(): void;
  step(dt: number): void;
  draw(): void;
  dispose(): void;
}
