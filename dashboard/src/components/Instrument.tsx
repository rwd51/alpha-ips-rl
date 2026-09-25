/** Layout shared by every instrument: header, stage, controls, notes. */
import type { ReactNode } from "react";
import type { EngineHandle } from "../engine/useEngine";

export function Stage(props: {
  handle: EngineHandle;
  canvases?: number;
  aspect: string;
  label: string;
  hud?: string;
  hint?: string;
  grab?: boolean;
  playOnTop?: boolean;
  noPlay?: boolean;
}) {
  const { handle, canvases = 1, aspect, label, hud, hint, grab, playOnTop, noPlay } = props;
  return (
    <div className="stage" ref={handle.stageRef} style={{ aspectRatio: aspect }}>
      {Array.from({ length: canvases }, (_, i) => {
        const main = i === canvases - 1;
        return (
          <canvas
            key={i}
            ref={handle.setCanvas(i)}
            className={grab ? "grab" : undefined}
            role={main ? "img" : undefined}
            aria-label={main ? label : undefined}
            aria-hidden={main ? undefined : true}
          />
        );
      })}
      {hud ? <div className="hud">{hud}</div> : null}
      {hint ? <span className="hint">{hint}</span> : null}
      {noPlay ? null : (
        <button
          className="play"
          type="button"
          style={playOnTop ? { top: 10, bottom: "auto" } : undefined}
          aria-pressed={!handle.playing}
          onClick={handle.togglePlay}
        >
          {handle.playing ? "Pause" : "Play"}
        </button>
      )}
    </div>
  );
}

export function Notes(props: { tryThis?: ReactNode[]; math?: ReactNode; analogy?: ReactNode; children?: ReactNode }) {
  return (
    <div className="notes">
      {props.tryThis ? (
        <>
          <h4>Try this</h4>
          <ul>
            {props.tryThis.map((t, i) => (
              <li key={i}>{t}</li>
            ))}
          </ul>
        </>
      ) : null}
      {props.math ? (
        <>
          <h4>The math</h4>
          <p>{props.math}</p>
        </>
      ) : null}
      {props.children}
      {props.analogy ? (
        <p className="analogy">
          <b>Analogy.</b> {props.analogy}
        </p>
      ) : null}
    </div>
  );
}

export function Instrument(props: {
  id: string;
  chips: string[];
  ghostChips?: string[];
  title: string;
  claim: ReactNode;
  stage: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="inst" id={props.id}>
      <header className="inst-head">
        <div className="prov">
          {props.chips.map((c) => (
            <span className="chip" key={c}>
              {c}
            </span>
          ))}
          {(props.ghostChips ?? []).map((c) => (
            <span className="chip ghost" key={c}>
              {c}
            </span>
          ))}
        </div>
        <h3>{props.title}</h3>
        <p className="claim">{props.claim}</p>
      </header>
      <div className="inst-body">
        {props.stage}
        <div className="panel">{props.children}</div>
      </div>
    </section>
  );
}

export function Group(props: { id: string; num: string; title: string; intro: ReactNode; children: ReactNode }) {
  return (
    <section className="group" id={props.id}>
      <div className="group-head">
        <span className="num">{props.num}</span>
        <h2>{props.title}</h2>
        <p>{props.intro}</p>
      </div>
      {props.children}
    </section>
  );
}
