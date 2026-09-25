/** Form controls shared by every instrument panel. */
import { Fragment, type ReactNode } from "react";
import type { Verdict } from "../engine/types";

export function Slider(props: {
  id: string;
  label: ReactNode;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  disabled?: boolean;
}) {
  const { id, label, min, max, step, value, onChange, format, disabled } = props;
  return (
    <div className="ctl">
      <label htmlFor={id}>{label}</label>
      <input
        type="range"
        id={id}
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      <output htmlFor={id}>{format ? format(value) : String(value)}</output>
    </div>
  );
}

export interface Option<T extends string> {
  value: T;
  label: string;
}

export function Segmented<T extends string>(props: {
  name: string;
  legend: ReactNode;
  options: Option<T>[];
  value: T;
  onChange: (v: T) => void;
}) {
  const { name, legend, options, value, onChange } = props;
  return (
    <fieldset className="seg">
      <legend>{legend}</legend>
      {options.map((o, i) => (
        <Fragment key={o.value}>
          <input
            type="radio"
            name={name}
            id={`${name}-${i}`}
            value={o.value}
            checked={value === o.value}
            onChange={() => onChange(o.value)}
          />
          <label htmlFor={`${name}-${i}`}>{o.label}</label>
        </Fragment>
      ))}
    </fieldset>
  );
}

export function Toggle(props: { id: string; label: ReactNode; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="switch" htmlFor={props.id}>
      <input type="checkbox" id={props.id} checked={props.checked} onChange={(e) => props.onChange(e.target.checked)} />
      {props.label}
    </label>
  );
}

export function VerdictChip({ v }: { v: Verdict | string | undefined }) {
  if (!v || typeof v === "string") return <span className="verdict">{v ?? "—"}</span>;
  const cls = v.ok === true ? "verdict good" : v.ok === false ? "verdict bad" : "verdict";
  return <span className={cls}>{v.text}</span>;
}

export function Readouts({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className="readouts">
      {items.map(([k, v], i) => (
        <div key={i}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Button(props: { onClick: () => void; children: ReactNode; id?: string }) {
  return (
    <button className="btn" type="button" id={props.id} onClick={props.onClick}>
      {props.children}
    </button>
  );
}

/** Plain-text readout value from an engine, or a dash. */
export function text(v: string | Verdict | undefined): string {
  if (v === undefined) return "—";
  return typeof v === "string" ? v : v.text;
}
