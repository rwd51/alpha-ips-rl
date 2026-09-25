/** Page chrome: sticky top bar, instrument index and footer. */
import { useLab } from "../engine/LabContext";

const NAV = [
  ["#g-dial", "Dial"],
  ["#g-groups", "Groups"],
  ["#g-numerics", "Numerics"],
  ["#g-estimators", "Estimators"],
  ["#g-analogies", "Analogies"],
];

export function TopBar() {
  const { paused, setPaused, linkAlpha, setLinkAlpha } = useLab();
  return (
    <header className="topbar">
      <div className="wrap">
        <a className="brand" href="#top">
          <i>α</i>-IPS Observatory
        </a>
        <nav className="topnav" aria-label="Sections">
          {NAV.map(([href, name]) => (
            <a key={href} href={href}>
              {name}
            </a>
          ))}
        </nav>
        <div className="topctl">
          <label className="switch" htmlFor="link-alpha">
            <input type="checkbox" id="link-alpha" checked={linkAlpha} onChange={(e) => setLinkAlpha(e.target.checked)} /> Link α across instruments
          </label>
          <button className="btn" type="button" aria-pressed={paused} onClick={() => setPaused(!paused)}>
            {paused ? "Resume all" : "Pause all"}
          </button>
        </div>
      </div>
    </header>
  );
}

const INDEX: [string, string, string][] = [
  ["tetra", "Four outcomes, one tetrahedron", "3-D · Exp 2 · tie race"],
  ["race", "The outcome race", "Exp 1–2 · tied tiers"],
  ["balls", "Balls into bins", "Exp 1 Fig 2 · Exp 2 Fig 4a"],
  ["ceiling", "The weight ceiling", "Exp 2–3 · Exp 5 Fig 3"],
  ["surface", "Survival surface", "3-D · Exp 2 Fig 4 · Exp 4"],
  ["atlas", "Phase atlas", "Exp 4 Figs 1–2"],
  ["stab", "Euler vs RK4", "Exp 1 Fig 3 · Exp 2 Fig 3"],
  ["amp", "Amplification landscape", "3-D · numerics"],
  ["newton", "Newton's tangent hops", "Exp 3 Fig 1"],
  ["est", "Estimator lab", "Exp 5 Figs 1–2"],
  ["ema", "The EMA oscillator", "Exp 5 Fig 5"],
  ["ifd", "Foragers and patches", "analogy · ecology"],
];

export function InstrumentIndex() {
  return (
    <nav className="index" aria-label="All instruments">
      {INDEX.map(([id, name, prov]) => (
        <a key={id} href={`#${id}`}>
          <b>{name}</b>
          <small>{prov}</small>
        </a>
      ))}
    </nav>
  );
}

export function Footer() {
  return (
    <footer>
      <div>
        Built on the <code>alpha-ips-rl</code> repository: equation (*) from <code>derivation.md</code>, the finite-group mean field from{" "}
        <code>src/finite_group.py</code> and <code>src/finite_group_general.py</code> (the five-outcome atlas is precomputed with it), estimator rules
        from <code>src/estimators.py</code>. Everything else is computed live in your browser.
      </div>
      <div>Base paper: Sinha, Elango &amp; Liu (2026), arXiv:2601.21669. Colours: Okabe–Ito, as in the project's figures.</div>
    </footer>
  );
}
