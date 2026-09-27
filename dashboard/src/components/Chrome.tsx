/** Page chrome: sticky top bar, presenter map and footer. */
import { useLab } from "../engine/LabContext";
import { PARTS } from "../presenters";

const NAV = PARTS.map((p) => [`#${p.id}`, p.label === "Overall" ? "Overall" : p.label.replace("Part ", "P")]);

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

/** Who presents which part, and the demos each speaker can open, in slide order. */
export function PresenterMap() {
  return (
    <nav className="pmap" aria-label="Presenters and their demos">
      {PARTS.map((p) => (
        <div className="pcard" key={p.id}>
          <a className="phead" href={`#${p.id}`}>
            <small>
              {p.label} · {p.slides}
            </small>
            <b>{p.title}</b>
            <span>
              {p.presenter} <em>{p.roll}</em>
            </span>
          </a>
          <ol>
            {p.demos.map(([id, name]) => (
              <li key={id}>
                <a href={`#${id}`}>{name}</a>
              </li>
            ))}
          </ol>
        </div>
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
