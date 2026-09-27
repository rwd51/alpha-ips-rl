/** The closing slide's cross-cutting result as jump links: every part meets the finite-group ceiling from a different side. */

const CARDS: [string, string, string, string][] = [
  ["Part 2 · finds it", "#ceiling", "w_G(p) → min(G, 1/ε)^α", "A group of G sees a rare outcome at most once, so the boost stops growing. Survival needs α > ln ρ / ln min(G, 1/ε)."],
  ["Part 3 · sharpens it", "#nested", "α_c(O5) = 0.886, not 0.580", "With five outcomes competing, the weakest needs a larger α than the two-outcome formula. At G ≤ 5 no α keeps it."],
  ["Part 4 · verifies it", "#surface", "0 mismatches, 19,120 points", "The sign of m = α ln min(G, 1/ε) − ln ρ decides survival across the whole four-parameter atlas. The 5 points exactly on m = 0 are extinct, as the strict inequality says."],
  ["Part 5 · explains it", "#distort", "ω(1)/ω(G) > ρ", "Only the rule's dynamic range decides survival. At α = 1 the curve is exactly (1 − (1 − p)^G)/p."],
];

export function CeilingRecap() {
  return (
    <section className="inst" id="recap">
      <header className="inst-head">
        <div className="prov">
          <span className="chip">All parts</span>
          <span className="chip ghost">closing slides</span>
        </div>
        <h3>One ceiling, four views</h3>
        <p className="claim">
          A finite group cannot apply an unbounded correction. Each card jumps to the demo where that part meets the ceiling; use them in order during the
          closing.
        </p>
      </header>
      <nav className="recap" aria-label="The ceiling in each part">
        {CARDS.map(([tag, href, formula, body]) => (
          <a key={href} href={href}>
            <small>{tag}</small>
            <b>
              <code>{formula}</code>
            </b>
            <span>{body}</span>
          </a>
        ))}
      </nav>
      <ol className="takeaways">
        <li>
          Pick G first. If min(G, 1/ε) is below r<sub>max</sub>/r<sub>min</sub>, no exponent and no amount of training keeps the weak outcomes (
          <a href="#ceiling">weight ceiling</a>).
        </li>
        <li>
          Do not set ε above 1/G without checking: it adds a second, independent ceiling (<a href="#surface">survival surface</a>, raise ε).
        </li>
        <li>
          For K outcomes the pairwise threshold is optimistic: at G = 16 the weakest of five needs α = 0.886, not 0.580 (<a href="#nested">nested solver</a>).
        </li>
        <li>
          A moving average buys the variance of a 19× larger group at β = 0.1, but training rings if β &lt; 4hλ (<a href="#ema">EMA oscillator</a>).
        </li>
      </ol>
    </section>
  );
}
