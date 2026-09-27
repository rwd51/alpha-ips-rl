/**
 * Who presents what: the defence deck (presentation/main.tex) in slide order,
 * with the instruments each speaker can demo. The top bar, the presenter map
 * and the section headers are all built from this list.
 */
export interface Part {
  id: string;
  label: string;
  slides: string;
  title: string;
  presenter: string;
  roll: string;
  /** [instrument id, instrument title] */
  demos: [string, string][];
}

export const PARTS: Part[] = [
  {
    id: "part1",
    label: "Part 1",
    slides: "slides 9–12",
    title: "Collapse baseline",
    presenter: "Ruwad Naswan",
    roll: "2105051",
    demos: [
      ["race", "The outcome race"],
      ["balls", "Balls into bins"],
      ["clock", "The collapse clock"],
    ],
  },
  {
    id: "part2",
    label: "Part 2",
    slides: "slides 13–19",
    title: "The diversity exponent",
    presenter: "Mohammad Raihan Rashid",
    roll: "2105046",
    demos: [
      ["tetra", "Four outcomes, one tetrahedron"],
      ["stab", "Euler vs RK4"],
      ["amp", "Amplification landscape"],
      ["ceiling", "The weight ceiling"],
    ],
  },
  {
    id: "part3",
    label: "Part 3",
    slides: "slides 20–23",
    title: "Root finding",
    presenter: "Sadia Binte Sayeed",
    roll: "2105045",
    demos: [
      ["roots", "The root-finder race"],
      ["newton", "Newton's tangent hops"],
      ["nested", "The nested solver"],
    ],
  },
  {
    id: "part4",
    label: "Part 4",
    slides: "slides 24–27",
    title: "Parameter atlas",
    presenter: "Md. Mehedi Hasan",
    roll: "2105052",
    demos: [
      ["surface", "Survival surface"],
      ["atlas", "Phase atlas"],
      ["interp", "Can a grid replace the solver?"],
    ],
  },
  {
    id: "part5",
    label: "Part 5",
    slides: "slides 28–32",
    title: "Estimator error",
    presenter: "Ahnaf Tahmid",
    roll: "2105041",
    demos: [
      ["est", "Estimator lab"],
      ["distort", "What the dynamics actually see"],
      ["ema", "The EMA oscillator"],
    ],
  },
  {
    id: "closing",
    label: "Overall",
    slides: "slides 33–36",
    title: "Overall findings",
    presenter: "Mohammad Raihan Rashid",
    roll: "2105046",
    demos: [
      ["recap", "One ceiling, four views"],
      ["ifd", "Foragers and patches"],
    ],
  },
];

export const partById = (id: string): Part => {
  const p = PARTS.find((x) => x.id === id);
  if (!p) throw new Error(`unknown part ${id}`);
  return p;
};
