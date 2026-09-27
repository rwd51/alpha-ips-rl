# Defence slides

`main.tex` is the project defence deck: beamer with the `metropolis` theme,
16:9, 35 pages.

## Building

Build with **XeLaTeX**, twice (the second pass fills in the frame count and the
section progress bar):

```
xelatex main.tex
xelatex main.tex
```

XeLaTeX rather than pdfLaTeX because metropolis uses Fira Sans through
`fontspec`. With pdfLaTeX it silently falls back to Computer Modern Sans and
looks noticeably worse.

Packages used: `beamer`, `metropolis`, `tikz` (+ `arrows.meta`, `positioning`,
`calc`, `fit`, `backgrounds`, `intersections`), `pgfplots` 1.18 with the
`fillbetween` library, `booktabs`, `array`, `amsmath`.

Every figure is drawn in TikZ or pgfplots, so there are no image files to keep
alongside the source.

## Authors

Listed on the title page, one per line:

- Ahnaf Tahmid (2105041)
- Sadia Binte Sayeed (2105045)
- Mohammad Raihan Rashid (2105046)
- Ruwad Naswan (2105051)
- Md. Mehedi Hasan (2105052)

## Structure

| Pages | Content |
|---|---|
| 1 | Title |
| 2--4 | The base paper: the collapse mechanism, IPS, and what it leaves open |
| 5--8 | Our problem statement, the model, and how the work split into five parts |
| 9--12 | Part 1: collapse baseline, ODE integration, order verification |
| 13--17 | Part 2: the alpha knob, eigenvalue spectrum, stability, the finite-G ceiling |
| 18--21 | Part 3: root finding on three formulations, conditioning, nested solver |
| 22--25 | Part 4: the four-parameter atlas and the interpolation audit |
| 26--30 | Part 5: estimator bias and variance, Richardson, golden section, moving averages |
| 31--34 | Overall findings, the cross-cutting ceiling result, and honest limits |
| 35 | Closing |

Each part follows the same three-beat shape: **Objective**, **Numerical
Methods** (which syllabus topics it used and how), then **Results and
Analysis**. Parts 2 and 5 take two Numerical Methods pages because they draw
on more of the course.

The section divider before each part is a natural handover point between
speakers.

## Sources for every number

Nothing on the slides is invented. Each figure and table traces back to a
committed write-up in the repository root:

- `RESULTS_exp1.md` through `RESULTS_exp5.md` for the measured values
- `derivation.md`, `derivation_exp3.md`, `derivation_exp4.md`,
  `derivation_exp5.md` for the mathematics

## Note on the one remaining LaTeX warning

The build reports a single `Overfull \vbox (13.8pt)` on the title page. It
comes from the metropolis title block itself and fires even on a minimal file
that only sets a title and a subtitle. It has no visual effect: nothing is
clipped or pushed off the page. Every other frame builds with no overfull or
underfull boxes.
