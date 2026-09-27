# α-IPS Observatory

Interactive, live simulations of the `alpha-ips-rl` project (CSE 402): eighteen
instruments, three of them in real-time 3-D, covering Experiments 1–5. The page
follows the defence deck (`../presentation/main.tex`): one section per part,
labelled with its presenter, then the overall findings.

Built with React + TypeScript + Vite. three.js draws the 3-D scenes; the fonts
are bundled, so it runs fully offline once installed.

## Run it

Requires Node.js 20.19+ or 22.12+.

```bash
cd dashboard
npm install        # first time only
npm run dev        # opens http://localhost:5173
```

Other scripts:

| command | what it does |
|---|---|
| `npm run build` | type-check, then build a static site into `dist/` |
| `npm run preview` | serve the built `dist/` locally |
| `npm run typecheck` | TypeScript check only |

`dist/` is a plain static site (relative paths), so it can be hosted anywhere,
for example GitHub Pages.

## How the code is organised

```
src/
  lib/          the project's math, ported from the Python in ../src
    model.ts        equation (*), RK4/Euler, p*, Jacobian rates   (dynamics.py, integrators.py)
    sampling.ts     seeded RNG, inverse-transform sampler, sampled update (rhs_sampled)
    finiteGroup.ts  binomial sums, weight rules, K=2 mean field   (finite_group.py, estimators.py)
    numerics.ts     RK amplification polynomial, RK4 stability limit
    canvas2d.ts, three3d.ts, color.ts, format.ts   drawing helpers
  engine/       how an instrument runs
    types.ts        the Engine interface every simulation implements
    useEngine.ts    React hook: canvases, animation loop, on-screen detection, resize, play/pause
    LabContext.tsx  "Pause all" and "Link α across instruments"
  components/   shared UI: sliders, segmented buttons, readouts, instrument layout
  presenters.ts who presents which part, and which demos belong to it
  sims/         one file per instrument (engine + React panel)
  data/         atlasK5.json, precomputed with src/finite_group_general.py
```

Each file in `sims/` has two parts:

- `createEngine()` owns the canvas and the simulation state. It receives plain
  parameters through `setParams` and reports readouts through `host.emit`.
- A React component owns the controls, passes their values to the engine with
  `useEngine`, and lays out the panel.

To add an instrument: copy a similar file in `sims/`, change the engine and the
controls, add the component to its part in `App.tsx`, and list it under that
part in `src/presenters.ts` so it shows up in the presenter map.

A walkthrough of every demo, grouped by presenter, is in
`../presentation/DEMO_GUIDE.md`.

## Where the numbers come from

Everything is computed live in the browser from the ported equations, except
`src/data/atlasK5.json`: the five-outcome finite-group stationary points on a
40 α × 17 G grid, solved with the repository's
`src/finite_group_general.py::meanfield_stationary`.
