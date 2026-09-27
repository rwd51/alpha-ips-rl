# Demo guide: the α-IPS Observatory during the defence

The dashboard in `dashboard/` follows the slide deck. Each part of the deck has
its own section on the page, labelled with its presenter, and each presenter has
three or four live demos. This guide says what each demo shows, which experiment
it comes from, and exactly what to click during your part.

## Before you present

```bash
cd dashboard
npm install      # first time only
npm run dev      # then open http://localhost:5173
```

- **Jump straight to a demo** by adding its id to the address, for example
  `http://localhost:5173/#nested`. The ids are listed in the tables below. The
  presenter map at the top of the page links to every demo too.
- **Pause all** (top right) freezes every animation. Use it while you talk over a
  slide, then press it again when you switch to the demo.
- **Link α across instruments** makes every α slider move together. Leave it off
  unless you want to show one α across several demos.
- Every demo is computed live in the browser from the same equations as the
  Python in `src/`. The only precomputed data is the five-outcome phase atlas.
- Each demo's panel has "Try this" steps and a short derivation. The scripts below
  are the same steps, in the order that fits the slides.

## Who shows what

| Part | Slides | Presenter | Demos (id) |
|---|---|---|---|
| Opening | 2–8 | whoever opens | hero triangle (`top`) |
| 1 · Collapse baseline | 9–12 | Ruwad Naswan | outcome race (`race`), balls into bins (`balls`), collapse clock (`clock`) |
| 2 · The diversity exponent | 13–17 | Mohammad Raihan Rashid | tetrahedron (`tetra`), Euler vs RK4 (`stab`), amplification landscape (`amp`), weight ceiling (`ceiling`) |
| 3 · Root finding | 18–21 | Sadia Binte Sayeed | root-finder race (`roots`), Newton's tangent hops (`newton`), nested solver (`nested`) |
| 4 · Parameter atlas | 22–25 | Md. Mehedi Hasan | survival surface (`surface`), phase atlas (`atlas`), interpolation audit (`interp`) |
| 5 · Estimator error | 26–30 | Ahnaf Tahmid | estimator lab (`est`), what the dynamics see (`distort`), EMA oscillator (`ema`) |
| Overall findings | 31–34 | Mohammad Raihan Rashid | one ceiling, four views (`recap`), foragers (`ifd`) |

Five of these are new in this version, so that every part has at least three
demos: `clock`, `roots`, `nested`, `interp` and `distort`. Each one was checked
against the numbers in its RESULTS file (listed with each demo below).

---

## Opening (slides 2–8): hero triangle, `#top`

**Shows.** Every policy over three outcomes is a point in a triangle. Particles
follow equation (\*) and gather at p\*. The dashed curve is where p\* sits for
every α.

**Script.**
1. Rewards 4 · 2 · 1, α = 1: particles settle at (0.571, 0.286, 0.143), which is
   r / Σr, the paper's IPS result.
2. Drag α toward 0: p\* runs into the O1 corner. That is collapse.
3. Drag α to 3: p\* moves toward the centre, near-uniform.

**Say.** "One exponent turns collapse into a dial: p\* ∝ r^(1/α)." This sets up
slide 6 (problem statement).

---

## Part 1 · Ruwad Naswan (slides 9–12)

### The outcome race, `#race` (Exp 1, Exp 2 Figs 1–2)

**Shows.** Bars of the current probabilities over a log-time chart. White dashes
are p\*.

**Script.**
1. Rewards "All tied", α = 0: nothing moves at all. This is slide 12's
   "the tie is provably flat" (max |Δp₁| = 1.11 × 10⁻¹⁶).
2. Rewards "Tiers 5 5 5 3 3 1 1", α = 0, "Start O1 slightly ahead" on: the weak
   tiers vanish, then O1 slowly takes the whole tie. A 0.1-logit head start is
   enough.
3. (Hand-over tease) α = 1: the tiers settle at 5/23, 3/23, 1/23 each.

### Balls into bins, `#balls` (Exp 1 Fig 2)

**Shows.** Each step draws G balls from the policy and updates it from the
counts. Below, the entropy of 32 parallel runs.

**Script.**
1. α = 0, K = 3, G = 16, "Fast-forward": the 32 entropy traces drop to 0 one by
   one, each run onto a different random winner. Noise alone collapses an exact
   tie (slide 12: winners 18.5 / 22.5 / 19.0 / 19.0 / 21.0 %, all within the 95%
   band).
2. Raise α to 0.25: the same noise, but the traces stay near 1.

### The collapse clock, `#clock` (Exp 1 Figs 1c and 2c), new

**Shows.** Left: p₁(t) for 12 reward gaps on a log clock. Right: collapse time
against the gap on log-log axes, with the least-squares line drawn as points
arrive.

**Script.**
1. "Reward gap Δ" mode, threshold 0.99: wait for the 12 points. The fit reads
   **t ≈ 53.94 · Δ^−1.000, R² = 0.99999994**, exactly slide 12's numbers.
2. Point at "theory 53.89 · Δ^−1": with two outcomes the logit gap obeys
   u′ = 2Δ·p₁p₂, which integrates in closed form, so the slope is exactly −1.
3. Switch the threshold to 0.9: the slope stays at −1 and only the prefactor
   changes.
4. "Group size G" mode: 32 sampled runs per G on an exact tie (h = 0.5,
   threshold 0.98, the Exp 1 settings). The medians rise with G and the fit gives
   an exponent near 1. The repository's 80-seed fit is **G^1.07** (1.066 after
   Exp 2's sampler fix). With 32 seeds expect 0.9–1.05, depending on the seed.

**Also available to Part 1.** The tetrahedron (`#tetra`) in "Tie race" mode with
rewards 5 5 5 1 and α = 0: every run ends in the corner of the tied outcome it
started ahead on.

---

## Part 2 · Mohammad Raihan Rashid (slides 13–17)

### Four outcomes, one tetrahedron, `#tetra` (Exp 2), 3-D

**Shows.** Every policy over four outcomes is a point in a tetrahedron. Drag to
rotate.

**Script.**
1. "Ideal flow", rewards 4 3 2 1, α = 1: particles converge on p\* =
   (0.4, 0.3, 0.2, 0.1).
2. Drag α down: the orange curve is p\*(α), and the dot slides toward the O1
   corner.
3. "Tie race", rewards 5 5 5 1: at α = 0 every colour ends in its own corner. At
   α = 0.02 the colours first pull apart, then drift back to an even split by
   t ≈ 1000.

### Euler vs RK4, `#stab` (Exp 1 Fig 3, Exp 2 Fig 3c)

**Shows.** Left: p₁(t) for Euler, RK4 and the exact flow. Right: the stability
regions in the complex plane, with the marker at −hλ.

**Script.**
1. α = 2 (λ = 8), h = 0.2: both methods converge.
2. Slide h to 0.3: hλ = 2.4 passes Euler's limit of 2. Euler bounces between 0
   and 1 while RK4 still converges (limit 2.785). The chips turn red and green.

### Amplification landscape, `#amp` (numerics), 3-D

**Shows.** log₁₀|R(z)| over the complex plane. The blue crater is where a method
is stable.

**Script.**
1. Step the order 1 → 4: the crater widens and reaches 2.785 on the real axis.
2. "exact eᶻ": the whole left half-plane is stable. Only the discretisation can
   blow up.

### The weight ceiling, `#ceiling` (Exp 2, Exp 3 Fig 3a, Exp 5 Fig 3)

**Shows.** The boost a finite group can apply, w(p)/w(1), against the ideal
p^−α. A probe draws one sampled group.

**Script.**
1. G = 16, α = 1, ρ = 4: the curve flattens at 16. Since 16 > 4, the weak outcome
   survives (green chip).
2. G = 4: the ceiling is 4 = ρ, so it dies. This is slide 17's "at G = 4 the
   paper's own α = 1 fails" (long-run p₁ = 0.9965 instead of 0.8).
3. Raise ε above 1/G: the clip, not the group, now sets the ceiling.

---

## Part 3 · Sadia Binte Sayeed (slides 18–21)

### The root-finder race, `#roots` (Exp 3 Fig 1a), new

**Shows.** Error against iteration for bisection, secant and Newton on the drift,
balance and log forms, all solving the same K = 2 condition. The strip below
shows where each iterate sits relative to u\*.

**Script.**
1. Defaults α = 1, ρ = 4, u₀ = 0 are slide 21's setting. The readouts show
   **bisection 56 (linear, ratio ≈ 0.5), secant 9 at order 1.61, Newton on the
   drift 5 at order 2.00, balance 6, log 1**.
2. Slide u₀ to 4: Newton on the drift runs off to infinity (arrow at the edge),
   while Newton on the balance form still lands on u\*.
3. Raise α above 1.11: the drift form converges again (Exp 3 found the cutoff at
   1.113).

A method's final step is a confirming step at round-off, so a count can
occasionally differ by one from the table.

### Newton's tangent hops, `#newton` (Exp 3 Fig 1b)

**Shows.** The residual curve of each form and Newton's tangent lines, plus the
basin over 300 starts.

**Script.**
1. Drift form, α = 0.5, start 7: the curve is flat out there, and the tangent
   flings Newton away.
2. Same start, balance form: steady hops, then quadratic snap-in.
3. Log form: one hop from anywhere.

### The nested solver, five outcomes, `#nested` (Exp 3 Fig 3, Exp 4 Fig 2), new

**Shows.** Bars of the finite-group stationary point for r = (5, 4, 3, 2, 1)
against the ideal p\*. Next to them, each outcome's own survival threshold: a
filled dot from the K = 5 mean field, a ring from the pairwise formula. Below,
the outer residual per iteration for Newton/Newton against bisection/bisection.

**Script.**
1. α = 1, G = 16: all five survive. The weakest one's threshold is **0.886, not
   the pairwise 0.580** (slide 21 table: 0.0805, 0.203, 0.419, 0.886; the demo
   computes the same four values).
2. Drop G to 5: the fifth outcome reads "lost at every α" (it needs G ≥ 6).
3. Point at the residual chart: Newton reaches 10⁻¹⁵ in about 8 outer steps,
   against about 41 for bisection, roughly 28× less total work. Slide 21 quotes
   8–20× from the repository's solver, whose tolerances are different.

---

## Part 4 · Md. Mehedi Hasan (slides 22–25)

### Survival surface, `#surface` (Exp 2 Fig 4c, Exp 4 Fig 1), 3-D

**Shows.** Height = the smallest α that keeps the weaker outcome alive, over
(G, ρ). Everything below the glowing plane at your α survives.

**Script.**
1. α = 1, ε = 10⁻³: the plane cuts the surface along G = ρ^(1/α). Set G = 16,
   ρ = 4: needed α_c = 0.5, so it survives.
2. Raise ε to 0.3: past G = 1/ε the surface turns into a plateau. With ε = 0.3
   the ceiling is 3.33 < 4, so no group size rescues ρ = 4 (slide 25's second
   table).

### Phase atlas, `#atlas` (Exp 4 Figs 1a, 2a)

**Shows.** Two outcomes: minority mass over (α, G), solved live, with the dashed
curve α = ln ρ / ln min(G, 1/ε). Five outcomes: the number of survivors, from the
repository's solver.

**Script.**
1. "Two": the bright edge sits exactly on the dashed curve. Exp 4 solved 19,125
   points with a solver that never uses the formula and found 0 mismatches.
2. "Five": at α = 1, G = 2 keeps 2 outcomes, G = 4 keeps 3, G = 8 keeps 4, and
   G = 16 keeps all 5.

### Can a grid replace the solver?, `#interp` (Exp 4 Fig 3), new

**Shows.** One slice through the atlas at G = 16: the solver's minority mass
(blue), the grid interpolant (orange), and the kinks (yellow: clip thresholds
ε = k/G; red: extinction m = 0). Below: the error at 400 held-out points,
coloured by cell type, and RMSE against grid size.

**Script.**
1. Default slice along α (ρ = 4, ε = 10⁻³): one kink, at α_c = 0.5. Only the
   red-shaded cell has large errors. Readouts: **order ≈ 2.0 in kink-free cells,
   ≈ 1.0 in the kinked cell**.
2. Step the grid 5 → 9 → 17 → 33: the inset lines follow the slope-2 and slope-1
   guides.
3. Switch to the ε slice: above 1/16 the clip thresholds are as dense as the
   grid, so almost every cell is kinked. That is the conclusion of Exp 4: decide
   survival from the closed-form margin, not from the grid.

**Numbers for the slide** (revised RESULTS_exp4, 17³ grid, 800 points):
kink-free orders 2.19 and 1.86, kinked 1.01 and 1.21, and kinked-cell RMSE 16×
the kink-free RMSE.

---

## Part 5 · Ahnaf Tahmid (slides 26–30)

### Estimator lab, `#est` (Exp 5 Figs 1–2)

**Shows.** The exact distribution of the clipped weight for one outcome, then
live samples. Below it, MSE against ε.

**Script.**
1. p = 0.1, G = 16: the empty group (probability 0.19, weight 1/ε = 1000) sits far
   right and carries most of the error.
2. Slide ε: MSE falls, then rises, with the minimum exactly at ε = p (the star).
   This is slide 30's ε\* = p.

### What the dynamics actually see, `#distort` (Exp 5 Fig 4, Fig 3c), new

**Shows.** The dynamics-level distortion D(p) = w_G(p)·p^α − 1 for the clip,
guarded Richardson, α-matched offset and Laplace rules. Left: against p. Right:
against G at a fixed p.

**Script.**
1. α = 1, G = 16: the clip lies on the wide grey line (1 − p)^G. It is exact up to
   the chance of missing the outcome entirely. Richardson is a straight line
   (error of order 1/(Gp)).
2. Readouts at α = 1: clip "faster than any power", Richardson "order ≈ 1 in
   1/G". The better estimator gives the dynamics a first-order error that the
   clip does not have. That is slide 30's "the better estimator trains worse"
   (G = 64: 0.0016 for the clip against 0.0294 for Richardson).
3. Move α to 2: the clip picks up its own 1/G term, and the α-matched offset
   becomes the best rule (order 2).

### The EMA oscillator, `#ema` (Exp 5 Fig 5)

**Shows.** Averaging p̂ over steps turns first-order settling into a damped
spring. The roots of s² + κs + κλ lie on a circle.

**Script.**
1. Small κ: the policy rings around p\* (underdamped).
2. κ = 4λ: the roots meet at −2λ, the fastest settling (critical damping, slide
   29).
3. Large κ: back to first-order behaviour.

---

## Overall findings · Mohammad Raihan Rashid (slides 31–34)

### One ceiling, four views, `#recap`

Four cards, one per part, each linking to the demo where that part meets the
finite-group ceiling (slide 33). Click them in order: weight ceiling → nested
solver → survival surface → dynamics distortion. Under the cards are the four
practitioner takeaways from slide 34, each linked to its demo.

### Foragers and patches, `#ifd` (analogy)

240 foragers choose among four food patches. Each moves to where intake
food / crowd^α is highest. At α = 0 they all pile into the richest patch
(collapse). At α = 1 they match food supply. In general they settle at
n ∝ r^(1/α), the same law as p\*. Good last demo before "Thank you".

---

## Deck numbers that changed after the slides were written

Mehedi's revision of Exp 4 (commit `9e8b7d6`, 27 Sep) changed some numbers that
the Part 4 slides still quote (the deck was committed earlier the same day).
Slide 25 is Mehedi's to update; slide 32 is already fixed.

| Slide | Deck says | RESULTS_exp4.md now says |
|---|---|---|
| 25 (Part 4 results) | misclassifications 0/19,125 | 0 mismatches at the 19,120 points with \|m\| > 10⁻⁹; the 5 points exactly on m = 0 are extinct, as the strict inequality requires |
| 25 | worst stationarity residual 1.18 × 10⁻¹⁵ | max relative balance residual 5.92 × 10⁻¹⁵ |
| 25 (interpolation audit) | smooth RMSE 2.5 × 10⁻³, order 1.58; near a kink 9.7 × 10⁻³, order 1.03 | kink-free cells RMSE 3.14 × 10⁻⁴, orders 2.19 and 1.86; kinked cells RMSE 1.02 × 10⁻², orders 1.01 and 1.21 (the old 1.58 mixed both regimes) |
| 32 (comparative analysis) | verified at 0/19,125 misclassifications | fixed: now says "checked on 19,125 settings with no exceptions" |

Slide 24 describes the kinks as "m = 0 and ε = 1/G". The revision found a kink
at every ε = k/G, not only at 1/G.
