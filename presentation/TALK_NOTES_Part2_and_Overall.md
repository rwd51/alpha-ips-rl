# Talk notes: Part 2 (slides 13–19) and Overall findings (slides 33–36)

Speaker: Mohammad Raihan Rashid. Plain-language points to say for each slide. The
numbers match the slides and `RESULTS_exp2.md`.

## The whole story in four lines

1. Normal training (Part 1) ends with one answer taking everything.
2. The paper fixes this by dividing each reward by p. We divide by p^α, so α
   works like a dial.
3. We checked where training ends up, how fast it gets there, and how big a
   solver step is safe.
4. Big surprise: a small group of samples puts a **cap** on the correction. If
   the cap is too low, the weaker answer dies no matter what.

---

## Slide 13: "Part 2: The Diversity Exponent" (divider)

> "Ruwad showed that normal training collapses. I'll show what happens when we
> turn the paper's fix into a dial."

## Slide 14: Objective (figure: training runs settling)

**Say what training is first.**
- "The model gives each answer a probability. At every step, answers that earn
  more reward are pushed up a little. After many steps nothing changes any more:
  that's where training ends."
- "The paper divides each reward by p, the answer's probability. We divide by p
  to the power α, so α is a dial. Training then ends at share ∝ r^(1/α)."

**Then explain the figure.**
- "Two answers. Rewards 4 and 1. The rewards are fixed; the probabilities are
  what training finds."
- "Every run starts at 50/50. Each curve is one run: the better answer's share
  over time."
- "Each α ends at a different level. At α = 0 the better answer takes 100%. At
  α = 0.5 it ends at 94%, which leaves 6% for the other: that's what '94/6'
  means. At α = 1 it's 80/20, the same as the rewards 4 : 1. At α = 2 it's
  67/33."
- "So we asked four questions: does training really end there, how fast, how big
  a step is safe, and what changes with a small group of samples?"

*Where the numbers come from: 4^(1/α) / (4^(1/α) + 1). α = 0.5 gives 16/17 =
0.94; α = 1 gives 4/5 = 0.8; α = 2 gives 2/3 = 0.67.*

## Slide 15: Numerical Methods (1/2) (figure: straight error lines)

- "Training is a differential equation. Setting the change to zero gives the end
  point."
- "Near the end point the error shrinks like e to the minus λ t. The λ values,
  the speeds, are the eigenvalues of one matrix."
- "One eigenvalue is always zero: adding the same number to every score changes
  nothing."
- Point at the figure: "This is the distance between where training is now and
  where it will end, on a log scale. On a log scale an exponential decay is a
  straight line, and its slope is the speed."
- "The four runs have very different speeds: from 0.36 to 45.6, more than 100
  times apart. But the time axis is multiplied by each run's *predicted* speed.
  If our prediction is right, all four lines must have the same slope, and they
  do. That's the check."
- "We fit the slope by least squares only inside the grey band. Below it the
  lines go flat: that's the computer's precision floor, about 10⁻¹⁵."
- If asked "is this the collapse?": "No. For α above 0 training does not
  collapse; it settles at a mix of answers. This plot shows how fast it settles.
  At α = 0 there's no exponential at all: the loser fades slowly, like 1/t."

## Slide 16: Numerical Methods (2/2) (figure: Euler jumping)

- "A solver moves in steps. Each step multiplies the error by some factor. If
  the factor is above 1, the error grows every step and the run blows up."
- "For Euler, h times λ must stay below 2. For RK4, below 2.785. We found 2.785
  by bisection, and bisection on the real simulation matched the theory within
  0.03%."
- Point at the figure: "Here λ = 8. With step 0.3, h·λ = 2.4, above Euler's
  limit: the orange Euler run jumps between 0 and 1. RK4 with the same step
  settles, and so does Euler with the smaller step 0.2."
- "Real training only sees G samples per step. Averaging over many groups gives
  an effective weight, and we solve for the end point with bisection. Every
  simulation is checked against that."

*Demo (30 s): `#stab`, α = 2. Move h from 0.2 to 0.3 and Euler starts jumping.*

## Slide 17: Results (1/3): The dial works

- Left figure: "Final share of the better answer for every α, for three reward
  pairs. Circles are simulations, lines are the formula. They agree to 15
  decimal places."
- "Left means collapse: the better answer takes 100%. Right means everyone gets
  almost the same share. α = 1, the dashed line, is the paper's fix."
- Right figure: "Five answers, rewards 5 down to 1. As α grows, the bars even
  out. At α = 1 each answer's share is exactly its reward over the total."

*Demo (20 s): `#tetra`, "Ideal flow". Drag α and watch the dot move.*

## Slide 18: Results (2/3): Speed and step size

- Left: "The dots are the speeds we measured. The lines are what the eigenvalues
  predicted, before running anything. They agree within 0.002% for two answers
  (0.09% for five)."
- Right: "The largest safe step for each α. The markers, found by bisection, sit
  on the theory lines. The dotted line is Part 1's step, 0.05: it becomes unsafe
  above α = 2.06."
- Bottom line: "Both ends of the dial are hard. At small α the speeds differ by
  a million times, called a stiff problem. At α = 0 there's no speed at all: the
  losing answer fades like 1/t."

## Slide 19: Results (3/3): A small group caps the boost (the key result)

- "Real training sees only G samples. A rare answer shows up at most once, so
  its probability is never estimated below 1/G. The boost has a cap: G to the
  power α."
- "The weaker answer survives only if the cap is bigger than the reward ratio.
  That's the boxed formula, and it's our own result."
- Middle figure: "Rewards 4 and 1. With groups of 16 or 64 (orange, green), the
  simulations follow the ideal curve. With a group of 4 (blue), the better answer
  keeps 100% until α passes 1: the weaker answer is dead. So even the paper's
  α = 1 is not enough with G = 4."
- Right figure: "Colour shows how much of the weaker answer survives, for every
  α and group size. White means dead. The dashed line is our formula, and it
  traces the edge."

*Demo (40 s): `#ceiling`. G = 16, α = 1, ρ = 4 gives a green "survives"; G = 4
turns it to "dies". Or use `#tetra` in "Sampled groups" with 5 5 5 1, α = 0.55,
G = 4: O4 dies. Raise G to 32 and it comes back.*

> Hand-over to Sadia: "I found the end point by running the simulation for a long
> time. Sadia solves for it directly, and shows the cap matters even more with
> more than two answers."

---

## Slide 33: "Overall Findings" (divider)

## Slide 34: Comparative Analysis

Read the table one row at a time: "the paper did this, we added this."

1. "Collapse: the paper proved it. We showed that with an exact tie nothing moves
   at all. It's the random sampling that picks the winner."
2. "IPS: the paper proved share ∝ reward. We confirmed it exactly and extended
   it to any α."
3. "Group size and clip: the paper only tried a few values. We derived the exact
   survival rule and checked it on 19,125 settings with no exceptions."
4. "Clipping: the paper named it as a limitation. We measured it exactly and
   found the best clip is ε = p."

> "We didn't rerun the paper's big language-model experiments, which need far
> more computing. We reproduced the mechanism in a small model where nothing else
> can explain the result."

## Slide 35: Cross-Cutting Result, the cap

- "The dashed line is the ideal boost: it keeps growing as an answer gets rarer.
  The solid lines are what a real group of 16 or 64 can give. They flatten,
  because a rare answer appears at most once."
- "Every part runs into this cap from a different side:
  - Part 2 found it.
  - Part 3 showed that more answers need a bigger α.
  - Part 4 checked the rule on 19,125 settings.
  - Part 5 showed that only the size of the cap decides survival."

*Demo (60 s): `#recap`. Click the four cards in order; each opens that part's
demo. Just point at the one number on each card.*

## Slide 36: Conclusions and Limitations

**Advice (left).**
1. "Choose the group size first. If the cap is smaller than best reward divided
   by worst reward, no α can save the weak answers."
2. "Keep the clip ε at or below 1/G, or it adds a second cap."
3. "With many answers the two-answer rule is too hopeful: the weakest of five
   needs α = 0.886, not 0.580."
4. "Averaging over training steps cuts noise for free, but with too little
   damping training starts to oscillate."

**What we don't claim (right).**
1. "We studied a simple model, not full GRPO. Our results are about the weight
   rule."
2. "Our improved correction is very accurate, but not exact."
3. "Averaging per answer needs a short list of possible answers. That's true
   here, not for generated text."
4. "We show convergence in simulations; we don't prove it for every α."

*Optional last demo (20 s): `#ifd`. At α = 0 all the animals crowd into the
richest patch; at α = 1 they spread in proportion to the food. "Same law,
different world." Then "Thank you".*

---

## If they ask...

**Why not just add an entropy bonus?**
"An entropy bonus pushes against collapse but keeps its cause. Our change moves
the end point itself, to a place we can write down exactly."

**Where does the cap come from?**
"A group of G samples sees a rare answer at best once, so its estimate is never
below 1/G. The weight 1/p̂^α can't go above G^α. The clip ε adds a second limit,
1/ε."

**Is the survival rule exact?**
"For two answers, yes, and Part 4 found no exceptions. For more answers it's
necessary but not enough; Part 3 shows you need a bigger α."

**Why is α = 0 special?**
"Every speed is proportional to α. At α = 0 the speed is zero, so the loser
fades slowly, like 1/t, instead of exponentially."

**Why do large α values need small steps?**
"The speeds grow with α, and a solver step must stay below 2 divided by the
fastest speed (2.785 for RK4)."

**Why doesn't the simulation match the prediction exactly (0.013)?**
"The prediction assumes tiny steps and infinite time. Real runs use a finite
step for a finite time, and the gap is largest near the survival boundary, where
everything is slowest."

**Is this the paper's result or yours?**
"The paper proved the collapse and the α = 1 fix. The dial, the speed and
stability analysis, the cap and the survival rule are ours. We searched the
literature and didn't find this rule stated elsewhere."

## Timing

Part 2: about 5 minutes (six slides), including 1.5 minutes of demos. Overall findings: about 3
minutes plus 1 minute of demo. If time is short, skip the `#tetra` demo on slide 17 and `#ifd`.
