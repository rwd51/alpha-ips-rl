# Talk notes: Part 2 (slides 13–17) and Overall findings (slides 31–34)

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

## Slide 14: Objective

- "The paper divides every reward by p, the probability of the answer. We divide
  by p to the power α. So α is a dial."
- "At α = 0 there is no fix, and one answer takes everything. At α = 1 we get the
  paper's fix, where each answer's share matches its reward. At large α all
  answers get almost the same share."
- "Training stops when every answer earns the same corrected reward. That gives
  one simple formula: share ∝ r to the power 1/α."
- Point at the plot: "Two answers, rewards 4 and 1. We predicted 94/6, 80/20 and
  67/33. The simulation hits these exactly, to machine precision."
- "So we asked four questions: does training really end up there, how fast, how
  big a step is safe, and what changes with a small group of samples?"

*Optional demo (20 s): `#tetra`. Drag α and watch the dot move from a corner to
the centre.*

## Slide 15: Numerical Methods (1/2)

- "Training is a differential equation. Setting the change to zero gives the end
  point."
- "**Eigenvalues tell us the speed.** Close to the end point, the error shrinks
  like e to the minus λ t. The λ values are the eigenvalues of one matrix. A big
  λ means fast, a small λ means slow."
- "One eigenvalue is always zero. Adding the same number to every score changes
  nothing, so nothing pulls back in that direction. That's harmless."
- "**Least squares measures the speed.** On a log scale the error is a straight
  line, and its slope is the speed. We fit the slope and compare it with the
  eigenvalue."
- "**Round-off decides where we fit.** Only between 10⁻⁶ and 10⁻¹². Above that the
  line isn't straight yet. Below it we hit the computer's precision floor, the
  flat part of the small plot."

## Slide 16: Numerical Methods (2/2)

- "**Stability.** Every solver step multiplies the error by some factor. If that
  factor is above 1, the error grows each step and the run blows up."
- "For Euler, h times λ must stay below 2. For RK4, below 2.785."
- "**Bisection.** 2.785 is the root of a cubic, and we computed it by bisection
  instead of looking it up. Then we used bisection on the real simulation to find
  the biggest step that still works. It matched the theory within 0.03%."
- "**Random sampling.** Real training doesn't know p. It draws a group of G
  samples and counts them. Averaged over many groups, this gives an effective
  weight in place of 1/p^α, and we solve for the end point with bisection again."
- "Every random simulation is checked against this average prediction."

*Demos (40 s):*
1. *`#stab`: set α = 2. At h = 0.2 both methods settle. Move h to 0.3: Euler
   starts jumping, RK4 still settles.*
2. *`#amp`: step the order 1 → 4 and watch the safe (blue) area grow.*

## Slide 17: Results and Analysis

**Left side.**
1. "The dial works, and we can predict the speed. The measured speed matched the
   eigenvalue within 0.002% over 27 runs."
2. "Both ends of the dial are hard. At small α the slowest and fastest rates
   differ by a million times, which is called a stiff problem. At large α every
   rate is huge, so the step has to be tiny. Part 1's step size breaks above
   α = 2.06."
3. "At α = 0 there is no speed at all: the losing answer fades slowly, like 1/t."

**Right side (the key result, slow down here).**
- "A group of G samples sees a rare answer at most once. So it can never estimate
  p below 1/G, and the boost 1/p^α has a **cap**."
- "The weaker answer survives only if the cap is bigger than the reward ratio.
  That's the boxed formula."
- Point at the table: "Rewards 4 and 1. With a group of 16, α must be above 0.5.
  With a group of 4, α must be above 1."
- "So with G = 4, even the paper's own α = 1 is **not enough**. The weaker answer
  dies."
- "This rule is our own result. The paper only tried a few group sizes; it never
  gave a rule."

*Demo (40 s): `#ceiling`. With G = 16, α = 1, ρ = 4 the curve flattens at 16,
and 16 > 4 gives a green "survives". Set G = 4: the cap equals 4 and the chip
turns to "dies".*

> Hand-over to Sadia: "I found the end point by running the simulation for a long
> time. Sadia solves for it directly, and shows the cap matters even more with
> more than two answers."

---

## Slide 31: "Overall Findings" (divider)

## Slide 32: Comparative Analysis

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

## Slide 33: Cross-Cutting Result, the cap

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

## Slide 34: Conclusions and Limitations

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

Part 2: about 4 minutes, including 2 minutes of demos. Overall findings: about 3
minutes plus 1 minute of demo. If time is short, skip `#amp` and `#ifd`.
