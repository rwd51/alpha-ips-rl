# Part 5, explained simply

## The setting

- The model gives every answer a probability `p`. Training nudges these
  probabilities, step by step.
- The paper's fix (IPS) divides each reward by `p`, so rare good answers are not
  wiped out. Our project divides by `p^α` instead, so `α` works like a dial.
- **The catch:** training never knows the true `p`. Each step it draws a small
  group of `G` samples, counts how often an answer appears (`n`), and guesses
  `p̂ = n/G`. The weight it uses is `max(p̂, ε)^(−α)`. The clip `ε` is a floor, so
  a count of zero does not give an infinite weight.
- A guess from a small group is often wrong. Part 5 asks three things: **how
  wrong is it, can we guess better, and does the error actually matter for
  training?**

## Goal 1: measure the error

- A guess can be wrong in two ways:
  - **bias:** it is off on average;
  - **variance:** it jumps around from one group to the next.
- **Taylor expansion** gives a formula for both. The key number is `Gp`, the
  expected number of times the answer shows up in the group. The more often it
  shows up, the smaller the error.
- **Best clip.** If `ε` is too small, rare answers get huge weights (noise). If
  `ε` is too big, weights get pushed too low (bias). So the error goes down,
  then up. **Golden-section search** finds the lowest point, and it is always at
  exactly `ε = p`, the answer's own true probability. We also proved this.
- **Why it matters:** the algorithm uses one `ε` for all answers, but each
  answer wants a different one. A common answer and a rare answer can never
  both get their best clip.

## Goal 2: three other ways to guess

- **Laplace smoothing:** pretend every answer was seen `λ` extra times. The
  guess is never zero, so there are no huge weights and less noise. But
  everything gets pulled towards "all answers equal", so rare answers get a
  weaker boost.
- **Richardson extrapolation:** the error behaves like `c/G`. Compute the guess
  on the whole group and on each half. Twice the whole-group guess, minus the
  average of the two half guesses, cancels the `c/G` part. What is left is much
  smaller, like `c/G²`. It needs a safety rule (a guard), because otherwise it
  can give a negative weight.
- **Moving average:** average the guess over many past steps. This gives less
  noise for free, but the average lags behind a policy that is still changing.
  Golden-section search picks the best averaging rate.
- **Result:** judged purely as guesses, Richardson and the moving average are
  better than the paper's clip.

## Goal 3: what training actually feels (the surprise)

- **Key fact:** if an answer is not in the group, its weight gets multiplied by
  zero. Training never uses it. Training only uses the weight from groups where
  the answer appeared at least once.
- In such a group the answer has one "sure hit". **Taylor expansion** on this
  kind of group finds two small errors pulling opposite ways:
  - the sure hit makes the answer look a bit more common, so its weight comes
    out **too small**;
  - the curve of the weight formula (as in Goal 1) makes it come out **too big**.
- At the paper's own setting, `α = 1`, these two are **exactly equal**, so they
  cancel. The paper's simple clip is almost perfect for training.
- Richardson removes the "too big" error, so the "too small" one is left alone.
  In real training it wins with small groups but loses with large ones. At
  `G = 64` it ends 0.029 away from the ideal, against 0.0016 for the paper's
  clip. Laplace is the worst at every group size.
- **Lesson:** a better guess does not always mean better training. What matters
  is the error training actually feels.

## A tiny example that shows Goal 3

Take a group of `G = 2`, a true `p = 0.5` (so the true weight is `1/p = 2`), and
a clip of `ε = 0.05` (chosen just to keep the numbers simple).

| The answer appears | Chance | Weight used | Counts in training? |
|---|---|---|---|
| 0 times | 0.25 | 1/0.05 = **20** | no (multiplied by 0) |
| 1 time | 0.50 | 1/0.5 = 2 | yes, half as much as the next row |
| 2 times | 0.25 | 1/1 = 1 | yes, fully |

- **Plain average of the weight:** `0.25×20 + 0.5×2 + 0.25×1 = 6.25`. That is
  three times the true value 2. This is what Goal 1 and Goal 2 measure.
- **What training feels:** the 20 row counts for nothing. Weight the other rows
  by how often the answer appeared: `(0.5×0.5×2 + 1×0.25×1) / 0.5 = 1.5`. That is
  much closer to 2. It also matches our exact formula `(1 − (1−p)^G)/p = 1.5`.

So the guess looks terrible on paper (6.25), but training barely notices
(1.5). That gap is the whole of Goal 3.

## Things that are easy to mix up

- `p` is the true probability (unknown). `p̂ = n/G` is the guess from one group.
  `ε` is a fixed number chosen in advance.
- "Error of the guess" (Goals 1 and 2) is not the same as "error training
  feels" (Goal 3).
- Richardson does not remove all error. It removes the `1/G` part and leaves a
  smaller `1/G²` part.
