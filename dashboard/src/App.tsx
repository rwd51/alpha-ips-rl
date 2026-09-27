import { useEffect } from "react";
import { LabProvider } from "./engine/LabContext";
import { Group } from "./components/Instrument";
import { Footer, PresenterMap, TopBar } from "./components/Chrome";
import { partById } from "./presenters";
import { Hero } from "./sims/SimplexFlow";
import { Tetrahedron } from "./sims/Tetrahedron";
import { OutcomeRace } from "./sims/OutcomeRace";
import { BallsIntoBins } from "./sims/BallsIntoBins";
import { CollapseClock } from "./sims/CollapseClock";
import { WeightCeiling } from "./sims/WeightCeiling";
import { SurvivalSurface } from "./sims/SurvivalSurface";
import { PhaseAtlas } from "./sims/PhaseAtlas";
import { InterpolationAudit } from "./sims/InterpolationAudit";
import { StabilityRace } from "./sims/StabilityRace";
import { AmplificationLandscape } from "./sims/AmplificationLandscape";
import { RootRace } from "./sims/RootRace";
import { NewtonHops } from "./sims/NewtonHops";
import { NestedSolver } from "./sims/NestedSolver";
import { EstimatorLab } from "./sims/EstimatorLab";
import { DynamicsDistortion } from "./sims/DynamicsDistortion";
import { EmaOscillator } from "./sims/EmaOscillator";
import { CeilingRecap } from "./sims/CeilingRecap";
import { Foragers } from "./sims/Foragers";

// sections follow the defence deck (presentation/main.tex); src/presenters.ts says who presents each
export default function App() {
  // the page renders after the browser has tried the #anchor, so a direct link like
  // .../#nested (one presenter's demo) needs to be scrolled to once the sections exist
  useEffect(() => {
    const id = decodeURIComponent(window.location.hash.slice(1));
    if (id) document.getElementById(id)?.scrollIntoView();
  }, []);

  return (
    <LabProvider>
      <TopBar />
      <main className="wrap" id="top">
        <Hero />
        <PresenterMap />

        <Group
          part={partById("part1")}
          intro="Expected-return training (α = 0) with exact probabilities and with sampled groups. An exact tie does not move at all without noise; with noise it collapses onto an arbitrary winner, on a clock set by the reward gap and the group size."
        >
          <OutcomeRace />
          <BallsIntoBins />
          <CollapseClock />
        </Group>

        <Group
          part={partById("part2")}
          intro="Scaling rewards by 1/p^α turns collapse into a dial: p* ∝ r^(1/α). The same exponent sets how fast training settles, which step sizes are stable, and, with a finite group, a ceiling on how much any outcome can be boosted."
        >
          <Tetrahedron />
          <StabilityRace />
          <AmplificationLandscape />
          <WeightCeiling />
        </Group>

        <Group
          part={partById("part3")}
          intro="Solve ż = 0 directly instead of integrating to it: bisection, secant and Newton on three equivalent forms of the same condition, then a nested solver for the finite-group fixed point with many outcomes."
        >
          <RootRace />
          <NewtonHops />
          <NestedSolver />
        </Group>

        <Group
          part={partById("part4")}
          intro="The survival boundary checked jointly over α, G, ε and the reward ratio, and extended to five outcomes. Then the question a practitioner asks: can a precomputed grid stand in for the solver?"
        >
          <SurvivalSurface />
          <PhaseAtlas />
          <InterpolationAudit />
        </Group>

        <Group
          part={partById("part5")}
          intro="Real training never knows p. It estimates p^−α from G samples. How wrong is that estimate, which of its errors do the learning dynamics notice, and what does averaging over steps buy?"
        >
          <EstimatorLab />
          <DynamicsDistortion />
          <EmaOscillator />
        </Group>

        <Group
          part={partById("closing")}
          intro="What the paper proved, what we derived on top, and the one result every part runs into from a different side. Then the same law in a different world."
        >
          <CeilingRecap />
          <Foragers />
        </Group>

        <Footer />
      </main>
    </LabProvider>
  );
}
