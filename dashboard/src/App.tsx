import { LabProvider } from "./engine/LabContext";
import { Group } from "./components/Instrument";
import { Footer, InstrumentIndex, TopBar } from "./components/Chrome";
import { Hero } from "./sims/SimplexFlow";
import { Tetrahedron } from "./sims/Tetrahedron";
import { OutcomeRace } from "./sims/OutcomeRace";
import { BallsIntoBins } from "./sims/BallsIntoBins";
import { WeightCeiling } from "./sims/WeightCeiling";
import { SurvivalSurface } from "./sims/SurvivalSurface";
import { PhaseAtlas } from "./sims/PhaseAtlas";
import { StabilityRace } from "./sims/StabilityRace";
import { AmplificationLandscape } from "./sims/AmplificationLandscape";
import { NewtonHops } from "./sims/NewtonHops";
import { EstimatorLab } from "./sims/EstimatorLab";
import { EmaOscillator } from "./sims/EmaOscillator";
import { Foragers } from "./sims/Foragers";

export default function App() {
  return (
    <LabProvider>
      <TopBar />
      <main className="wrap" id="top">
        <Hero />
        <InstrumentIndex />

        <Group
          id="g-dial"
          num="I · THE DIAL"
          title="Where training settles, and how it gets there"
          intro="The ideal flow of equation (*), with exact probabilities and no sampling noise. α = 0 is standard expected-return training (Experiment 1); α = 1 is the paper's inverse probability scaling."
        >
          <Tetrahedron />
          <OutcomeRace />
        </Group>

        <Group
          id="g-groups"
          num="II · SAMPLING AND GROUP SIZE"
          title="What a finite group of G samples can and cannot see"
          intro="Real training never knows p. It estimates p̂ from G samples, and that single fact caps how much any outcome can be boosted."
        >
          <BallsIntoBins />
          <WeightCeiling />
          <SurvivalSurface />
          <PhaseAtlas />
        </Group>

        <Group
          id="g-numerics"
          num="III · NUMERICS"
          title="The solvers behind the curves"
          intro="Euler, RK4 and Newton–Raphson, run on the project's own equation, so you can see where each one works and where it breaks."
        >
          <StabilityRace />
          <AmplificationLandscape />
          <NewtonHops />
        </Group>

        <Group
          id="g-estimators"
          num="IV · ESTIMATORS"
          title="How far p̂ is from p, and whether it matters"
          intro="Experiment 5 measured the bias and variance of the weight estimate exactly, then asked which of those errors the learning dynamics actually notice."
        >
          <EstimatorLab />
          <EmaOscillator />
        </Group>

        <Group
          id="g-analogies"
          num="V · ANALOGIES"
          title="The same law in other worlds"
          intro="p* ∝ r^(1/α) is not unique to reinforcement learning. Here it appears in animals choosing where to feed."
        >
          <Foragers />
        </Group>

        <Footer />
      </main>
    </LabProvider>
  );
}
