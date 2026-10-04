# Literature review: outcome-level mode collapse, inverse probability scaling (IPS) and α-IPS

*Prepared 2026-10-03 (re-checked 2026-10-04) for CSE 402, Section A, Group 04. Base paper: Sinha, Elango & Liu, arXiv:2601.21669 (ICML 2026 poster, <https://icml.cc/virtual/2026/poster/63579>). Our report: `report/report.tex`.*

## Summary

- **The base paper's reference list:** 43 references in total. I checked every link (arXiv API, Crossref, publisher pages). One exception: the OpenReview page of DeRL (An et al.) did not open because of a bot check, but the work is confirmed to exist. The base paper itself is now accepted at **ICML 2026** (poster).
- **Wider literature:** I listed **151 more works** (papers and books), each one verified (mostly 2023–2026), plus older classic work (psychology, ecology, evolutionary computation, statistics). No reference is made up.
- **The biggest news (it reduces our novelty):** the "α knob", i.e. r/p^α and p* ∝ r^{1/α}, is not new. (1) **Hu et al. 2026** (ACL Findings 2026; on arXiv before the base paper) multiply GRPO's advantage by exactly **1/f^α**, α∈[0,1]. (2) **Lin & Ie 2026**, Theorem 2.1, proves that the optimum of the Tsallis q-log objective is the escort distribution ∝ (weight)^{1/q}. This is our S1. (3) **Guu et al. 2017** saw the same "rich-get-richer" problem and used p^β weights. (4) In GFlowNets the R(x)^β target has been in use since 2021.
- **IPS itself is a new form of older ideas:** the IPS gradient = on-policy **DPG / forward KL** (Parshakova 2019, Khalifa 2021). **DRA-GRPO (May 2025)** divides the reward by a soft count within the group, which is effectively IPS-GRPO. Older forms exist too: fitness sharing (1987), the **matching law** (Herrnstein 1961), the **ideal free distribution** (Fretwell & Lucas 1969). In all of them the rule is "reward ÷ crowding" and the equilibrium is p ∝ r. The s of Baum's generalized matching law equals our 1/α, and Sutherland's interference m equals our α.
- **Our α=1 exact identity** w_G(p) = (1−(1−p)^G)/p also already exists: in Theorem 2 of **MaxRL** (Tajwar et al., ICML 2026), and as a classical negative binomial moment (Chao & Strawderman 1972). The size-biasing form w_G = E[ω(1+B)] is the Bernstein representation of **RL2ML** (Zheng 2026).
- **What looks new (not found anywhere):** (a) the finite-group survival limit ω(1)/ω(G) > r1/r2 and α > ln(r1/r2)/ln min{G,1/ε}; (b) the general-K extinction thresholds; (c) the explicit rate λ = α‖r‖_{1/α}μ of the α-flow and the Euler/RK4 step-size limits (the tools are old, the application is new); (d) in the base paper's Table 4 the clip never acts when ε < 1/G, so the entries coincide. These should be presented as our main contributions.
- **Main competitors:** the base paper's three baselines: GRPO, FlowRL, REINVENT. Among newer methods: MARA (GX-Chen, ICLR 2026), Outcome-based exploration (Song et al. 2025), Unlikeliness reward (He et al. 2025), Uniqueness-aware RL (Hu et al. 2026), DRA-GRPO, GAPO, DARLING, DMPO, DPH-RL, MaxRL, pass@k training, GFlowNet fine-tuning.
- **Name clash:** in off-policy learning "IPS" means inverse propensity scoring. The **"IPS-α"** estimator π/π0^α of Aouali et al. (ICML 2023) can be confused with our name "α-IPS". The report needs a footnote.

## What you need / what to do

1. **Read these 5 first (in this order):** Lin & Ie 2026 → Hu et al. 2026 → Tajwar et al. 2026 (MaxRL) → Zheng 2026 (RL2ML) → Guu et al. 2017. Why to read each one is in Section 6.
2. **What to change in the report** (details in Section 4):
   - Abstract and the first Contributions bullet: they say "We generalize IPS with a tunable exponent". Write "we study an α-generalized IPS weight, a member of a known family", and cite Hu 2026, Lin & Ie 2026, Bengio 2021 (R^β) and Baum 1974.
   - Call **S1 (p* ∝ r^{1/α})** "known". Our new part is the dynamics: rate, step size, finite G.
   - With **S5 (α=1 exactness)**, cite MaxRL and Chao & Strawderman, and write "outcome-level version".
   - With **w_G(p)=E[ω(1+B)]**, cite RL2ML.
   - With the **1/t convergence at α=0**, cite Mei et al. 2020; with the **RK4 limit 2.785**, Hairer & Wanner 1996; with **"rich-get-richer"**, Guu et al. 2017.
   - Describe the **S4 survival limit** as "to our knowledge new". Also make clear that it is an idealized mean-field result (group-REINFORCE; no GRPO std normalization and no PPO clip). Soften the word "universal", e.g. to "general (any count-based weight rule)".
3. **Adding citations:** `next_phase/01_literature.bib` has 27 entries, and their keys do not clash with the keys in `report/references.bib`. The simplest way: in `report.tex`, change `\bibliography{references}` to `\bibliography{references,../next_phase/01_literature}`. Alternatively, copy the file to the end of references.bib. "ICML 2026" can be added to the `sinha2026` entry.
4. **Where to put Related Work:** paste the LaTeX text of Section 5 (`\section{Related Work}`) into `report.tex` after Contributions, just before `\section{Methodology}`. Every `\cite` key is in the bib files (`sinha2026` comes from the existing bib).
5. **Be careful:** Section 2.13 has 6 mathematical connections (e.g. the α-flow is the gradient flow of Σ r·ln_α p). I checked these numerically myself. If you put them in the report, present them as "we note". Check the global-convergence argument (the α<1 boundary) yourself before putting it in the report; the critical review (`02_critical_review.md`, §2.3) now gives a full proof, including the boundary case.

---

## 0. How this review was done

- **Base-paper references (Section 1)** were taken from the PDF (`pdftotext`). Every arXiv ID was checked against the arXiv API (title, authors, date, comments). Every DOI was checked against the Crossref API. Remaining items (PMLR, JMLR, OpenReview) were checked on the proceedings pages or by web search.
- **Additional literature (Sections 2–4)** was found by web search and then verified one by one: arXiv IDs through the arXiv API, DOIs through Crossref, and venues through proceedings pages or the arXiv "comments" field. For every paper whose exact mechanism matters to our novelty claims, the PDF itself was downloaded and the relevant equation was read. These included Aouali 2023, Lin & Ie 2026, MaxRL, RL2ML, Hu 2026, DRA-GRPO, Guu 2017, Mohri 2026, GAPO, He 2025, Song 2025, Bengio 2021, Malkin 2022, Kim 2024, Khalifa 2021, GX-Chen 2025, FlowRL, Multi-Objective GFlowNets, Strehl 2010, Bottou 2013, Swaminathan & Joachims 2015, Fauvergue 2006 and Sanchez & Gillespie 2022.
- A venue is stated only when it was confirmed (proceedings page, journal DOI, or the authors' own arXiv comment). Otherwise only the arXiv ID is given.
- Items that could not be fully verified are listed in Appendix A. No reference in this file was taken from memory without a check.

---

## 1. The base paper's reference list (complete: 43 references)

The base paper cites **43** works. They are grouped below by theme. "Used for" gives the section and claim in the base paper (sections: 1 Intro, 2.1 Mode collapse, 2.2 Flow-based, 2.3 LLM post-training, 3 Theory, 4 Method, 5 Experiments).

### 1.1 Application domains that motivate outcome-multimodal RL

| # | Reference | Venue / ID | Link | Used for in the base paper |
|---|---|---|---|---|
| 1 | Simmons-Edler, Miltner & Seung (2018). *Program Synthesis Through Reinforcement Learning Guided Tree Search* | arXiv:1806.02932 | <https://arxiv.org/abs/1806.02932> | §1: program synthesis as an example of a domain with many equally good outcomes |
| 2 | Park, Ahn, Choi & Kim (2025). *Mol-AIR: Molecular RL with Adaptive Intrinsic Rewards for Goal-directed Molecular Generation* | J. Chem. Inf. Model. 65(5):2283–2296 | <https://doi.org/10.1021/acs.jcim.4c01669> (arXiv:2403.20109) | §1: molecular discovery as a multimodal domain (also an intrinsic-reward exploration method) |
| 3 | Grinsztajn, Furelos-Blanco, Surana, Bonnet & Barrett (2023). *Winner Takes It All: Training Performant RL Populations for Combinatorial Optimization* | arXiv:2210.03475 | <https://arxiv.org/abs/2210.03475> | §1: combinatorial generation (population-based diversity) |
| 4 | Romera-Paredes et al. (2024). *Mathematical discoveries from program search with large language models* (FunSearch) | Nature 625(7995):468–475 | <https://doi.org/10.1038/s41586-023-06924-6> | §1: discovery tasks need coverage of many alternatives |
| 5 | Castro, Tomasev et al. (2025). *Discovering Symbolic Cognitive Models from Human and Animal Behavior* | bioRxiv 2025 (also ICML 2025, PMLR 267) | <https://doi.org/10.1101/2025.02.05.636732>, <https://proceedings.mlr.press/v267/castro25a.html> | §1: same point (LLM-guided program search) |
| 6 | Novikov et al. (2025). *AlphaEvolve: A coding agent for scientific and algorithmic discovery* | arXiv:2506.13131 | <https://arxiv.org/abs/2506.13131> | §1: same point |

### 1.2 Evidence that reward optimization reduces outcome diversity

| # | Reference | Venue / ID | Link | Used for |
|---|---|---|---|---|
| 7 | Hu, E. J., Jain, Elmoznino, Kaddar, Lajoie, Bengio & Malkin (2024). *Amortizing intractable inference in large language models* | ICLR 2024 (arXiv:2310.04363) | <https://openreview.net/forum?id=Ouj6p4ca60> | §1: RL fine-tuning collapses onto few outcomes; also GFlowNet fine-tuning of LLMs (a competitor) |
| 8 | Gao, Schulman & Hilton (2023). *Scaling Laws for Reward Model Overoptimization* | ICML 2023, PMLR 202:10835–10866 | <https://proceedings.mlr.press/v202/gao23h.html> | §1 collapse; §2.3 preference optimization favours high-probability responses |
| 9 | GX-Chen, Prakash, Guo, Fergus & Ranganath (2025). *KL-Regularized RL is Designed to Mode Collapse* | arXiv:2510.20817; ICLR 2026 (as "...for Generative Modelling Is Designed to Mode Collapse") | <https://arxiv.org/abs/2510.20817> | §1: source of the term "outcome-level mode collapse"; §2.1 KL constraints |
| 10 | Kirk et al. (2024). *Understanding the Effects of RLHF on LLM Generalisation and Diversity* | arXiv:2310.06452 | <https://arxiv.org/abs/2310.06452> | §2.3: RLHF reduces output diversity |
| 11 | Slocum, Parker-Sartori & Hadfield-Menell (2025). *Diverse Preference Learning for Capabilities and Alignment* | arXiv:2511.08594 (journal-ref: ICLR 2025) | <https://arxiv.org/abs/2511.08594> | §2.3: same |
| 12 | Padmakumar & He (2024). *Does Writing with Language Models Reduce Content Diversity?* | ICLR 2024 (arXiv:2309.05196) | <https://arxiv.org/abs/2309.05196> | §2.3: homogenization of outputs |

### 1.3 The "usual explanations" the base paper argues against

| # | Reference | Venue / ID | Link | Used for |
|---|---|---|---|---|
| 13 | Song, Kempe & Munos (2025). *Outcome-based Exploration for LLM Reasoning* | arXiv:2509.06941 | <https://arxiv.org/abs/2509.06941> | §1 "insufficient exploration"; §2.1 exploration bonuses |
| 14 | Wang, S. et al. (2025). *Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL for LLM Reasoning* | NeurIPS 2025 (arXiv:2506.01939) | <https://arxiv.org/abs/2506.01939> | §1 "insufficient exploration" |
| 15 | Wang, C., Jiang, Yang, Liu & Chen (2023). *Beyond Reverse KL: Generalizing DPO with Diverse Divergence Constraints* | arXiv:2309.16240 | <https://arxiv.org/abs/2309.16240> | §1 "suboptimal regularization" |
| 16 | Dohare, Lan & Mahmood (2023). *Overcoming Policy Collapse in Deep RL* | EWRL 2023 | <https://openreview.net/forum?id=m9Jfdz4ymO> | §1 "poor choice of hyperparameters" |

### 1.4 Mode collapse in GANs

| # | Reference | Venue / ID | Link | Used for |
|---|---|---|---|---|
| 17 | Goodfellow et al. (2020). *Generative Adversarial Networks* | Commun. ACM 63(11):139–144 (original: arXiv:1406.2661) | <https://doi.org/10.1145/3422622> | §2.1: origin of the term "mode collapse" |
| 18 | Kossale, Airaj & Darouichi (2022). *Mode Collapse in Generative Adversarial Networks: An Overview* | ICOA 2022 (IEEE) | <https://doi.org/10.1109/ICOA55659.2022.9934291> | §2.1: same |

### 1.5 Exploration, entropy and regularization remedies

| # | Reference | Venue / ID | Link | Used for |
|---|---|---|---|---|
| 19 | Haarnoja, Zhou, Abbeel & Levine (2018). *Soft Actor-Critic* | ICML 2018, PMLR 80:1861–1870 | <https://proceedings.mlr.press/v80/haarnoja18b.html> | §2.1: entropy regularization (MaxEnt RL) |
| 20 | Cheng et al. (2025). *Reasoning with Exploration: An Entropy Perspective* | arXiv:2506.14758 (AAAI 2026) | <https://arxiv.org/abs/2506.14758> | §2.1: entropy-based fix for LLM RL |
| 21 | Ahmed, Le Roux, Norouzi & Schuurmans (2019). *Understanding the Impact of Entropy on Policy Optimization* | ICML 2019, PMLR 97:151–160 | <https://proceedings.mlr.press/v97/ahmed19a.html> | §2.1: entropy regularization |
| 22 | Zhou, R. et al. (2025). *Co-GRPO: Co-Optimized GRPO for Masked Diffusion Model* | arXiv:2512.22288 | <https://arxiv.org/abs/2512.22288> | §2.1: cited for "increasing entropy in the training data". Its abstract is about jointly optimizing a masked-diffusion model and its decoding schedule, so the link is weak (see notes) |
| 23 | Cui et al. (2025). *The Entropy Mechanism of RL for Reasoning Language Models* | arXiv:2505.22617 | <https://arxiv.org/abs/2505.22617> | §2.1: entropy collapse (Clip-Cov, KL-Cov) |
| 24 | Rietz & Stork (2023). *Diversity for Contingency: Learning Diverse Behaviors for Efficient Adaptation and Transfer* | arXiv:2310.07493 (IROS 2023 workshop) | <https://arxiv.org/abs/2310.07493> | §2.1: "constraints between successive policies" (each new policy must be unlikely under the earlier ones) |
| 25 | An, Qin, Kong & Flamant (2025). *DeRL: Diverse-exploration RL for LLMs improves mathematical reasoning* | OpenReview ZIYYeTkZQ4 | <https://openreview.net/forum?id=ZIYYeTkZQ4> | §2.1: reward shaping (an LLM judge rewards approaches unlike earlier ones) |
| 26 | Davoodabadi, Dijujin & Baghshah (2024). *PreND: Enhancing Intrinsic Motivation in RL through Pre-trained Network Distillation* | arXiv:2410.01745 | <https://arxiv.org/abs/2410.01745> | §2.1: intrinsic motivation |

### 1.6 Flow-based and energy-based distribution matching

| # | Reference | Venue / ID | Link | Used for |
|---|---|---|---|---|
| 27 | Bengio, Y., Lahlou, Deleu, Hu, Tiwari & Bengio, E. (2023). *GFlowNet Foundations* | JMLR 24(210):1–55 (arXiv:2111.09266) | <https://jmlr.org/papers/v24/22-0364.html> | §2.2: GFlowNets sample terminal states in proportion to reward |
| 28 | Zhu, X. et al. (2025). *FlowRL: Matching Reward Distributions for LLM Reasoning* | arXiv:2509.15207 | <https://arxiv.org/abs/2509.15207> | §2.2 flow-based RL; **baseline** in §5.1–5.2 |
| 29 | Chao, Feng, Sun, Lee, See & Lee (2024). *Maximum Entropy RL via Energy-Based Normalizing Flow* | NeurIPS 2024 (arXiv:2405.13629) | <https://arxiv.org/abs/2405.13629> | §2.2: energy-based approaches |

### 1.7 LLM post-training, RLHF, reward hacking and decoding

| # | Reference | Venue / ID | Link | Used for |
|---|---|---|---|---|
| 30 | Ouyang et al. (2022). *Training language models to follow instructions with human feedback* | NeurIPS 35:27730–27744 (arXiv:2203.02155) | <https://arxiv.org/abs/2203.02155> | §1 LLM post-training; §2.3 RLHF, strong KL regularization, reward misspecification |
| 31 | Bai et al. (2022). *Constitutional AI: Harmlessness from AI Feedback* | arXiv:2212.08073 | <https://arxiv.org/abs/2212.08073> | §2.3: RLHF/RLAIF context |
| 32 | Rafailov et al. (2024). *Direct Preference Optimization* | arXiv:2305.18290 (NeurIPS 2023) | <https://arxiv.org/abs/2305.18290> | §2.3: preference fine-tuning favours high-probability responses |
| 33 | Ziegler et al. (2020). *Fine-Tuning Language Models from Human Preferences* | arXiv:1909.08593 | <https://arxiv.org/abs/1909.08593> | §2.3: KL regularization |
| 34 | Stiennon et al. (2020). *Learning to summarize from human feedback* | NeurIPS 2020 (arXiv:2009.01325) | <https://arxiv.org/abs/2009.01325> | §2.3: early stopping |
| 35 | Holtzman, Buys, Du, Forbes & Choi (2020). *The Curious Case of Neural Text Degeneration* | ICLR 2020 (arXiv:1904.09751) | <https://arxiv.org/abs/1904.09751> | §2.3: decoding-time heuristics (nucleus sampling, temperature) |
| 36 | Amodei et al. (2016). *Concrete Problems in AI Safety* | arXiv:1606.06565 | <https://arxiv.org/abs/1606.06565> | §2.3: reward hacking and misspecification |

### 1.8 Optimizer

| # | Reference | Venue / ID | Link | Used for |
|---|---|---|---|---|
| 37 | Shao et al. (2024). *DeepSeekMath* (introduces GRPO) | arXiv:2402.03300 | <https://arxiv.org/abs/2402.03300> | §1 (the only place GRPO is cited); GRPO is the optimizer IPS is plugged into (§4.4, Algorithm 1) and the main **baseline** (§5) |

### 1.9 Benchmarks and molecular design

| # | Reference | Venue / ID | Link | Used for |
|---|---|---|---|---|
| 38 | Chen, T. et al. (2025). *HypoSpace* (cited title: "Evaluating LLM creativity as set-valued hypothesis generators under underdetermination") | arXiv:2510.15614 (title changed in v3) | <https://arxiv.org/abs/2510.15614> | §5.2: benchmark with a finite set of admissible answers |
| 39 | Moret et al. (2023). *Leveraging molecular structure and bioactivity with chemical language models for de novo drug design* | Nat. Commun. 14:114 | <https://doi.org/10.1038/s41467-022-35692-6> | §5: chemical language models |
| 40 | Guo, J. et al. (2025). *Generative Molecular Design with Steerable and Granular Synthesizability Control* | arXiv:2505.08774 | <https://arxiv.org/abs/2505.08774> | §5.3: source of the SYNTH and ALL-AMIDE rewards |
| 41 | Olivecrona, Blaschke, Engkvist & Chen (2017). *Molecular de-novo design through deep RL* (REINVENT) | J. Cheminform. 9:48 (arXiv:1704.07555) | <https://doi.org/10.1186/s13321-017-0235-x> | §5.3: **REINVENT baseline** |
| 42 | Guo & Schwaller (2024). *Saturn: Sample-efficient Generative Molecular Design using Memory Manipulation* | arXiv:2405.17066 | <https://arxiv.org/abs/2405.17066> | §5.3: REINVENT-style pipeline whose code the base paper uses |
| 43 | Gao, W., Fu, Sun & Coley (2022). *Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization* (PMO) | arXiv:2206.12411 | <https://arxiv.org/abs/2206.12411> | §5.3: "REINVENT is state of the art on standard benchmarks" |

### 1.10 Notes on the base paper's bibliography (useful for the critical-review agent)

- **Typo:** the author line on page 1 reads "Dinabo Liu"; the correspondence line and arXiv say Dianbo Liu.
- **HypoSpace (#38)** now has a different title on arXiv (v3: "A Diagnostic Benchmark for Set-Valued Hypothesis Generation under Underdetermination and Sublinear Coverage Bounds").
- **Castro et al. (#5)** is also published at ICML 2025, so it could be cited from there.
- **Co-GRPO (#22)** is cited for "increasing entropy in the training data". Its abstract does not mention entropy or diversity, so the citation looks misplaced.
- The theory section (§3–4, App. A) cites **no prior work at all**, although several closely related results exist. These are missing citations in the base paper:
  - the "rich-get-richer" loop and the p^β fix (Guu et al. 2017);
  - the O(1/t) rate of softmax policy gradient (Mei et al. 2020);
  - IPS as an on-policy distributional policy gradient / forward KL (Parshakova et al. 2019; Khalifa et al. 2021);
  - the path-multiplicity bias π(x) ∝ n(x)R(x) that motivated GFlowNets, which is exactly the base paper's §6 experiment with 35 vs 7 paths (Bengio et al. 2021; Deleu et al. 2024);
  - DRA-GRPO (May 2025), which already divides GRPO rewards by a soft within-group count;
  - Hu et al. (Jan 2026), with uniqueness weights f^(−α);
  - classical "reward ÷ crowding" equilibria (fitness sharing, the matching law, the ideal free distribution).

  All of these are in Section 2.

---

## 2. Broader literature (2016–2026, emphasis on 2023–2026), with classics where they matter

The "Relation" column says how each work relates to IPS (α = 1) or our α-IPS (r/p^α, p* ∝ r^{1/α}).

### 2.1 Diversity / mode collapse in RL for LLM reasoning (RLVR, GRPO variants)

**Evidence and diagnosis**

| Paper | Venue | Link | Key idea | Relation to IPS / α-IPS |
|---|---|---|---|---|
| Yue et al. 2025, *Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* | NeurIPS 2025 (oral) | <https://arxiv.org/abs/2504.13837> | RLVR raises pass@1, but the base model wins at large k: RL narrows coverage | Main empirical motivation for diversity-preserving objectives |
| Dang et al. 2025, *Weight Ensembling Improves Reasoning in Language Models* | COLM 2025 | <https://arxiv.org/abs/2504.10478> | Pass@k collapses during SFT of reasoning models while pass@1 improves; interpolating with an early checkpoint (WiSE-FT) recovers pass@k | Evidence of collapse |
| Wu et al. 2025, *The Invisible Leash: Why RLVR May or May Not Escape Its Origin* | arXiv | <https://arxiv.org/abs/2507.14843> | RLVR mostly reweights inside the base model's support and can shrink it | Evidence (support shrinkage) |
| Liu, M. et al. 2025, *ProRL* | arXiv | <https://arxiv.org/abs/2505.24864> | Prolonged RL with KL control and resets can expand reasoning boundaries | Counterpoint |
| Liu, Z. et al. 2025, *Understanding R1-Zero-Like Training* (Dr. GRPO) | arXiv | <https://arxiv.org/abs/2503.20783> | An optimization bias in GRPO's normalization inflates response length (especially of wrong answers); Dr. GRPO removes the length and std normalizations | What our mean field leaves out (std normalization) |
| Davis & Recht 2025, *What is the objective of reasoning with RL?* | arXiv | <https://arxiv.org/abs/2510.13651> | Binary-reward RL algorithms are stochastic gradient ascent on a monotone transform of p: GRPO ↔ arcsin√p, rejection sampling ↔ log p | A tool for adding GRPO's normalization to our mean field |
| Bay & Yearick 2026, *GRPO, Dr. GRPO, and DAPO Are Three Operations on One Number* | arXiv | <https://arxiv.org/abs/2607.00152> | All three methods adjust the group standard deviation; the "group-std identity" | Same use as above |
| Barakat et al. 2026, *Why Pass@k Optimization Can Degrade Pass@1* | arXiv | <https://arxiv.org/abs/2602.21189> | Prompt interference under pass@k objectives | A caution for set-level objectives |
| Zhu, X. et al. 2025, *The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning* | NeurIPS 2025 | <https://arxiv.org/abs/2506.01347> | Positive-sample reinforcement sharpens (hurts pass@k); negative-only keeps diversity; W-REINFORCE | Same "probability multiplier on correct samples" diagnosis, a different fix |
| Karan & Du 2025, *Reasoning with Sampling: Your Base Model is Smarter Than You Think* | arXiv | <https://arxiv.org/abs/2510.14901> | Inference-time MCMC sampling from p_base^α (power distribution) | The reverse knob: sharpening a distribution with a power |

**Methods that keep many good answers (direct competitors)**

| Paper | Venue | Link | Key idea | Relation to IPS / α-IPS |
|---|---|---|---|---|
| **Hu, Z. et al. 2026, *Rewarding the Rare: Uniqueness-Aware RL for Creative Problem Solving in LLMs*** | Findings of ACL 2026 (arXiv Jan 13, 2026) | <https://arxiv.org/abs/2601.08763> | An LLM judge clusters rollouts by strategy; the GRPO advantage is multiplied by **w = 1/f^α**, where f is the cluster size and **α ∈ [0,1]**; weights lie in [K^(−α), 1] | **Closest algorithmic precedent of α-IPS.** Same count-power weight (applied to advantages, on strategy clusters, α ≤ 1, no stationary theory) |
| **Chen, X. et al. 2025, *DRA-GRPO*** | ACL 2026 (per arXiv comment; v1 May 14, 2025) | <https://arxiv.org/abs/2505.09655> | R̃ = R / (1 + Σ_{j≠i} s(o_i, o_j)): reward divided by a kernel "soft count" of similar completions; v5 frames it explicitly as inverse propensity scoring | **IPS-GRPO precedent.** With s = 1{same outcome}, R̃ = R/n_o = R/(G·p̂). Predates the base paper; not cited there |
| He, Fried & Welleck 2025, *Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening* | arXiv | <https://arxiv.org/abs/2506.02355> | "Rank bias" in GRPO; correct samples get r·(1 − β_rank·(G − rank)/G), where rank is by likelihood | A rank-based, bounded proxy for 1/p |
| Song, Kempe & Munos 2025 (also base #13) | arXiv | <https://arxiv.org/abs/2509.06941> | UCB-style bonus over historical answer counts; batch penalty −(1/n)·Σ_{j≠i} 1{a_i = a_j} | Additive count terms instead of a multiplicative (G/n)^α |
| Anschel et al. 2025, *Group-Aware RL for Output Diversity in LLMs* (GAPO) | EMNLP 2025 | <https://aclanthology.org/2025.emnlp-main.1649/> | Frequency-aware group reward 1 − abs(f − 1/L) for valid items (L = number of valid items) | Targets the uniform distribution over valid answers (like α → ∞ for binary rewards); needs L |
| Li, T. et al. 2025, *Jointly Reinforcing Diversity and Quality* (DARLING) | arXiv | <https://arxiv.org/abs/2509.02534> | A learned semantic-equivalence partition; reward = quality × diversity | Needs a learned classifier |
| Mohri, Schneider & Wu 2026, *Distributional Alignment Games for Answer-Level Fine-Tuning* | arXiv | <https://arxiv.org/abs/2604.27166> | Answer-level entropy maximization gives the reward −log ν̂(answer); the Gini index gives 1 − ν̂ ("Diversity-GRPO") | Additive inverse-frequency rewards, derived from an objective; IPS is the multiplicative counterpart |
| Li, X. et al. 2026, *Beyond Mode Collapse: Distribution Matching for Diverse Reasoning* (DMPO) | arXiv | <https://arxiv.org/abs/2605.19461> | Group-level target ∝ reward over the sampled trajectories; forward-KL alignment | The forward-KL view of IPS, done at the group level |
| Plyusov et al. 2026, *F-GRPO: Don't Let Your Policy Learn the Obvious and Forget the Rare* | arXiv | <https://arxiv.org/abs/2602.06717> | Probability of "tail-miss" events as a function of group size (non-monotone); focal-loss-style down-weighting of high-success groups | **Closest finite-G analysis** to our S4 (different question: missing rare correct samples, not a survival threshold) |
| Yao et al. 2025, *Diversity-Aware Policy Optimization for LLM Reasoning* | arXiv | <https://arxiv.org/abs/2505.23433> | Token-level diversity objective applied to positive samples | Entropy-style regularizer |
| Wan et al. 2026, *DSDR: Dual-Scale Diversity Regularization* | arXiv | <https://arxiv.org/abs/2602.19895> | Diversity among correct trajectories (global) plus entropy within correct trajectories (local) | Regularizer-based |
| Hamid et al. 2025, *Polychromic Objectives for RL* | arXiv | <https://arxiv.org/abs/2509.25424> | Set-level objective rewarding success and diversity; PPO with vine sampling | Actor-critic, set-level |
| Tuyls et al. 2025, *Representation-Based Exploration for Language Models* | ICLR 2026 | <https://arxiv.org/abs/2510.11686> | Novelty bonus computed from the base model's hidden states | Exploration bonus |
| Zhang, X. et al. 2025, *Count Counts* (MERCI) | ICLR 2026 | <https://arxiv.org/abs/2510.16614> | Pseudo-count intrinsic reward from a coin-flipping network | Count-based bonus (additive) |
| Wang, T. et al. 2026, *Anchored Policy Optimization* | arXiv | <https://arxiv.org/abs/2602.05717> | "Recursive space contraction"; anchors to the reference model's high-confidence support | Support-coverage regularizer |
| Walder & Karkhanis 2025, *Pass@K Policy Optimization* (PKPO) | arXiv | <https://arxiv.org/abs/2505.15201> | Unbiased low-variance pass@k estimators that become a reward transformation | Set-level objective; no target distribution |
| Chen, Z. et al. 2025, *Pass@k Training* | arXiv | <https://arxiv.org/abs/2508.10751> | Pass@k as the reward, with an analytic advantage | Same |
| Tang, Zheng, Synnaeve & Munos 2025, *Optimizing LMs for Inference Time Objectives using RL* | ICML 2025 | <https://arxiv.org/abs/2503.19595> | RL for pass@k and majority-vote objectives | Same |
| Chow et al. 2024, *Inference-Aware Fine-Tuning for Best-of-N Sampling* | arXiv | <https://arxiv.org/abs/2412.15287> | Best-of-N-aware objectives | Same |
| Yu, Q. et al. 2025, *DAPO* | arXiv | <https://arxiv.org/abs/2503.14476> | Clip-higher (stops entropy collapse), dynamic sampling | Treats the symptom; can be combined with α-IPS |
| Zhang, J. et al. 2025, *Verbalized Sampling* | arXiv | <https://arxiv.org/abs/2510.01171> | Mode collapse traced to typicality bias in preference data; a prompting fix | Inference-time, no training |
| Lanchantin et al. 2025, *Diverse Preference Optimization* (DivPO) | arXiv | <https://arxiv.org/abs/2501.18101> | Preference pairs chosen for diversity | Preference-learning analogue |
| Li, Z. et al. 2025, *Preserving Diversity in SFT of LLMs* (GEM) | ICLR 2025 | <https://arxiv.org/abs/2408.16673> | Game-theoretic SFT objective, equivalent to reverse KL with entropy regularization | SFT analogue |

**Objective families and finite-group estimators (most important for our theory)**

| Paper | Venue | Link | Key idea | Relation to our report |
|---|---|---|---|---|
| **Lin & Ie 2026, *How Fast Should a Model Commit to Supervision? Training Reasoning Models on the Tsallis Loss Continuum*** | arXiv (Apr 2026) | <https://arxiv.org/abs/2604.25907> | Loss family J_Q = −log_q P (Tsallis q-log), from RLVR (q=0) to log-likelihood (q=1); gradient = **P^(−q) × RL gradient**; **Thm 2.1: the categorical minimizer is the escort distribution θ_j ∝ α_j^{1/q}**; gradient-flow escape times Ω(1/p0) vs Θ(log 1/p0); estimator bias O(q/(M P^q)) | **Our S1 is their Thm 2.1** (with α ↔ q, r ↔ α_j). Our α-flow is the gradient flow of Σ r·ln_α p (Section 2.13, C2). They work per prompt; we work per outcome and also cover α > 1 |
| **Tajwar et al. 2026, *Maximum Likelihood RL* (MaxRL)** | ICML 2026 | <https://arxiv.org/abs/2602.02710> | Expected reward is a first-order approximation of log-likelihood; the truncated objective −Σ_{k≤T}(1−p)^k/k; **Thm 2: the 1/K-normalized estimator has mean ∇p·(1 − (1−p)^N)/p** | **Our α=1 identity w_G = (1−(1−p)^G)/p is their Thm 2**, here applied per outcome (Section 2.13, C4) |
| **Zheng 2026, *RL2ML: Finite-Rollout Surrogate Objectives*** | arXiv (May 2026) | <https://arxiv.org/abs/2605.30154> | **Bernstein representation:** any estimator that depends on a group only through the success count has population weight Σ_m β_m B_{m,N−1}(p); "group-level update scale"; sub- and supercritical regimes | **Our size-biasing identity w_G(p) = E[ω(1+B)] is this representation.** Also relevant to S4 |
| Sun, K. et al. 2026, *CurveRL* | arXiv | <https://arxiv.org/abs/2605.24331> | Prompt reweighting as the functional derivative of a utility of pass rates | Same "objective = utility of p" view |
| Wang, Z. et al. 2026, *Gradients Must Earn Their Influence* (DEFT) | arXiv | <https://arxiv.org/abs/2602.11424> | Deformed-log (Tsallis-type) token-level SFT; gate p^α | Token-level relative of the α family |
| Li, G. et al. 2025, *Beyond Log Likelihood: Probability-Based Objectives for SFT* | ICML 2026 (per arXiv comment) | <https://arxiv.org/abs/2510.00526> | −p and −p^k versus −log p objectives across model capability | Same family (the q-log endpoints) |

**Papers that cite the base paper (found through a citation index; all checked on arXiv):** Liu, Y. et al. 2026, *MEDS: Memory-Enhanced Dynamic Reward Shaping* (<https://arxiv.org/abs/2604.11297>); Hayes et al. 2026, *Beyond the Best Guess: ... Evolution Strategies* (<https://arxiv.org/abs/2608.12679>); Zhou, J. et al. 2026, *PIVOT* (<https://arxiv.org/abs/2609.35303>). None of them analyses IPS itself.

### 2.2 GFlowNets, FlowRL and reward temperature (p ∝ R^β)

GFlowNets learn a sequential sampler with p(x) ∝ R(x). In practice they very often use a **reward exponent β**, sampling **p(x) ∝ R(x)^β**. This is **exactly our stationary family with β = 1/α**: α < 1 ↔ β > 1 (sharper), α > 1 ↔ β < 1 (flatter).

| Paper | Venue | Link | What it does with the exponent / relation |
|---|---|---|---|
| Bengio, E., Jain, Korablyov, Precup & Bengio, Y. 2021, *Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation* | NeurIPS 2021 | <https://arxiv.org/abs/2106.04399> | Trains on R(x)^β with **β = 4** for molecules. **Proposition: sampling action sequences ∝ R gives π(x) ∝ n(x)R(x) when n(x) paths lead to x.** This is the path-multiplicity bias in the base paper's §6 (35 vs 7 paths) |
| Bengio, Y. et al. 2023, *GFlowNet Foundations* (base #27) | JMLR | <https://jmlr.org/papers/v24/22-0364.html> | Theory |
| Malkin et al. 2022, *Trajectory Balance* | NeurIPS 2022 | <https://arxiv.org/abs/2201.13259> | Reward exponents **β ∈ {4, 8, 10, 16}** (molecules) and β ∈ {2,3,4} in other tasks. Note: arXiv now lists the first author as E. S. Whitammer |
| Jain et al. 2022, *Biological Sequence Design with GFlowNets* | ICML 2022 | <https://arxiv.org/abs/2203.04115> | "β: reward exponent R(x)^β" is a tuned hyperparameter |
| Jain et al. 2023, *Multi-Objective GFlowNets* | ICML 2023 | <https://arxiv.org/abs/2210.12765> | Preference-conditional sampling π(x \| ω) ∝ R(x \| ω)^β; the paper studies β ∈ {16, 32, 48, 96} and notes that changing β trades diversity for reward |
| **Kim, M. et al. 2024, *Learning to Scale Logits for Temperature-Conditional GFlowNets*** | ICML 2024 | <https://arxiv.org/abs/2310.02823> | **p(x \| β) ∝ R(x)^β, one network for all β** ("temperature-conditional"). This is the GFlowNet version of a learnable "diversity knob" |
| Kim, M. et al. 2024, *Local Search GFlowNets* | ICLR 2024 | <https://arxiv.org/abs/2310.02710> | Better mode discovery |
| Zhang, D. et al. 2023, *Let the Flows Tell* | NeurIPS 2023 | <https://arxiv.org/abs/2305.17010> | Combinatorial optimization with GFlowNets |
| Malkin et al. 2023, *GFlowNets and Variational Inference* | ICLR 2023 | <https://arxiv.org/abs/2210.00580> | Links trajectory balance to variational inference |
| Tiapkin et al. 2024, *GFlowNets as Entropy-Regularized RL* | AISTATS 2024 | <https://arxiv.org/abs/2310.12934> | A GFlowNet is soft RL with a specific reward correction |
| Deleu et al. 2024, *Discrete Probabilistic Inference as Control in Multi-path Environments* | UAI 2024 | <https://arxiv.org/abs/2402.10309> | Corrects MaxEnt-RL rewards so the marginal over objects is ∝ reward despite multiple paths (the bias in the base paper's §6) |
| Mohammadpour et al. 2024, *MaxEnt GFlowNets with Soft Q-learning* | AISTATS 2024 | <https://arxiv.org/abs/2312.14331> | Same link from the soft-Q side |
| Lee, S. et al. 2025, *Learning diverse attacks on LLMs for robust red-teaming* | ICLR 2025 | <https://arxiv.org/abs/2405.18540> | GFlowNet fine-tuning of LLMs for diversity |
| Yu, F. et al. 2025, *Flow of Reasoning* | ICML 2025 | <https://arxiv.org/abs/2406.05673> | GFlowNet-style training for divergent reasoning |
| Bartoldson et al. 2025, *Trajectory Balance with Asynchrony* | NeurIPS 2025 | <https://arxiv.org/abs/2503.18929> | Scalable trajectory-balance post-training |
| Zhu, X. et al. 2025, FlowRL (base #28) | arXiv | <https://arxiv.org/abs/2509.15207> | Reverse KL (trajectory balance) to a target built from exp(β·r), with a learned partition function Z_φ(x); the final objective adds the reference model as a prior, i.e. target ∝ π_ref·exp(β·r) |

### 2.3 Maximum-entropy RL, entropy regularization and softmax policy-gradient theory

| Paper | Venue | Link | Relation |
|---|---|---|---|
| Williams & Peng 1991, *Function optimization using connectionist RL algorithms* | Connection Science 3(3):241–268 | <https://doi.org/10.1080/09540099108946587> | Origin of the entropy bonus in policy gradients |
| Williams 1992, *REINFORCE* | Machine Learning 8:229–256 | <https://doi.org/10.1007/BF00992696> | The score-function gradient that IPS reweights |
| Mnih et al. 2016, *A3C* | ICML 2016 | <https://arxiv.org/abs/1602.01783> | Entropy bonus in deep RL |
| Haarnoja et al. 2017, *RL with Deep Energy-Based Policies* | arXiv | <https://arxiv.org/abs/1702.08165> | Soft Q-learning: multimodal Boltzmann policies |
| Levine 2018, *RL and Control as Probabilistic Inference* | arXiv | <https://arxiv.org/abs/1805.00909> | The KL / MaxEnt-as-inference view |
| **Mei, Xiao, Szepesvári & Schuurmans 2020, *On the Global Convergence Rates of Softmax Policy Gradient Methods*** | ICML 2020, PMLR 119:6820–6829 | <https://proceedings.mlr.press/v119/mei20b.html> | **O(1/t) and a matching Ω(1/t) lower bound for exact softmax PG** (= our α = 0 algebraic rate); linear rate with entropy regularization |
| Mei et al. 2020, *Escaping the Gravitational Pull of Softmax* | NeurIPS 2020 | <https://proceedings.neurips.cc/paper/2020/hash/f1cf2a082126bf02de0b307778ce73a7-Abstract.html> | "Softmax gravity well" (sensitivity to initialization), "softmax damping" for log-probabilities; proposes the escort transform |
| Agarwal, Kakade, Lee & Mahajan 2019, *On the Theory of Policy Gradient Methods* | arXiv | <https://arxiv.org/abs/1908.00261> | Convergence of softmax / natural PG |
| Cen et al. 2022, *Fast Global Convergence of NPG with Entropy Regularization* | Operations Research 70(4):2563–2578 | <https://arxiv.org/abs/2007.06558> | Linear rates under entropy regularization |
| Mei et al. 2024, *Stochastic Gradient Succeeds for Bandits* | arXiv (correction of the ICML 2023 version) | <https://arxiv.org/abs/2402.17235> | The stochastic gradient bandit converges to the best arm (i.e., collapses) at O(1/t) |
| Mei et al. 2024, *Small steps no more* | NeurIPS 2024 | <https://arxiv.org/abs/2502.07141> | Global convergence for any constant learning rate |
| Ding & Soricut 2017, *Cold-Start RL with Softmax Policy Gradient* | NeurIPS 2017 | <https://arxiv.org/abs/1709.09346> | Softmax value function that combines policy gradient with maximum-likelihood-style training (no warm start needed) |
| Lee, Choi & Oh 2018, *Sparse MDPs with Causal Sparse Tsallis Entropy Regularization* | arXiv | <https://arxiv.org/abs/1709.06293> | Tsallis entropy regularizer |
| Lee et al. 2019, *Tsallis RL: A Unified Framework for MaxEnt RL* | arXiv | <https://arxiv.org/abs/1902.00137> | Tsallis entropic index as a knob |
| Nachum, Chow & Ghavamzadeh 2018, *Path Consistency Learning in Tsallis Entropy Regularized MDPs* | arXiv | <https://arxiv.org/abs/1802.03501> | Same |

### 2.4 KL-regularized RL, distribution matching, Tsallis / α- / f-divergences

| Paper | Venue | Link | Relation |
|---|---|---|---|
| Norouzi et al. 2016, *Reward Augmented Maximum Likelihood* (RAML) | NIPS 2016 | <https://arxiv.org/abs/1609.00150> | Forward KL to exp(r/τ) (mass-covering) versus the reverse KL of RL |
| **Guu, Pasupat, Liu & Liang 2017, *From Language to Programs: Bridging RL and Maximum Marginal Likelihood*** | ACL 2017, pp. 1051–1062 | <https://doi.org/10.18653/v1/P17-1097> | **"The rich get richer, and the poor get poorer"**: both RL and MML favour already-likely programs. Fix: **β-meritocratic gradient weights q_β ∝ p^β over reward-earning programs**; β = 0 gives every correct program equal weight. In our notation the gradient weight is r·p^(1−α), so **β = 1 − α**. An IPS-like idea with a tunable exponent, 9 years earlier |
| Tan et al. 2018, *Connecting the Dots Between MLE and RL for Sequence Prediction* | arXiv | <https://arxiv.org/abs/1811.09740> | One family that interpolates MLE, RAML and RL |
| **Parshakova, Andreoli & Dymetman 2019, *Distributional RL for Energy-Based Sequential Models*** (DPG) | arXiv (OptRL workshop, NeurIPS 2019) | <https://arxiv.org/abs/1912.08517> | **Distributional policy gradient: ∇CE(p, π_θ) = −E_q[(p/q)∇log π_θ].** With q = π_θ and target P = r this is exactly the IPS update (up to the constant 1/Z) |
| **Khalifa, Elsahar & Dymetman 2021, *A Distributional Approach to Controlled Text Generation*** (GDC) | ICLR 2021 | <https://arxiv.org/abs/2012.11635> | KL-adaptive DPG; states the DPG identity explicitly |
| Korbak, Perez & Buckley 2022, *RL with KL penalties is better viewed as Bayesian inference* | Findings of EMNLP 2022 | <https://arxiv.org/abs/2205.11275> | KL-regularized RL targets π_ref·exp(r/β) |
| Korbak et al. 2022, *On RL and Distribution Matching for Fine-Tuning LMs with no Catastrophic Forgetting* | arXiv | <https://arxiv.org/abs/2206.00761> | RL versus distribution matching |
| Go et al. 2023, *Aligning LMs with Preferences through f-divergence Minimization* (f-DPG) | arXiv | <https://arxiv.org/abs/2302.08215> | Any f-divergence toward an explicit target; different divergences give different alignment–diversity trade-offs (Jensen–Shannon is often a good balance) |
| GX-Chen et al. 2025/2026 (base #9) | ICLR 2026 | <https://arxiv.org/abs/2510.20817> | Reverse or forward KL picks the family of targets; low β with equal rewards gives unimodal targets; **MARA** fix |
| Li, L. et al. 2025, *The Choice of Divergence* (DPH-RL) | arXiv | <https://arxiv.org/abs/2509.07430> | Mass-covering f-divergences (forward KL, JS) toward the initial policy keep pass@k |
| Kruszewski et al. 2025, *Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity* | ICLR 2026 | <https://arxiv.org/abs/2512.05962> | Target = base model filtered to correct answers; **α-divergence family** to tune precision versus diversity |
| Li & Turner 2016, *Rényi Divergence Variational Inference* | NIPS 2016 | <https://arxiv.org/abs/1602.02311> | α-family between mode-seeking and mass-covering |
| Hernández-Lobato et al. 2016, *Black-box α-divergence Minimization* | ICML 2016 | <https://arxiv.org/abs/1511.03243> | Same |
| Cressie & Read 1984, *Multinomial Goodness-of-Fit Tests* (power divergence) | JRSS-B 46(3):440–464 | <https://doi.org/10.1111/j.2517-6161.1984.tb01318.x> | Power-divergence family |
| Tsallis 1988, *Possible generalization of Boltzmann–Gibbs statistics* | J. Stat. Phys. 52:479–487 | <https://doi.org/10.1007/BF01016429> | Origin of the q-logarithm ln_q |
| Beck & Schlögl 1993, *Thermodynamics of Chaotic Systems* (book) | CUP | <https://doi.org/10.1017/CBO9780511524585> | **Escort distributions** p_i^q/Σp^q, the name for the p* ∝ r^{1/α} family |
| **Ferrari & Yang 2010, *Maximum Lq-likelihood estimation*** | Ann. Statist. 38(2):753–783 | <https://doi.org/10.1214/09-AOS687> | Maximizes Σ ln_q f(x_i). **Our α-IPS objective Σ r·ln_α p is a reward-weighted Lq-likelihood** |
| Zhang & Sabuncu 2018, *Generalized Cross Entropy Loss* | NeurIPS 2018 | <https://arxiv.org/abs/1805.07836> | L_q = (1 − p^q)/q interpolates CE and MAE, the same family at the label level |
| Peters & Schaal 2007, *RL by reward-weighted regression* | ICML 2007 | <https://doi.org/10.1145/1273496.1273590> | Reward-weighted maximum likelihood |
| Zelikman et al. 2022, *STaR* | arXiv | <https://arxiv.org/abs/2203.14465> | Rejection-sampling fine-tuning (a log-p objective, per Davis & Recht) |
| Singh et al. 2023, *Beyond Human Data* (ReST-EM) | TMLR | <https://arxiv.org/abs/2312.06585> | Same |
| Dong et al. 2023, *RAFT* | TMLR | <https://arxiv.org/abs/2304.06767> | Same |

### 2.5 Quality-diversity, niching and population methods

| Paper | Venue | Link | Relation |
|---|---|---|---|
| **Goldberg & Richardson 1987, *Genetic Algorithms with Sharing for Multimodal Function Optimization*** | Proc. 2nd ICGA (Lawrence Erlbaum), pp. 41–49 | (no DOI; confirmed through the survey below and web search) | **Fitness sharing: f_shared = f / (niche count).** The population settles on each peak in proportion to its height. **This is the α = 1 rule (reward ÷ count) in evolutionary computation, ~40 years earlier** |
| Wong 2015, *Evolutionary Multimodal Optimization: A Short Survey* | arXiv | <https://arxiv.org/abs/1508.00457> | States the sharing equilibrium ("each peak receiving a fraction of the population in proportion to the height of that peak") |
| Horn, Goldberg & Deb 1994, *Implicit Niching in a Learning Classifier System: Nature's Way* | Evolutionary Computation 2(1):37–66 | <https://doi.org/10.1162/evco.1994.2.1.37> | Resource sharing gives proportional allocation |
| Lehman & Stanley 2011, *Abandoning Objectives: Novelty Search* | Evolutionary Computation 19(2):189–223 | <https://doi.org/10.1162/EVCO_a_00025> | Novelty instead of objective |
| Mouret & Clune 2015, *Illuminating search spaces by mapping elites* (MAP-Elites) | arXiv | <https://arxiv.org/abs/1504.04909> | Quality-diversity archive |
| Pugh, Soros & Stanley 2016, *Quality Diversity: A New Frontier for Evolutionary Computation* | Front. Robot. AI 3:40 | <https://doi.org/10.3389/frobt.2016.00040> | QD survey |
| Eysenbach et al. 2018, *Diversity is All You Need* (DIAYN) | arXiv | <https://arxiv.org/abs/1802.06070> | Skill diversity through mutual information |
| Parker-Holder et al. 2020, *Effective Diversity in Population Based RL* | NeurIPS 2020 | <https://arxiv.org/abs/2002.00632> | Population diversity (determinant) |
| Zhou, Z. et al. 2022, *Reward-Switching Policy Optimization* | ICLR 2022 | <https://arxiv.org/abs/2204.02246> | Iteratively discovers novel strategies |

### 2.6 Count-based exploration (our ω(n) = (G/n)^α is a multiplicative count weight)

| Paper | Venue | Link | Relation |
|---|---|---|---|
| Bellemare et al. 2016, *Unifying Count-Based Exploration and Intrinsic Motivation* | arXiv | <https://arxiv.org/abs/1606.01868> | Additive bonus ∝ 1/√N(s). α-IPS uses a multiplicative (G/n)^α; α = 1/2 looks like the √ count bonus |
| Tang, H. et al. 2017, *#Exploration* | NIPS 2017 | <https://arxiv.org/abs/1611.04717> | Hash-based counts |
| Burda et al. 2018, *Random Network Distillation* | arXiv | <https://arxiv.org/abs/1810.12894> | Novelty via prediction error |
| Song et al. 2025; Zhang, X. et al. 2025 (MERCI) | see 2.1 | | Count bonuses for LLM answers / trajectories |

### 2.7 "IPS" in bandits and off-policy evaluation (name clash, and the clipping analogy)

| Paper | Venue | Link | Relation |
|---|---|---|---|
| Horvitz & Thompson 1952, *A Generalization of Sampling Without Replacement from a Finite Universe* | JASA 47(260):663–685 | <https://doi.org/10.1080/01621459.1952.10483446> | Origin of inverse-probability weighting |
| Rosenbaum & Rubin 1983, *The central role of the propensity score* | Biometrika 70(1):41–55 | <https://doi.org/10.1093/biomet/70.1.41> | Propensity scores |
| Ionides 2008, *Truncated Importance Sampling* | JCGS 17(2):295–311 | <https://doi.org/10.1198/106186008X320456> | Truncating weights trades bias for variance (compare the clip ε) |
| **Strehl, Langford, Kakade & Li 2010, *Learning from Logged Implicit Exploration Data*** | arXiv | <https://arxiv.org/abs/1003.0120> | **Importance weight 1/max{π̂(a\|x), τ}** with an *estimated* propensity π̂ and a threshold τ, and an analysis of how τ trades bias against variance. **This is exactly the clipped form 1/max(p̂, ε) of IPS-GRPO's Eq. 9** (α = 1), in off-policy evaluation |
| Dudík, Langford & Li 2011, *Doubly Robust Policy Evaluation and Learning* | ICML 2011 | <https://arxiv.org/abs/1103.4601> | Doubly robust |
| Bottou et al. 2013, *Counterfactual Reasoning and Learning Systems* | arXiv | <https://arxiv.org/abs/1209.2355> | Clipped importance weights in a real system (Bing ad placement) |
| Swaminathan & Joachims 2015, *Counterfactual Risk Minimization* | arXiv | <https://arxiv.org/abs/1502.02362> | IPS with weight clipping (constant M) plus a variance penalty (CRM principle, POEM) |
| **Aouali, Brunel, Rohde & Korba 2023, *Exponential Smoothing for Off-Policy Learning*** | ICML 2023 | <https://arxiv.org/abs/2305.15877> | **Defines "IPS-α": weight π(a\|x)/π0(a\|x)^α, α ∈ [0,1]** (α = 1 is standard IPS). The same tunable inverse-probability exponent, used for bias–variance control in off-policy learning. **Name clash with our "α-IPS"** |
| Auer, Cesa-Bianchi, Freund & Schapire 2002, *The Nonstochastic Multiarmed Bandit Problem* (EXP3) | SIAM J. Comput. 32(1):48–77 | <https://doi.org/10.1137/S0097539701398375> | Importance-weighted reward x/p for the chosen arm. There 1/p removes bias; in IPS-GRPO, 1/p **changes the objective** |
| Schaul et al. 2016, *Prioritized Experience Replay* | ICLR 2016 | <https://arxiv.org/abs/1511.05952> | Priorities p_i^α and importance weights (N·P(i))^(−β): a tunable inverse-probability exponent in RL, used for bias correction |
| Russo et al. 2018, *A Tutorial on Thompson Sampling* | Found. Trends ML 11(1) | <https://arxiv.org/abs/1707.02038> | Thompson sampling is "probability matching" in a different sense (probability of being optimal, not ∝ reward) |

### 2.8 Replicator dynamics and evolutionary game theory

| Paper | Venue | Link | Relation |
|---|---|---|---|
| Taylor & Jonker 1978, *Evolutionary stable strategies and game dynamics* | Math. Biosci. 40:145–156 | <https://doi.org/10.1016/0025-5564(78)90077-9> | The replicator equation ṗ_i = p_i(f_i − f̄) |
| Hofbauer & Sigmund 1998, *Evolutionary Games and Population Dynamics* | CUP | <https://doi.org/10.1017/CBO9781139173179> | Textbook |
| Börgers & Sarin 1997, *Learning Through Reinforcement and Replicator Dynamics* | J. Econ. Theory 77:1–14 | <https://doi.org/10.1006/jeth.1997.2319> | Cross's reinforcement learning tends to the replicator equation in continuous time |
| Sato & Crutchfield 2003, *Coupled Replicator Equations for the Dynamics of Learning in Multiagent Systems* | Phys. Rev. E 67:015206 | <https://doi.org/10.1103/PhysRevE.67.015206> | Boltzmann Q-learning gives replicator dynamics plus an entropy term |
| Bloembergen, Tuyls, Hennes & Kaisers 2015, *Evolutionary Dynamics of Multi-Agent Learning: A Survey* | JAIR 53:659–697 | <https://doi.org/10.1613/jair.4818> | Survey |
| Kimura 1962, *On the probability of fixation of mutant genes in a population* | Genetics 47(6):713–719 | <https://doi.org/10.1093/genetics/47.6.713> | Neutral drift: with equal fitness, a finite population still fixes on one type by chance. This is the right analogy for our "collapse at an exact tie is noise-driven" (Experiment 1) |

How the α = 0 flow relates to replicator dynamics:

- In probability space, our α = 0 logit flow ż_i = p_i(r_i − r̄) becomes ṗ_i = p_i(p_i a_i − Σ_k p_k² a_k). This is a replicator equation whose "fitness" p_i·a_i is amplified by popularity.
- The exact replicator equation ṗ_i = p_i(r_i − r̄) is what natural-gradient / Hedge updates (ż = r) or Cross learning give.
- With the IPS fitness f_i = r_i/p_i, the replicator equation becomes ṗ_i = r_i − p_i·Σr. This converges to p ∝ r, which is the ideal-free-distribution equilibrium in Section 2.9.

### 2.9 Outside machine learning: matching law, ideal free distribution, probability matching

| Paper | Venue | Link | Relation |
|---|---|---|---|
| **Herrnstein 1961, *Relative and absolute strength of response as a function of frequency of reinforcement*** | JEAB 4(3):267–272 | <https://doi.org/10.1901/jeab.1961.4-267> | **Matching law:** B_1/(B_1+B_2) = R_1/(R_1+R_2), i.e. behaviour ∝ reinforcement = **p ∝ r (our α = 1)** |
| Herrnstein 1970, *On the law of effect* | JEAB 13(2):243–266 | <https://doi.org/10.1901/jeab.1970.13-243> | Quantitative law of effect |
| **Baum 1974, *On two types of deviation from the matching law: bias and undermatching*** | JEAB 22(1):231–242 | <https://doi.org/10.1901/jeab.1974.22-231> | **Generalized matching law B_1/B_2 = b·(R_1/R_2)^s** ⇒ our **α = 1/s**: undermatching (s < 1) ↔ α > 1, overmatching (s > 1) ↔ α < 1, strict matching ↔ α = 1 |
| Baum 1979, *Matching, undermatching, and overmatching in studies of choice* | JEAB 32(2):269–281 | <https://doi.org/10.1901/jeab.1979.32-269> | Reanalysis of 103 data sets from 23 studies: for response ratios the slope s usually falls below 1 (undermatching); for time ratios it scatters around 1 |
| Vaughan 1981, *Melioration, matching, and maximization* | JEAB 36(2):141–149 | <https://doi.org/10.1901/jeab.1981.36-141> | **Melioration:** behaviour shifts toward the option with the higher **local rate of reinforcement, i.e. reinforcers obtained per unit of behaviour allocated to that option (≈ r/p)**. This is the IPS reward |
| Herrnstein & Prelec 1991, *Melioration: A Theory of Distributed Choice* | J. Econ. Perspect. 5(3):137–156 | <https://doi.org/10.1257/jep.5.3.137> | Accessible review |
| Loewenstein & Seung 2006, *Operant matching is a generic outcome of synaptic plasticity based on the covariance between reward and neural activity* | PNAS 103(41):15224–15229 | <https://doi.org/10.1073/pnas.0505220103> | Covariance (REINFORCE-like) learning rules converge to matching |
| Sakai & Fukai 2008, *The Actor-Critic Learning Is Behind the Matching Law* | Neural Comput. 20(1):227–251 | <https://doi.org/10.1162/neco.2008.20.1.227> | Actor-critic learning yields matching instead of maximizing |
| Loewenstein, Prelec & Seung 2009, *Operant Matching as a Nash Equilibrium of an Intertemporal Game* | Neural Comput. 21(10):2755–2773 | <https://doi.org/10.1162/neco.2009.09-08-854> | Game-theoretic view of matching |
| **Fretwell & Lucas 1969, *On territorial behavior and other factors influencing habitat distribution in birds. I.*** | Acta Biotheoretica 19:16–36 (Crossref date 1969; often cited as 1970) | <https://doi.org/10.1007/BF01601953> | **Ideal free distribution (IFD)**: individuals settle so that every occupied habitat gives equal payoff |
| **Sutherland 1983, *Aggregation and the "ideal free" distribution*** | J. Anim. Ecol. 52(3):821–828 | <https://doi.org/10.2307/4456> | **Interference IFD model:** per-capita gain on patch i is W_i = Q_i·n_i^(−m) (Q_i = patch quality, n_i = number of competitors, m = interference coefficient). Equal gains give **n_i ∝ Q_i^{1/m}**, which is exactly our p* ∝ r^{1/α} with **α = m**. m = 1 gives matching, m < 1 overmatching, m > 1 undermatching (as stated by Sanchez & Gillespie 2022, below) |
| Hassell & Varley 1969, *New inductive population model for insect parasites* | Nature 223:1133–1137 | <https://doi.org/10.1038/2231133a0> | Origin of the interference constant m |
| Tregenza, Parker & Thompson 1996, *Interference and the ideal free distribution: models and tests* | Behav. Ecol. 7(4):379–386 | <https://doi.org/10.1093/beheco/7.4.379> | Reviews the assumptions and predictions of five interference-type distribution models |
| Parker & Sutherland 1986, *Ideal free distributions when individuals differ in competitive ability: phenotype-limited ideal free models* | Anim. Behav. 34(4):1222–1242 | <https://doi.org/10.1016/S0003-3472(86)80182-8> | Source cited for the interference form W_i = Q·n_i^(−m) |
| Sanchez & Gillespie 2022, *Dispersal and distribution of a generalist predator in habitats with multiple food resources* | Front. Ecol. Evol. 10 | <https://doi.org/10.3389/fevo.2022.977689> | Writes the interference model W_i = Q·n_i^(−m) and states: m = 1 gives perfect matching, m < 1 overmatching, m > 1 undermatching. **This is the verified source for the m ↔ α mapping above** |
| Tregenza 1995, *Building on the Ideal Free Distribution* | Adv. Ecol. Res. 26:253–307 | <https://doi.org/10.1016/S0065-2504(08)60067-7> | Review of IFD theory and tests |
| Kennedy & Gray 1993, *Can ecological theory predict the distribution of foraging animals? A critical analysis of experiments on the IFD* | Oikos 68(1):158 | <https://doi.org/10.2307/3545322> | Undermatching is common in IFD experiments |
| Fagen 1987, *A generalized habitat matching rule* | Evol. Ecol. 1:5–10 | <https://doi.org/10.1007/BF02067264> | Generalized habitat matching |
| Fauvergue et al. 2006, *Habitat assessment by parasitoids* | Behav. Ecol. 17(4):522–531 | <https://doi.org/10.1093/beheco/arj063> | States clearly: per-capita intake = input rate / competitor density, and the equilibrium is the **input-matching rule** (share of competitors = share of resources) |
| Vulkan 2000, *An Economist's Perspective on Probability Matching* | J. Econ. Surv. 14(1):101–118 | <https://doi.org/10.1111/1467-6419.00106> | Probability matching: choice frequency ≈ outcome probability |
| Shanks, Tunney & McCarthy 2002, *A re-examination of probability matching and rational choice* | J. Behav. Decis. Mak. 15(3):233–250 | <https://doi.org/10.1002/bdm.413> | Same |

**Why this matters for us.** The IPS reward r(o)/p(o) is the *per-capita share* of a resource that is split among everyone who chose o. The equilibrium p ∝ r is the input-matching rule of the ideal free distribution and Herrnstein's matching law. The α family is Baum's generalized matching law (s = 1/α) and Sutherland's interference model (m = α). These give a ready-made, intuitive interpretation of our "diversity knob". They are also prior art, so S1 should not be presented as new.

### 2.10 The same "frequency to a power" knob elsewhere in ML

| Paper | Venue | Link | The knob |
|---|---|---|---|
| Mikolov et al. 2013, *Distributed Representations of Words and Phrases* | arXiv (NeurIPS 2013) | <https://arxiv.org/abs/1310.4546> | Negative sampling from U(w)^{3/4} |
| Lample & Conneau 2019, *Cross-lingual Language Model Pretraining* (XLM) | arXiv | <https://arxiv.org/abs/1901.07291> | Language sampling q_i ∝ p_i^α, α = 0.5 |
| Conneau et al. 2020, *XLM-R* | ACL 2020 | <https://arxiv.org/abs/1911.02116> | Same with α = 0.3 |
| Arivazhagan et al. 2019, *Massively Multilingual NMT in the Wild* | arXiv | <https://arxiv.org/abs/1907.05019> | Temperature sampling p^{1/T}, T = 5 |
| Cui, Y. et al. 2019, *Class-Balanced Loss Based on Effective Number of Samples* | arXiv (CVPR 2019) | <https://arxiv.org/abs/1901.05555> | Class weights from the "effective number" of samples (a smoothed inverse frequency) |
| Menon et al. 2021, *Long-tail learning via logit adjustment* | ICLR 2021 | <https://arxiv.org/abs/2007.07314> | Logit adjustment τ·log(prior) |

### 2.11 Background for our numerical / probabilistic derivations

| Paper | Venue | Link | Use in our report |
|---|---|---|---|
| Hairer & Wanner 1996, *Solving ODEs II: Stiff and Differential-Algebraic Problems* (2nd ed.) | Springer | <https://doi.org/10.1007/978-3-642-05221-7> | Stability regions of explicit RK methods (RK4 real interval ≈ [−2.785, 0]) for S3 |
| Stephan 1945, *The expected value and variance of the reciprocal and other negative powers of a positive Bernoullian variate* | Ann. Math. Stat. 16(1):50–61 | <https://doi.org/10.1214/aoms/1177731170> | Moments of 1/X for binomial X (our delta-method bias b, s) |
| **Chao & Strawderman 1972, *Negative Moments of Positive Random Variables*** | JASA 67(338):429–431 | <https://doi.org/10.1080/01621459.1972.10482404> | E[(X+a)^(−1)] by integrating the probability generating function, worked out for the binomial and Poisson distributions. With a = 1 and B ~ Bin(G−1, p) it gives E[1/(1+B)] = (1−q^G)/(Gp), **which is our α = 1 identity** |
| Znidaric 2005/2009, *Asymptotic expansion for inverse moments of binomial and Poisson distributions* | Open Stat. Prob. J. 1:7–10 (arXiv math/0511226) | <https://arxiv.org/abs/math/0511226> | Asymptotic expansions of 1/X moments, compare our D expansion |

### 2.12 Direct answer: has anyone already proposed r/p^α or a tunable inverse-probability exponent?

**Yes, in several forms. None is identical to our setting, but together they cover the device and the target family.**

| Prior work | What exactly | How close to α-IPS |
|---|---|---|
| **Hu, Z. et al. 2026** (Jan 2026, ACL Findings) | GRPO advantage × **f^(−α)**, f = strategy-cluster count within the group, α ∈ [0,1] | **Very close** (count-power weight inside GRPO). Differences: it multiplies the normalized *advantage*, not the reward; clusters come from an LLM judge; α ≤ 1; empirical only (no stationary law, rates or survival limit) |
| **DRA-GRPO** (May 2025) | R / (1 + soft count of similar completions) | **IPS-GRPO at α = 1** with a kernel instead of exact counts (no exponent) |
| **Guu et al. 2017** | Gradient weights q_β ∝ p^β over reward-earning programs, β ∈ [0,1] | Same tunable exponent on probability (β = 1 − α), for binary rewards, per-example normalized |
| **Lin & Ie 2026** | Per-prompt amplification P^(−q), q ∈ [0,1]; escort minimizer ∝ α_j^{1/q} | Same objective family; per prompt (success probability), not per outcome |
| **Aouali et al. 2023** ("IPS-α") | π/π0^α in off-policy learning | Same device, different purpose (bias–variance in off-policy evaluation) |
| **Schaul et al. 2016** (PER) | Importance weights (N·P(i))^(−β) | Same device, different purpose (replay correction) |
| **Strehl et al. 2010** | Weight 1/max{π̂, τ} with an estimated propensity π̂ and threshold τ | The **clipped** form of IPS-GRPO's weight (α = 1, ε = τ), including its bias–variance discussion |
| **GFlowNets** (2021–) | Target p ∝ R^β | Same **target** family, different mechanism (flow balance, learned Z) |
| **Baum 1974; Sutherland 1983** | p_1/p_2 = b·(r_1/r_2)^s; share ∝ Q^{1/m} | Same **equilibrium** family in psychology and ecology |

**What we did not find anywhere:**

- the specific on-policy, outcome-level update r/max(p̂, ε)^α with α > 1 allowed, analysed as a dynamical system;
- the finite-group survival threshold ω(1)/ω(G) > r_1/r_2 (and its clipped form α > ln(r_1/r_2)/ln min{G, 1/ε});
- the general-K extinction thresholds;
- the explicit linear rates and step-size limits of this flow.

### 2.13 Mathematical connections we found (our own observations, checked numerically)

These are not taken from a paper. They are short derivations that link our report to the literature above. Each was checked numerically in a quick script: finite-difference gradients agreed to ~1e-9, and a Monte Carlo test of (C4) to ~1e-3 (sampling noise).

- **(C1) IPS is forward-KL / maximum-likelihood matching.** With p̄ = stopgrad(p), the IPS flow ż = r − p·R equals ∇_z Σ_k r_k log p_k(z). The base paper's potential Ψ(z) = log Σ e^{z_k} − Σ p*_k z_k is **exactly the cross-entropy H(p*, p(z))** (difference 0.0 numerically). So IPS performs gradient descent on R·KL(p* ‖ p) + const. This is the on-policy DPG of Parshakova et al. 2019 / Khalifa et al. 2021 with target P = r.
- **(C2) α-IPS is a Tsallis / Lq-likelihood gradient flow.** Our flow ż_i = r_i p_i^{1−α} − p_i Σ_k r_k p_k^{1−α} equals ∇_z L_α with **L_α = Σ_k r_k ln_α(p_k)**, ln_α(u) = (u^{1−α} − 1)/(1−α). L_α is strictly concave in p for every α > 0, so its unique maximizer on the simplex is the escort distribution p* ∝ r^{1/α} (Lin & Ie 2026, Thm 2.1, for q ≤ 1; the same Lagrange argument works for α > 1). This also gives a **Lyapunov function for all α > 0** (dL_α/dt = ‖∇_z L_α‖² ≥ 0), which generalizes the base paper's Ψ. Global convergence still needs a boundary argument for α < 1, where ln_α(0) is finite. Check this before claiming it.
- **(C3) The fixed point is MaxEnt RL on log-reward.** p* ∝ r^{1/α} maximizes Σ_o p_o ln r_o + α·H(p): an outcome-level entropy-regularized objective with reward ln r and temperature α. It is also the GFlowNet target R^β with β = 1/α. Both flows share the fixed point but have different dynamics: α-IPS uses the *multiplicative* reward r·p^(−α) = exp(ln r − α ln p), MaxEnt the *additive* reward ln r − α ln p.
- **(C4) IPS-GRPO at α = 1 is per-outcome MaxRL.** For ε ≤ 1/G, p̂·ω(n) = 1{n ≥ 1}, so the mean sampled update is r_i φ_i − p_i Σ_k r_k φ_k with φ = 1 − (1−p)^G (the probability that the outcome appears in the group). This is exactly ∇_z of **F = −Σ_k r_k Σ_{j=1}^{G} (1−p_k)^j / j**, the reward-weighted, per-outcome version of MaxRL's truncated objective (Tajwar et al. 2026, Eq. 6). Our "exact up to the miss probability" statement is therefore MaxRL's Theorem 2 applied outcome by outcome.
- **(C5) Every count-based rule gives a separable program, concave when ω is non-increasing.** For any weight rule ω, the mean field is the gradient flow of Σ_k r_k Φ(p_k) with Φ′ = w_G (a Bernstein polynomial, as in Zheng 2026). If ω is non-increasing (as for the clipped rule), w_G is decreasing and Φ is concave. Rules with negative or increasing weights, such as naive Richardson, break this. Our rest-point conditions (r_i w_G(p_i) = S for survivors, r_i w_G(0) ≤ S for extinct outcomes) are the **KKT conditions** of this program, and **S4 is its boundary KKT condition** r_2 w_G(0) > r_1 w_G(1), i.e. ω(1)/ω(G) > r_1/r_2. This can help prove the single-sign-change conjecture for K > 2 (left open in the report).
- **(C6) Ecology reading.** The replicator equation with IPS fitness f_i = r_i/p_i is ṗ_i = r_i − p_i·Σr, which converges to p ∝ r (input matching). With fitness r_i/p_i^m it converges to p ∝ r^{1/m} (Sutherland's interference IFD).

---

## 3. Competitors: methods that keep many good outcomes

"Target" is the distribution over outcomes the method aims for at convergence, when one is defined. "Aux?" means an auxiliary model, critic or judge is needed.

| Method (year, venue) | Mechanism | Target over outcomes | Aux? | Extra cost | Relation to IPS and to α-IPS |
|---|---|---|---|---|---|
| **GRPO** (Shao et al. 2024) — base-paper baseline | Group-normalized advantage (r − mean)/std, PPO clip, KL | Collapses to the argmax (for binary rewards ≈ ascent on arcsin√p, Davis & Recht 2025) | No | — | α = 0 reference point |
| **FlowRL** (Zhu et al. 2025) — base-paper baseline | Reverse KL / trajectory balance to a reward-tempered target with a learned partition function Z_φ(x) | ∝ π_ref·exp(β·r) (Boltzmann with a reference prior) | Learned partition function | Small (extra head) | Same goal (distribution matching); IPS avoids learning Z by using the observed p̂; exponential in r, not a power |
| **REINVENT** (Olivecrona et al. 2017) + diversity filter / memory (Saturn) — base-paper baseline | Augmented likelihood log π_prior + σ·score; scaffold-bucket penalty, experience replay | ≈ π_prior·exp(σ·score), plus a hard cap on repeated scaffolds | Prior model, memory | Small | The scaffold bucket is a crude, hard count-based limit; IPS is a smooth count weight |
| Entropy bonus / MaxEnt RL (Williams & Peng 1991; Haarnoja et al. 2018; Cheng et al. 2025) | + τ·H(π) | ∝ exp(r/τ) | No (critic in SAC) | None | Additive −τ·log p vs multiplicative p^(−α); same fixed-point family on log r (C3) |
| Clip-higher / Clip-Cov / KL-Cov (Yu et al. 2025; Cui et al. 2025) | Keep token entropy from collapsing | Unchanged (collapse delayed) | No | None | Treats the symptom; orthogonal, can be combined |
| KL to a reference (Ziegler et al. 2019; Ouyang et al. 2022) | r − β·log(π/π_ref) | π_ref·exp(r/β) | Frozen reference | One extra forward pass | Often unimodal for small β (GX-Chen) |
| **MARA** (GX-Chen et al. 2025, ICLR 2026) | Augments rewards of high-quality samples so the KL target puts equal high mass on all good modes | Flat over high-reward modes, ≈ π_ref elsewhere | Frozen reference | Negligible | **Strongest "fix the target" competitor**; works through the KL target rather than 1/p scaling |
| GFlowNet fine-tuning (Hu, E. J. et al. 2024; Bengio et al. 2021; Kim et al. 2024) | Flow balance (TB / SubTB), often off-policy replay | ∝ R^β | Learned Z / flows | Moderate | **Same target family as α-IPS (β = 1/α)**, different estimator |
| **DRA-GRPO** (Chen et al. 2025, ACL 2026) | R/(1 + Σ_j similarity) | Not derived (≈ reward-proportional over semantic clusters) | Small embedding model | Small | **≈ IPS-GRPO with a soft count** (precedent) |
| **Uniqueness-aware RL** (Hu, Z. et al. 2026, ACL Findings) | GRPO advantage × f^(−α), α ∈ [0,1] | Not derived | LLM judge for strategy clusters | Large (judge calls) | **Closest precedent of α-IPS** (count-power weight) |
| Outcome-based exploration (Song et al. 2025) | UCB bonus on historical outcome counts; batch penalty −(1/n)Σ 1{same answer} | Not derived | Count tables | Negligible | Additive count terms; our weight is multiplicative (G/n)^α |
| Unlikeliness reward (He et al. 2025) | r·(1 − β_rank·(G − rank)/G) on correct samples | Not derived | No | Negligible | Bounded, rank-based proxy for 1/p |
| GAPO (Anschel et al. 2025, EMNLP) | Group reward 1 − abs(f − 1/L) | Uniform over L valid items | Needs the valid set | Negligible | Uniform target (like α → ∞, binary rewards) |
| DARLING (Li, T. et al. 2025) | Quality × learned semantic diversity | Not derived | Learned partition classifier | Moderate | Explicit diversity reward |
| Diversity-GRPO (Mohri et al. 2026) | −log ν̂ (entropy) or 1 − ν̂ (Gini) answer-level reward | Answer-level entropy-regularized optimum | No | Negligible | Additive inverse-frequency; derived from an objective, as ours is |
| DMPO (Li, X. et al. 2026) | Group-level target ∝ reward; forward-KL alignment | ∝ r within the group | No | Small | Forward-KL view of IPS done per group |
| DPH-RL (Li, L. et al. 2025) | Forward-KL / JS to the initial policy | Keeps base coverage | Initial-policy samples | Small | Regularizer, not reward-proportional |
| α-divergence filtering (Kruszewski et al. 2025, ICLR 2026) | α-divergence to "base filtered to correct" | π_base restricted to correct answers | Base samples | Moderate | Another tunable knob (α-divergence) toward a different target |
| Pass@k training / PKPO / inference-time objectives (Chen 2025; Walder & Karkhanis 2025; Tang et al. 2025) | Optimize pass@k with unbiased estimators | Set-level objective, no outcome target | No | Negligible | Rewards diversity within k samples |
| MaxRL (Tajwar et al. 2026, ICML) | 1/K-normalized average over successful samples | Maximizes log P(success) (truncated at N) | No | Negligible | Per-prompt 1/p; IPS-GRPO(α=1) = per-outcome MaxRL (C4) |
| Tsallis J_Q, GARL/PAFT (Lin & Ie 2026) | Amplify the RL gradient by P^(−q) | Escort ∝ α_j^{1/q} (categorical) | No | Negligible | Same family per prompt; α ↔ q |
| Polychromic PPO (Hamid et al. 2025) | Set-level success + diversity objective, vine sampling | Set-level | Critic (PPO) | Vine rollouts | Actor-critic (where IPS does not apply directly) |
| Fitness sharing (Goldberg & Richardson 1987) | f / niche count | ∝ f across peaks | Distance kernel | O(N²) distances | Classical α = 1 analogue |
| **α-IPS (ours)** | r / max(p̂, ε)^α inside group-REINFORCE | ∝ r^{1/α} (finite-G mean field: r_i w_G(p_i) = S) | No | Negligible | Adds an explicit stability / survival theory |

---

## 4. Positioning: what in our report is known, what appears new, what to soften

### 4.1 Claim-by-claim

| Our claim (where in `report.tex`) | Status in the literature | What to do |
|---|---|---|
| "We generalize IPS with a tunable exponent α, r̃ = r/p^α" (Abstract; Contribution 1) | **Not new as a device.** Hu, Z. et al. 2026 (f^(−α) in GRPO); Guu et al. 2017 (p^β); Aouali et al. 2023 (π/π0^α); Schaul et al. 2016 (PER) | **Soften:** "We study an α-generalized IPS weight r/p^α (a member of a known family, cf. [hu2026rewarding, guu2017language])" |
| (S1) p* ∝ r^{1/α}, "α is a diversity knob" (Sec. 2.2.2) | **Known.** Escort / tempered family: Lin & Ie 2026 Thm 2.1; GFlowNet R^β (Bengio 2021, Kim 2024); generalized matching law (Baum 1974); interference IFD (Sutherland 1983); Lq-likelihood (Ferrari & Yang 2010) | **Cite and present as known.** Our part is that the stop-gradient update converges to it (C2), plus the dynamics |
| α-generalized flow, Eq. (flow) | Not stated in this form, but it is the gradient flow of Σ r·ln_α p (C2), the reward-weighted Tsallis objective of Lin & Ie | Add one sentence noting the potential L_α; it gives a Lyapunov function |
| (S2) λ_j = α‖r‖_{1/α}μ_j | **Not found.** Simple linearization; related rate analyses: Mei et al. 2020 (softmax PG), Lin & Ie 2026 (escape times) | Keep as ours; cite the related analyses |
| (S3) Euler hλ < 2, RK4 hλ < 2.785 | Standard numerical analysis (Hairer & Wanner 1996); new application to this flow | Cite the textbook; call it an application |
| "Large α is stiff, small α is slow" | New framing; consistent with Lin & Ie's point that larger amplification speeds escape but hurts estimators | Keep |
| α = 0 is singular: zero drift at an exact tie; collapse at a tie is noise-driven | Elementary but correct, and it corrects the base paper (whose §3 argument says the more likely outcome moves faster even with equal rewards, although every a_i = 0 then). It is the analogue of neutral drift (Kimura 1962) | Keep; optionally mention the drift analogy |
| α = 0 converges algebraically, p_2 ≈ 1/(2Δt) | **Known rate:** O(1/t) and Ω(1/t) for exact softmax PG (Mei et al. 2020) | **Cite Mei et al.**; the exact K = 2 implicit solution (u + sinh u = Δt + c) is a nice small addition (not found stated, but elementary) |
| (S4) Survival limit ω(1)/ω(G) > r_1/r_2; α_c = ln(r_1/r_2)/ln min{G, 1/ε} | **Not found anywhere** (closest: F-GRPO tail-miss analysis; RL2ML update scales; Hu 2026 notes weights in [K^(−α), 1] but draws no survival conclusion) | **Our main novelty.** Say "to our knowledge new". Qualify: idealized mean field (h → 0), group-REINFORCE without GRPO's std normalization or PPO clipping. Replace "universal" by "general (any count-based weight rule)". Optionally add the KKT reading (C5) |
| General-K extinction thresholds and the α → ∞ criterion | Not found | Keep; the single-sign-change conjecture may be provable with (C5) |
| w_G(p) = E[ω(1+B)] (size-biasing) | **Known structure:** Bernstein representation of count-only estimators (Zheng 2026, Thm 2.1) | Cite RL2ML |
| (S5) At α = 1, w_G = (1−(1−p)^G)/p, D = −(1−p)^G | **Known in a closely related form:** MaxRL Thm 2 (binary reward, per prompt); a classical negative moment of the binomial distribution (Chao & Strawderman 1972) | **Cite both**; present ours as the outcome-level counterpart (C4) |
| Delta-method bias b, spread s; D expansion; offset c* = (1−α)/2 | Delta method and inverse moments are standard (Stephan 1945; Znidaric 2005); Lin & Ie give an O(q/(M P^q)) ratio-estimator bias. The offset rule appears new (minor) | Cite the background; keep the offset rule |
| MSE-optimal clip ε* = p | The clipped weight 1/max{π̂, τ} and its bias–variance role go back to Strehl et al. 2010 (off-policy evaluation). The exact optimum ε* = p was not found; it needs the unknown p, so it is a diagnostic rather than a rule | Keep, framed as a diagnostic; cite Strehl et al. for the clipped form |
| Base paper's Table 4 entries coincide because the clip never fires when ε < 1/G | **New observation about the base paper** (simple and convincing) | Keep (good for the critique) |
| Laplace smoothing raises α_c; naive Richardson gives negative weights | Not found; minor | Keep |
| EMA estimator gives a damped oscillator, optimum κ = 4λ | Standard second-order system analysis; new application | Keep |
| "Removing the multiplier also helps Newton" | Observation about our solver | Keep |

### 4.2 Overall assessment (honest)

1. **Not novel:** the exponent device r/p^α, the target family p* ∝ r^{1/α}, the "diversity knob" framing, the α = 1 exactness identity, and the size-biasing identity. All have published antecedents. The two most important are Hu et al. 2026 (algorithm) and Lin & Ie 2026 (theory). In psychology and ecology the same law is decades old (matching law 1961, generalized form 1974, interference ideal free distribution 1983).
2. **Appears novel (within our idealized model):** the finite-group survival law and the general-K extinction thresholds; the explicit rate and step-size stability law of the α-flow; the diagnosis of the base paper's clipping ablation; the estimator-design observations (offset rule, Laplace, Richardson, EMA).
3. **What makes our report valuable despite (1):** it is the only work we found that treats this family as a **numerical-analysis object**: rates, stiffness, step-size limits, root-finding formulations, Monte Carlo validation, and an exact finite-G mean field with a sharp survival threshold. That is a legitimate contribution for a numerical-analysis course project. It should be framed as "a quantitative stability/survival analysis of a known family", not as "a new generalization".
4. **Base paper's own novelty (for the critique):** the base paper does not cite Guu 2017, DPG (Parshakova 2019, Khalifa 2021), DRA-GRPO (2025), Bengio 2021 (path multiplicity), Mei 2020, Strehl 2010 (the same clipped weight 1/max{π̂, τ}), fitness sharing or the matching law. Its contribution is best described as "a clear objective-level diagnosis plus a drop-in GRPO estimator", not a new principle.

### 4.3 Concrete wording changes (suggested)

- **Abstract**, replace "We generalize IPS with a tunable exponent α, r̃ = r/p^α, and use numerical analysis to derive ..." by: *"We study an α-generalized IPS weight r̃ = r/p^α, whose stationary policy p* ∝ r^{1/α} belongs to a known family of tempered (escort) targets, and use numerical analysis to derive ..."*
- **Contribution 1**: *"An analysis of the α-generalized update, which we show is the stop-gradient form of the reward-weighted Tsallis objective Σ_o r(o) ln_α p(o) whose optimum is p* ∝ r^{1/α} [lin2026tsallis, bengio2021flow, baum1974two, hu2026rewarding]."*
- **Box 1 / (S5)**: append *"the outcome-level counterpart of the MaxRL identity [tajwar2026maxrl]; cf. [chao1972negative]."*
- **Theorem "Universal survival limit"**: rename to *"General survival limit"* and add *"To our knowledge this threshold has not been stated before; related finite-group analyses are [zheng2026rl2ml, plyusov2026fgrpo]."*
- **Section 2.2.4 (α = 0)**: add *"consistent with the Θ(1/t) rate of exact softmax policy gradient [mei2020global]."*
- **Section 2.2.3**: cite [hairer1996solving] for the RK4 interval.
- **Footnote on naming**: *"Not to be confused with inverse propensity scoring in off-policy evaluation, where an 'IPS-α' estimator divides by π0^α [aouali2023exponential]."*

---

## 5. Ready-to-paste Related Work section and BibTeX

Paste this into `report/report.tex` just **before** `\section{Methodology}` (after the Contributions list). It cites `sinha2026` from the existing `references.bib` and 27 keys from `next_phase/01_literature.bib`. Length: about 490 words. **Compile-tested** (2026-10-04) with the report's own `acmart.cls` and `ACM-Reference-Format.bst`: no LaTeX errors, all 28 citations resolved. BibTeX only prints the usual warnings about optional publisher/address/page fields of some `@inproceedings` entries.

```latex
\section{Related Work}
\label{sec:related}

\paragraph{Diversity collapse in RL post-training.}
GRPO~\cite{shao2024deepseekmath} and related verifiable-reward methods raise
pass@1 but often narrow the set of answers a model produces, so that the base
model wins at large $k$~\cite{yue2025does}. Proposed remedies add exploration
signals: UCB-style bonuses over final answers and penalties for answers repeated
within a batch~\cite{song2025outcome}, a rank-based bonus for unlikely correct
proofs~\cite{he2025rewarding}, rewards divided by a kernel estimate of a
sample's density in its group~\cite{chen2025dra}, or a uniqueness weight
$f^{-\alpha}$ on GRPO advantages, where $f$ is the size of a strategy cluster
formed by an LLM judge~\cite{hu2026rewarding}. Others change the target: MARA
reshapes rewards so that the KL-regularized optimum covers every high-reward
mode~\cite{gxchen2026kl}, and FlowRL matches a reward-induced distribution
through a learned partition function~\cite{zhu2025flowrl}. Our base
paper~\cite{sinha2026} locates the cause in the probability multiplier of the
policy gradient. The same rich-get-richer loop was described by Guu et
al.~\cite{guu2017language}, whose $\beta$-meritocratic update weights
reward-earning programs by $p^{\beta}$ (in our notation $\beta=1-\alpha$), and
for exact softmax policy gradients Mei et al.~\cite{mei2020global} prove the
$\Theta(1/t)$ rate that we observe at $\alpha=0$.

\paragraph{Distribution matching and tempered targets.}
With $p$ held fixed, the IPS update $\sum_o r(o)\nabla\log p_\theta(o)$ is the
on-policy distributional policy gradient for the target $p\propto r$, that is,
the gradient of a forward-KL (cross-entropy)
objective~\cite{parshakova2019distributional,khalifa2021distributional}.
GFlowNets were introduced partly to remove the path-multiplicity bias of
maximum-entropy RL~\cite{bengio2021flow,deleu2024discrete}, and they routinely
sample from tempered targets $p\propto R^{\beta}$~\cite{bengio2021flow,kim2024learning};
our stationary law $p^\star\propto r^{1/\alpha}$ is this family with
$\beta=1/\alpha$. Our $\alpha$-flow is also gradient ascent on the
reward-weighted Tsallis objective $\sum_o r(o)\ln_\alpha p(o)$. Lin and
Ie~\cite{lin2026tsallis} show that the optimum of this family in a categorical
model is an escort distribution, and they analyse the amplification factor
$P^{-q}$ at the level of whole prompts.

\paragraph{Finite groups.}
For estimators that depend on a group only through counts, the population
weight is a Bernstein polynomial in $p$~\cite{zheng2026rl2ml}. MaxRL's $1/K$
estimator has weight $(1-(1-p)^N)/p$~\cite{tajwar2026maxrl}, a negative
moment of the binomial distribution~\cite{chao1972negative}; our identity
$w_G(p)=(1-(1-p)^G)/p$ for IPS-GRPO at $\alpha=1$ is the outcome-level version
of this fact. Plyusov et al.~\cite{plyusov2026fgrpo} study how finite groups
miss rare correct samples. To our knowledge, however, the survival condition
$\omega(1)/\omega(G)>r_1/r_2$, and its clipped form
$\alpha>\ln(r_1/r_2)/\ln\min\{G,1/\varepsilon\}$, have not been stated before.
A note on naming: in off-policy learning ``IPS'' means inverse propensity
scoring, and the ``IPS-$\alpha$'' estimator of Aouali et
al.~\cite{aouali2023exponential} divides by $\pi_0^{\alpha}$. That use of the
exponent is different from ours.

\paragraph{Matching outside machine learning.}
Reward-proportional allocation is an old equilibrium. Herrnstein's matching
law~\cite{herrnstein1961relative} and Baum's generalized form
$B_1/B_2=b\,(R_1/R_2)^{s}$~\cite{baum1974two} correspond to $\alpha=1/s$, so
under-matching ($s<1$) is our $\alpha>1$. In the continuous-input version of
the ideal free distribution~\cite{fretwell1969territorial,sutherland1983aggregation},
each individual receives the input rate divided by the number of competitors,
which is the IPS reward $r/p$. In Sutherland's interference
model~\cite{sutherland1983aggregation} the per-capita gain on a patch of
quality $Q$ with $n$ competitors is $Q\,n^{-m}$, so equal gains give
$n\propto Q^{1/m}$: matching for $m=1$ and under-matching for $m>1$, i.e.\ our
$\alpha=m$. Fitness sharing in genetic
algorithms divides fitness by a niche count and spreads a population over
peaks in proportion to their height~\cite{goldberg1987genetic}. Our step-size
limits use the standard stability intervals of explicit Runge--Kutta
methods~\cite{hairer1996solving}.
```

**BibTeX** (27 entries; identical to `next_phase/01_literature.bib`; no key clashes with `report/references.bib`):

```bibtex
% ---- next_phase/01_literature.bib : recommended citations for the alpha-IPS report ----
% Existing key in report/references.bib that the Related Work also uses: sinha2026
% (optionally update it: ICML 2026 poster, https://icml.cc/virtual/2026/poster/63579)

@misc{shao2024deepseekmath,
  author       = {Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao and Bi, Xiao and Zhang, Haowei and Zhang, Mingchuan and Li, Y. K. and Wu, Y. and Guo, Daya},
  title        = {{DeepSeekMath}: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  year         = {2024},
  howpublished = {arXiv preprint arXiv:2402.03300},
  url          = {https://arxiv.org/abs/2402.03300}
}

@inproceedings{yue2025does,
  author    = {Yue, Yang and Chen, Zhiqi and Lu, Rui and Zhao, Andrew and Wang, Zhaokai and Yue, Yang and Song, Shiji and Huang, Gao},
  title     = {Does Reinforcement Learning Really Incentivize Reasoning Capacity in {LLMs} Beyond the Base Model?},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {arXiv:2504.13837},
  url       = {https://arxiv.org/abs/2504.13837}
}

@misc{song2025outcome,
  author       = {Song, Yuda and Kempe, Julia and Munos, R{\'e}mi},
  title        = {Outcome-based Exploration for {LLM} Reasoning},
  year         = {2025},
  howpublished = {arXiv preprint arXiv:2509.06941},
  url          = {https://arxiv.org/abs/2509.06941}
}

@misc{he2025rewarding,
  author       = {He, Andre and Fried, Daniel and Welleck, Sean},
  title        = {Rewarding the Unlikely: Lifting {GRPO} Beyond Distribution Sharpening},
  year         = {2025},
  howpublished = {arXiv preprint arXiv:2506.02355},
  url          = {https://arxiv.org/abs/2506.02355}
}

@misc{chen2025dra,
  author       = {Chen, Xiwen and Zhu, Wenhui and Qiu, Peijie and Dong, Xuanzhao and Wang, Hao and Wu, Haiyu and Li, Huayu and Sotiras, Aristeidis and Wang, Yalin and Razi, Abolfazl},
  title        = {{DRA-GRPO}: Your {GRPO} Needs to Know Diverse Reasoning Paths for Mathematical Reasoning},
  year         = {2025},
  howpublished = {arXiv preprint arXiv:2505.09655 (ACL 2026)},
  url          = {https://arxiv.org/abs/2505.09655}
}

@inproceedings{hu2026rewarding,
  author    = {Hu, Zhiyuan and Wang, Yucheng and He, Yufei and Wu, Jiaying and Zhao, Yilun and Ng, See-Kiong and Breazeal, Cynthia and Luu, Anh Tuan and Park, Hae Won and Hooi, Bryan},
  title     = {Rewarding the Rare: Uniqueness-Aware {RL} for Creative Problem Solving in {LLMs}},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages     = {39765--39790},
  year      = {2026},
  note      = {arXiv:2601.08763},
  url       = {https://aclanthology.org/2026.findings-acl.1982/}
}

@inproceedings{gxchen2026kl,
  author    = {GX-Chen, Anthony and Prakash, Jatin and Guo, Jeff and Fergus, Rob and Ranganath, Rajesh},
  title     = {{KL}-Regularized Reinforcement Learning for Generative Modelling Is Designed to Mode Collapse},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {arXiv:2510.20817},
  url       = {https://arxiv.org/abs/2510.20817}
}

@misc{zhu2025flowrl,
  author       = {Zhu, Xuekai and Cheng, Daixuan and Zhang, Dinghuai and Li, Hengli and Zhang, Kaiyan and Jiang, Che and Sun, Youbang and Hua, Ermo and Zuo, Yuxin and Lv, Xingtai and others},
  title        = {{FlowRL}: Matching Reward Distributions for {LLM} Reasoning},
  year         = {2025},
  howpublished = {arXiv preprint arXiv:2509.15207},
  url          = {https://arxiv.org/abs/2509.15207}
}

@inproceedings{guu2017language,
  author    = {Guu, Kelvin and Pasupat, Panupong and Liu, Evan Zheran and Liang, Percy},
  title     = {From Language to Programs: Bridging Reinforcement Learning and Maximum Marginal Likelihood},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {1051--1062},
  year      = {2017},
  doi       = {10.18653/v1/P17-1097}
}

@inproceedings{mei2020global,
  author    = {Mei, Jincheng and Xiao, Chenjun and Szepesv{\'a}ri, Csaba and Schuurmans, Dale},
  title     = {On the Global Convergence Rates of Softmax Policy Gradient Methods},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  pages     = {6820--6829},
  year      = {2020},
  url       = {https://proceedings.mlr.press/v119/mei20b.html}
}

@misc{parshakova2019distributional,
  author       = {Parshakova, Tetiana and Andreoli, Jean-Marc and Dymetman, Marc},
  title        = {Distributional Reinforcement Learning for Energy-Based Sequential Models},
  year         = {2019},
  howpublished = {arXiv preprint arXiv:1912.08517 (OptRL Workshop at NeurIPS 2019)},
  url          = {https://arxiv.org/abs/1912.08517}
}

@inproceedings{khalifa2021distributional,
  author    = {Khalifa, Muhammad and Elsahar, Hady and Dymetman, Marc},
  title     = {A Distributional Approach to Controlled Text Generation},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021},
  url       = {https://arxiv.org/abs/2012.11635}
}

@inproceedings{bengio2021flow,
  author    = {Bengio, Emmanuel and Jain, Moksh and Korablyov, Maksym and Precup, Doina and Bengio, Yoshua},
  title     = {Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {34},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.04399}
}

@inproceedings{kim2024learning,
  author    = {Kim, Minsu and Ko, Joohwan and Yun, Taeyoung and Zhang, Dinghuai and Pan, Ling and Kim, Woochang and Park, Jinkyoo and Bengio, Emmanuel and Bengio, Yoshua},
  title     = {Learning to Scale Logits for Temperature-Conditional {GFlowNets}},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2310.02823}
}

@inproceedings{deleu2024discrete,
  author    = {Deleu, Tristan and Nouri, Padideh and Malkin, Nikolay and Precup, Doina and Bengio, Yoshua},
  title     = {Discrete Probabilistic Inference as Control in Multi-path Environments},
  booktitle = {Conference on Uncertainty in Artificial Intelligence (UAI)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2402.10309}
}

@misc{lin2026tsallis,
  author       = {Lin, Chu-Cheng and Ie, Eugene},
  title        = {How Fast Should a Model Commit to Supervision? Training Reasoning Models on the {Tsallis} Loss Continuum},
  year         = {2026},
  howpublished = {arXiv preprint arXiv:2604.25907},
  url          = {https://arxiv.org/abs/2604.25907}
}

@misc{zheng2026rl2ml,
  author       = {Zheng, Yifu},
  title        = {{RL2ML}: Finite-Rollout Surrogate Objectives from Reinforcement Learning to Maximum Likelihood},
  year         = {2026},
  howpublished = {arXiv preprint arXiv:2605.30154},
  url          = {https://arxiv.org/abs/2605.30154}
}

@inproceedings{tajwar2026maxrl,
  author    = {Tajwar, Fahim and Zeng, Guanning and Zhou, Yueer and Song, Yuda and Arora, Daman and Jiang, Yiding and Schneider, Jeff and Salakhutdinov, Ruslan and Feng, Haiwen and Zanette, Andrea},
  title     = {Maximum Likelihood Reinforcement Learning},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {arXiv:2602.02710},
  url       = {https://arxiv.org/abs/2602.02710}
}

@misc{plyusov2026fgrpo,
  author       = {Plyusov, Daniil and Gorbatovski, Alexey and Shaposhnikov, Boris and Sinii, Viacheslav and Malakhov, Alexey and Korotyshova, Daria and Gavrilov, Daniil},
  title        = {{F-GRPO}: Don't Let Your Policy Learn the Obvious and Forget the Rare},
  year         = {2026},
  howpublished = {arXiv preprint arXiv:2602.06717},
  url          = {https://arxiv.org/abs/2602.06717}
}

@article{chao1972negative,
  author  = {Chao, M. T. and Strawderman, W. E.},
  title   = {Negative Moments of Positive Random Variables},
  journal = {Journal of the American Statistical Association},
  volume  = {67},
  number  = {338},
  pages   = {429--431},
  year    = {1972},
  doi     = {10.1080/01621459.1972.10482404}
}

@inproceedings{aouali2023exponential,
  author    = {Aouali, Imad and Brunel, Victor-Emmanuel and Rohde, David and Korba, Anna},
  title     = {Exponential Smoothing for Off-Policy Learning},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {202},
  year      = {2023},
  url       = {https://arxiv.org/abs/2305.15877}
}

@article{herrnstein1961relative,
  author  = {Herrnstein, Richard J.},
  title   = {Relative and Absolute Strength of Response as a Function of Frequency of Reinforcement},
  journal = {Journal of the Experimental Analysis of Behavior},
  volume  = {4},
  number  = {3},
  pages   = {267--272},
  year    = {1961},
  doi     = {10.1901/jeab.1961.4-267}
}

@article{baum1974two,
  author  = {Baum, William M.},
  title   = {On Two Types of Deviation from the Matching Law: Bias and Undermatching},
  journal = {Journal of the Experimental Analysis of Behavior},
  volume  = {22},
  number  = {1},
  pages   = {231--242},
  year    = {1974},
  doi     = {10.1901/jeab.1974.22-231}
}

@article{fretwell1969territorial,
  author  = {Fretwell, Stephen D. and Lucas, Henry L.},
  title   = {On Territorial Behavior and Other Factors Influencing Habitat Distribution in Birds. {I}. {T}heoretical Development},
  journal = {Acta Biotheoretica},
  volume  = {19},
  number  = {1},
  pages   = {16--36},
  year    = {1969},
  doi     = {10.1007/BF01601953}
}

@article{sutherland1983aggregation,
  author  = {Sutherland, William J.},
  title   = {Aggregation and the `Ideal Free' Distribution},
  journal = {Journal of Animal Ecology},
  volume  = {52},
  number  = {3},
  pages   = {821--828},
  year    = {1983},
  doi     = {10.2307/4456}
}

@inproceedings{goldberg1987genetic,
  author    = {Goldberg, David E. and Richardson, Jon},
  title     = {Genetic Algorithms with Sharing for Multimodal Function Optimization},
  booktitle = {Proceedings of the Second International Conference on Genetic Algorithms and Their Application},
  pages     = {41--49},
  publisher = {Lawrence Erlbaum Associates},
  year      = {1987}
}

@book{hairer1996solving,
  author    = {Hairer, Ernst and Wanner, Gerhard},
  title     = {Solving Ordinary Differential Equations {II}: Stiff and Differential-Algebraic Problems},
  edition   = {Second},
  series    = {Springer Series in Computational Mathematics},
  publisher = {Springer},
  address   = {Berlin, Heidelberg},
  year      = {1996},
  doi       = {10.1007/978-3-642-05221-7}
}
```

---

## 6. What to read first, and what is still missing

### 6.1 Read these five first (in this order)

1. **Lin & Ie 2026, *Tsallis Loss Continuum*** (<https://arxiv.org/abs/2604.25907>). The same objective family as our α-flow. Their Thm 2.1 is our S1, and their gradient-flow and estimator-bias analysis is the closest published counterpart of our S2/S5. We must cite it and say clearly how we differ: per outcome instead of per prompt, α > 1 allowed, finite-G survival.
2. **Hu, Z. et al. 2026, *Rewarding the Rare*** (<https://arxiv.org/abs/2601.08763>). The closest algorithm: GRPO with count-power weights f^(−α). It shows the α-knob is already used in practice, without theory. A natural experiment for us is to compare their rule with ours in the bandit.
3. **Tajwar et al. 2026, *MaxRL*** (<https://arxiv.org/abs/2602.02710>). Their Thm 2 is our α = 1 identity, and their "objective–estimator alignment" view is the cleanest language for our S5.
4. **Zheng 2026, *RL2ML*** (<https://arxiv.org/abs/2605.30154>). The Bernstein representation is our w_G = E[ω(1+B)], and their sub- and supercritical update scales are the natural setting for extending our survival limit.
5. **Guu et al. 2017** (<https://doi.org/10.18653/v1/P17-1097>). Short and readable. It is the original "rich get richer" diagnosis with a p^β fix, and it shows the base paper's principle is older than 2026.

Next tier: GX-Chen et al. (MARA); DRA-GRPO; Song et al. 2025; Bengio et al. 2021 (path multiplicity, R^β); Mei et al. 2020 (1/t rate); Baum 1974 and Sutherland 1983 (interpretation of α); Plyusov et al. 2026 (F-GRPO).

### 6.2 What is still missing in our work (suggestions)

1. **GRPO's std normalization and PPO clipping** are not in our mean field (stated in our limitations). Davis & Recht 2025 (GRPO ↔ arcsin√p) and Bay & Yearick 2026 (group-std identity) give the tools to add the normalization as a transform of p.
2. **No comparison with the closest competitors**, even in the bandit. Implementing the weight rules of Hu et al. (f^(−α) on advantages), DRA-GRPO (soft counts), Song et al. (batch penalty), Mohri et al. (−log ν̂, 1 − ν̂) and MaxRL in our simulator would directly test whether our survival law predicts their behaviour. It is cheap and would strengthen the report a lot.
3. **A Lyapunov / global-convergence statement for all α > 0** via L_α = Σ r·ln_α p (C2). Check the boundary case α < 1 first.
4. **The single-sign-change conjecture for K > 2** may follow from the separable concave program in (C5) (monotone comparative statics of KKT supports). This is worth a try.
5. **Neural / LLM experiments** (HypoSpace, hyper-grid) remain future work, as the report already says.
6. **A short interpretation paragraph** (matching law / IFD / fitness sharing) would help non-ML readers. The Related Work text above already contains one.

---

## Appendix A. Verification log and items not fully verified

- **Base-paper references:** 43 of 43 located. 37 of them (plus the base paper itself) were matched through the arXiv API (titles, authors, dates). 7 DOIs were matched through Crossref (Kossale, Moret, Olivecrona, Romera-Paredes, Goodfellow CACM, Park/JCIM, Castro bioRxiv). PMLR and JMLR pages were matched for Ahmed, Haarnoja, Gao and Bengio. Castro is also on the ICML 2025 proceedings page.
- **Not fully verified:**
  - *An et al., DeRL* (OpenReview ZIYYeTkZQ4): the OpenReview page and API are behind a bot check. The work's existence is confirmed only through a Rutgers seminar listing (speaker Chenyang An, Nov 2025), not through the paper itself.
  - *Dohare et al. 2023*: confirmed by web search only (OpenReview blocked).
  - *Base paper at ICML 2026*: confirmed by the icml.cc poster page (search result) and a paper index (Lacuna). The OpenReview forum could not be opened.
- **Venues deliberately not stated** (only secondary sources): FlowRL (claimed ICLR 2026), Song et al. 2025, DAPO, DARLING, Polychromic, He et al. 2025.
- **Second-hand details:**
  - Sutherland 1983's page range 821–828 comes from a citing paper (Fauvergue et al. 2006); Crossref gives only the start page.
  - The interference form W_i = Q·n_i^(−m) and the "m = 1 matching, m < 1 overmatching, m > 1 undermatching" statement were read in the full text of Sanchez & Gillespie 2022, which attributes the form to Parker & Sutherland 1986. Sutherland's 1983 PDF itself was not read; Fauvergue et al. 2006 confirm that "Sutherland's (1983) model" has an interference coefficient m.
  - Kennedy & Gray 1993 is cited only for "undermatching is common". Its metadata were verified; the content claim comes via Fauvergue et al. 2006 and Earn & Johnstone 1997 (*A systematic error in tests of ideal free theory*, Proc. R. Soc. B 264:1671–1675, <https://doi.org/10.1098/rspb.1997.0232>).
  - Chao & Strawderman 1972: that the paper works out E[1/(X+A)] for the binomial and Poisson distributions comes from an abstract summary; the formula (1−q^G)/(Gp) itself was re-derived and checked numerically by us.
- **Corrections made in the re-check of 2026-10-04:**
  - Strehl et al. 2010 author order (Kakade before Li).
  - Tregenza et al. 1996 co-author (G. A. Parker, not "Hack").
  - FlowRL's target includes the reference model (∝ π_ref·exp(β·r)).
  - The "typical s ≈ 0.8" value for Baum 1979 was removed (only a secondary page gave it).
  - The Multi-Objective GFlowNets description was corrected: β is a fixed reward exponent and conditioning is on preferences.
  - The descriptions of Dang 2025, Dr. GRPO, GEM, Go 2023, Ding & Soricut 2017 and CRM were aligned with the abstracts or PDFs.
  - Stale "Section 2.12" pointers were changed to 2.13.
- **Name note:** arXiv now lists the first author of *Trajectory Balance* (arXiv:2201.13259) and *GFlowNets and Variational Inference* (arXiv:2210.00580) as "Esmeralda S. Whitammer". The proceedings list N. Malkin.
- **Mathematical connections (C1)–(C6)** are our own derivations, checked numerically. They are not claims from the cited papers, except where a paper is named.
