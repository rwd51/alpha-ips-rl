# Critical review of the α-IPS derivations

## Summary

- **Overall verdict:** the algebra of our derivations is correct almost everywhere. I re-derived 47 claims one by one and checked them numerically with 13 scripts. Result: 25 correct (✓), 16 need a condition or better wording (⚠), 6 wrong (✗). The problems are mostly not in the formulas but in the assumptions and in the scope of the claims.
- **What is solid:** the α-flow, $p^\star\propto r^{1/\alpha}$, $J^\star$ and the rates, the RK4 constant 2.7853, the exact α=0 solution (53.89/Δ), the size-biasing identity, $w_G=(1-(1-p)^G)/p$ at α=1, $\varepsilon^\star=p$, $\lambda^\star$, $c^\star=(1-\alpha)/2$, the distortion $D$, the EMA variance and lag: all re-derived and matched. I found no wrong number in the report (53.89, 1.6, 0.886, 0.445, $7.5\times10^{-15}$ ...).
- **The biggest problem (Major):** our Monte Carlo agent (`rhs_sampled`) is actually group-REINFORCE without a baseline, not IPS-GRPO. IPS-GRPO has GRPO's group-mean baseline (with or without std normalization). Then the K=2 survival limit becomes $\alpha>\ln(r_1/r_2)/\ln(G-1)$, not $\ln G$. At α=1, r=(4,1), G=5 our model keeps the minority ($p_2=0.073$), but IPS-GRPO loses it. So the formula the report states in the name of "IPS-GRPO" is not right.
- **The critique of the base paper is only half right:** if all rewards are equal, the deterministic flow stands still; that is right. But if only the good outcomes are tied and at least one worse outcome exists, the deterministic flow itself collapses inside the tie ($p_2/p_1\sim t^{-1/2}$). So "collapse comes only from noise" is an overclaim. The mechanism of the noise-driven collapse is not a "Pólya urn rich-get-richer" either: it is an exact martingale (neutral drift), and the collapse is not permanent (52% of seeds later change winner).
- **Local vs global:** (S3) "stable iff $h\lambda_{\max}<2$" is true only near $p^\star$. For α>1 the stiffness near the boundary is unbounded; with α=2, h=0.05, starting from $p_2(0)=0.01$, Euler overflows.
- **Mean field vs real (finite-h) training:** with a finite step the survival condition changes to $G(1-e^{-2hr_2\omega(1)/G})>2hr_1\omega(G)$. The theorem's "any weight rule" is wrong; ω must be monotone. With naive Richardson the minority does not simply die: the dynamics become bistable, and from a uniform start it can be the better outcome that dies. At finite β the EMA is not "unbiased in steady state"; at α=1 it is more biased than the paper's clip.
- **Good news (new proofs):** the α-flow is the exact gradient flow of the potential $F_\alpha=\sum_kr_kp_k^{1-\alpha}/(1-\alpha)$, so global convergence can be proved for every α>0 (the report only has a local linearization). The "single sign change of $h_j(\alpha)$", which the report calls "numerical, not proven", now has a 6-line proof. The finite-G target is the same under any parameterization (multi-step, neural).
- **A new caution:** with finite G, a nearly dead outcome needs time $\approx 1/(c\,p_j(0))$ to come back (only $\log$ in ideal IPS): from $p_2(0)=10^{-8}$, $4.2\times10^6$ against 8.5. So (S4) is a statement about stationary points; it gives no guarantee that the outcome comes back within a practical time.

## What you need / what to do

In order of priority (the first one first):

1. **(Major, ~1 day) Fix the model's name and the IPS-GRPO claim.**
   - What is needed: new code (a new file rather than a change to an existing one, e.g. `src/grpo_variants.py`, with a group-mean baseline (B) variant and a mean+std (N) variant; template: `next_phase/case2_checks/c05_grpo_variants.py`). Re-run Exp 2 Fig 4b–c with B/N (~10 minutes).
   - Math: a new proposition, $\alpha_c^{\rm GRPO}=\ln\rho/\ln(G-1)$; at α=1 "the baseline costs exactly one sample", i.e. $m^{B}(G)\equiv m^{V}(G-1)$ (proof in §2.1).
   - What changes in the report: in the abstract, "which for the paper's clipped estimator reads..." and "loses an outcome whenever $G\le r_1/r_2$"; point (ii) of report §2.2.5; "It does what GRPO does" in report §2.3.4; Key finding 5; row 5 of Table 6; the Conclusion. The exact sentences are given in §4.
2. **(Major, ~half a day) Soften the critique of the base paper.** Only writing and the c03 result (deterministic collapse in a partial tie) are needed. What changes: the Intro's "is argued from a deterministic flow...", "therefore describes the stochastic process" in report §2.2.4, Key finding 1, row 2 of Table 6, and the "Pólya-urn" part of `derivation.md`. Add the martingale and the closed form $E[t_c]=(G/(hr^2))(U^2/2+\cosh U-1)$ (c04: agrees within 2–6%).
3. **(Major, ~half a day, math only) Add the global convergence theorem and the potential.** The proof is written out in §2.3. Put it in `derivation.md` and in report §2.2.3. Also mention that the base paper's KL potential does not work for K≥3, α≠1.
4. **(Major, ~2 hours) Reword (S3) and the "Design rule" box.** State that it is local; along the path, $\max\lambda_{\max}(-J)$ is what matters. Optionally show the c02 result as a figure.
5. **(Major, ~half a day) Add the finite-h survival criterion (*).** Rewrite the "Mean-field limit" paragraph in Limitations. Re-run c06 (~15–20 minutes, 7 cores).
6. **(Major, ~1 hour) Fix the theorem's hypothesis.** Add "non-increasing ω" and rename "universal". Also fix the naive-Richardson corollary (c12), the EMA claim (c07) and the explanation of the "factor 555" (c09 #10).
7. **(Major/scope, ~2 hours) Add a new paragraph to Limitations.** Topics: finite-G recovery is hyperbolic (c10); with binary rewards (S4) is vacuous, and for K≫G the restoring force shrinks by a factor ≈ $G(G-1)/(2K^2)$ (c13); with a KL term there is no extinction; portability via the potential (c11).
8. **(Proof upgrade, ~1 hour)** Remove the "Numerical, not proven" limitation and give the single-sign-change proof (§2.8, c08). The monotonicity of the support can be proved as well.
9. **(Presentation, ~1 hour)** Small corrections: collapse threshold 0.95 vs 0.98 (Exp 1 Fig 2c); the paper's Eq. (4) → (5); $\lVert r\rVert_{1/\alpha}$ is a quasi-norm for α>1; "A proof that α=0..." → "A derivation"; the wording of the 19,125-point check.

Total time: math and writing ~3–4 days; new simulations ~30 minutes of CPU (all scripts are ready in `next_phase/case2_checks/`).

*Note on references: "report §2.x.y", "Table 6" and "Case N" refer to the section, table and case numbers of the compiled report (`report/report.pdf`); "l. NNN" are line numbers in `report/report.tex`; a bare "§N" refers to a section of this review.*

---

**Scope.** I re-derived every main claim in `derivation.md`, `derivation_exp3.md`, `derivation_exp4.md`, `derivation_exp5.md`, the report's Methodology section and Box 1 (`report/report.tex`), and the `RESULTS_exp*.md` write-ups. I also read the implementing code (`src/dynamics.py`, `src/finite_group*.py`, `src/meanfield_rule.py`, `src/estimators.py`, `src/ema.py`) and the base paper (arXiv:2601.21669, Sec. 3, 4, App. A). Thirteen scripts in `next_phase/case2_checks/` check the claims numerically (Section 3). No repository file was modified.

**Convention.** "Vanilla" (V) is the report's Monte Carlo update `rhs_sampled`: group-REINFORCE with IPS weights and no baseline. "B" adds GRPO's group-mean baseline. "N" adds GRPO's group-mean baseline and std normalization. $\omega(n)$ is the weight given to an outcome seen $n$ times, $w_G(p)=\mathbb E[\omega(1+B)]$ with $B\sim\mathrm{Bin}(G-1,p)$, $\rho=r_1/r_2$, and $m$ is the margin $\alpha\ln\min\{G,1/\varepsilon\}-\ln\rho$.

## 1. Verdict table

Status: ✓ correct · ⚠ needs qualification (or fills a gap) · ✗ wrong as stated. Severity: Critical / Major / Minor / Presentation.

| # | Claim | Where | Status | Severity | One-line fix |
|---|---|---|---|---|---|
| 1 | Stop-gradient α-flow $\dot z_i=r_ip_i^{1-\alpha}-p_i\sum_kr_kp_k^{1-\alpha}$ | derivation.md; report Eq. (3) | ✓ | — | Add that it equals $\nabla_zF_\alpha$ (§2.3) |
| 2 | α=0 and α=1 reduce to the paper's flows | derivation.md; report §2.2.1 | ✓ | Presentation | The paper's IPS flow is its Eq. (5), not Eq. (4); $\dot z_i=p_ia_i$ is App. A.1 Eq. (3), not main-text Eq. (2) |
| 3 | Unique interior rest point $p^\star\propto r^{1/\alpha}$ (S1) | derivation.md; report Eq. (4) | ✓ | — | — |
| 4 | Convergence to $p^\star$ (only a local linearization is given) | report §2.2.3, Table 6 | ⚠ gap | Major | Add the global theorem via the potential $F_\alpha$ (§2.3, c01) |
| 5 | $J^\star=-\alpha\lVert r\rVert_{1/\alpha}(\mathrm{diag}\,p^\star-p^\star p^{\star\top})$; rates; K=2, tie and small-α formulas | derivation.md; Eq. (5) | ✓ | — | $J(z)$ is symmetric at every $z$ (a Hessian), not only at $p^\star$ |
| 6 | $\lVert r\rVert_{1/\alpha}$ called a norm | Box 1 | ⚠ | Presentation | It is a quasi-norm for α>1 |
| 7 | (S3) "explicit update is stable iff $h\lambda_{\max}<2$ / 2.7853" | Box 1; Case 5 | ⚠ local only; false globally for α>1 | Major | Reword as local; global needs $h<2/\max_{\rm path}\lambda_{\max}(-J)$ (§2.4, c02) |
| 8 | RK4 limit 2.7853 | derivation.md | ✓ | — | — |
| 9 | Exact tie ⇒ $\dot z\equiv0$ | derivation.md; Case 1 | ✓ | — | — |
| 10 | "Collapse at equal rewards comes only from sampling noise" (critique of the paper) | Intro; report §2.2.4; Key finding 1; Table 6 | ✗ true only for a full tie | Major | A partial tie collapses deterministically, $p_2/p_1\sim t^{-1/2}$ (§2.2, c03) |
| 11 | Noise mechanism is a "Pólya-urn rich-get-richer" loop amplified by $p_i$ | derivation.md (l. 86–88); exp1_collapse.py docstring | ✗ | Minor | It is an exact martingale (neutral drift); add the closed-form $E[t_c]$ (§2.2, c04) |
| 12 | Every seed collapses (implicitly: for good) | Case 2 | ⚠ | Minor | Null-recurrent: winners switch (52% of seeds in $2\times10^5$ steps) |
| 13 | α=0 exact solution, $t_c=53.89/\Delta$, $p_2\approx1/(2\Delta t)$ | derivation.md; Eq. (6) | ✓ | — | — |
| 14 | "A proof that α=0 is a singular limit" | Contributions | ✓ | Presentation | Call it a derivation |
| 15 | Mean field of `rhs_sampled`: $m_i=p_i(r_iw_G(p_i)-S)$ | derivation.md; Eq. (8) | ✓ | — | Add that $m=\nabla_zF_G$ (§2.3) |
| 16 | Size-biasing identity; $w_G(0)=\omega(1)$, $w_G(1)=\omega(G)$ | derivation_exp3/5 | ✓ | — | — |
| 17 | "$w_G$ is strictly decreasing" (for any rule) | report §2.2.5 | ⚠ | Minor | True iff ω is non-increasing and non-constant; false for naive Richardson |
| 18 | K=2 survival theorem "for any weight rule ω" | Theorem; (S4) | ⚠ | Major | Assume ω non-increasing and ρ>1; add uniqueness and global stability (§2.6) |
| 19 | Naive Richardson ⇒ "the minority dies unconditionally" | Corollary; derivation_exp5 §4 | ✗ | Minor | The mean field is bistable; from a uniform start the better outcome dies (c12) |
| 20 | Name "universal survival limit" | Abstract; report §2.2.5; Conclusion | ⚠ over-claim | Major | It is a K=2, mean-field, no-baseline, monotone-rule criterion; rename it |
| 21 | The survival formula describes IPS-GRPO ("IPS-GRPO at G=4 loses…", "loses an outcome whenever $G\le r_1/r_2$", "does what GRPO does") | Abstract; report §2.2.5(ii); report §2.3.4; Key finding 5; Table 6 | ✗ for IPS-GRPO's update | Major (Critical if kept as is) | With GRPO's baseline, $\alpha_c=\ln\rho/\ln(G-1)$; at α=1, baseline at $G$ ≡ vanilla at $G-1$ (§2.1, c05, c09 #9) |
| 22 | Mean-field survival carries over to the finite-h Monte Carlo agent | report §2.3.4; Case 9; Limitations | ⚠ | Major | Define survival as positive recurrence; use criterion (*); the boundary shifts by ≈ $hr_2\omega(1)/G$ (§2.5, c06) |
| 23 | "Any α>0 prevents the noise-driven collapse" | Key finding 2; Case 9 | ⚠ | Minor | At $h=0.5$, $G=16$ only for $\alpha>0.0116$ (c06 b) |
| 24 | "Monte Carlo gaps ≤ 0.013 are finite-h bias" | Limitations | ⚠ | Minor | Near $m=0$ part of the gap is finite-T (algebraic transients) and part is the finite-h boundary shift |
| 25 | (missing) recovery of a nearly extinct outcome | — | ⚠ gap | Major (practical) | Finite-G recovery time is ≈ $1/(c\,p_j(0))$, against logarithmic for ideal IPS (§2.7, c10) |
| 26 | KKT conditions, unique solution, bracketed nested solver | derivation_exp3; report §2.2.6 | ✓ | — | Justify them as the KKT conditions of the strictly concave $F_G$ |
| 27 | Runner-up $\alpha_c$ equals the K=2 formula; weaker outcomes need a strictly larger α | derivation_exp3 | ✓ | — | — |
| 28 | α→∞ criterion $\sum_{i\in A,i\ne j}(r_j/r_i)^{1/(G-1)}>\lvert A\rvert-2$ | derivation_exp3; report §2.2.6 | ✓ | Minor | State the equality case and $G<1/\varepsilon$ |
| 29 | Single sign change of $h_j(\alpha)$ "checked, not proven" | Limitations; Future work | ✓ provable | — | Add the short proof (§2.8, c08) |
| 30 | Support grows with α and with G | Case 12 | ✓ provable (in α; in G when $\varepsilon\le1/G$) | Minor | State it as a proposition, not only as a grid observation |
| 31 | Delta-method bias and spread, Eq. (9) | report §2.2.7; derivation_exp5 §1 | ⚠ (algebra correct, regime unstated) | Minor | Say "fixed $p,\varepsilon$, $G\to\infty$"; the remainder contains $(1-p)^G\varepsilon^{-\alpha}$ |
| 32 | MSE-optimal clip $\varepsilon^\star=p$ | report §2.2.7; derivation_exp5 §2 | ✓ | Minor (relevance) | The MSE is dominated by $n=0$ draws, which the update never uses |
| 33 | Add-λ optimum $\lambda^\star=Kp(1-p)/(1-Kp)^2$ | derivation_exp5 | ✓ | — | — |
| 34 | α=1 identity $w_G=(1-(1-p)^G)/p$ | Eq. (11) | ✓ (needs $\varepsilon\le1/G$, which is stated) | — | — |
| 35 | Distortion $D=\alpha(\alpha-1)(1-p)/(2Gp)$, "vanishes at α=1 and nowhere else" | Eq. (10); derivation_exp5 | ✓ | Presentation | It also vanishes (trivially) at α=0 |
| 36 | Richardson order 1→2; guard bounds; $D_R=-\alpha(1-p)/(Gp)$ | derivation_exp5 §4–5.1 | ✓ | — | — |
| 37 | Offset $c^\star=(1-\alpha)/2$ and range $((G-c)/(1-c))^\alpha$ | derivation_exp5 §5.1 | ✓ | — | — |
| 38 | EMA variance factor $\beta/(2-\beta)$, lag $(1-\beta)\delta/\beta$, MSE(β) | report §2.2.7 | ✓ (static p) | — | — |
| 39 | "An EMA is unbiased in steady state, unlike ε and λ" | derivation_exp5 §6.3; RESULTS_exp5 Fig 5 | ✗ at finite β | Minor | $D_{\rm EMA}\approx\alpha\beta\frac{1-p}{Gp}[\frac{\alpha+1}{2(2-\beta)}-1]$; at α=1 this is larger than the clip's $-(1-p)^G$ (c07) |
| 40 | EMA damped oscillator; $\partial F/\partial z=0$; $\kappa^\star=4\lambda$; rate $2\lambda$ | Eq. (12) | ✓ in the limit $h\to0$, $\beta=\kappa h$ | — | State the joint limit |
| 41 | "EMA RMSE 0.077 vs 42.6 for the paper's clip" (factor 555) | Case 15; RESULTS_exp5 | ⚠ misleading | Minor | 100.0% of the clip's MSE is the $n=0$ draw; the update-relevant RMSE is ≈ 0.47 (c09 #10) |
| 42 | Kink jump $\Delta_k$; interpolation orders | derivation_exp4 | ✓ | — | — |
| 43 | $\varepsilon<1/G$ ⇒ the clip never fires, so the paper's Table 4 entries coincide | report §2.2.5(i); Table 6 | ✓ | — | — |
| 44 | "$\varepsilon^\star=p$ explains why Table 4 has no single winner" | derivation_exp5 §2; RESULTS_exp5 | ⚠ speculative | Minor | HypoSpace uses binary rewards (ρ=1); soften |
| 45 | Modelling assumptions (deterministic $r_i>0$, tabular logits, one step, no KL) | report §2.1; Limitations | ⚠ scope | Major | Add the portability result (c11) and the binary-reward, KL and $K\gg G$ caveats (§2.10) |
| 46 | Collapse declared at $\max_ip_i>0.95$ | report §2.3.4 vs Exp 1 Fig 2c code (0.98) | ✗ inconsistent | Presentation | State both thresholds |
| 47 | "The rule holds … at all 19,125 hypergrid points" as validation | Key finding 5 | ⚠ wording | Presentation | It confirms the solver and the theorem's algebra, not training dynamics |

**Counts:** 47 claims · ✓ 25 · ⚠ 16 · ✗ 6.

---

## 2. Detailed issues, re-derivations and fixes

### 2.1 The Monte Carlo agent is not IPS-GRPO; GRPO's baseline shifts the survival limit (rows 20–21, Major)

**Claim.** Report §2.2.5: "(ii) Even the paper's $\alpha=1$ needs $G>r_1/r_2$: with $r=(4,1)$, IPS-GRPO at $G=4$ loses the weaker outcome." The abstract says the paper's rule "loses an outcome whenever $G\le r_1/r_2$". report §2.3.4 says the agent "does what GRPO does".

**Re-derivation of the report's update.** With $G$ samples $o_1,\dots,o_G$ and IPS rewards $\tilde R_g=r_{o_g}\omega(n_{o_g})$, REINFORCE without a baseline is
$$\hat g_i=\frac1G\sum_g\tilde R_g\,\partial_{z_i}\ln p_{o_g}=\frac1G\sum_g\tilde R_g(\delta_{i,o_g}-p_i)=\hat p_i\tilde R_i-p_i\sum_k\hat p_k\tilde R_k .$$
This is exactly `src/dynamics.py::rhs_sampled`, so the code matches the stated model (✓).

**What IPS-GRPO computes.** The paper's Algorithm 1, line 6, says: "Run a standard GRPO update using $\tilde r_g$ (all other terms and steps remains unchanged)". GRPO uses $A_g=(\tilde R_g-\hat\mu)/\hat\sigma$, with $\hat\mu,\hat\sigma$ the group mean and std. For one on-policy step (ratio 1, no KL), using $\sum_k\hat p_kA_k=0$:
$$\hat g^{B}_i=\hat p_i(\tilde R_i-\hat\mu),\qquad \hat g^{N}_i=\hat p_i(\tilde R_i-\hat\mu)/\hat\sigma,\qquad \hat\mu=\sum_k\hat p_k\tilde R_k .$$
Two structural differences follow. First, an outcome absent from the group gets **no** update (the push-down term $-p_i\sum_k$ disappears). Second, the baseline is computed from the same group, so it is correlated with the outcome's own count.

**K=2.** Here $\hat g^B_1=\hat p_1\hat p_2(\tilde R_1-\tilde R_2)$. Since $\hat\sigma=\lvert\tilde R_1-\tilde R_2\rvert\sqrt{\hat p_1\hat p_2}$, the std-normalized step is $\hat g^N_1=\sqrt{\hat p_1\hat p_2}\,\mathrm{sign}(\tilde R_1-\tilde R_2)$: the reward magnitudes drop out. Applying size-biasing twice ($n(G-n)\binom Gn p^nq^{G-n}=G(G-1)pq\binom{G-2}{n-1}p^{n-1}q^{G-1-n}$) gives
$$m^B_1=\tfrac{G-1}{G}\,pq\Big[r_1\,\mathbb E\,\omega\big(1+\mathrm{Bin}(G-2,p)\big)-r_2\,\mathbb E\,\omega\big(1+\mathrm{Bin}(G-2,q)\big)\Big].$$
As $q=p_2\to0$, only groups with exactly one minority sample matter (probability $\approx Gq$):
$$m^B_1\approx q\tfrac{G-1}{G}\big[r_1\omega(G-1)-r_2\omega(1)\big],\qquad m^N_1\approx q\sqrt{G-1}\;\mathrm{sign}\big[r_1\omega(G-1)-r_2\omega(1)\big].$$
So for both B and N the minority survives iff $r_2\omega(1)>r_1\omega(G-1)$. For the clip with $\varepsilon\le1/G$ we have $\omega(1)/\omega(G-1)=(G-1)^\alpha$, hence
$$\boxed{\alpha>\alpha_c^{\rm GRPO}=\frac{\ln(r_1/r_2)}{\ln(G-1)}}\qquad(G\ge3;\ \text{at }G=2\text{ no }\alpha\text{ works}),$$
and in general $\omega(1)/\omega(G-1)=\big(\max\{\tfrac{G-1}G,\varepsilon\}/\max\{\tfrac1G,\varepsilon\}\big)^\alpha$. The vanilla limit is $\ln\rho/\ln\min\{G,1/\varepsilon\}$.

**Exact identity at α=1, any K ($\varepsilon\le1/G$).** At α=1, $\hat p_i\tilde R_i=r_i\mathbf 1[n_i>0]$. Using $\mathbb E[\hat p_i\mathbf 1[n_k>0]]=p_i(1-(1-p_k)^{G-1})$ for $k\ne i$,
$$m^B_i(G)=r_i\big(1-(1-p_i)^G\big)-p_i\Big[r_i+\sum_{k\ne i}r_k\big(1-(1-p_k)^{G-1}\big)\Big]\equiv m^V_i(G-1)=r_i\big(1-(1-p_i)^{G-1}\big)-p_i\sum_kr_k\big(1-(1-p_k)^{G-1}\big),$$
because the difference is $r_ip_i(1-p_i)^{G-1}-p_ir_i(1-p_i)^{G-1}=0$. **At the paper's α=1, a group-mean baseline costs exactly one sample:** every finite-G result of the report holds for the baseline update with $G\to G-1$. For example, the K=5 target at $G=16$ is the vanilla $G=15$ one, with O5 mass 0.0122 instead of 0.0187.

**Numerical confirmation.**
- c05 (a): exact K=2 thresholds. V matches $\ln4/\ln G$, and B and N both match $\ln4/\ln(G-1)$ to four decimals for every G from 2 to 64.
- c05 (b): Monte Carlo at α=1, r=(4,1), G=5. Long-run $p_2$ is 0.064 for V (mean field 0.073), but 0.004 for B and 0.006 for N (mean field 0: extinct).
- c09 #9: the identity $m^B(G)=m^V(G-1)$ holds to $2.7\times10^{-15}$ over 2000 random $(K,r,p,G)$.
- c05 (c): K=5, r=(5,4,3,2,1), α=1. O5 mass is 0.0187 (V), 0.0122 (B) and 0.0174 (N) at G=16, and 0.0572, 0.0563 and 0.0566 at G=32. At G=8, O5 is extinct for V and still decaying for B and N at $t=2\times10^4$ ($8\times10^{-6}$, $4\times10^{-5}$).
- c05 (a) also finds a single interior root of the K=2 drift for V, B and N on every grid tested, so the boundary criterion decides survival for B and N as well.

The std normalization also moves interior targets at small G ($p_2=0.186$ for N against 0.115 for V at G=6). It makes the update magnitude non-vanishing at $p^\star$ (K=2: $\lvert\hat g_1\rvert=\sqrt{\hat p_1\hat p_2}$), so the rate and step-size laws (S2)–(S3) do not describe IPS-GRPO near its fixed point: the update is a sign-like (median-type) iteration that jitters at $O(h)$.

**What survives.** "G=4 loses the minority at α=1, r=(4,1)" holds under all three updates. "$G\le r_1/r_2$" must become "$G\le1+r_1/r_2$" for IPS-GRPO, and the design rule $G>(r_1/r_2)^{1/\alpha}$ becomes $G>1+(r_1/r_2)^{1/\alpha}$ (for $\varepsilon\le1/G$).

**Fix.**
1. Call the agent "group-REINFORCE without baseline" everywhere.
2. Add the proposition above and the α=1 identity.
3. Re-run Exp 2 Fig 4b–c with B and N (c05 is a ready template, ~10 min).
4. Put the GRPO thresholds next to (S4).

A leave-one-out (RLOO) baseline was not tested. It is not automatically unbiased here, because $\tilde R_g$ depends on the whole group through the counts, so it would need its own derivation.

### 2.2 The equal-reward critique of the base paper, and the noise mechanism (rows 10–12, Major/Minor)

**The paper's exact text (Sec. 3 and 3.1).**
- "We show that outcome-level mode collapse is not an artifact of poor exploration, finite samples, or optimization noise, but a structural property of the expected-return objective itself."
- Step 1: "Exact ties require perfectly equal probabilities and advantages and are unstable; in practice they are broken immediately by random initialization or sampling noise."
- "This argument does not depend on rewards being unequal. Even when rewards (and thus advantages) are equal, the initial probability multiplier $p_o(t)$ ensures that the slightly more likely outcome receives a larger update."

**Re-derivation.** From the paper's Eq. (2), $\frac{d}{dt}\ln\frac{p_i}{p_j}=p_ia_i-p_ja_j$. If $r_i=r_j=r_{\rm top}$, then $a_i=a_j=a:=r_{\rm top}-\bar r$ and
$$\frac{d}{dt}\ln\frac{p_i}{p_j}=(p_i-p_j)\,a .$$
- **Full tie** (all K rewards equal): $a\equiv0$, so the flow is frozen (the report is right; the paper's sentence is false here, and initial asymmetry alone does nothing).
- **Partial tie** (tied best outcomes plus at least one worse outcome with $p>0$): $a>0$, so the deterministic flow amplifies $\lvert p_i-p_j\rvert$ (the paper is right). With $p_1\to1$ and $r_3<r_{\rm top}$, the worse outcome decays as $p_3\approx1/(2(r_{\rm top}-r_3)t)$, so $\frac{d}{dt}\ln\frac{p_1}{p_2}\approx(r_{\rm top}-r_3)p_3\approx\frac1{2t}$ and $p_2/p_1\sim Ct^{-1/2}$: deterministic, but algebraic, collapse inside the tie.

c03, with r=(1,1,0) and $p_0=(0.36,0.34,0.30)$, gives $p_2/p_1$ = 0.90, 0.61, 0.16, 0.018 at $t=10,10^3,10^5,10^7$, with tail slope −0.48 approaching −1/2. A symmetric start stays symmetric, and the full tie stays frozen.

**Fair verdict.** The paper over-states the full-tie case ("not an artifact of finite samples or optimization noise"). The report over-states in the opposite direction ("collapse comes only from sampling noise"). The realistic multimodal case is the partial tie: several equally good answers plus worse ones, e.g. valid and invalid hypotheses with binary rewards. There the paper's deterministic mechanism does operate.

**Noise mechanism (row 11).** At a full tie with α=0, $\hat g_i=r(\hat p_i-p_i)$, so $\mathbb E[\hat g\mid z]=0$ exactly: the logits are a martingale and nothing is amplified in expectation. For K=2, $\Delta u=2hr(\hat p_1-p_1)$ with variance $4h^2r^2p(1-p)/G$, which vanishes at the vertices (sticky boundaries). The diffusion approximation, with generator $\tfrac12\sigma^2(u)\partial_u^2$, $\sigma^2=\frac{4h^2r^2}{G}\frac1{2(1+\cosh u)}$, solving $\tfrac12\sigma^2T''=-1$, $T(\pm U)=0$, gives
$$\mathbb E[t_c]=\frac{G}{h\,r^2}\Big(\frac{U^2}2+\cosh U-1\Big),\qquad U=\ln\frac\theta{1-\theta}\quad(\text{time units }t=\text{steps}\times h).$$
This is exactly linear in G. c04 finds it within 2–6% of 2000-seed means for G=4 to 64, with a log-log slope of 0.997; the report's fitted $G^{1.07}$ is a noisy estimate of exactly 1.

A martingale with bounded increments either converges or has $\limsup u=+\infty$ and $\liminf u=-\infty$ a.s. Convergence is impossible because the conditional variance is bounded away from 0 on compacts, so **the "collapse" is not absorbing**. In c04, 51.7% of seeds switched winner within $2\times10^5$ steps (G=4), while 99.9% of the time is spent near a vertex. A Pólya urn is also the wrong analogy: its fraction converges to a random interior (Beta) limit. RESULTS_exp1 already calls the G-scaling "genetic-drift-style", and that is the right picture: neutral drift with sticky boundaries.

**Fix.** Replacement text is in §4 (items 3, 7, 14, 17, 25). Add c03 as a figure and the closed form as a validated prediction.

### 2.3 Missing global convergence: the α-flow is a gradient flow (row 4, Major gap with a fix)

**Proposition A (potential).** For every $\alpha\ge0$, $\dot z=\nabla_zF_\alpha(z)$ with
$$F_\alpha(z)=\frac{1}{1-\alpha}\sum_kr_kp_k(z)^{1-\alpha}\ (\alpha\ne1),\qquad F_1(z)=\sum_kr_k\ln p_k(z).$$
*Proof.* $\partial_{z_i}F=\sum_kr_kp_k^{-\alpha}p_k(\delta_{ik}-p_i)=r_ip_i^{1-\alpha}-p_i\sum_kr_kp_k^{1-\alpha}$. ∎

c01 (a) confirms this to $2.4\times10^{-9}$.

Consequences:
- **Real spectrum.** $J(z)=\nabla^2F_\alpha$ is symmetric at every $z$, not only at $p^\star$, so its spectrum is real along the whole path (c01 b: asymmetry $1.8\times10^{-16}$).
- **Interpretation.** Stop-gradient IPS is gradient ascent on $\frac1{1-\alpha}\mathbb E_p[r/p^\alpha]$. For α>1 this is gradient *descent* on $\mathbb E_p[r/p^\alpha]$, which is worth stating.
- **Unique maximizer.** $f_\alpha(p)=\sum r_kp_k^{1-\alpha}/(1-\alpha)$ has Hessian $\mathrm{diag}(-\alpha r_kp_k^{-\alpha-1})$. It is strictly concave on the simplex for every α>0, and $p^\star$ is its unique maximizer (Lagrange condition $r_kp_k^{-\alpha}=\mu$).

**Theorem B (global convergence).** Let $r_k>0$ for all k and α>0. Then $p(t)\to p^\star$ from every $z(0)$.

*Proof.* $\dot F=\lVert\dot z\rVert^2\ge0$.

*Case α≥1.* $f_\alpha\to-\infty$ on $\partial\Delta$, so $\{f_\alpha\ge f_\alpha(p(0))\}$ is a compact subset of the open simplex. Hence $z(t)$ (modulo the gauge $\mathbf1$) stays in a compact set, and LaSalle gives $p(t)\to\{p^\star\}$.

*Case 0<α<1.* $f_\alpha$ is continuous and bounded on the closed simplex, and the entries of $\nabla^2F$ are bounded by $C\sum r$. So $\dot z$ is uniformly continuous in t and $\int\lVert\dot z\rVert^2<\infty$, which gives $\lVert\dot z(t)\rVert\to0$ (Barbalat). The extended field vanishes on the closed simplex exactly on the finite set $E=\{p^{\star(A)}\}$ of face equilibria ($\propto r^{1/\alpha}$ on a face A). The ω-limit set is connected and contained in E, so $p(t)\to q=p^{\star(A)}$ for some A. Suppose $A\ne[K]$ and take $j\notin A$; write $\delta=p_A-q_A$.
1. $F(t)\uparrow f(q)$ forces $\sum_{j\notin A}r_jp_j^{1-\alpha}\le C\lvert\delta\rvert^2$, because the linear term is $-S_q\sum p_j$ and the face is strictly concave.
2. Near q, $\dot z_j>0$ (since $r_jp_j^{-\alpha}\to\infty$), so $\frac{d}{dt}\ln p_j=\dot z_j-p\cdot\dot z\ge-C'\lvert\delta\rvert$.
3. The in-face dynamics read $\dot\delta=-\Lambda_A\delta+O(\lvert\delta\rvert^2)$ with $\Lambda_A\succ0$, so $\lvert\delta\rvert\le Ce^{-\lambda t}$.

Therefore $\ln p_j(t)\ge\ln p_j(t_0)-C'\!\int\lvert\delta\rvert>-\infty$, which contradicts $p_j\to0$. ∎

**The base paper's potential does not extend.** $\Psi=\ln\sum e^{z_k}-\sum p^\star_kz_k=\mathrm{KL}(p^\star\Vert p)+$const satisfies $\dot\Psi=\sum(p_i-p^\star_i)\dot z_i$. This is ≤0 for K=2 (any α) and for α=1 (any K). But c01 (c) finds $\dot\Psi>0$ at 316 of 20000 random states for K=3, α=0.3, and at 5 of 20000 for α=1.5. Along trajectories from $p_{\min}(0)\approx10^{-9}$, KL increased at 486 of the sampled time steps at α=0.2 (51 at α=0.5) while $F_\alpha$ never decreased, and every run converged ($\ell_1\le2.6\times10^{-10}$; c01 d). $F_\alpha$ is the right Lyapunov function.

**Same structure for the finite-G mean field.** $m=\nabla_zF_G$ with
$$F_G(z)=\frac1G\sum_ir_i\,\mathbb E_{n_i\sim\mathrm{Bin}(G,p_i)}\Big[\sum_{k=1}^{n_i}\omega(k)\Big],\qquad \Phi_G'(p)=w_G(p),$$
using $\frac{d}{dp}\mathbb E_{\mathrm{Bin}(G,p)}[g(n)]=G\,\mathbb E_{\mathrm{Bin}(G-1,p)}[g(n+1)-g(n)]$. c01 (e) confirms this to $6.9\times10^{-9}$. For the clip at α=1 with $\varepsilon\le1/G$, $F_G=\sum_ir_i\mathbb E[H_{n_i}]$, with $H_n$ the harmonic number. For general α it is $G^{\alpha-1}\sum_ir_i\mathbb E[H^{(\alpha)}_{n_i}]$, with $H^{(\alpha)}_n=\sum_{k\le n}k^{-\alpha}$.

Interpretation: each outcome earns a concave, saturating utility of its count. Survival means the first copy of the minority, $r_2\omega(1)$, is worth more than the G-th copy of the majority, $r_1\omega(G)$. The same proof gives global convergence to the KKT point, with the invasion rate $r_jw_G(0)-S_q>0$ replacing $r_jp_j^{1-\alpha}$. "Global" can still mean very slow. In c01 (e), started from $p_{\min}(0)\approx10^{-9}$, the G=16, α=1 mean field is still at $\ell_1=0.42$ from its target at $t=4000$ while $F_G$ keeps increasing. This is the hyperbolic recovery of §2.7, not a failure of the theorem.

**Fix.** Add Proposition A and Theorem B to `derivation.md` and to report §2.2.3, and change Table 6's "Global convergence, no rate" row accordingly.

### 2.4 (S3) is a local statement (row 7, Major)

**Re-derivation.** The Euler map $\Phi_h(z)=z+hf(z)$ has $D\Phi_h(z^\star)=I+hJ^\star$, with eigenvalues $1-h\lambda_j$, plus 1 in the gauge direction, which f never excites because $\mathbf1^\top f=0$. So $h\lambda_{\max}<2$ gives local asymptotic stability, $h\lambda_{\max}>2$ gives instability, and equality is undecided. For RK4 the limit is the root of $x^3-4x^2+12x-24$, $x^\star=2.7852935634$; c02 confirms $\lvert R(-x)\rvert\le1$ on $[0,x^\star]$.

Nothing here constrains the basin. For α>1, $\partial\dot z_i/\partial z_i\approx(1-\alpha)r_ip_i^{1-\alpha}\to-\infty$ as $p_i\to0$, so the local stiffness is unbounded.

**Check (c02).** r=(4,1), α=2, $\lambda^\star=8$, Euler with h=0.05 ($h\lambda^\star=0.4$, "stable" by (S3)):

| $p_2(0)$ | $\sup_{\rm path}\lambda_{\max}(-J)$ | Euler, h=0.05 |
|---|---|---|
| ≥ 0.1 | ≤ 18.9 | converges |
| $10^{-2}$ | 198 | overflows at step 3 |
| $10^{-4}$ | 19998 | overflows at step 1 |

With the code's `p_floor=1e-12`, the iteration instead gets stuck at $\ell_1$ error 0.67. In general, h must scale like $p_{\min}(0)^{\alpha-1}$.

**Fix.** Use replacement items 4 and 18 in §4. The report's own Exp 2 caveat (RK4 at h=0.3 diverging from uniform) belongs in Box 1.

### 2.5 Mean field versus the finite-h stochastic process (rows 22–24, Major)

**Gap.** The report compares the h→0 mean-field fixed point with time averages of the constant-step agent. Constant-step stochastic approximation (Kushner–Yin, Ch. 8; Benaïm 1999) justifies this away from the boundary: the iterates track the ODE over finite horizons, and the stationary law concentrates near the ODE attractor with $O(h)$ bias. "Survival" still needs a definition, because $p_2$ never hits 0 in finite time.

**Definition.** The minority *survives* if the chain $u_t=z_1-z_2$ is positive recurrent (time-averaged $p_2>0$). It is *extinct* otherwise: transient ($u\to\infty$) or null recurrent (time-averaged $p_2\to0$).

**Derivation (K=2, vanilla).** As $q=p_2\to0$:
- a homogeneous group (probability ≈ $1-Gq$) moves u up by $c\,q$, with $c=2hr_1\omega(G)$;
- a group with one minority sample (probability ≈ $Gq$) moves u down by $J=2hr_2\omega(1)/G$.

Both rates are proportional to q. Under the time change $d\tau=q\,dt$, u becomes a spectrally negative compound-Poisson process with drift c, jump rate G and jump size J. The Lundberg equation $c\theta=G(1-e^{-\theta J})$ gives a stationary tail $e^{-\theta^\star u}$ in τ. Real time spends $1/q=e^u$ steps per unit of τ, so the real-time density is $\propto e^{(1-\theta^\star)u}$, normalizable iff $\theta^\star>1$:
$$(*)\qquad G\Big(1-e^{-2hr_2\omega(1)/G}\Big)>2hr_1\,\omega(G).$$
As $h\to0$ this becomes the report's $r_2\omega(1)>r_1\omega(G)$. At finite h it needs $m\gtrsim-\ln(1-J/2)\approx hr_2\omega(1)/G$, which is $hr_2G^{\alpha-1}$ for the clip.

**Check (c06).** Vanilla update, 48 seeds, averages over the window $(T/2,T]$.

(a) r=(4,1), α=1, G=5: margin $m=0.223>0$, mean-field $p_2=0.0735$ at every h. Criterion (*) predicts survival only for $h<0.233$.

| h | θ* | (*) | $\langle p_2\rangle$, T=2e4 | 8e4 | 3.2e5 | 1.28e6 |
|---|---|---|---|---|---|---|
| 0.05 | 4.64 | holds | 0.0604 | 0.0633 | 0.0639 | 0.0637 |
| 0.10 | 2.32 | holds | 0.0514 | 0.0515 | 0.0517 | 0.0512 |
| 0.20 | 1.16 | holds (barely) | 0.0270 | 0.0224 | 0.0211 | 0.0195 |
| 0.30 | 0.77 | fails | 0.0123 | 0.0073 | 0.0056 | 0.0039 |
| 0.50 | 0.46 | fails | 0.0022 | 0.0012 | 0.0006 | 0.0002 |
| 1.00 | 0.23 | fails | 0.0004 | 0.0002 | 0.0000 | 0.0000 |

Where (*) holds clearly, the average is stable. Its $O(h)$ offset from 0.0735 is the finite-h bias the report mentions. At $h=0.2$ ($\theta^\star=1.16$) the stationary tail $\propto e^{-0.16u}$ is very heavy, so the average still drifts. Where (*) fails, $p_2\to0$ even though the mean field predicts 0.0735.

(b) Exact tie, G=16, h=0.5 (the Exp 1/2 setting). Criterion (*) predicts a persistent noise collapse for $\alpha<0.0116$. Share of time with $\max_ip_i\ge0.95$:

| α | 0 | 0.005 | 0.010 | 0.015 | 0.025 | 0.05 | 0.25 |
|---|---|---|---|---|---|---|---|
| θ* | 0 | 0.44 | 0.87 | 1.29 | 2.09 | 3.96 | 12.75 |
| T=5e4 | 0.983 | 0.911 | 0.716 | 0.389 | 0.094 | 0.004 | 0.000 |
| T=8e5 | 0.999 | 0.984 | 0.845 | 0.481 | 0.117 | 0.004 | 0.000 |

The share climbs toward 1 exactly where (*) fails (α ≤ 0.010) and levels off where it holds (α ≥ 0.015). So "any α>0 prevents the noise-driven collapse" is a mean-field statement. At a finite step it needs α above an h-dependent minimum; the report's tested values (≥ 0.25) are far above it.

Near $m=0$ the decay is algebraic, so a second-half average over 30k steps reports a transient. For example, Exp 2's "$p_1=0.9965$ at G=4" sits at the grid point α=1.003, where $m=0.0042$. There the mean field gives $p_2=0.00166$, while (*) fails at h=0.05 (it needs $m\gtrsim0.05$), so the agent's long-run $p_2$ tends to 0. The reported 0.0035 is neither of these limits; it is a finite-time transient.

**Fix.** Use replacement item 19 in §4, and add (*) as a proposition with c06 as its validation. State the Monte Carlo comparisons as "away from the boundary ($m\gg hr_2\omega(1)/G$)".

### 2.6 The survival theorem: hypotheses, the naive-Richardson corollary, "universal" (rows 17–20)

**Corrected statement.** Let ρ>1 and let $\omega:\{1..G\}\to(0,\infty)$ be non-increasing with $\omega(1)>\omega(G)$. Then:
- $w_G'(p)=(G-1)\mathbb E_{\mathrm{Bin}(G-2,p)}[\omega(2+B)-\omega(1+B)]<0$ on (0,1);
- $f(p)=r_1w_G(p)-r_2w_G(1-p)$ is strictly decreasing, with $f(\tfrac12)>0$;
- since $\dot u=2p_1p_2f(p_1)$: (a) if $\omega(1)/\omega(G)>\rho$, there is exactly one interior rest point, and it attracts every interior start; (b) otherwise $p_1\to1$, with the minority decaying algebraically: $p_2\sim1/(2f(1)\,t)$ when $f(1)>0$, and $p_2\sim t^{-1/2}$ at equality, because the finite-G drift near the boundary is $\propto p_2$.

If ρ=1 and ω is constant, every point is a rest point, so the stated "iff" fails at ρ=1.

**Counterexample to "any weight rule" (c12).** For naive Richardson, $w_G<0$ on $p\in[0,0.778]$ at G=8 and on $[0,0.505]$ at G=16. The K=2 balance has a single interior root, and it is **unstable**. From the uniform start the mean field sends $p_1\to0$: the better outcome dies for both ρ=1.5 and ρ=4. The Monte Carlo agent is bistable (32-seed long-run $\bar p_1$ = 0.50 and 0.47 at G=8; 0.81 and 0.97 at G=16). "The minority dies unconditionally" is therefore false. Naive Richardson is still harmful, but in a different way.

**"Universal".** The criterion is exact for K=2, the vanilla update, count rules with non-increasing ω, and the mean field. With GRPO's baseline it changes (§2.1); for K>2 it is the runner-up criterion only; at finite h it shifts (§2.5). "Dynamic-range survival criterion" is an honest name.

### 2.7 Finite-G recovery is hyperbolic; many tied outcomes (row 25, Major for practice)

**Derivation.** Near a face, $\dot z_j=p_j\big(r_jw_G(p_j)-S\big)\approx p_j\big(r_j\omega(1)-S_q\big)$. Because the weight saturates at $\omega(1)$, the probability multiplier that IPS was meant to remove returns for $p_j\ll1/G$. Hence $\dot p_j\approx c\,p_j^2$ and the recovery time is $\approx1/(c\,p_j(0))$, with $c=2(r_2\omega(1)-r_1\omega(G))$ for K=2. Ideal IPS has $\dot z_j\to r_j$, so its recovery time is logarithmic.

**Check (c10).** K=2, r=(4,1), G=16, α=1 (c=24), time to reach $p_2^\star/2$:

| $p_2(0)$ | finite-G mean field | $1/(c\,p_2(0))$ | ideal IPS |
|---|---|---|---|
| $10^{-2}$ | 5.4 | 4.2 | 1.54 |
| $10^{-4}$ | 420 | 417 | 3.87 |
| $10^{-6}$ | 41673 | 41667 | 6.18 |
| $10^{-8}$ | $4.17\times10^6$ | $4.17\times10^6$ | 8.48 |

The Monte Carlo agent agrees: median 43.7 against 44.1 predicted from $10^{-3}$, and 411 against 420 from $10^{-4}$. For K=5, O5 needs $3.5\times10^5$ time units from $10^{-6}$, against 8.9 for ideal IPS.

So (S4) is a statement about the stationary point, not about reachability.

**K≫G tied outcomes (c13).** With binary rewards, every valid answer has r=1, so ρ=1 and (S4) is vacuous. What finite G changes is the restoring force toward uniform: $\lambda_G/\lambda_\infty\to G(G-1)/(2K^2)$ at α=1 (c13: 0.0018 at K=256, G=16, exactly the formula). Diversity among many valid answers is protected only weakly against noise. This is the regime of the paper's HypoSpace tasks.

### 2.8 General K: upgrades (rows 26–30)

- **KKT.** The conditions are the exact KKT conditions of maximizing the strictly concave $f_G(p)=\sum r_i\Phi_G(p_i)$ over the simplex. $\Phi_G'(0)=\omega(1)$ is finite, so boundary maximizers occur. Existence and uniqueness of the solution the nested solver finds are therefore rigorous.
- **Single sign change (now a proof).** Let $m_G=\min\{G,1/\varepsilon\}$ and $\tilde w_\alpha(p)=w_G(p)/m_G^\alpha=\mathbb E\big[(\max\{1/G,\varepsilon\}/\max\{(1+B)/G,\varepsilon\})^\alpha\big]$. The base lies in (0,1], so $\tilde w_\alpha$ is non-increasing in α, with $\tilde w_\alpha(0)=1$ for all α. Hence each $p_i(s)$ solving $r_i\tilde w_\alpha(p_i)=s$ is non-increasing in α. Since $T(s;\alpha)=\sum_ip_i(s;\alpha)-1$ is non-increasing in both arguments, the root $s^\star(\alpha)=S^\star(\alpha)/m_G^\alpha$ is non-increasing. Therefore $h_j(\alpha)=\ln r_j-\ln s^\star(\alpha)$ is **non-decreasing**: at most one sign change, from − to +. ∎ The α→∞ criterion then decides exactly whether a finite threshold exists (strict inequality case).
- **Support monotonicity.** The support is non-decreasing in α. For $\varepsilon\le1/G$ it is also non-decreasing in G, because $\mathrm{Bin}(G-1,p)$ increases stochastically with G.
- **Check (c08).** The test covers 600 random cases: K=3–8, G ∈ {2,3,5,8,16}, ε ∈ {10⁻³, 0.15}, each at 60 values of α ∈ [0.02, 40]. $s^\star(\alpha)$ never increased. None of the 2770 curves $h_j(\alpha)$ decreased or changed sign more than once. The α→∞ criterion agreed with the sign of $h_j(40)$ in 2216/2216 cases with $\varepsilon<1/G$. The fast solver used there matches the repository solver to $2\times10^{-10}$ in $p$.

### 2.9 Estimator section (rows 31–41)

- **Delta method (row 31).** The unclipped $\mathbb E[\hat p^{-\alpha}]$ is infinite, since $P(\hat p=0)=(1-p)^G>0$. Eq. (9) is an asymptotic expansion of the *clipped* weight for fixed p and ε as $G\to\infty$, with remainder $O(G^{-2})+O((1-p)^G\varepsilon^{-\alpha})$, non-uniform in p. The derivation file says this; the report's Eq. (9) should too.
- **$\varepsilon^\star=p$ (row 32).** The proof is correct: MSE is piecewise smooth and continuous at $\varepsilon=n/G$, with derivative sign equal to $\mathrm{sign}(\varepsilon-p)$ (c09 #3: brute-force optimum within the grid resolution).
  - But at G=16, p=0.1, α=1 the $n=0$ draw carries **100.0%** of the MSE (c09 #10), and the update multiplies that weight by $\hat p=0$.
  - For the dynamics-relevant (size-biased) error, the minimizer is $\varepsilon=p$ only when $p>1/G$; for $p\le1/G$ every $\varepsilon\le1/G$ is optimal, because the clip is inactive on $1+B\ge1$.
- **"Factor 555" (row 41).** The relevant comparisons are the conditional RMSE (0.479), the size-biased RMSE (0.468), and the update term $\hat p\,\omega(n)$ itself (mean 0.815, sd 0.389). The EMA is still better, but not by 555×.
- **EMA bias (row 39).** The current group enters both $\hat p_t$ and $\bar p_t$. A second-order expansion for static p gives
  $$D_{\rm EMA}\approx\alpha\beta\frac{1-p}{Gp}\Big[\frac{\alpha+1}{2(2-\beta)}-1\Big].$$
  This reduces to the clip's $\alpha(\alpha-1)/2$ coefficient at β=1 and vanishes as β→0. c07 (Monte Carlo, 200k chains) confirms it: at α=1, G=16, p=0.5, β=0.1, $D_{\rm EMA}=-0.00299$ against $-0.00296$, while the plain clip has $-1.5\times10^{-5}$. "Unbiased in steady state" holds only in the limit $h\to0$, $\beta=\kappa h$ used for Eq. (12).
- **Correct as stated.** Rows 33–38, 40 and 42 were re-derived and confirmed: λ*, the α=1 identity (it fails by up to 58% if ε=0.2>1/G; c09 #4), $D$ coefficients (c09 #5: 1.0014 vs 1, 3.0073 vs 3, offset ≈ 0), $D_R$, $c^\star$, EMA variance and lag, ∂F/∂z=0 at the fixed point, and the kink jumps (c09 #8 to 5 digits).

### 2.10 Modelling assumptions (row 45, Major for scope)

**(a) Rewards.**
- With $r_j=0$ (binary rewards), $p^\star_j=0$ lies on the boundary and $\dot z_j=-p_jS$ decays algebraically, so (S2)–(S3) do not apply in that direction.
- Among valid answers ρ=1, so (S4) is vacuous; §2.7 describes the regime that matters.
- Stochastic rewards with $\mathbb E[R\mid o]=r_o$ leave the mean field unchanged (if R is independent of the counts given o) and add noise.
- Negative rewards make $r^{1/\alpha}$ undefined and flip the IPS boost; the rewards must be shifted to be positive.

**(b) Parameterization and multi-step policies (c11).** For the vanilla update and *any* differentiable policy, size-biasing applied per trajectory gives $\mathbb E[\text{update}]=\nabla_\theta F_G(p(\theta))$, and in the ideal case $\nabla_\theta F_\alpha(p(\theta))$. This was checked on a two-level tree MDP with shared prefixes and two paths per outcome: $7.7\times10^{-10}$ ideal, $2.3\times10^{-9}$ for finite G (exact $5^G$ enumeration). Monte Carlo training of the tree policy (vanilla update, h=0.05, 32 seeds) lands on the *bandit* KKT target. At G=4, α=1 it gives (0.640, 0.360, 0.0004) against (0.639, 0.361, 0); at G=8, α=1 it gives (0.543, 0.351, 0.106) against (0.541, 0.350, 0.109). The ideal target is (0.5, 0.333, 0.167), so the finite-G distortion, including the extinction of C at G=4, carries over to the multi-step policy. At α=2, G=4 there is a gap at h=0.05, but it is finite-h bias: the ℓ1 gap is 0.067, 0.023 and 0.0083 at h=0.05, 0.02 and 0.01, roughly linear in h. The *target* is therefore independent of the parameterization and of path multiplicity whenever it is representable. The *dynamics* are not: $\dot z=J_\theta J_\theta^\top\nabla_zF$ (an NTK-like kernel), so rates, step sizes and spurious local maxima in θ depend on the network. GRPO's std normalization breaks the potential structure.

**(c) KL regularization.** $\dot z_i=p_i\big[r_iw_G(p_i)-\beta\ln(p_i/\pi_i)-(S-\beta\,\mathrm{KL})\big]$. The term $-\beta\ln(p_i/\pi_i)\to+\infty$ as $p_i\to0$, so no outcome goes extinct for β>0. Below the threshold (K=2) the minority keeps $p_2\approx\frac{\pi_2}{\pi_1}\exp\{-(r_1\omega(G)-r_2\omega(1))/\beta\}$. (S4) becomes a smooth crossover of width O(β). The paper's HypoSpace runs use KL to the pretrained model.

**(d) Transfer to LLMs.** The additional differences are $K\gg G$ (§2.7), std normalization (§2.1), PPO clipping, several epochs per batch, and token-level credit assignment. The conclusions concern the mechanism; the numerical thresholds shift.

### 2.11 Consistency and presentation (rows 2, 6, 14, 35, 44, 46, 47)

- Report §2.3.4 declares collapse at $\max_ip_i>0.95$, but Exp 1 Fig 2c (`exp1_collapse.py`, `threshold=0.98`) uses 0.98.
- The paper's IPS flow is Eq. (5) (main text), not Eq. (4); $\dot z_i=p_ia_i$ is App. A.1 Eq. (3).
- $\lVert r\rVert_{1/\alpha}$ is a quasi-norm for α>1.
- "A proof that α=0 is a singular limit" → "a derivation".
- "$D$ vanishes at α=1 and nowhere else": it is also trivially zero at α=0.
- The 19,125-point check is a solver-against-theorem consistency test, not evidence about training.
- Explaining Table 4's lack of a single winner through $\varepsilon^\star=p$ is speculative: HypoSpace rewards are binary. The "identical entries because $\varepsilon<1/G$" explanation is solid (c09 #11).

---

## 3. Numerical checks

All scripts are in `next_phase/case2_checks/`. Run them from the repository root, for example:

```
cd /home/sadia/4-1/402/Project/alpha-ips-rl
python3 -u next_phase/case2_checks/c05_grpo_variants.py > next_phase/case2_checks/c05_output.txt
```

Each `cNN_output.txt` next to the scripts holds the output quoted below. Seeds are fixed. Runtimes are for this 8-core laptop: c06 uses 7 processes; all others use one core.

| Script | What it checks | Runtime |
|---|---|---|
| c01_potential_lyapunov.py | $\dot z=\nabla F_\alpha$; $J$ symmetric everywhere; KL fails for K≥3, α≠1; global convergence; finite-G potential | ~10 s |
| c02_step_size_global.py | RK4 constant; (S3) fails globally for α>1 | ~5 s |
| c03_partial_tie.py | deterministic collapse inside a partial tie | ~5 s |
| c04_tie_noise_martingale.py | martingale; closed-form $E[t_c]$; recurrence | ~1 min |
| c05_grpo_variants.py | V vs B vs N: thresholds, Monte Carlo, K=5 | ~5–10 min |
| c06_finite_h_survival.py | finite-h survival criterion (*) | ~15–20 min on 7 cores (less when run alone) |
| c07_ema_bias.py | EMA dynamics-level bias at finite β | ~5–10 min |
| c08_hj_monotone.py | $s^\star(\alpha)$ non-increasing; single sign change | ~5 min |
| c09_misc_formulas.py | 11 formula and number re-checks | ~10 s |
| c10_recovery_time.py | hyperbolic recovery at finite G | ~1–2 min |
| c11_mdp_portability.py | potential on a tree MDP; Monte Carlo against the bandit target; h-shrink test | ~5–8 min |
| c12_naive_richardson.py | naive Richardson bistability | ~1 min |
| c13_many_outcomes.py | restoring rate for $K\gg G$ | seconds |

**Outputs (verbatim, from `cNN_output.txt`).**

`c01_potential_lyapunov.py`

```text
(a) max rel |rhs_alpha - grad_z F_alpha|      = 2.41e-09 (finite-difference level)
(b) max rel asymmetry of jacobian_alpha(z)     = 1.84e-16 (J is a Hessian everywhere)
(c) sign of dKL(p*||p)/dt at random states (20000 per cell):
    K=2: a=0.3:    0 KL-increases | a=0.7:    0 KL-increases | a=1.0:    0 KL-increases | a=1.5:    0 KL-increases | a=3.0:    0 KL-increases
    K=3: a=0.3:  316 KL-increases | a=0.7:   29 KL-increases | a=1.0:    0 KL-increases | a=1.5:    5 KL-increases | a=3.0:    0 KL-increases
    K=5: a=0.3:  441 KL-increases | a=0.7:   28 KL-increases | a=1.0:    0 KL-increases | a=1.5:    5 KL-increases | a=3.0:    0 KL-increases
(d) global convergence from near-boundary starts (LSODA, rtol 1e-10), r=(5,4,3,2,1):
    alpha=0.2: T=30/lambda_min=  99295.3: max l1(p(T),p*) = 3.7e-14; F decreases: 0; KL increases: 486
    alpha=0.5: T=30/lambda_min=    367.2: max l1(p(T),p*) = 2.6e-10; F decreases: 0; KL increases: 51
    alpha=1.0: T=30/lambda_min=     26.1: max l1(p(T),p*) = 1.2e-12; F decreases: 0; KL increases: 0
    alpha=2.0: T=30/lambda_min=      1.6: max l1(p(T),p*) = 1.7e-12; F decreases: 0; KL increases: 0
    alpha=3.0: T=30/lambda_min=      0.2: max l1(p(T),p*) = 1.1e-12; F decreases: 0; KL increases: 0
(e) max rel |finite-G mean drift - grad_z F_G| = 6.94e-09
    finite-G mean field from near-boundary starts, r=(5,4,3,2,1), eps=1e-3:
      G= 4, alpha=1.0: support 3/5, max l1(p(T), pi_KKT) = 2.6e-04, F_G decreases: 0
      G=16, alpha=1.0: support 5/5, max l1(p(T), pi_KKT) = 4.2e-01, F_G decreases: 0
      G= 8, alpha=0.5: support 3/5, max l1(p(T), pi_KKT) = 5.6e-04, F_G decreases: 0
      G=16, alpha=2.0: support 5/5, max l1(p(T), pi_KKT) = 8.7e-16, F_G decreases: 0
```

`c02_step_size_global.py`

```text
(a) RK4 real-axis limit x* = 2.7852935634;  max |R(-x)| on [0,x*] = 1.000000000000; min R(-x) = 0.2704
(b) r=(4,1), alpha=2: p* = [0.6667 0.3333], lambda* = 8.0000
    p2(0)=  5e-01: sup_path lambda_max(-J) =       10.0  ->  2/sup = 2.00e-01;  Euler h=0.05: l1 err 0e+00 (with p_floor=1e-12: l1 err after 4000 steps 0.00); Euler h=0.2: l1 err 0e+00 (with p_floor=1e-12: l1 err after 4000 steps 0.00)
    p2(0)=  1e-01: sup_path lambda_max(-J) =       18.9  ->  2/sup = 1.06e-01;  Euler h=0.05: l1 err 0e+00 (with p_floor=1e-12: l1 err after 4000 steps 0.00); Euler h=0.2: overflow at step 4 (with p_floor=1e-12: l1 err after 4000 steps 1.33)
    p2(0)=  1e-02: sup_path lambda_max(-J) =      198.1  ->  2/sup = 1.01e-02;  Euler h=0.05: overflow at step 3 (with p_floor=1e-12: l1 err after 4000 steps 0.67); Euler h=0.2: overflow at step 2 (with p_floor=1e-12: l1 err after 4000 steps 0.67)
    p2(0)=  1e-03: sup_path lambda_max(-J) =     1998.0  ->  2/sup = 1.00e-03;  Euler h=0.05: overflow at step 2 (with p_floor=1e-12: l1 err after 4000 steps 0.67); Euler h=0.2: overflow at step 2 (with p_floor=1e-12: l1 err after 4000 steps 0.67)
    p2(0)=  1e-04: sup_path lambda_max(-J) =    19998.0  ->  2/sup = 1.00e-04;  Euler h=0.05: overflow at step 1 (with p_floor=1e-12: l1 err after 4000 steps 0.67); Euler h=0.2: overflow at step 1 (with p_floor=1e-12: l1 err after 4000 steps 0.67)
```

`c03_partial_tie.py`

```text
r=(1,1,1.0), p0=(0.36, 0.34, 0.3): t=1e1: p2/p1=9.444e-01, t=1e3: p2/p1=9.444e-01, t=1e5: p2/p1=9.444e-01, t=1e7: p2/p1=9.444e-01;  tail slope d log(p2/p1)/d log t = +0.000
r=(1,1,0.0), p0=(0.36, 0.34, 0.3): t=1e1: p2/p1=9.026e-01, t=1e3: p2/p1=6.143e-01, t=1e5: p2/p1=1.569e-01, t=1e7: p2/p1=1.762e-02;  tail slope d log(p2/p1)/d log t = -0.481
r=(1,1,0.5), p0=(0.36, 0.34, 0.3): t=1e1: p2/p1=9.199e-01, t=1e3: p2/p1=6.782e-01, t=1e5: p2/p1=2.090e-01, t=1e7: p2/p1=2.487e-02;  tail slope d log(p2/p1)/d log t = -0.470
r=(1,1,0.9), p0=(0.36, 0.34, 0.3): t=1e1: p2/p1=9.389e-01, t=1e3: p2/p1=7.974e-01, t=1e5: p2/p1=3.659e-01, t=1e7: p2/p1=5.488e-02;  tail slope d log(p2/p1)/d log t = -0.420
r=(1,1,0.0), p0=(0.35, 0.35, 0.3): t=1e1: p2/p1=1.000e+00, t=1e3: p2/p1=1.000e+00, t=1e5: p2/p1=1.000e+00, t=1e7: p2/p1=1.000e+00;  tail slope d log(p2/p1)/d log t = +0.000
```

`c04_tie_noise_martingale.py`

```text
(a) u0=0.8, G=16: mean Delta u = -3.01e-05 +- 2.6e-04 (SE); Var = 0.01338 vs 4h^2r^2p(1-p)/G = 0.01337
(b) first passage to max p >= 0.98, h=0.5, r=1 (time units):
     G | formula mean | MC mean (SE)      | MC median | report median (80 seeds, censored)
     4 |          249 |      265 (   4)  |       210 | 177   (uncensored 100.0%)
     8 |          497 |      508 (   9)  |       402 | 383   (uncensored 100.0%)
    16 |          995 |     1035 (  17)  |       820 | 654   (uncensored 100.0%)
    32 |         1989 |     2040 (  34)  |      1535 | 2051   (uncensored 100.0%)
    64 |         3979 |     4187 (  68)  |      3316 | 3066   (uncensored 100.0%)
    log-log slope of MC mean vs G = 0.997  (diffusion theory: exactly 1)
    G=16 with src.dynamics.rhs_sampled (1000 seeds): mean 1009, median 778
(c) G=4, h=0.5, 400 seeds, 200000 steps: seeds that switched winner at least once: 51.7%; mean switches per seed 1.12; time share with max p >= 0.95 in 2nd half: 99.9%
```

`c05_grpo_variants.py`

```text
(a) K=2, r=(4,1), eps=1e-3: critical alpha (minority survives above it)
     G | ln4/lnG | V (exact) | ln4/ln(G-1) | B (exact) | N (exact)
     2 |  2.0000 |    2.0000 |         inf |       inf |       inf
     3 |  1.2619 |    1.2619 |      2.0000 |    2.0000 |    2.0000
     4 |  1.0000 |    1.0000 |      1.2619 |    1.2619 |    1.2619
     5 |  0.8614 |    0.8614 |      1.0000 |    1.0000 |    1.0000
     6 |  0.7737 |    0.7737 |      0.8614 |    0.8614 |    0.8614
     8 |  0.6667 |    0.6667 |      0.7124 |    0.7124 |    0.7124
    16 |  0.5000 |    0.5000 |      0.5119 |    0.5119 |    0.5119
    32 |  0.4000 |    0.4000 |      0.4037 |    0.4037 |    0.4037
    64 |  0.3333 |    0.3333 |      0.3346 |    0.3346 |    0.3346
    max number of interior roots of the K=2 drift on a grid (G in 3..16, alpha in 0.7..2.5): {'V': 1, 'B': 1, 'N': 1}
    K=2 stationary p2 at alpha = 1 (exact mean field):
      G=  4: V: 0 (extinct), B: 0 (extinct), N: 0 (extinct)   [repo K2 solver, V: 0.0000]
      G=  5: V: 0.0734, B: 0 (extinct), N: 0 (extinct)   [repo K2 solver, V: 0.0735]
      G=  6: V: 0.1148, B: 0.0734, N: 0.1864   [repo K2 solver, V: 0.1149]
      G=  8: V: 0.1569, B: 0.1403, N: 0.1332   [repo K2 solver, V: 0.1570]
      G= 16: V: 0.1949, B: 0.1936, N: 0.2038   [repo K2 solver, V: 0.1950]
      G= 64: V: 0.1999, B: 0.1999, N: 0.1919   [repo K2 solver, V: 0.2000]
      G=256: V: 0.1999, B: 0.1999, N: 0.2004   [repo K2 solver, V: 0.2000]
(b) Monte Carlo, r=(4,1), alpha=1, h=0.05, 40k steps, 32 seeds; long-run p2 (2nd half):
    G= 3: V: MC 0.0003 / MF 0.0000 | B: MC 0.0002 / MF 0.0000 | N: MC 0.0002 / MF 0.0000
    G= 4: V: MC 0.0031 / MF 0.0000 | B: MC 0.0004 / MF 0.0000 | N: MC 0.0002 / MF 0.0000
    G= 5: V: MC 0.0643 / MF 0.0734 | B: MC 0.0041 / MF 0.0000 | N: MC 0.0059 / MF 0.0000
    G= 6: V: MC 0.1103 / MF 0.1148 | B: MC 0.0713 / MF 0.0734 | N: MC 0.1855 / MF 0.1864
    G= 8: V: MC 0.1555 / MF 0.1569 | B: MC 0.1382 / MF 0.1403 | N: MC 0.1319 / MF 0.1332
    G=16: V: MC 0.1950 / MF 0.1949 | B: MC 0.1933 / MF 0.1936 | N: MC 0.2034 / MF 0.2038
(c) K=5, r=(5,4,3,2,1), alpha=1, exact mean field (ODE to t=2e4 from uniform):
    G= 8: V(KKT): [0.4028 0.3109 0.2074 0.0789 0.    ]  B: [4.1717e-01 3.1813e-01 2.0453e-01 6.0163e-02 8.4875e-06]  N: [4.0755e-01 3.0646e-01 2.0177e-01 8.4182e-02 3.9570e-05]
    G=16: V(KKT): [0.3583 0.2856 0.2102 0.1272 0.0187]  B: [0.3623 0.2884 0.2114 0.1257 0.0122]  N: [0.3618 0.286  0.2083 0.1265 0.0174]
    G=32: V(KKT): [0.3373 0.2698 0.2022 0.1335 0.0572]  B: [0.3377 0.2701 0.2024 0.1335 0.0563]  N: [0.3401 0.2708 0.2013 0.1312 0.0566]
```

`c06_finite_h_survival.py`

```text
(a) r=(4,1), alpha=1, G=5: margin m = 0.223, mean-field p2 = 0.0735
    h=0.05: (*) holds (theta*= 4.64)  |  T=  20000: <p2>=0.0604  T=  80000: <p2>=0.0633  T= 320000: <p2>=0.0639  T=1280000: <p2>=0.0637
    h=0.10: (*) holds (theta*= 2.32)  |  T=  20000: <p2>=0.0514  T=  80000: <p2>=0.0515  T= 320000: <p2>=0.0517  T=1280000: <p2>=0.0512
    h=0.20: (*) holds (theta*= 1.16)  |  T=  20000: <p2>=0.0270  T=  80000: <p2>=0.0224  T= 320000: <p2>=0.0211  T=1280000: <p2>=0.0195
    h=0.30: (*) FAILS (theta*= 0.77)  |  T=  20000: <p2>=0.0123  T=  80000: <p2>=0.0073  T= 320000: <p2>=0.0056  T=1280000: <p2>=0.0039
    h=0.50: (*) FAILS (theta*= 0.46)  |  T=  20000: <p2>=0.0022  T=  80000: <p2>=0.0012  T= 320000: <p2>=0.0006  T=1280000: <p2>=0.0002
    h=1.00: (*) FAILS (theta*= 0.23)  |  T=  20000: <p2>=0.0004  T=  80000: <p2>=0.0002  T= 320000: <p2>=0.0000  T=1280000: <p2>=0.0000
(b) tie r=(1,1), G=16, h=0.5: (*) predicts persistent noise-collapse for alpha < 0.0116
    alpha=0.000: (*) FAILS (theta*=  0.00) | T= 50000: share(max p>=.95)=0.983  T=200000: share(max p>=.95)=0.998  T=800000: share(max p>=.95)=0.999
    alpha=0.005: (*) FAILS (theta*=  0.44) | T= 50000: share(max p>=.95)=0.911  T=200000: share(max p>=.95)=0.966  T=800000: share(max p>=.95)=0.984
    alpha=0.010: (*) FAILS (theta*=  0.87) | T= 50000: share(max p>=.95)=0.716  T=200000: share(max p>=.95)=0.799  T=800000: share(max p>=.95)=0.845
    alpha=0.015: (*) holds (theta*=  1.29) | T= 50000: share(max p>=.95)=0.389  T=200000: share(max p>=.95)=0.474  T=800000: share(max p>=.95)=0.481
    alpha=0.025: (*) holds (theta*=  2.09) | T= 50000: share(max p>=.95)=0.094  T=200000: share(max p>=.95)=0.121  T=800000: share(max p>=.95)=0.117
    alpha=0.050: (*) holds (theta*=  3.96) | T= 50000: share(max p>=.95)=0.004  T=200000: share(max p>=.95)=0.004  T=800000: share(max p>=.95)=0.004
    alpha=0.250: (*) holds (theta*= 12.75) | T= 50000: share(max p>=.95)=0.000  T=200000: share(max p>=.95)=0.000  T=800000: share(max p>=.95)=0.000
```

`c07_ema_bias.py`

```text
 alpha  G    p     beta | D_EMA (MC +- SE)        | delta-method | plain clip D
  1.0   16  0.30   0.02 | -0.00147 +- 0.00001     | -0.00144     | -0.003323
  1.0   16  0.30   0.10 | -0.00699 +- 0.00001     | -0.00691     | -0.003323
  1.0   16  0.30   0.30 | -0.01883 +- 0.00000     | -0.01801     | -0.003323
  1.0   16  0.50   0.02 | -0.00060 +- 0.00001     | -0.00062     | -0.000015
  1.0   16  0.50   0.10 | -0.00299 +- 0.00000     | -0.00296     | -0.000015
  1.0   16  0.50   0.30 | -0.00798 +- 0.00000     | -0.00772     | -0.000015
  1.0   64  0.30   0.02 | -0.00037 +- 0.00001     | -0.00036     | -0.000000
  1.0   64  0.30   0.10 | -0.00173 +- 0.00000     | -0.00173     | -0.000000
  1.0   64  0.30   0.30 | -0.00455 +- 0.00000     | -0.00450     | -0.000000
  2.0   16  0.30   0.02 | -0.00141 +- 0.00004     | -0.00141     | +0.209509
  2.0   16  0.30   0.10 | -0.00636 +- 0.00004     | -0.00614     | +0.209509
  2.0   16  0.30   0.30 | -0.01286 +- 0.00004     | -0.01029     | +0.209509
  2.0   16  0.50   0.02 | -0.00064 +- 0.00003     | -0.00061     | +0.079396
  2.0   16  0.50   0.10 | -0.00271 +- 0.00003     | -0.00263     | +0.079396
  2.0   16  0.50   0.30 | -0.00508 +- 0.00003     | -0.00441     | +0.079396
  2.0   64  0.30   0.02 | -0.00035 +- 0.00002     | -0.00035     | +0.040208
  2.0   64  0.30   0.10 | -0.00153 +- 0.00002     | -0.00154     | +0.040208
  2.0   64  0.30   0.30 | -0.00271 +- 0.00002     | -0.00257     | +0.040208
```

`c08_hj_monotone.py`

```text
fast solver vs repository solver (30 random cases): max |dp| = 2.1e-10, max |d ln S| = 1.4e-09
600 (r, G, eps) cases x 60 alphas in [0.02, 40], 2770 outcome curves:
  cases where s*(alpha) = S*/m_G^alpha increased:      0
  curves h_j(alpha) that decreased anywhere:          0
  curves with more than one sign change:              0
  alpha->inf criterion vs sign of h_j(40) (eps<1/G):   2216/2216 agree
```

`c09_misc_formulas.py`

```text
1  (u+sinh u) from 0.1 to ln 99 = 53.8899  (report: 53.89)
2  lambda(r=(4,1), alpha=1) = 1.600000  (report: 1.6)
   tie K=3, alpha=0.5: rates [0.490748 0.490748] vs alpha*rbar*K^(alpha-1) = 0.490748
   tie K=5, alpha=2.0: rates [17. 17. 17. 17.] vs alpha*rbar*K^(alpha-1) = 17.000000
3  eps* = p by brute force (grid ratio 1.0046): max |eps*/p - 1| = 2.08e-03
   add-lambda lambda* = Kp(1-p)/(1-Kp)^2 by brute force: max rel diff 6.6e-06
4  size-biasing: max rel |phi_G/p - E[omega(1+B)]| = 1.9e-15
   alpha=1 closed form vs binomial sum (G=12): 2.0e-13
   ... with eps = 0.2 > 1/G the identity fails by up to 0.58 (needs eps <= 1/G)
5  alpha=0.5: G p D/(1-p): clip -0.1250 (theory -0.1250); offset c*: +0.00002 (theory 0)
5  alpha=2.0: G p D/(1-p): clip +1.0014 (theory +1.0000); offset c*: -0.00027 (theory 0)
5  alpha=3.0: G p D/(1-p): clip +3.0073 (theory +3.0000); offset c*: -0.00106 (theory 0)
6  G= 8 alpha=0.5: naive w_G monotone=False (max rise 6.06e-02), interior roots rho=1.5/4: [0, 0]; guarded monotone=True
6  G= 8 alpha=1.0: naive w_G monotone=False (max rise 2.42e+00), interior roots rho=1.5/4: [1, 1]; guarded monotone=True
6  G= 8 alpha=2.0: naive w_G monotone=False (max rise 2.49e+03), interior roots rho=1.5/4: [1, 1]; guarded monotone=True
6  G=16 alpha=0.5: naive w_G monotone=False (max rise 1.03e-01), interior roots rho=1.5/4: [0, 0]; guarded monotone=True
6  G=16 alpha=1.0: naive w_G monotone=False (max rise 4.66e+00), interior roots rho=1.5/4: [1, 1]; guarded monotone=True
6  G=16 alpha=2.0: naive w_G monotone=False (max rise 4.97e+03), interior roots rho=1.5/4: [1, 1]; guarded monotone=True
6  G=64 alpha=0.5: naive w_G monotone=False (max rise 1.81e-01), interior roots rho=1.5/4: [2, 0]; guarded monotone=True
6  G=64 alpha=1.0: naive w_G monotone=False (max rise 1.44e+01), interior roots rho=1.5/4: [2, 0]; guarded monotone=True
6  G=64 alpha=2.0: naive w_G monotone=False (max rise 1.91e+04), interior roots rho=1.5/4: [2, 0]; guarded monotone=True
7  alpha_c(O5, G=16) = 0.8862 (report 0.886); pairwise ln5/ln16 = 0.5805
   minimum G keeping O4 at some alpha: 3
   minimum G keeping O5 at some alpha: 6
   alpha=4, r=(4,1): p2 by G G=8: 0.4334, G=10: 0.4419, G=12: 0.4447, G=14: 0.4440, G=16: 0.4413  (G->inf: 0.4142)
8  kink at eps=1/16: slope jump -0.03938 vs Delta_k -0.03938
8  kink at eps=2/16: slope jump -0.14546 vs Delta_k -0.14546
8  kink at eps=3/16: slope jump -0.33434 vs Delta_k -0.33434
9  alpha=1: max |m_baseline(G) - m_vanilla(G-1)| over 2000 random (K,r,p,G) = 2.7e-15
   => K=5, r=(5,4,3,2,1): baseline at G=16 has the vanilla G=15 target [0.3623 0.2884 0.2114 0.1257 0.0122]
10 clip, G=16, p=0.1, alpha=1: relative RMSE of omega(n) incl. n=0 = 42.6 (report 42.6); share of MSE from n=0: 1.000
   conditional on n>=1: 0.479;  size-biased (what the mean field sees): 0.468;  update term p^ omega(n): mean 0.8147, sd 0.3885
11 paper Table 4: clip can fire (eps > 1/G)?  entries
   G= 4: fires [False, False, False]  entries (17.63, 17.63, 17.63)
   G= 8: fires [False, False, True]  entries (27.5, 27.5, 41.99)
   G=16: fires [False, True, True]  entries (41.02, 33.05, 43.91)
   G=32: fires [False, True, True]  entries (32.13, 35.27, 25.62)
   G=64: fires [False, True, True]  entries (36.52, 39.12, 42.19)
```

`c10_recovery_time.py`

```text
(a) K=2, r=(4,1), alpha=1, G=16: mean-field p2* = 0.1950; target p2*/2; c = 24.0
    p2(0)=1e-02: finite-G mean field t =          5.4  (1/(c p2(0)) =          4.2);  ideal IPS t =   1.54
    p2(0)=1e-04: finite-G mean field t =        420.3  (1/(c p2(0)) =        416.7);  ideal IPS t =   3.87
    p2(0)=1e-06: finite-G mean field t =      41672.6  (1/(c p2(0)) =      41666.7);  ideal IPS t =   6.18
    p2(0)=1e-08: finite-G mean field t =    4166674.9  (1/(c p2(0)) =    4166666.7);  ideal IPS t =   8.48
(b) Monte Carlo agent (rhs_sampled), h=0.05, 64 seeds: time to reach p2 >= p2*/2
    p2(0)=1e-03: median t =     43.7, recovered 100% (mean-field prediction 44.1)
    p2(0)=1e-04: median t =    410.8, recovered 100% (mean-field prediction 420.3)
(c) K=5, G=16, alpha=1: O5 from 1e-6 to p5*/2 = 0.0093: finite-G mean field t = 348086 (invasion rate r5 w_G(0) - S* = 2.06); ideal IPS t = 8.92
```

`c11_mdp_portability.py`

```text
(a) ideal IPS: max rel |E[update] - grad F_alpha(p(theta))| = 7.7e-10
    finite G (exact 5^G enumeration): max rel |E[update] - grad F_G(p(theta))| = 2.3e-09
(b) G=4, alpha=1.0: tree MC long-run outcome dist [6.399e-01 3.598e-01 4.000e-04] | bandit KKT target [0.6389 0.3611 0.    ] | ideal r^(1/alpha) [0.5    0.3333 0.1667]
(b) G=8, alpha=1.0: tree MC long-run outcome dist [0.5428 0.3508 0.1064] | bandit KKT target [0.5412 0.35   0.1089] | ideal r^(1/alpha) [0.5    0.3333 0.1667]
(b) G=4, alpha=2.0: tree MC long-run outcome dist [0.5365 0.3906 0.0729] | bandit KKT target [0.5136 0.3754 0.1109] | ideal r^(1/alpha) [0.4177 0.3411 0.2412]
(c) G=4, alpha=2, h=0.05: tree MC [0.5342 0.3882 0.0776] | bandit KKT [0.5136 0.3754 0.1109] | l1 gap 0.0667
(c) G=4, alpha=2, h=0.02: tree MC [0.5205 0.3799 0.0995] | bandit KKT [0.5136 0.3754 0.1109] | l1 gap 0.0228
(c) G=4, alpha=2, h=0.01: tree MC [0.5163 0.3769 0.1068] | bandit KKT [0.5136 0.3754 0.1109] | l1 gap 0.0083
```

`c12_naive_richardson.py`

```text
G=8, alpha=1, naive Richardson: w_G(0)=-486.0, w_G(1)=1.00, w_G<0 on p in [0.000, 0.778]
   r=(1.5, 1.0): rest points of f in (0,1): [(0.5242, 'unstable')]; mean field from p1=0.5 -> p1=0.0000; MC (h=0.01, 32 seeds) long-run p1 = 0.5000
   r=(4.0, 1.0): rest points of f in (0,1): [(0.5816, 'unstable')]; mean field from p1=0.5 -> p1=0.0000; MC (h=0.01, 32 seeds) long-run p1 = 0.4687
G=16, alpha=1, naive Richardson: w_G(0)=-472.0, w_G(1)=1.00, w_G<0 on p in [0.000, 0.505]
   r=(1.5, 1.0): rest points of f in (0,1): [(0.5008, 'unstable')]; mean field from p1=0.5 -> p1=0.0000; MC (h=0.01, 32 seeds) long-run p1 = 0.8125
   r=(4.0, 1.0): rest points of f in (0,1): [(0.5027, 'unstable')]; mean field from p1=0.5 -> p1=0.0000; MC (h=0.01, 32 seeds) long-run p1 = 0.9687
```

`c13_many_outcomes.py`

```text
  K   G  alpha | slowest restoring rate: finite-G MF | ideal IPS | ratio | G(G-1)/(2K^2)
    4   8  1.0  |      0.63292                        |         1 |  0.6329 |  1.7500
   16   8  1.0  |      0.08503                        |         1 |  0.0850 |  0.1094
   64   8  1.0  |     0.006421                        |         1 |  0.0064 |  0.0068
  256   8  1.0  |   0.00042062                        |         1 |  0.0004 |  0.0004
    4  16  1.0  |      0.93652                        |         1 |  0.9365 |  7.5000
   16  16  1.0  |      0.26411                        |         1 |  0.2641 |  0.4688
   64  16  1.0  |     0.025334                        |         1 |  0.0253 |  0.0293
  256  16  1.0  |    0.0017656                        |         1 |  0.0018 |  0.0018
    4   8  2.0  |       5.6208                        |         8 |  0.7026 |     nan
   16   8  2.0  |      0.96112                        |        32 |  0.0300 |     nan
   64   8  2.0  |      0.07597                        |       128 |  0.0006 |     nan
  256   8  2.0  |    0.0050299                        |       512 |  0.0000 |     nan
    4  16  2.0  |       10.395                        |         8 |  1.2994 |     nan
   16  16  2.0  |       5.4813                        |        32 |  0.1713 |     nan
   64  16  2.0  |      0.58808                        |       128 |  0.0046 |     nan
  256  16  2.0  |     0.042029                        |       512 |  0.0001 |     nan
```


**Did any check contradict the report?** Yes. Simulations contradict five of the six ✗ claims: rows 10 (c03), 11 (c04), 19 (c12), 21 (c05, c09 #9) and 39 (c07). They also contradict the "any α>0" reading of row 23 (c06 b). The sixth ✗, row 46, is a text/code inconsistency found by reading `exp1_collapse.py`. No reported number was contradicted: every value I re-computed matched, including 53.89, 1.6, 0.886, the 0.445 peak at G=12, $\varepsilon^\star/p$, the α=1 identity and the $D$ coefficients.

---

## 4. Suggested replacement text for the report

Line numbers refer to `report/report.tex`.

1. **Abstract (l. 94–95).** *Current:* "and a \emph{universal survival limit} for finite groups of $G$ rollouts, which for the paper's clipped estimator reads $\alpha>\ln(r_1/r_2)/\ln\min\{G,1/\varepsilon\}$." → *New:* "and a finite-group survival criterion for the two-outcome mean field of group-REINFORCE without a baseline, which for the paper's clipped estimator reads $\alpha>\ln(r_1/r_2)/\ln\min\{G,1/\varepsilon\}$ (with GRPO's group-mean baseline, $G$ is replaced by $G-1$)."
2. **Abstract (l. 105–106).** *Current:* "…exact up to the miss probability $(1-p)^G$, but loses an outcome whenever $G\le r_1/r_2$." → *New:* "…exact up to the miss probability $(1-p)^G$, but without a baseline it loses the weaker of two outcomes whenever $\min\{G,1/\varepsilon\}\le r_1/r_2$, and with GRPO's baseline whenever $G\le1+r_1/r_2$."
3. **Intro (l. 178–179).** *Current:* "Its claim that collapse happens ``even with equal rewards'' is argued from a deterministic flow whose drift is exactly zero at a tie." → *New:* "Its claim that collapse happens ``even with equal rewards'' holds when the tied outcomes coexist with worse ones, because their common advantage is then positive; it fails when all outcomes are tied, where the deterministic drift is exactly zero and collapse needs sampling noise."
4. **Box 1 (S3) (l. 195).** *Current:* "An explicit update is stable iff $h\lambda_{\max}<2$ (Euler) or $h\lambda_{\max}<2.7853$ (RK4)." → *New:* "The fixed point of an explicit update is locally asymptotically stable if $h\lambda_{\max}<2$ (Euler) or $h\lambda_{\max}<2.7853$ (RK4), and unstable if the inequality is reversed. Along the path the local rate $\lambda_{\max}(-J(z))$ can be far larger; for $\alpha>1$ it is unbounded near the boundary."
5. **Box 1 (S4) title (l. 202–203).** *Current:* "(S4) Universal survival limit. With finite groups of $G$ rollouts, outcome 2 survives iff" → *New:* "(S4) Dynamic-range survival criterion (two outcomes, mean field, no-baseline update, non-increasing $\omega$). Outcome 2 survives iff"
6. **Contributions (l. 252).** *Current:* "A proof that $\alpha=0$ is a singular limit: at an exact tie the deterministic flow does not move, and convergence turns from exponential to algebraic." → *New:* "A derivation showing that $\alpha=0$ is a singular limit: at a full tie the deterministic flow does not move, and with a reward gap convergence turns from exponential to algebraic."
7. **Report §2.2.4 (l. 462–464).** *Current:* "The paper's claim that collapse happens ``even with equal rewards'' therefore describes the \emph{stochastic} process, where the finite group never reports the exact mean reward." → *New:* "At a full tie the paper's claim therefore describes only the stochastic process. There the expected update is exactly zero, so the logits perform a martingale random walk whose noise vanishes at the vertices. Runs drift to a vertex in mean time $(G/(hr^2))(U^2/2+\cosh U-1)$, $U=\ln\frac{\theta}{1-\theta}$, and later switch. If the tied outcomes coexist with a worse one, the deterministic flow itself separates them, $\frac{d}{dt}\ln\frac{p_1}{p_2}=(p_1-p_2)(r_{\rm top}-\bar r)$, with $p_2/p_1\sim t^{-1/2}$."
8. **Theorem (l. 510).** *Current:* "Let $K=2$ and $\rho=r_1/r_2\ge1$. For any weight rule $\omega$, the mean field has an interior rest point (the minority survives) iff" → *New:* "Let $K=2$, $\rho=r_1/r_2>1$, and let $\omega$ be non-increasing on $\{1,\dots,G\}$ with $\omega(1)>\omega(G)$. The mean field of the no-baseline update has an interior rest point, which is then unique and attracts every interior start, iff"
9. **Proof (l. 520).** *Current:* "Since $w_G$ is monotone, a root in $(\tfrac12,1)$ exists iff $f(1)<0$." → *New:* "Because $\omega$ is non-increasing, $w_G'(p)=(G-1)\mathbb E_{\mathrm{Bin}(G-2,p)}[\omega(2+B)-\omega(1+B)]<0$, so $f$ is strictly decreasing and has a root in $(\tfrac12,1)$ iff $f(1)<0$. Since $\dot u=2p_1p_2f(p_1)$, the root attracts every interior start."
10. **Report §2.2.5 (ii) (l. 530–531).** *Current:* "Even the paper's $\alpha=1$ needs $G>r_1/r_2$: with $r=(4,1)$, IPS-GRPO at $G=4$ loses the weaker outcome." → *New:* "Even at $\alpha=1$ the no-baseline update needs $G>r_1/r_2$, and GRPO's group-mean baseline needs $G>1+r_1/r_2$: with $r=(4,1)$ both lose the weaker outcome at $G=4$, and IPS-GRPO loses it at $G=5$ as well."
11. **Corollary (l. 535–538).** *Current:* "…Naive split-half Richardson gives a singleton the negative weight $\omega_R(1)=…$, so the minority dies unconditionally." → *New:* "…Naive split-half Richardson gives a singleton the negative weight $\omega_R(1)=…$; its $w_G$ is negative over a wide range of $p$ and not monotone, so the theorem does not apply. The two-outcome mean field becomes bistable, and from a uniform start it can drive the better outcome extinct."
12. **Report §2.3.4 (l. 686).** *Current:* "It does what GRPO does: it samples a group, estimates $\hat p$ from counts, and takes one gradient step." → *New:* "It samples a group, estimates $\hat p$ from counts, and takes one REINFORCE step without a baseline. Unlike GRPO it does not subtract the group mean or divide by the group standard deviation, which changes the finite-$G$ thresholds (Section X)."
13. **"What is measured" (l. 718).** *Current:* "\emph{Collapse} is declared when $\max_ip_i>0.95$" → *New:* "\emph{Collapse} is declared when $\max_ip_i\ge0.95$ ($0.98$ for the collapse-time scan of Case 2c)".
14. **Key finding 1 (l. 1116–1117).** *Current:* "Collapse at equal rewards comes from sampling noise, not from the objective." → *New:* "At a full tie, collapse comes from sampling noise; when the tied best outcomes coexist with worse ones, the objective itself separates them, but only algebraically."
15. **Key finding 2 (l. 1123–1124).** *Current:* "any $\alpha>0$ prevents the noise-driven collapse" → *New:* "every tested $\alpha\ge0.25$ prevents the noise-driven collapse (at a finite step a minimum $\alpha$ is needed: $\alpha>0.0116$ for $G=16$, $h=0.5$)".
16. **Key finding 5 (l. 1136–1138).** *Current:* "At the paper's own $\alpha=1$, a group of $G=4$ loses an answer worth a quarter of the best one ($p_1=0.9965$)." → *New:* "At $\alpha=1$ a group of $G=4$ loses an answer worth a quarter of the best one ($p_1=0.9965$) under our no-baseline update; with GRPO's baseline this happens up to $G=5$."
17. **Table 6 (l. 1173).** *Current:* "Deterministic flow is flat to $10^{-16}$ at a tie; collapse comes only from sampling noise" → *New:* "At a full tie the flow is flat to $10^{-16}$ and collapse is noise-driven (a martingale); with worse outcomes present the flow separates tied outcomes deterministically ($p_2/p_1\sim t^{-1/2}$)".
18. **Design rule (l. 1193–1194).** *Current:* "keep the learning rate $h<2/\lambda_{\max}$ (Euler) along the whole path (S2)--(S3)." → *New:* "keep $h<2/\max_t\lambda_{\max}(-J(z(t)))$ (Euler). $J$ is symmetric, so this is computable along the path; for $\alpha>1$, initialise away from the boundary."
19. **Limitations (l. 1247–1249).** *Current:* "\emph{Mean-field limit.} The finite-$G$ theory is the $h\to0$ limit of the sampled update. Monte Carlo gaps of up to 0.013 are finite-$h$ bias, and they shrink with $h$ as measured." → *New:* "\emph{Mean-field limit.} The finite-$G$ theory is the $h\to0$ limit. At a finite step the two-outcome survival condition becomes $G(1-e^{-2hr_2\omega(1)/G})>2hr_1\omega(G)$, which shifts the boundary by about $hr_2\omega(1)/G$ in $m$. Near $m=0$ the approach is algebraic, so finite runs report transients."
20. **Limitations (l. 1251–1254).** *Current:* "\emph{Numerical, not proven.} The single sign change of $h_j(\alpha)$…was checked on grids, not proven." → *New:* "The single sign change of $h_j(\alpha)$ is proven: $S^\star(\alpha)/\min\{G,1/\varepsilon\}^\alpha$ is non-increasing in $\alpha$ (Appendix)."
21. **Limitations (l. 1242–1245).** *Current:* "All results are for the outcome-selection bandit with a group-REINFORCE update. GRPO's per-group advantage standardization and PPO clipping are not modeled, so the conclusions concern the weight rule, not GRPO's other machinery." → *New:* "All results are for the outcome-selection bandit with a no-baseline group-REINFORCE update. GRPO's group-mean baseline shifts the survival limit to $\ln\rho/\ln(G-1)$, and its standardization makes the update sign-like near $p^\star$, so (S2)–(S4) are not IPS-GRPO's laws without these corrections. PPO clipping is not modeled, and a KL term would remove extinction altogether."
22. **Conclusion (l. 1233).** *Current:* "the universal finite-group survival limit $\omega(1)/\omega(G)>r_1/r_2$" → *New:* "the finite-group survival criterion $\omega(1)/\omega(G)>r_1/r_2$ (two outcomes, no-baseline update)".
23. **Conclusion (l. 1239).** *Current:* "and where they fail ($G\le r_1/r_2$)" → *New:* "and where they fail ($G\le r_1/r_2$ without a baseline, $G\le1+r_1/r_2$ with GRPO's)".
24. **Case 15 (l. 1081).** *Current:* "EMA RMSE 0.077 vs.\ 42.6 for the paper's clip." → *New:* "EMA RMSE 0.077 vs.\ 42.6 for the paper's clip; the 42.6 comes entirely from empty groups, whose weight the update never uses (0.47 conditional on a hit)."
25. **Report §2.2.1 (l. 398).** *Current:* "the paper's IPS Eq.~(4)" → *New:* "the paper's IPS Eq.~(5)".
26. **derivation.md, "An important subtlety".** *Current:* "a Pólya-urn-style rich-get-richer mechanism" → *New:* "neutral drift: the expected update is exactly zero and the noise vanishes at the vertices, so runs drift to a vertex and stay there for a long time, but not for ever".
27. **derivation_exp5.md §6.3.** *Current:* "an EMA is unbiased in steady state, unlike $\varepsilon$ and $\lambda$" → *New:* "in the joint limit $h\to0$, $\beta=\kappa h$ the EMA leaves the fixed point unchanged; at finite $\beta$ its dynamics-level distortion is $\approx\alpha\beta\frac{1-p}{Gp}[\frac{\alpha+1}{2(2-\beta)}-1]$, which at $\alpha=1$ exceeds the clip's $-(1-p)^G$."

---

## 5. Prioritized fix plan

| Priority | Fix | Needs | Time |
|---|---|---|---|
| 1 | IPS-GRPO relabelling plus the GRPO-baseline proposition (§2.1) | math (done here); new `src/grpo_variants.py`; re-run Exp 2 Fig 4b–c with V/B/N (template c05) | ~1 day |
| 2 | Equal-reward critique: partial tie and martingale mechanism (§2.2) | text; c03/c04 figures | ~½ day |
| 3 | Potential $F_\alpha$, Theorem B, finite-G potential $F_G$ (§2.3) | math (proof given) | ~½ day |
| 4 | (S3) local wording; design rule (§2.4) | text; optional c02 figure | ~2 h |
| 5 | Finite-h survival criterion (*) (§2.5) | math (given); c06 as validation | ~½ day |
| 6 | Theorem hypotheses; naive Richardson; EMA; "factor 555" (§2.6, §2.9) | text; c07/c09/c12 numbers | ~1 h |
| 7 | Limitations: recovery time, K≫G, KL, binary rewards, portability (§2.7, §2.10) | text; c10/c11/c13 | ~2 h |
| 8 | Single-sign-change proof and support monotonicity (§2.8) | math (given); c08 | ~1 h |
| 9 | Presentation items (§2.11) | text | ~1 h |

Total: about 3–4 days of writing and math, plus about 30 minutes of CPU for the new runs.
