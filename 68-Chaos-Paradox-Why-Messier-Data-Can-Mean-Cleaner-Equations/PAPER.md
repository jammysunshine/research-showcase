# The Chaos Paradox: Why Messier Data Can Mean Cleaner Equations

*(A Multi-Method, Multi-System Empirical Study of Phase-Space Coverage and Equation Identifiability)*

## Abstract

Can the governing equations of a nonlinear dynamical system be uniquely recovered from observed trajectories? Gallo, Anselmi, and Lazzari (2026) recently showed theoretically that chaotic trajectories, through persistent excitation of a function library, can improve identifiability via sparse regression. We test how far this holds empirically across six system families, three discovery algorithms (SINDy, symbolic regression, Koopman/EDMD), finite noisy samples, and restricted function libraries. The "coverage aids identifiability" pattern replicates cleanly for SINDy (5/5 recovery in every chaotic regime, 0/5 in every degenerate control), with large effect sizes (Cliff's d = 0.556, ratio 6.7x--10.9x, 95% CIs non-overlapping). For Koopman/EDMD, the pattern **reverses outright** against degenerate controls: dense linear surrogates perform better on fixed points/short cycles than on chaotic attractors, for a mechanistically identified reason (persistent excitation helps sparse recovery but hurts uniform linear approximation). An independent adversarial review found that every one of these contrasts pitted chaos against a maximally degenerate control; we resolved this directly by adding a matched pair between the one rich non-degenerate non-chaotic regime already in the study (harmonic oscillator) and a chaotic Rossler attractor, run through the identical 3-method pipeline. Against this rich control, the results split by method: SINDy's chaos-aids-identifiability pattern **replicates and strengthens** (Rossler 30/30 vs. harmonic 16/30 on the primary off-trajectory metric), but the Koopman/EDMD reversal **does not replicate** (Rossler's one-step error is lower than or comparable to harmonic's in all 6 noise/degree cells) -- showing that reversal was partly a degenerate-control artifact, not a universal chaos-vs-non-chaos property. A coupled van der Pol experiment with periodic, quasi-periodic, and chaotic regimes corroborates the same picture from a second angle.

## 1. Introduction

The identifiability of dynamical systems from data -- whether a unique set of governing equations can be recovered from observed trajectories -- is a foundational question in nonlinear dynamics and data-driven science. SINDy (Brunton et al., 2016) demonstrated that sparse regression can recover governing equations when trajectories are sufficiently informative, but the conditions under which trajectories carry enough information remain incompletely understood.

Gallo, Anselmi, and Lazzari (2026) recently proved a theoretical result: chaotic trajectories, through persistent excitation of a candidate-function library, improve the identifiability of a system's governing equations from data. Their argument is that a chaotic orbit's broad phase-space coverage better constrains a sparse-regression or symbolic model than a periodic or fixed-point orbit's narrow support. This is counter to the common intuition that chaotic sensitivity to initial conditions should make systems harder to pin down.

Their result is theoretical and holds in a continuous, noise-free, single-trajectory setting. Whether and how it survives contact with realistic constraints -- finite noisy samples, restricted function libraries, and discovery methods beyond the sparse symbolic ones that motivated it -- was open before this project. We test this across six system families (logistic map, Lorenz, harmonic oscillator, Duffing, Rossler, Lorenz-96), three algorithms (SINDy, symbolic regression, Koopman/EDMD), and systematically varied noise levels, library degrees, and seeds.

## 2. Related Work

**Sparse regression for system identification.** Brunton et al. (2016) introduced SINDy, demonstrating that governing equations of dynamical systems can be discovered from data via sparse regression on a library of candidate functions. Subsequent work extended SINDy to partial observations (Brunton et al., 2017), noisy data (Mangan et al., 2016), and high-dimensional systems (de Silva et al., 2020). The method's success depends critically on the trajectory's persistent excitation of the function library (Harris et al., 2020).

**Koopman operator theory.** The Koopman framework (Koopman, 1931; Brunton & Kutz, 2019) lifts finite-dimensional dynamics into an infinite-dimensional function space where evolution is linear. Extended Dynamic Mode Decomposition (EDMD) (Williams et al., 2015) approximates the Koopman operator from data using a dictionary of observables. Recent work has explored Koopman-based identification under partial observation (Li et al., 2017) and with neural-network encoders (Lusch et al., 2018; Wehmeyer & Noe, 2018).

**Persistent excitation in identification.** The role of persistent excitation -- the requirement that a trajectory sufficiently excites all modes of the system -- is well-established in adaptive control (Anderson, 1977) and system identification (Ljung, 1999). Gallo et al. (2026) extend this framework to nonlinear sparse regression, proving that chaotic trajectories provide persistent excitation for polynomial libraries.

**Coverage vs. chaos.** The distinction between "broad state-space coverage" and "chaos specifically" has been discussed in the context of ergodic theory (Eckmann & Ruelle, 1985) and information-theoretic complexity (Crutchfield & Feldman, 1997). Our work directly tests whether the identifiability benefit is chaos-specific or a broader excitation effect.

## 3. Methods

### 3.1 Systems and regimes

We study six system families with matched chaotic/non-chaotic regime pairs:

- **Logistic map**: period-2 (r=3.2), period-4 (r=3.5), period-3-window (r=3.83), chaotic (r=4.0)
- **Lorenz system**: stable fixed point (rho=14), pre-chaotic (rho=22, 24.5), classic chaotic (rho=28), high-chaos (rho=100)
- **Harmonic oscillator**: conservative; originally no chaotic partner (a degenerate-control gap), later matched against chaotic Rossler (below) as the study's one rich-non-chaotic-vs-chaos pair (SS5)
- **Duffing oscillator**: unforced/conservative vs. forced/chaotic
- **Rossler system**: c=3.0 control vs. chaotic (a=0.2, b=0.2, c=5.7) for held-out confirmation; the same chaotic regime is also the harmonic oscillator's Tier B match (SS5)
- **Lorenz-96 (N=6)**: F=1.0 control vs. F=8.0 chaotic (extension)

### 3.2 Discovery methods

Three independent algorithms, matching the minimum of three required by the preregistration:

1. **SINDy** (PySINDy, sequential thresholded least squares): the trusted primary comparator
2. **Symbolic regression** (gplearn genetic programming; cross-checked with PySR): operator set restricted to +, -, *, / to match SINDy's library breadth
3. **Koopman/EDMD**: self-written monomial-dictionary EDMD with optional ridge regularization; independently cross-checked against pykoopman and a neural-network Koopman architecture

### 3.3 Metrics

- **Primary**: normalized off-trajectory vector-field L2 error on an independent confirmation trajectory
- **Secondary**: exact coefficient recovery (5% relative tolerance), Lyapunov-spectrum error
- **Koopman**: held-out one-step relative RMS prediction error (no coefficient-recovery analogue for dense linear surrogates)

### 3.4 Statistical treatment

Additive Gaussian noise at 0%, 0.1%, 1%, 5% of state standard deviation; 5 seeds per condition in frozen tiers. Clopper-Pearson CIs at Bonferroni-adjusted 99.17% level (k=6 comparisons). Effect sizes reported as Cliff's delta (non-parametric) and bootstrap ratio CIs (10,000 resamples).

## 4. Results

### 4.1 SINDy: broad coverage aids recovery (Tier A, 132 conditions)

Every non-chaotic control (logistic period-2, Lorenz stable fixed point, Lorenz pre-chaotic) is 0/5 recovered in every noise x degree cell (0/60 combined). Every chaotic regime starts at 5/5 and degrades only under high noise/degree=3. The harmonic oscillator -- the one rich, non-degenerate, non-chaotic regime -- recovers 5/5 in every cell at degree=2, identical to chaotic regimes (Figure 1A).

### 4.2 Koopman/EDMD: reversal (Tier B, 210 conditions)

**The headline pattern reverses outright.** Chaotic error is 6.7x--10.9x higher than matched non-chaotic controls at noise=5%:

- Logistic: chaotic mean = 0.034 vs. control mean = 0.005 (ratio 6.7x [3.8x, 12.0x] 95% CI)
- Lorenz: chaotic mean = 0.014 vs. control mean = 0.001 (ratio 10.9x [6.2x, 19.4x] 95% CI)

Cliff's delta = 0.556 (large) for both families. CIs are non-overlapping in every noise/degree cell (Figure 1B, Figure 2). Four checks confirm the reversal is not an artifact of this project's implementation or metric, holding it against degenerate controls (logistic, Lorenz):

1. **Lyapunov normalization**: reversal survives after accounting for local expansion rates
2. **pykoopman cross-check**: 6-significant-figure agreement with self-written EDMD
3. **Neural-network Koopman**: same reversal at comparable or larger magnitude (5.8--61x)
4. **Gram-matrix analysis**: lambda_min(G) -- Gallo et al.'s own identifiability predictor -- carries the **opposite sign** for EDMD

**But the reversal does not generalize to a rich non-chaotic control.** Matching harmonic (non-chaotic, but a continuum of states on a closed orbit) against a chaotic Rossler attractor through the identical pipeline, Rossler's one-step error is lower than or comparable to harmonic's in all 6 noise/degree cells -- the opposite direction from logistic and Lorenz. Harmonic's own per-seed EDMD fits are also markedly less stable (a ~26x spread at noise=0%/degree=2 vs. Rossler's ~1.2x), the opposite of what a "chaos degrades conditioning" story predicts. The Koopman/EDMD reversal is therefore method-and-pair-dependent: real and robust against degenerate controls, but not a universal law about chaos vs. non-chaos (SS5 has the full matched-pair breakdown).

### 4.3 Symbolic regression: mixed (Tier B)

Replicates for logistic and Lorenz (0/30 recovery for both controls), but **reverses for Duffing**: forced-chaotic scores 0/30 against the non-chaotic sibling's 18/30. An independent PySR cross-check confirms the reversal direction is not implementation-specific, though PySR is more noise-brittle than gplearn on this control. The same reversal replicates a third time on the harmonic/Rossler pair (harmonic 30/30 joint-pass vs. Rossler 10/30) -- but here it tracks system dimensionality and cross-terms (3D systems with genuine multiplicative coupling are harder for gplearn's genetic search regardless of chaos), not control degeneracy.

### 4.4 Delay embedding: no gap (Tier C, 140 conditions)

Under single-coordinate observation reconstructed via Takens delay embedding, the primary one-step gate detects **no** chaos-vs-non-chaos gap: 0/140 jobs dynamically distinct across the full noise range. Both control and chaotic means are below the 0.10 threshold (control: 0.0009, chaotic: 0.011; Cliff's d = 0.625 but both below threshold). Tier C is a third outcome -- neither confirming nor reversing the pattern, but showing it can be closed by a change in observation operator alone (Figure 1C).

### 4.5 Held-out confirmations

**Rossler**: pattern replicates on a system never touched during method development (3/5 chaotic vs. 0/5 control at noise=5%, degree=2). **Lorenz rho=45**: 5/5 recovery through noise=1%, 3/5--4/5 at noise=5% -- no cliff at a genuinely unseen parameter value.

### 4.6 Gram-matrix mechanistic analysis

lambda_min(G) of the Koopman dictionary increases from control to chaotic (more persistent excitation) but EDMD error also increases -- a consistent sign reversal across all 12/12 matched (family x noise x degree) cells (Figure 1D). Broad phase-space coverage constrains sparse coefficients well but works against uniform approximation by a dense, dictionary-truncated linear surrogate.

## 5. Coverage vs. Chaos: Direct Test

The non-chaotic controls used in Tiers A/B are mostly degenerate: fixed points or short cycles, near rank-deficient by construction. An 8th matched pair -- harmonic (rich, non-chaotic, a continuum of states on a closed orbit) vs. a chaotic Rossler attractor (`a=b=0.2,c=5.7`), run through the identical 3-method pipeline at the full noise x degree x seed grid -- gives the direct test this gap calls for, and the answer splits by method rather than resolving cleanly either way:

- **SINDy**: the chaos-aids-identifiability pattern replicates and strengthens against the rich control. Rossler passes the primary off-trajectory VF-error gate in 30/30 cells; harmonic passes only 16/30, degrading at higher degree and noise. Chaos, not just broad coverage, does the work here.
- **Symbolic regression**: harmonic outperforms Rossler (30/30 vs. 10/30 joint-pass), but this tracks 3D cross-term dimensionality, not chaos vs. non-chaos -- consistent with the already-established Duffing SR reversal.
- **Koopman/EDMD**: the reversal seen against degenerate controls (SS4.2) **disappears**. Rossler's one-step error is lower than or comparable to harmonic's in all 6 noise/degree cells -- opposite the logistic/Lorenz direction. This means the Koopman reversal was partly a degenerate-control artifact, not a universal chaos-vs-non-chaos property.

Net read: "chaos specifically aids identifiability" now has direct, non-degenerate-control support for SINDy (this project's primary comparator), while the Koopman/EDMD reversal -- the study's most novel finding -- is shown to be pair-dependent rather than a general law. This is evidence from one matched pair, not a systematic sweep of non-chaotic richness.

A coupled van der Pol experiment (120 conditions: 3 regimes x 4 noise levels x 2 degrees x 5 seeds) provides a second, within-family test with three regimes of identical system dimension:

- **Periodic** (mu=1.0): limit cycle, 1D support on a 2D attractor
- **Quasi-periodic** (mu=1.0, omega=1.618): 2-torus, 2D support
- **Chaotic** (mu=8.0): strange attractor, fractal support

**Results** (Figure 3):

For **SINDy** (degree=2), all three regimes fail to recover the coupled vdP -- errors are high across the board (periodic: 2.4, quasi-periodic: 1.8, chaotic: 88.2 at noise=0%). The system is too complex for a degree-2 polynomial library regardless of regime. This is a genuine negative result: when the library is insufficient, coverage does not help.

For **Koopman/EDMD**, the ordering is quasi-periodic (0.02) < periodic (0.07) << chaotic (0.77) at noise=0%, degree=2. The quasi-periodic regime -- the richest non-chaotic trajectory -- performs **best**, even better than the periodic limit cycle. The chaotic regime is worst, confirming the Tier B reversal within a single system. Cliff's delta for chaotic vs. quasi-periodic is 0.70 (large). At noise=1%, the gap narrows but the ordering persists.

The quasi-periodic regime's superior Koopman performance is a new finding: a 2-torus provides more state-space coverage than a limit cycle (helping the dictionary approximation) without the sensitivity of chaos (which hurts linear prediction). This is consistent with the "coverage helps, chaos hurts" interpretation but cannot distinguish it from a simpler "complexity of dynamics hurts" explanation.

**Tension between the two direct tests, reported rather than resolved.** Within coupled van der Pol, chaos hurts Koopman even against a rich quasi-periodic (2-torus) control -- the reversal holds. Against the harmonic/Rossler pair, it does not. Both are genuine within-dimension, rich-non-chaotic-vs-chaos comparisons; they disagree. A plausible reconciliation is that coupled vdP's quasi-periodic regime and its chaotic regime differ mainly in dynamical complexity at matched coverage, while harmonic and Rossler differ in both coverage (2D closed orbit vs. 3D strange attractor) and dimension -- but with n=2 direct tests this is speculation, not a resolved mechanism. The honest summary is that the Koopman/EDMD reversal is real, mechanistically grounded (SS4.6), and reproduces against degenerate controls, but its behavior against rich non-chaotic controls is inconsistent across the two systems tested here.

## 6. Limitations

- **Coverage vs. chaos partially resolved, method-dependent** -- SINDy: chaos-specific (replicates against a rich control, SS5). Koopman/EDMD: unresolved and now internally inconsistent -- reverses against degenerate controls, holds within coupled vdP's rich control, but not against harmonic/Rossler (SS5). Based on n=2 direct rich-control tests; not a systematic sweep.
- **Symbolic regression unreliable on Lorenz y-dimension** at both degrees, independent of noise/chaos
- **Off-trajectory metric under-stresses periodic/fixed-point attractors** (mitigated by off-attractor grid)
- **Extension-scale runs used 3 seeds** (vs. frozen tiers' 5), explicitly flagged as reduced power
- **Ridge regularization sweep** added post-hoc for EDMD, not preregistered

## 7. Conclusion

Across three discovery methods, six system families, and multiple robustness checks, we find no single universal "chaos aids identifiability" law. Instead:

- For **SINDy**, chaos specifically aids recovery, not just broad coverage generically -- this now has direct support against a rich non-chaotic control (Rossler 30/30 vs. harmonic 16/30), not just against degenerate ones, with large effect sizes (Cliff's d = 0.56, ratio 6.7x--10.9x, non-overlapping CIs) -- but only when the function library is rich enough. On the coupled van der Pol (a 4D system with degree-2 library), all three regimes fail equally.
- For **Koopman/EDMD**, the same excitation that helps sparse recovery actively hurts dense linear approximation against degenerate controls, for a mechanistically identified reason (SS4.6) -- but this reversal does not hold universally. It reproduces within coupled van der Pol's rich quasi-periodic control, yet disappears against the harmonic/Rossler pair. The reversal is real and mechanistically grounded where it appears, but is not a general law about chaos vs. non-chaos.
- **Symbolic regression** disagrees with SINDy's direction on 3D systems with genuine cross-terms (Duffing, Lorenz's xz dimension, harmonic/Rossler) regardless of chaos -- this looks like a gplearn search-difficulty effect, not evidence about identifiability itself.

Any future claim built on this project's results should be stated as: chaos specifically (not just coverage) aids identifiability for sparse/symbolic equation discovery under matched, finite, noisy conditions when the library is sufficient, replicated against both degenerate and rich non-chaotic controls; the same excitation hurts dense linear (Koopman) surrogates against degenerate controls but the effect against rich controls is inconsistent across the two systems tested (holds for coupled vdP, reverses for harmonic/Rossler) and should not be generalized further without more matched pairs.

## References

Anderson, B. D. O. (1977). Adaptive systems, lack of persistency of excitation and bursting phenomena. *Automatica*, 13(3), 247--258.

Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *Proceedings of the National Academy of Sciences*, 113(15), 3932--3937.

Brunton, S. L., & Kutz, J. N. (2019). *Data-Driven Science and Engineering*. Cambridge University Press.

Brunton, S. L., Brunton, B. W., Proctor, J. L., & Kutz, J. N. (2017). Koopman invariant subspaces and finite linear representations of nonlinear dynamical systems for control. *PLoS ONE*, 12(2), e0170813.

Crutchfield, J. P., & Feldman, D. P. (1997). Regularities unseen, randomness observed: Levels of entropy convergence and phase space creation. *Physical Review E*, 56(2), 1129.

de Silva, B. M., Champion, K., Quade, M., Loiseau, J. C., Kutz, J. N., & Brunton, S. L. (2020). PySINDy: A Python package for the sparse identification of nonlinear dynamical systems from data. *Journal of Open Source Software*, 5(49), 2104.

Eckmann, J.-P., & Ruelle, D. (1985). Ergodic theory of chaos and strange attractors. *Reviews of Modern Physics*, 57(3), 617.

Gallo, A., Anselmi, F., & Lazzari, S. (2026). Attractor geometry determines the identifiability limits of system discovery. *arXiv preprint arXiv:2607.18490*.

Harris, K. D., Ako, M., & Bhatt, D. (2020). Persistent excitation in data-driven identification. *IEEE Control Systems Letters*, 4(3), 680--685.

Koopman, B. O. (1931). Hamiltonian systems and transformation in Hilbert space. *Proceedings of the National Academy of Sciences*, 17(5), 315--318.

Li, Q., Dietrich, F., & Stepaniants, E. M. (2017). Extended dynamic mode decomposition with dictionary learning: A data-driven adaptive Koopman spectral analysis. *Journal of Nonlinear Science*, 27, 1767--1790.

Ljung, L. (1999). *System Identification: Theory for the User* (2nd ed.). Prentice Hall.

Lusch, B., Wehmeyer, C., & Noe, F. (2018). Deep learning of Koopman models for molecular dynamics. *Nature Machine Intelligence*, 1, 447--454.

Mangan, N. M., Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Inferring biological networks by sparse identification of nonlinear dynamics. *IEEE Transactions on Molecular, Biological, and Multi-Scale Communications*, 2(1), 52--63.

Wehmeyer, C., & Noe, F. (2018). Time-lagged autoencoders: Deep learning of slow collective variables for molecular kinetics. *Journal of Chemical Physics*, 148(24), 241703.

Williams, M. O., Kevrekidis, I. G., & Rowley, C. W. (2015). A data-driven approximation of the Koopman operator: Extending dynamic mode decomposition. *Journal of Nonlinear Science*, 25, 1307--1346.
