# Executive Summary

## Question

Can chaotic dynamics improve the recoverability of governing equations from trajectory data, and if so, for which methods and under what conditions?

## Motivation

Gallo, Anselmi, and Lazzari (arXiv:2607.18490, Jul 2026) proved theoretically that chaotic trajectories, via persistent phase-space excitation, can make sparse equation recovery *easier* than periodic or fixed-point trajectories. Whether this holds empirically under finite noisy samples, restricted function libraries, and discovery methods beyond sparse regression was untested.

## Design

- **Systems:** 5 families (logistic map, Lorenz, harmonic oscillator, Duffing, Rössler) + Lorenz-96 (N=6) as an extension, spanning maps and flows, conservative and dissipative dynamics.
- **Discovery methods:** 3 families — SINDy (sparse regression), symbolic regression (gplearn, independently cross-checked with PySR), Koopman/EDMD (dense linear surrogate, self-written + pykoopman cross-check + neural-network Koopman variant).
- **Regimes:** Matched chaotic/non-chaotic pairs per family, with 12 regimes in the Tier A SINDy grid and 7 matched pairs in the Tier B cross-method comparison.
- **Conditions:** Noise {0%, 0.1%, 1%, 5%}, library degrees {2, 3}, 5 seeds per condition (3 in extensions), 660+ total fits in frozen tiers alone.
- **Confirmation:** Held-out Rössler system (untouched until main study concluded), held-out Lorenz rho=45.0 parameter regime.
- **All local CPU, no cloud, no paid APIs.**

## Key Findings

### 1. Chaos aids identifiability for SINDy — clean, universal, replicated out-of-sample

Across every system, regime, noise level, and library degree tested, SINDy recovers true coefficients from chaotic trajectories and fails to recover from degenerate non-chaotic controls (fixed points, short cycles). At grid scale (Tier A: 132 conditions), every non-chaotic control is 0/5 recovered in all 8 of its noise × degree cells (0/60 combined); every chaotic regime starts at 5/5 and degrades gracefully, never failing outright. Replicated on the genuinely untouched Rössler confirmation family and at the held-out Lorenz rho=45.0.

### 2. Symbolic regression replicates for logistic/Lorenz but reverses for Duffing

Under the corrected joint criterion (degree-constrained AND dynamically non-vacuous), gplearn replicates chaos-aids for logistic and Lorenz (0/30 recovery for both controls). But for the Duffing pair, forced-chaotic is 0/30 while its non-chaotic control is 18/30 — the chaotic regime is *worse* than its control. An independent PySR cross-check confirms the reversal's direction is not gplearn-specific (PySR: 0/9 forced vs. 3/9 unforced at noise=0%).

### 3. Koopman/EDMD reverses the headline pattern — the most important finding

One-step prediction error is **6–11x higher** in chaotic regimes than their matched non-chaotic controls, in every tested pair at every noise level. This reversal survives four independent robustness checks:
- Lyapunov-rate normalization (eliminates trajectory-divergence confounding)
- pykoopman cross-implementation (6-figure agreement on Lorenz chaotic, rules out implementation bug)
- Neural-network Koopman (same reversal at comparable-or-larger magnitude, rules out fixed-dictionary explanation)
- Gram-matrix mechanistic analysis (Gallo et al.'s λ_min predictor carries the **opposite sign** for EDMD: more excitation → worse approximation)

**Qualitative account:** broad phase-space coverage helps constrain a sparse coefficient vector (SINDy/SR) but hurts uniform approximation by a dense, dictionary-truncated linear surrogate (EDMD).

### 4. Partial observation via delay embedding erases the gap entirely

Under single-coordinate delay-embedded observation, the primary one-step metric detects **no** chaos-vs-non-chaos gap across all 210 conditions (7 pairs × 3 noise levels × 2 degrees × 5 seeds), up to 5% noise. The observation operator, not the dynamics, determines identifiability.

### 5. The "rich non-chaotic control" problem

The independent adversarial review found that all matched-pair contrasts pit chaos against maximally *degenerate* controls (fixed points, 2-cycles). The one rich non-chaotic regime tested (harmonic oscillator) recovers as cleanly as chaotic regimes. The project's own data better supports "broad state-space coverage aids identifiability" than "chaos specifically."

## Calibrated Conclusion

Chaos aids identifiability for sparse/symbolic equation discovery under matched finite noisy conditions, replicated out-of-sample. The same excitation that helps sparse recovery actively hurts dense linear (Koopman) surrogates, for a mechanistically identified reason. Whether the sparse-recovery benefit is chaos-specific or a broader excitation effect remains an open question. No single universal "chaos aids identifiability" law exists across methods.

## Evidence Level

**Candidate contribution** (PROMPT.md evidence ladder level 3): preregistered result crosses thresholds, survives untouched confirmation (Rössler) and held-out parameter regime (Lorenz rho=45.0), uncertainty analysis (Bonferroni-adjusted CIs), sensitivity analysis (noise sweep, degree sweep, trajectory-length sub-sweep), and independent adversarial review. Not yet independently rerun by a separate lab (level 4 not reached).

## Artifacts

All code, results, and analysis in this repository. `reports/TECHNICAL_REPORT.md` for the full technical synthesis. `PAPER.md` / `PAPER_DRAFT.md` for the paper-quality write-up.
