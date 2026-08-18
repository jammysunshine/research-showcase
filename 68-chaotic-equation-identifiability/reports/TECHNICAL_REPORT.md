# Chaotic Equation Identifiability: A Technical Report

Preprint-style write-up, target audience: dynamical-systems / SciML researchers working on sparse equation discovery and system identification (PROJECT_CHARTER.md). This document synthesizes the project; full detail, raw numbers, and machine-readable evidence live in RESULTS.md, CLAIM_LEDGER.md, EVIDENCE_INDEX.md, DECISION_LOG.md, and LIMITATIONS.md, all cited by section below.

## 1. Question

When can the governing equations of a nonlinear dynamical system be uniquely recovered from observed trajectories, and does chaos specifically — as opposed to any sufficiently rich, state-space-covering trajectory — help or hurt that recovery? (PROJECT_CHARTER.md §Primary question.)

Prior work (arXiv:2511.08860) proves discoverability in continuous/analytic function space from a single noise-free trajectory but does not address the finite-sample, noisy, restricted-library, or partial-observation regime this project targets, and no existing work empirically benchmarks the chaos-helps-identifiability claim across discovery methods under matched noise/sampling (PRIOR_ART.md).

## 2. Method

Matched-pair design: for each system family, a chaotic (or otherwise rich) regime is contrasted against a non-chaotic control at identical noise levels, library degree, and seeds, using SINDy/STLSQ as the primary discovery method (trusted comparator, PROJECT_CHARTER.md), with symbolic regression (gplearn) and Koopman/EDMD as secondary methods. Identifiability is operationalized via: (1) exact coefficient recovery within a frozen 5% relative-error tolerance; (2) off-trajectory vector-field L2 error on an independent confirmation trajectory and an off-attractor evaluation grid, frozen at 10%; (3) Lyapunov-spectrum error and invariant-measure TV distance as secondary diagnostics (PREREGISTRATION.md §6-§8). All thresholds and hyperparameters (STLSQ_THRESHOLD=0.1, COEFF_TOL=0.05, VF_ERR_TOL=0.10) were frozen before the main study and never moved post-hoc across any tier — independently verified in the adversarial review (§5 below).

Four systems were developed pre-registration (logistic map, Lorenz, undamped harmonic oscillator, Duffing oscillator); Rössler was held out untouched until the confirmation stage (PREREGISTRATION.md §1, §11).

## 3. Results

**Tier A (SINDy, full-state, 132 conditions).** Every non-chaotic, degenerate control (fixed point, short periodic cycle) is 0/5 coefficient-recovered in every noise×degree cell; every chaotic/past-onset regime starts at 5/5 and degrades gracefully at higher noise/library degree. Two disclosed anomalies (a pure library-degree flip for one logistic regime; total failure for Duffing forced-chaotic, structurally inapplicable for coefficient recovery — see correction in §5) do not change the core pattern. (RESULTS.md "Tier A".)

**Tier B (3 discovery methods, 7 matched pairs, 630 fits).** SINDy and corrected symbolic regression replicate Tier A's pattern (with one Duffing-pair reversal under symbolic regression). **Koopman/EDMD one-step prediction error reverses the pattern outright** — the chaotic regime is 6-11x *worse* than its matched non-chaotic control at 1-5% noise, in both regime pairs tested. This is the project's single most important qualifying finding: "chaos aids identifiability" is not a method-agnostic law. (RESULTS.md "Tier B".)

**Tier C (partial observation via delay embedding, SINDy only, 140 conditions).** Under single-coordinate, delay-embedded observation, the primary one-step-from-ground-truth metric detects **no identifiability gap at all** (0/140 jobs flagged dynamically distinct) at noise {0%, 1%}. A secondary rollout diagnostic shows a large gap but is confounded by chaotic Lyapunov-time sensitivity and is explicitly not used as evidence. A third, genuinely distinct outcome alongside Tier B's "holds"/"reverses" split. (RESULTS.md "Tier C".)

**Held-out Rössler confirmation (SINDy, full-state, 16 conditions, run exactly once).** The Tier A pattern replicates on a system never touched during method development: at noise=5%, chaotic coefficient recovery is 3/5 (degree=2) / 2/5 (degree=3) vs. 0/5 for the matched non-chaotic control at both degrees. First out-of-sample generalization evidence in the project. (RESULTS.md "Held-out Rössler confirmation".)

**Held-out Lorenz parameter regime (rho=45.0, run exactly once).** A never-tested Lorenz value shows coefficient-recovery degradation of the same order as the preregistered chaotic regime (rho=28) — no cliff or new failure mode. Discharges PREREGISTRATION.md §1's "held-out parameter regime per developed family" clause for the project's central backbone system. (RESULTS.md "Held-out parameter regime, Lorenz family".)

## 4. Adversarial counterexamples

A conserved-quantity gauge-freedom construction for the undamped harmonic oscillator is a **confirmed** counterexample: an alternative vector field matches the true dynamics to machine precision on the observed level set while diverging 425.7% off it, well within the frozen degree-3 polynomial library (COUNTEREXAMPLES.md, CLAIM_LEDGER.md). A tent-map/logistic-map coordinate-conjugacy construction was attempted, found invalid (the only observationally-equivalent alternative is algebraically the true map itself), and preserved as a documented failed attempt rather than deleted, per PROMPT.md's claim-discipline requirement to preserve negative findings.

## 5. Independent adversarial review

A dedicated, read-only review pass (PROMPT.md's completion-gate "independent review" item) independently recomputed every headline number from raw result JSON and verified confound-mitigation code directly rather than trusting documentation. Verdict: **"the frozen main study + confirmation family substantively satisfies the independent-review gate."** No threshold-gaming, cherry-picking, or cross-tier inconsistency was found. One finding materially narrows how the headline should be stated and is incorporated into this report and LIMITATIONS.md #7:

> The matched-pair contrasts carrying the "chaos aids identifiability" claim all pit chaos against *maximally degenerate* non-chaotic controls (fixed points, short cycles — near rank-deficient by construction), never against the one rich-but-non-chaotic regime in the study, the harmonic oscillator — which recovers coefficients as cleanly as the chaotic regimes (5/5 at degree=2, every noise level) and was never given a chaotic partner. A **persistent-excitation / state-space-coverage** mechanism is at least as well supported by this project's own data as a chaos-specific one.

Two further open items, disclosed rather than resolved: PREREGISTRATION.md §9's promised multiplicity correction across the study's hypothesis-test set was never actually applied (LIMITATIONS.md #6); the Rössler run's blinding-preserving smoke test left no committed artifact, so that specific claim rests on contemporaneous self-report rather than an independently checkable trail (same section).

## 6. Headline claim, precisely stated

**Not:** "chaos aids identifiability."

**Better supported by the evidence actually collected:** trajectories that persistently excite a broad region of state space — whether via chaotic ergodicity or a rich non-chaotic orbit — support substantially better equation recovery than degenerate trajectories (fixed points, short periodic cycles), *under SINDy and (mostly) symbolic regression*. This effect is **method-dependent**: it reverses under Koopman/EDMD one-step prediction error, and is **undetectable** on the primary metric under delay-embedded partial observation at the noise levels tested. It generalizes to a genuinely held-out system family (Rössler) and a held-out parameter regime (Lorenz rho=45.0) under the conditions where it was originally observed (SINDy, full-state).

## 7. Evidence level (PROMPT.md's ladder)

**Candidate contribution.** The finding is numerically supported across multiple systems, multiple discovery methods (with an honestly-reported reversal), a held-out family, and a held-out regime, and has survived one genuinely independent adversarial review. It falls short of **confirmed contribution** because: (a) the "chaos-specific" framing versus the "persistent-excitation" framing has not been directly tested by a matched chaotic-vs-rich-non-chaotic pair (§5); (b) the promised multiplicity correction across the fixed hypothesis-test set was not executed; (c) independent review to date is a single review pass, not a second independent replication by a separate method/team. These are exactly the gaps a reader should weigh before citing "chaos aids identifiability" as an established result from this work.

## 8. Reproducibility

All results are produced by versioned scripts against pinned dependencies and fixed seeds (REPRODUCIBILITY_MANIFEST.md). Every headline number in this report is backed by a committed JSON result file and cited in EVIDENCE_INDEX.md. `PYTHONPATH=. .venv/bin/python -m pytest tests/ -q` reproduces the full test suite (28/28 passing at time of writing, independently re-verified during the adversarial review).

## 9. Fallback contribution, delivered regardless of headline framing

Independent of how the identifiability question above is ultimately read, this project also delivers PROJECT_CHARTER.md's fallback contribution: a documented, reproducible adversarial counterexample (§4), a benchmark harness spanning 5 system families × 3 discovery methods × multiple observation regimes (`experiments/`, `src/`), and a fully-logged decision trail (DECISION_LOG.md) that others can extend or contest.
