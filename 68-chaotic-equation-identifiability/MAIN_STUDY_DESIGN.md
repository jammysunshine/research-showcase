# Main Study Design — Frozen Comparator/Regime Matrix

Frozen before any main-study run is examined, per PROMPT.md claim discipline. Operationalizes PREREGISTRATION.md §1–§4 into a concrete, locally-computable run list. Any deviation after results are seen must be logged in DECISION_LOG.md as post-hoc.

## 1. Why a subset, not the full cross-product

PREREGISTRATION.md §1's system/regime table, crossed with §3's 3 observation operators, §4's 3 sample lengths, >=5 seeds, and 3 discovery methods, is combinatorially large (12 core regimes x ~4 noise levels x 3 observation modes x 3 lengths x 5 seeds x 3 methods = tens of thousands of fits). PROJECT_CHARTER.md's compute ceiling is local CPU only (16GB Mac, no cloud). The pilot showed symbolic regression (gplearn, population=3000/generations=25, up to 5 degree-retries, 3 dimensions) is the bottleneck — order-of-minutes per (regime, noise, seed) fit — while SINDy and Koopman/EDMD are each order-of-seconds. Running the full cross-product through symbolic regression alone is infeasible within this project's local-compute budget.

This is a scope-limiting design decision, made before any main-study result is examined, and is explicitly logged (see DECISION_LOG.md entry "Main-study matrix scoping"). It is not a post-hoc adjustment. PREREGISTRATION.md §1 itself anticipates this: "expand toward 50+ regimes only after pilot passes" — the design below is the first frozen expansion past the pilot, not the final ceiling.

## 2. Three-tier design

**Tier A — Full grid, SINDy only (cheap, trusted primary comparator per §7).**
All 12 core regimes x full-state noise levels {0%, 0.1%, 1%, 5%} x medium length (N=5,000) x 5 seeds x both degree-2 and degree-3 libraries (per LIMITATIONS.md #2 precedent). This is the backbone chaos-vs-non-chaos comparison and the length/noise sensitivity sweep (short N=500 and long N=50,000 also run, SINDy only, at 0% and 1% noise, to characterize sample-size sensitivity without the combinatorial explosion of running every method at every length).

Regimes (12): logistic {r=3.2, 3.5, 3.83, 4.0}; Lorenz {rho=14, 22, 24.5, 28, 100}; harmonic oscillator (1 conservative regime); Duffing {unforced, forced-chaotic}.

**Tier B — Representative subset, all three methods (SINDy, symbolic regression, Koopman/EDMD).**
One matched chaotic/non-chaotic (or conservative-control) pair per system family, chosen as the clearest contrast from Tier A once it completes, at medium length, full-state noise levels {0%, 1%, 5%} (drop 0.1% — Tier A shows 0% and 1% already bracket the interesting behavior), 5 seeds, both degree-2 and degree-3 libraries:
- Logistic: r=3.2 (period-2 control) vs r=4.0 (chaotic) — already run in the pilot; re-run here under the frozen main-study harness rather than reusing pilot numbers, per PREREGISTRATION.md §11's spirit of not reusing pilot-development data for confirmatory claims.
- Lorenz: rho=14 (fixed-point control) vs rho=28 (classic chaotic) — same rationale.
- Harmonic oscillator: the single conservative regime (non-identifiable control, first-integral case).
- Duffing: unforced (conservative) vs forced-chaotic.

8 regimes x 3 noise levels x 5 seeds x 2 degrees x 3 methods = 720 fits. At symbolic regression's observed per-fit cost this is the dominant cost of the whole study; budgeted as the main study's single largest compute item.

**Pinned hyperparameters for Tier B symbolic regression (resolves LIMITATIONS.md #4's "no pre-registered policy" for the main study): population=3000, generations=25, max_degree_retries=5 — the smoke test's values, not `fit_symbolic_regression()`'s undocumented defaults (population=2000/generations=20).** Measured cost check: a single degree/dimension fit at these settings takes ~28s (worst case ~150s across all 5 retries); a 3-dimension Lorenz-scale fit ≈200-250s; Tier B's 240 SR fits (8 regimes x 3 noise x 5 seeds x 2 degrees) totals roughly 8-15 CPU-hours — feasible on a 16GB Mac within this project's multi-week timeframe. Logged in DECISION_LOG.md alongside this document's freeze entry, per LIMITATIONS.md #4.

**Tier C — Partial-observation stress test (§3's designed non-identifiability case).**
Single-coordinate observation (x(t) only, delay-embedded) for the same 8 Tier-B regime pairs, medium length, noise levels {0%, 1%}, 5 seeds, SINDy only (delay-embedding reconstruction + all 3 discovery methods is out of scope for this first pass; symbolic regression / Koopman on delay-embedded partial observations is flagged as future work, not silently dropped — see §5 below).

## 3. Held-out confirmation family

Rössler system (dx/dt = -y-z, dy/dt = x+a*y, dz/dt = b+z(x-c); classic chaotic parameters a=0.2, b=0.2, c=5.7, plus one non-chaotic control regime at reduced c). Selected over a coordinate-transformed Lorenz because it is a genuinely distinct vector field, not a relabeling of an already-used system, avoiding the coordinate-representation-ambiguity issue that sank `src/counterexamples_conjugacy.py` (COUNTEREXAMPLES.md). Implementation (simulator + Lyapunov check) is written and tested now (§4) but its trajectories are not inspected, plotted, fit, or discussed until the falsification/independent-replication gate, per PREREGISTRATION.md §11. Any accidental early inspection is logged in DECISION_LOG.md and the regime is replaced with a re-parameterized variant.

## 4. Immediate next actions (this design's execution plan)

1. **CLOSED 2026-08-16** (see DECISION_LOG.md same date). `src/simulators.py` gained Rössler (`rossler_rhs`/`rossler_jacobian`/`rossler_trajectory`/`rossler_lyapunov_spectrum`), harmonic oscillator (`harmonic_rhs`/`harmonic_jacobian`/`harmonic_trajectory`), and Duffing (`duffing_unforced_*` 2-state conservative, `duffing_forced_*` 3-state autonomous-embedded chaotic) — the three system families Tier A/B name but the repo did not yet implement. Tested (`tests/test_simulators.py`, 7 new tests, 28/28 total passing): Rössler chaotic largest LE ≈0.10 (literature ≈0.07) vs. negative for the c=3.0 non-chaotic control; harmonic/Duffing-unforced conserve energy to ~1e-11; Duffing forced-chaotic shows a clean positive/zero/negative spectrum. Rössler trajectories are not fit, plotted, or otherwise inspected for discovery purposes — only the literature-comparison Lyapunov numeric check above, which does not violate §11's blinding (documented in the test file and in DECISION_LOG.md).
2. **Close both §7 gaps before any Tier A run is executed** (not deferred to "before results are confirmatory" — that phrasing let a follower reach confirmation without ever closing them, corrected here):
   a. **Module sub-part CLOSED 2026-08-16** (see DECISION_LOG.md same date). `src/metrics_dynamical.py` implements `lyapunov_spectrum_error`, `invariant_measure_tv_distance`, `bifurcation_location_error`, reusing `src/simulators.py`'s Benettin-QR machinery (refactored into `generic_map_lyapunov_exponent`/`generic_ode_lyapunov_spectrum`), tested (12/12 passing) and independently verified. **Wiring sub-part still OPEN**: "wire it into every Tier A/B/C run's output record" cannot happen until `experiments/main_study.py` (step 4) exists — treat that wiring as part of step 4's completion criteria, not yet done. Known scope limit (not previously disclosed): `bifurcation_location_error` only supports 1D discrete-map families (logistic); no ODE/continuous-parameter analogue exists yet for Lorenz/Duffing bifurcation structure — acceptable for now since Tier A/B's bifurcation-relevant regimes are the logistic family, but must be revisited if a Tier gains an ODE bifurcation requirement.
   b. **Module sub-part CLOSED 2026-08-16** (see DECISION_LOG.md same date). `src/discovery_naive_baseline.py`'s `fit_naive_polynomial` reuses `ps.PolynomialLibrary` and does ordinary dense least squares (no STLSQ), tested (4/4 passing) including the required "denser than SINDy" demonstration (31/60 vs 6/60 nonzero coefficients on the same noised Lorenz data). **Wiring sub-part still OPEN**: not yet called from any Tier A/B/C run because `experiments/main_study.py` does not exist; wiring is part of step 4's completion criteria.
3. **Module sub-part CLOSED 2026-08-16** (see DECISION_LOG.md "Off-attractor evaluation grid"; wiring sub-part still OPEN, folded into step 4). Chose option (i) from the two options this step originally posed: `experiments/pilot_chaos_vs_periodic.py` now computes an explicit off-attractor evaluation-grid VF-error metric (500 points from a system-appropriate wider-than-attractor domain: trajectory bounding box for Lorenz, analytic domain `[0,1]` for the logistic map) alongside the confirmation-trajectory metric. Confirmed on a real run: fixes the flagged logistic-period-2 disagreement (0/5 grid-VF-recovered now correctly matches 0/5 coefficient recovery, vs. the old metric's misleading 5/5). Residual, disclosed limitation: VF-error metrics (confirmation and grid alike) can still read "recovered" in a few chaotic-regime cells where exact coefficient recovery has degraded (logistic degree=3/noise=5%, Lorenz degree=3/noise=1%) — consistent with VF error being more forgiving than exact-match, not a defect in the grid construction; the grid's value-add is specific to periodic/fixed-point regimes. Wiring this same grid-metric logic into `experiments/main_study.py`'s per-regime output record (§4 step 4) remains open, same status as steps 2a/2b's wiring.
4. Implement `experiments/main_study.py`: parameterized runner covering Tiers A/B/C, writing results to `experiments/main_study_results/` (one JSON per tier, mirroring the pilot's format) plus a run manifest (system, regime, noise, length, seed, method, degree, wall-clock time) for reproducibility and future compute-budget accounting. Steps 2a/2b/3 above must be wired in before this step is considered complete.
5. Run Tier A first (cheapest, backbone result) end-to-end, verify, commit.
6. Run Tier B (dominant cost) with the discovery-method fits chunked/checkpointed so a partial run is resumable rather than needing to restart from zero if interrupted.
7. Run Tier C.
8. Only after Tiers A-C are complete and frozen: unseal and run the Rössler confirmation family (§3), exactly once, per PREREGISTRATION.md §11.

## 5. Explicitly out of scope for this pass (not silently dropped)

- Symbolic regression / Koopman on partial (delay-embedded) observations — Tier C is SINDy-only; flagged for a future extension, not claimed as covered.
- The full 50+-regime expansion PREREGISTRATION.md §1 gestures at post-pilot — Tiers A-C are the first frozen expansion, not that full expansion.
- RBF-dictionary Koopman/EDMD (§2's "secondary configuration") — monomial dictionary only in this pass.
- Naive (non-sparse) least-squares baseline (§7) — module implemented 2026-08-16 (`src/discovery_naive_baseline.py`, tested), but not yet wired into `experiments/main_study.py` (which does not exist yet); wire in before Tier A results are treated as confirmatory, since §7 requires it as a comparator.

## 6. Compliance check against PREREGISTRATION.md

- §1: core regimes match the frozen table; held-out family selected and untouched. Full 50+-regime expansion explicitly deferred, not silently skipped (§5 above).
- §2: both degree-2 and degree-3 libraries run throughout, consistent with the LIMITATIONS.md #2 closure.
- §3: full-state noise-free and noisy covered in Tiers A/B; partial observation covered in Tier C (SINDy only, gap logged).
- §4: medium length is the default for Tiers B/C; short/long sensitivity covered in Tier A only (SINDy), logged as a scope choice, not silently narrowed.
- §5: train/confirmation split used throughout (per the pilot's now-fixed off-trajectory metric); confirmation family untouched until falsification gate. Known exception for periodic/point regimes tracked via §4 step 3 (LIMITATIONS.md #5).
- §6/§8: both primary (off-trajectory VF error) and secondary (coefficient recovery) metrics computed and reported for every run, per the pilot's LIMITATIONS.md #5 fix. Lyapunov/invariant-measure/bifurcation metrics from §6 are a **blocking prerequisite** (§4 step 2a) — wired in before `experiments/main_study.py` is considered complete, not deferred past Tier A execution.
- §7: sparse (SINDy/STLSQ) trusted comparator covered; naive least-squares baseline is a **blocking prerequisite** (§4 step 2b) — wired in before `experiments/main_study.py` is considered complete, not deferred past Tier A execution.
- §9: 5 seeds throughout, matching §4's floor; mean +/- std reporting and the frozen main study's Bonferroni-style correction across the fixed Tier A/B/C hypothesis set to be applied at synthesis time, not per-run.
- §11: held-out family untouched, enforced by this document itself gating step 6 above behind steps 1-5 completing.

## 7. Two known gaps this design does not yet close — both are execution-plan blockers, not optional polish

**Updated 2026-08-16 (see DECISION_LOG.md same date for the full reconciliation):**

1. §6's Lyapunov-spectrum / invariant-measure / bifurcation-structure secondary metrics: **module CLOSED** — implemented in `src/metrics_dynamical.py`, reusable, tested (12/12), independently verified by an adversarial verifier and re-verified in this reconciliation pass. **Wiring into every Tier A/B/C run's output record remains OPEN** — cannot happen before `experiments/main_study.py` (§4 step 4) exists; treat as part of step 4's completion criteria, not a standalone prerequisite that blocks step 4 from starting.
2. §7's naive (non-sparse) least-squares baseline: **module CLOSED** — implemented in `src/discovery_naive_baseline.py`, tested (4/4), including the required denser-than-SINDy demonstration. **Wiring into every Tier A/B/C run remains OPEN**, same reasoning as above.
3. §4 step 3 (periodic/point-attractor off-trajectory-metric decision, LIMITATIONS.md #5) — **module sub-part CLOSED 2026-08-16**, wiring sub-part open (see step 3 above and DECISION_LOG.md).

Revised gate: the two metrics/baseline **modules**, plus the off-attractor-grid metric decision (step 3), now exist and are trustworthy, so `experiments/main_study.py` (§4 step 4) may begin being built with them as dependencies. Tier A execution (§4 step 5) still may not proceed until step 4's implementation actually wires all three (Lyapunov/invariant-measure/bifurcation metrics, naive baseline, off-attractor grid) into every run's output record. That wiring is not done yet.

## 8. Adversarial critic pass (2026-08-16)

An independent critic agent reviewed this document before freeze. Findings incorporated: (a) Tier B's symbolic-regression hyperparameters are now pinned explicitly (§2, population=3000/generations=25) with a measured cost estimate (~28s/fit, ~8-15 CPU-hours for Tier B), resolving LIMITATIONS.md #4's "no pre-registered policy" gap for the main study; (b) §4's execution plan was reordered so the §7 blockers (Lyapunov/invariant-measure/bifurcation metrics, naive baseline) and LIMITATIONS.md #5's periodic-attractor metric gap are closed *before* Tier A runs, not left as prose a reader could skip; (c) confirmed via `grep -rn "rossler\|Rössler\|Rossler" src/ experiments/ scripts/ tests/` — zero hits, the held-out family is genuinely untouched; (d) the Rössler Lyapunov sanity-check test (§4 step 1) is clarified as a literature-comparison numeric check, not an inspection of discovery-relevant behavior, so it does not violate §11. See DECISION_LOG.md for the freeze entry.
