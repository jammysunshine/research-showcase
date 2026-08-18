# Results

## Pilot: chaos vs. non-chaotic regime, matched system (exploratory, not frozen main study)

`experiments/pilot_chaos_vs_periodic.py`, raw output in `experiments/pilot_chaos_vs_periodic_results.json`. 5 seeds per condition, SINDy (degree-2 polynomial library, STLSQ threshold 0.1), 5% relative-coefficient recovery tolerance per `PREREGISTRATION.md` SS6. Noise regimes per `PREREGISTRATION.md` SS4: additive Gaussian noise at 0%, 0.1%, 1%, 5% of state standard deviation.

**Lorenz non-chaotic control fix:** the original pilot compared rho=14 vs rho=28 using the whole trajectory, including the pre-convergence transient — a regime-selection confound (see `DECISION_LOG.md`, "Pilot revealed regime-selection confound"). Fixed by discarding the first 50% of every simulated trajectory (both regimes, for symmetry) and fitting SINDy only on the post-transient tail. For rho=14 the tail sits on the stable fixed point (genuinely non-exciting); for rho=28 the tail is still on the chaotic attractor. Noise is added to the full pre-discard trajectory (so the noise scale is tied to the system's natural amplitude and matched between conditions), then the transient is dropped.

| System | Regime | Noise (% of state std) | Recovered / seeds |
|---|---|---|---|
| Logistic map | r=3.2 (period-2) | 0 / 0.1 / 1 / 5 | 0/5 at every noise level |
| Logistic map | r=4.0 (chaotic) | 0 / 0.1 / 1 / 5 | 5/5 at every noise level |
| Lorenz | rho=14 (stable fixed point, post-transient tail) | 0 / 0.1 / 1 / 5 | 0/5 at every noise level |
| Lorenz | rho=28 (classic chaotic, post-transient tail) | 0 / 0.1 / 1 / 5 | 5/5 at every noise level |

### Logistic map: clean matched-system identifiability gap, robust to noise
At r=3.2, the trajectory converges to a period-2 orbit visiting only 2 distinct x-values. The regression design matrix (columns [1, x, x^2]) collapses to rank <=2 on 2 unique support points, so the 3-coefficient quadratic model is underdetermined — least-squares returns a minimum-norm solution that does not match the true map (x_{n+1}=r x - r x^2) in every seed, at every noise level tested (max relative error ~1.05-1.4, far above the 5% tolerance). At r=4.0 (chaotic, ergodic on [0,1]), the orbit visits a continuum of x-values and the same regression recovers the true coefficients to <5% error every time; error grows monotonically with noise (~0.01% at 0% noise, ~2% at 5% noise) but stays inside tolerance at all four levels tested.

This is a mechanistic, not merely correlational, confirmation of the persistent-excitation account of chaos-aided identifiability noted in `PRIOR_ART.md` SS5 (Koopman/PE literature): the periodic orbit fails to persistently excite the candidate-function basis, independent of noise or optimizer choice.

### Lorenz: gap replicates cleanly once the transient-discard fix is applied
With both regimes fit only on their post-transient tail, rho=14's tail is (numerically) sitting on the fixed point: SINDy's STLSQ optimizer thresholds essentially all coefficients to zero (a UserWarning — "Sparsity parameter is too big... eliminated all coefficients" — fires on every rho=14 run), giving 0/5 recovery at every noise level (max relative error ~1.0-1.75, i.e. the discovered "model" is degenerate/near-zero, nowhere near the true sigma/rho/beta). rho=28's tail remains on the chaotic attractor and recovers all three coefficients to well under 5% error at every noise level (~0.03% at 0% noise, ~3% at 5% noise). This directly mirrors the logistic map's mechanism and result: matched system, matched library, matched seeds and noise, chaotic regime identifiable and non-chaotic regime not, across four noise levels each.

### Status (superseded in part — see below)
Pilot only — informs the frozen main study design per `PREREGISTRATION.md`; not a preregistered confirmatory result (single discovery method, 2 systems, not yet run against the frozen comparator set or an untouched confirmation system/trajectory). The Lorenz regime-selection confound identified after the first pilot run is fixed and the gap replicates; see `DECISION_LOG.md` for both the original confound finding and this correction.

**The above table used a degree-2 SINDy library and only the secondary (in-sample coefficient) metric — both since corrected below (LIMITATIONS.md #2, #5). Raw data preserved verbatim in `experiments/pilot_chaos_vs_periodic_results_degree2.json`; do not treat it as replaced, per PROMPT.md's "report all preregistered outcomes."**

---

## Pilot correction: degree-3 library (PREREGISTRATION.md SS2) + off-trajectory primary metric (SS8)

`experiments/pilot_chaos_vs_periodic.py` (rewritten), raw output in `experiments/pilot_chaos_vs_periodic_results.json`. Same 5 seeds, same 4 noise levels, same systems/regimes as above. Two corrections applied and reported side by side with the original, per PROMPT.md claim discipline:

- **Gap #2 (LIMITATIONS.md):** SINDy's `PolynomialLibrary` is now run at **both degree=2 (original) and degree=3 (PREREGISTRATION.md SS2's frozen value)**, exposing the higher-degree candidate terms (x^3, x^2y, xyz, etc.) that the narrower degree-2 library never gave STLSQ a chance to wrongly keep or correctly zero out.
- **Gap #5 (LIMITATIONS.md):** for every fitted model, an **independent confirmation trajectory** (different initial condition, same system/parameters/regime/noise level, drawn from a distinct RNG seed) is generated via `src/simulators.py`, and the fitted model's vector field is evaluated against the true analytic vector field (`lorenz_rhs` / `logistic_map`) at every confirmation-trajectory point. The normalized L2 vector-field error (PREREGISTRATION.md SS6) is the **primary** metric; >10% = dynamically distinct / non-recovery.

### Coefficient-recovery (secondary metric): seeds recovered / 5, by library degree

| System | Regime | Noise | degree=2 | degree=3 |
|---|---|---|---|---|
| Lorenz | rho=14 (fixed pt) | 0/0.1/1/5% | 0/0/0/0 | 0/0/0/0 |
| Lorenz | rho=28 (chaotic) | 0/0.1/1/5% | 5/5/5/5 | 5/5/**3**/**3** |
| Logistic | r=3.2 (period-2) | 0/0.1/1/5% | 0/0/0/0 | 0/0/0/0 |
| Logistic | r=4.0 (chaotic) | 0/0.1/1/5% | 5/5/5/5 | 5/5/5/**2** |

**Degree-3 finding (honest, changes the story):** the non-chaotic regimes' 0/5 result is unaffected by library degree — a rank-deficient/non-exciting regression stays degenerate regardless of how many spurious terms are offered. But the chaotic regimes' "5/5 at every noise level" headline from the degree-2 pilot **does not hold at degree=3**: Lorenz chaotic drops to 3/5 at 1% and 5% noise (2 of 5 seeds have max relative coefficient error 1.26 and 1.62 — grossly wrong, not near-misses — vs. <=0.03 for the other 3 seeds), and logistic-map chaotic drops to 2/5 at 5% noise. This confirms LIMITATIONS.md #2's prediction: the degree-2 library mechanically inflated apparent recovery cleanliness by never exercising STLSQ's ability to wrongly retain a spurious higher-degree term under noise. The **qualitative direction of the gap is unchanged** (chaotic regime coefficient-recovers at least as often as non-chaotic at every condition tested, and strictly more often at every condition except degree=3/noise=0.1%+ where it is still 3-5/5 vs 0/5) but the degree-3, higher-noise chaotic cells are no longer clean 5/5s, and this should not be reported as a uniformly clean result going forward.

### Off-trajectory vector-field error on confirmation trajectory (PRIMARY metric): seeds passing (<=10% normalized L2 error) / 5

| System | Regime | Noise | degree=2 | degree=3 |
|---|---|---|---|---|
| Lorenz | rho=14 (fixed pt) | 0/0.1/1/5% | 0/0/0/0 | 0/0/0/0 |
| Lorenz | rho=28 (chaotic) | 0/0.1/1/5% | 5/5/5/5 | 5/5/5/**3** |
| Logistic | r=3.2 (period-2) | 0/0.1/1/5% | **5/5/5/5** | **5/5/5/5** |
| Logistic | r=4.0 (chaotic) | 0/0.1/1/5% | 5/5/5/5 | 5/5/5/5 |

**Primary-metric finding (honest, changes the story for the logistic map):**

- **Lorenz:** the primary metric broadly tracks the secondary one. rho=14's degenerate fit has vector-field error = exactly 1.0 (100%) at every seed and noise level — total failure on both metrics. rho=28's confirmation VF error stays low (<3%) except at degree=3/noise=5%, where the same 2 seeds that failed coefficient recovery also exceed the 10% VF-error threshold (18-20% error), so the primary metric independently confirms the degree-3/high-noise degradation found above.
- **Logistic map — the primary metric does NOT reproduce the period-2 non-recovery gap.** At r=3.2, off-trajectory confirmation VF error is tiny at every noise level (~1e-15 noise-free, ~1.4% at 5% noise) — well under the 10% threshold — even though coefficient recovery is 0/5 at every condition. Mechanism: an independent-IC confirmation trajectory for a period-2 map converges (after the 500-step transient) onto the *same* 2-point cycle as the training trajectory, because a deterministic map's periodic attractor is basin-wide, not IC-specific. The confirmation trajectory therefore only re-tests the fitted model at the same ~2 support points it was fit on (where the underdetermined minimum-norm least-squares solution passes through exactly), rather than stress-testing extrapolation to unseen states. **This is a genuine limitation of "independent-IC, same-regime" as an off-trajectory design for point/periodic attractors**, not evidence that the identifiability gap is spurious — the mechanism (rank-deficient design matrix, non-unique coefficients) is unchanged and independently confirmed by the exact-form coefficient check, but the specific primary-metric operationalization in PREREGISTRATION.md SS5 ("independent initial condition, same system/parameters") does not stress the non-chaotic logistic regime the way it stresses the non-chaotic Lorenz regime (whose fixed point is even less exciting: the fit degenerates entirely rather than merely being underdetermined-but-locally-exact). Logged as an open item in LIMITATIONS.md.
- **Logistic chaotic at 5% noise:** primary metric still passes 5/5 even where secondary metric only passes 2/5 (degree=3) — the aggregate confirmation-trajectory VF-L2 error is a more forgiving/holistic test than requiring each of 3 individual coefficients to be within 5% relative error; both are legitimate per PREREGISTRATION.md SS8 (primary + secondary logged together, not one replacing the other) and are reported as such.

### Revised headline
The chaos-vs-non-chaos identifiability gap **replicates for Lorenz on both metrics and both library degrees**, though the chaotic regime's coefficient-recovery cleanliness is noise-and-degree sensitive (3/5, not 5/5, at degree=3/noise>=1%). For the **logistic map**, the gap is real and robust on the secondary (coefficient) metric at both degrees, but **does not show up on the primary (off-trajectory) metric as currently operationalized**, because the confirmation-trajectory design does not exercise off-support states for a periodic attractor. This is reported as an open methodological gap (LIMITATIONS.md), not resolved by asserting either metric is "the real one" — PREREGISTRATION.md SS8 requires both be logged.

### Status
Pilot only — informs the frozen main study design; not a preregistered confirmatory result. See `DECISION_LOG.md` for the degree-2→degree-3 and secondary→primary-metric corrections and their honestly-reported effects.

## Tier A: frozen main study, SINDy-only, 12-regime grid (`MAIN_STUDY_DESIGN.md` §4 step 5)

`experiments/main_study.py`, raw output `experiments/main_study_results/tier_a_results.json`/`tier_a_manifest.json`. 132/132 conditions completed, zero errors, 5 seeds each (logistic, Lorenz, harmonic, Duffing unforced/forced), degree=2 and degree=3 polynomial libraries, 4 noise levels {0, 0.1%, 1%, 5%} per `PREREGISTRATION.md` §4, plus a short/long trajectory-length sensitivity sub-sweep (logistic, Lorenz, degree=3, noise in {0, 1%}). See `DECISION_LOG.md` "Tier A first successful full completion" for the three prior failed/stalled attempts and their fixes.

**n_recovered/5 (coefficient match), by regime × noise; vfc = VF-confirmation recovered/5, vfg = VF-off-attractor-grid recovered/5:**

Logistic map:
| Regime | degree | 0% | 0.1% | 1% | 5% |
|---|---|---|---|---|---|
| period_2 (non-chaotic) | 2 | 0 | 0 | 0 | 0 |
| period_2 (non-chaotic) | 3 | 0 | 0 | 0 | 0 |
| period_4 | 2 | 5 | 5 | 5 | 0 |
| period_4 | 3 | 5 | 5 | 0 | 0 |
| period_3_window | 2 | 5 | 5 | 5 | 5 |
| period_3_window | 3 | **0** | **0** | **0** | **0** |
| chaotic | 2 | 5 | 5 | 5 | 5 |
| chaotic | 3 | 5 | 5 | 5 | 3 |

Lorenz:
| Regime | degree | 0% | 0.1% | 1% | 5% |
|---|---|---|---|---|---|
| stable_fixed_point (non-chaotic) | 2 | 0 | 0 | 0 | 0 |
| stable_fixed_point (non-chaotic) | 3 | 0 | 0 | 0 | 0 |
| pre_chaotic (non-chaotic) | 2 | 0 | 0 | 0 | 0 |
| pre_chaotic (non-chaotic) | 3 | 0 | 0 | 0 | 0 |
| near_onset (rho=24.5) | 2 | 5 | 5 | 5 | 3 |
| near_onset (rho=24.5) | 3 | 5 | 5 | 4 | 3 |
| classic_chaotic (rho=28) | 2 | 5 | 5 | 5 | 5 |
| classic_chaotic (rho=28) | 3 | 5 | 5 | 3 | 3 |
| high_rho_chaotic (rho=100) | 2 | 5 | 5 | 5 | 5 |
| high_rho_chaotic (rho=100) | 3 | 5 | 5 | 5 | 5 |

**Core finding replicates at Tier A scale:** every non-chaotic control (logistic period_2, Lorenz stable_fixed_point, Lorenz pre_chaotic) is 0/5 in every one of its 8 noise×degree cells. Every clearly-chaotic regime (logistic chaotic, Lorenz classic_chaotic/high_rho_chaotic) starts at 5/5 and degrades only at higher noise and/or degree=3, never fails outright. Lorenz near_onset (rho=24.5, past the chaos onset) behaves like the chaotic regimes, not the controls.

**Anomalies, reported honestly, not smoothed over:**
- **Logistic `period_3_window` (periodic window inside the chaotic band): ROOT-CAUSED, 2026-08-16.** 5/5 at degree=2 in every noise cell, but **0/5 at degree=3 in every noise cell, including noise=0%** — a clean, deterministic flip driven entirely by library degree, not noise. Cause: the post-transient trajectory at r=3.83 collapses onto exactly 3 unique support points (a period-3 orbit); the degree-2 design matrix (3 columns) is exactly determined by 3 points and recovers the true coefficients, but the degree-3 design matrix (4 columns) is still only rank 3 over this support — underdetermined — so `np.linalg.lstsq` returns an arbitrary minimum-norm solution unrelated to the true coefficients, even though it still interpolates the training/confirmation points exactly (hence both VF-error metrics stay 5/5 — the fit is locally accurate, just structurally wrong). **This is the same rank-deficiency/persistent-excitation mechanism already documented for the `period_2` control (LIMITATIONS.md #7), not a new failure mode** — it only shows up here because a period-3 orbit's 3 support points cross the identifiability threshold between degree=2 (3 params, exactly determined) and degree=3 (4 params, underdetermined). Full derivation: DECISION_LOG.md "Tier A anomaly root-caused: logistic period_3_window degree-3 flip".
- **Logistic `period_4`:** degree=2 holds 5/5 through noise=1% then drops to 0/5 at noise=5%; degree=3 drops to 0/5 already at noise=1%. Ordinary noise/degree sensitivity, no anomaly.
- Harmonic oscillator (conservative, 1 regime): degree=2 is 5/5 coefficient-recovered at every noise level, but VF-off-attractor-grid recovery drops to 0/5 at noise≥1% (metric disagreement, same pattern as LIMITATIONS.md #5). Degree=3 regresses sharply to mostly 0/5–1/5 coefficient recovery even at noise=0% — a real degradation, not noise-driven.
- Duffing unforced (conservative, 2 regimes collapsed to 1 in the grid): degree=2 (missing the true cubic term) is 0/5 coefficient at every noise level but ~4/5 VF-confirmation passes anyway — the confirmation-trajectory metric doesn't detect the missing term near the training support, while VF-grid correctly reads 0/5 off-support. Degree=3 gets a clean 5/5 at noise=0% then collapses to 0/5 at any nonzero noise.
- **Duffing forced-chaotic:** 0/5 on every metric (coefficient, VF-confirmation, VF-grid), both degrees, every noise level including 0% — a total, honestly-reported negative result; the fit never recovers the forced system in this grid.
- Length-sensitivity sub-sweep: qualitatively unchanged for logistic across short/long trajectories. For Lorenz, `stable_fixed_point`/`pre_chaotic` stay 0/5 at both lengths; the chaotic-adjacent regimes trend toward cleaner 5/5 recovery at the long length vs. more mixed results at the short length — consistent with longer sampling better exciting the attractor, not a surprising result.

**Off-attractor grid bounds fix and rerun (2026-08-16).** Independent review flagged the harmonic/Duffing-unforced VF-off-attractor-grid bounds (hardcoded `±3.0`/`±2.0`) as arbitrary hand-picked round numbers (LIMITATIONS.md #5). Fixed by deriving the grid bounds from each trajectory's own amplitude scale (`OFF_ATTRACTOR_GRID_SCALE * np.abs(states).max(axis=0)`, `OFF_ATTRACTOR_GRID_SCALE=3.0`), matching the principled approach already used for Lorenz/Rössler's own bounding box. All 132 Tier A conditions were rerun against the corrected bounds (990.9s wall-clock, 132/132, zero errors). Result: every harmonic/Duffing-unforced pass/fail cell in the tables above is **identical** to the pre-fix run — zero flips — while the underlying VF-off-attractor-grid error values genuinely shifted (e.g. harmonic degree=3/noise=0%/seed=3: 2.772→4.445; Duffing unforced degree=2/noise=0%: ~2.0→~1.3 across all 5 seeds), confirming the fix was actually applied rather than silently no-op'd. No headline conclusion in this section changes. Full comparison: `DECISION_LOG.md` "Off-attractor grid bounds derived from amplitude scale".

### Status
Tier A complete — 132/132 conditions, first-ever successful full completion this project (`DECISION_LOG.md`). Confirms the core chaos-aids-identifiability pattern at grid scale across 2 systems/12 regimes/2 degrees/4 noise levels. `period_3_window` degree-3 anomaly root-caused above (rank-deficiency, same mechanism as `period_2`); Duffing forced-chaotic's `recovered=None` is structurally-inapplicable-by-design, not a numeric failure (independent review, LIMITATIONS.md #6/RESULTS.md "Independent adversarial review"). (Historical note: Tier B/C, described below, had not yet run when this status line was first written.)

## Tier B: frozen main study, 3 discovery methods × 7 matched regime pairs (`MAIN_STUDY_DESIGN.md` §2/§4 step 6)

`experiments/main_study_tier_b.py`, raw output `experiments/main_study_results/tier_b_results.json`/`tier_b_manifest.json`. Originally 210/210 conditions (7 regime items), zero job failures; extended 2026-08-17 with an 8th item, `rossler` (chaotic, `a=b=0.2,c=5.7`) matched against the existing `harmonic` (conservative) regime — see "Harmonic/Rössler matched pair" subsection below — bringing the grid to **240/240 conditions, zero job failures**. Grid: 8 regime items (logistic period_2/chaotic, Lorenz stable_fixed_point/classic_chaotic, harmonic conservative, Duffing unforced/forced-chaotic conservative, rossler chaotic) × noise {0%, 1%, 5%} × library degree {2, 3} × 5 seeds. Each job shares one generated trajectory across all 3 methods (SINDy, symbolic regression via gplearn, Koopman/EDMD) to avoid an RNG-provenance confound (`DECISION_LOG.md` "Tier B shared-trajectory-generation design choice"). The design doc's own regime list originally summed to 7 pairs, not the 8 its prose stated (`DECISION_LOG.md` "Tier B regime-count discrepancy") — the 2026-08-17 addition now makes the actual grid match that original 8-pair intent, via a genuinely new pair rather than a fabricated one.

**SINDy coefficient recovery (n_recovered/5), collapsed by noise (degree collapsed) and by degree (noise collapsed):**

| Regime | 0% | 1% | 5% | deg2 | deg3 |
|---|---|---|---|---|---|
| logistic period_2 (control) | 0/10 | 0/10 | 0/10 | 0/15 | 0/15 |
| logistic chaotic | 10/10 | 10/10 | 8/10 | 15/15 | 13/15 |
| lorenz stable_fixed_point (control) | 0/10 | 0/10 | 0/10 | 0/15 | 0/15 |
| lorenz classic_chaotic | 10/10 | 8/10 | 8/10 | 15/15 | 11/15 |
| harmonic conservative (no pair) | — | — | — | 15/15 | 1/15 |
| duffing_unforced conservative (no pair) | 5/10 | 0/10 | 0/10 | 0/15 | 5/15 |
| duffing_forced forced_chaotic | n/a (`recovered=None`, library_mismatch_expected by design — no ground-truth coefficients under a forced library) | | | | |

**Core Tier A finding replicates cleanly under SINDy**: both non-chaotic controls (logistic period_2, Lorenz stable_fixed_point) are a hard floor of **0/60 combined across every noise×degree cell**, while their matched chaotic regimes stay at or near ceiling until high noise/degree erode them. Duffing has no SINDy-comparable pair (forced has no coefficient-recovery concept).

**Symbolic regression — the raw `degree_ok` proxy is misleading and must not be used alone.** `degree_ok=True` on every state dimension only means gplearn found *some* polynomial within the degree cap — not that it's dynamically correct. Cross-checked against the study's own VF-confirmation accuracy gate (`dynamically_distinct`): for **lorenz stable_fixed_point, 30/30 raw "successes" are simultaneously `dynamically_distinct=True`** — every structural pass is a vacuous fit near a barely-moving fixed point. Using the honest joint criterion (`degree_ok` all True **and** `dynamically_distinct=False`):

| Regime | joint-pass / 30 |
|---|---|
| logistic chaotic | 30/30 |
| logistic period_2 (control) | 4/30 |
| lorenz classic_chaotic | 14/30 |
| lorenz stable_fixed_point (control) | 0/30 |
| duffing_unforced conservative | 18/30 |
| duffing_forced forced_chaotic | **0/30** |

Under the corrected metric, chaos-aids-identifiability replicates strongly for logistic and Lorenz (0/30 for both controls), but **reverses for the Duffing pair**: forced-chaotic scores 0/30 real recovery against its non-chaotic sibling's 18/30 — Duffing forced is worse than its own control, not just a floor. This is consistent with Tier A's Duffing forced-chaotic total-failure finding, now shown to generalize beyond SINDy.

**Koopman/EDMD — a genuine reversal of the headline pattern, reported prominently, not smoothed over.** Using one-step prediction error (`one_step_rel_rms_err`, lower = more identifiable) at matched noise level, the **chaotic regime is consistently and substantially worse than its non-chaotic control**:
- Logistic, noise=5%: chaotic mean 0.086 vs. period_2 0.015 (deg=2) / 0.011 (deg=3) — chaotic error ~6–8× higher.
- Lorenz, noise=5%: classic_chaotic 0.036 vs. stable_fixed_point 0.0033 — chaotic error ~11× higher.
- Same direction at noise=1% in both families.

Plausible mechanism: chaotic trajectories amplify state noise through exponential sensitivity before EDMD's linear operator can average it out; near a fixed point/low-period orbit the same additive noise stays small relative to a far more confined state space. `spectral_rel_err` (K-matrix eigenvalues vs. true Jacobian) does not show the same clean reversal — mixed by family (favors chaos in Lorenz, favors the control in logistic, both with noise/degree-driven blowups from near-singular K matrices in low-variance regimes; K-condition-number reaches ~2.9e19 and 2 records produce `NaN` rollout error for Lorenz stable_fixed_point at deg=3/noise=0%).

**Mechanistic follow-up test (2026-08-16, `experiments/tier_b_koopman_normalization.py`): is the reversal actually the same divergence-rate confound already diagnosed for Tier C's rollout metric, rather than a genuine EDMD-specific effect?** Re-expressed `one_step_rel_rms_err` as a ratio to each regime's own largest-Lyapunov-exponent-derived local expansion factor (`exp(lambda_max * dt)`, computed once per regime from the true dynamics via the existing Benettin-QR machinery — no new simulation). Result: the reversal **survives normalization in both matched pairs, at every noise/degree cell tested**. For Lorenz (the stronger, continuous-time pair) the correction is negligible — the observation interval is far below the Lyapunov timescale, so the raw ~10.5-11x reversal is essentially untouched by normalization. For the logistic map, normalization removes roughly half the effect (raw 5.0-11.4x -> normalized 2.5-5.7x, matching the exact 2x per-iterate expansion factor at r=4) but a genuine >2.5x reversal remains in every cell. Conclusion: this is NOT primarily a trajectory-divergence-rate artifact of the kind that confounds Tier C's rollout metric — it is reported as a genuine Koopman/EDMD-specific identifiability pathology. Full numbers: `experiments/main_study_results/tier_b_koopman_normalization_results.json`/`..._summary.json`, `DECISION_LOG.md` "Koopman/EDMD one-step-error reversal: Lyapunov-normalization mechanistic test".

**Independent-implementation cross-check (2026-08-17): is the reversal a bug in the self-written EDMD?** `src/discovery_koopman.py`'s Koopman/EDMD arm is a self-written fallback (`pykoopman` originally failed to install under this project's Python 3.14 venv). Since the reversal is the paper's most novel finding and rests entirely on that homemade implementation, it was cross-checked against the third-party `pykoopman` library, installed separately under a Python 3.11 venv (`.venv_pykoopman311`). `pykoopman.Koopman(observables=Polynomial(degree=d, include_bias=True), regressor=EDMD())` — matched on dictionary degree and bias-inclusion to this project's own monomial dictionary — was fit on identical clean (noise=0%, seed=0) trajectories for both Tier B matched pairs. Held-out one-step relative RMS error for Lorenz classic_chaotic, the regime driving the headline reversal, agrees with the self-written implementation to **6 significant figures** at both degree 2 (3.596263e-05 both) and degree 3 (7.582637e-08 both); the other three regimes (both at/near machine-precision noise floor, as expected for the exactly-polynomial logistic map and the near-fixed-point Lorenz case) also agree. Training-residual magnitudes differ by orders of magnitude between the two implementations (pykoopman's `EDMD` regressor uses internal SVD-based rank handling different from this project's plain `lstsq`), so residual is not used as a comparison metric — only held-out one-step error, which is regressor-implementation-invariant. This rules out "implementation bug" as an explanation for the reversal. Scripts: `experiments/koopman_crosscheck_generate.py` (main venv) / `experiments/koopman_crosscheck_pykoopman.py` (`.venv_pykoopman311`). Full write-up: `DECISION_LOG.md` "Independent-implementation cross-check of self-written EDMD against pykoopman".

**Mechanistic theory (2026-08-17): does Gallo et al.'s (arXiv:2607.18490) own persistent-excitation predictor explain the reversal?** That paper predicts SINDy/PySR's identifiability ceiling from lambda_min(M), the smallest eigenvalue of a feature-library moment/Gram matrix built along the trajectory — more excitation (larger lambda_min) predicts better identifiability. Tested the analogous quantity built from EDMD's own monomial dictionary, `Phi(x)^T Phi(x)/n`, across all 120 Tier B matched-pair jobs (`experiments/koopman_gram_matrix_analysis.py`, bit-for-bit-reproduced trajectories). The specific hypothesis tested first — that chaos degrades EDMD's regression *conditioning* (cond(G) = lambda_max/lambda_min) and that this explains the reversal — is **refuted**: chaotic regimes have systematically *better* (lower) cond(G) than their non-chaotic control in all 12 (family x noise x degree) cells, and log-log correlation of cond(G) against one-step error is the wrong sign for that story (logistic rho=-0.131 n.s., lorenz rho=-0.423 p=7.6e-4). A distinct, cleaner finding survives: **lambda_min(G) itself carries the opposite predictive sign for EDMD than Gallo et al.'s theory gives for SINDy/PySR** — moving from a non-chaotic control to its chaotic pair raises lambda_min(G) (genuinely more persistent excitation) AND raises one-step error, in all 12/12 matched cells with zero exceptions, both families; within-family rank correlation of log(lambda_min) vs log(error) is positive and significant (logistic rho=+0.397 p=4.3e-3; lorenz rho=+0.399 p=1.6e-2). Reading: the same phase-space breadth that helps constrain a sparse coefficient vector (SINDy/SR) works against uniform approximation by EDMD's dense, dictionary-truncated linear surrogate — stated as a plausible qualitative account, not a derived bound. Full write-up including the refuted hypothesis: `DECISION_LOG.md` "Gram-matrix mechanistic theory for the Koopman reversal"; raw data `experiments/main_study_results/koopman_gram_matrix_results.json`.

**Anomalies, investigated and characterized, none change a headline conclusion:**
- **Lorenz classic_chaotic, symbolic regression, y-dimension only:** 6/30 jobs fail `degree_ok` on dimension 1 specifically (scattered across noise/degree/seed, no pattern), consistent with gplearn's genetic search occasionally failing to converge on the one cross-term (xz) equation in this system within 5 retries — ordinary SR stochastic variance.
- **Duffing_unforced, symbolic regression, degree=2 only:** 5/30 jobs fail `degree_ok`, exclusively at the degree cap that structurally excludes the true cubic term (v̇ = −αx − βx³); this is expected library mismatch, not an anomaly, and self-resolves to 5/5 at degree=3 in every noise cell. Sub-finding: only 5/15 degree=2 attempts even land on the (wrong but degree-2) true-adjacent structure; the retry loop's first-structural-success acceptance criterion, not accuracy, decides when it stops.
- **Duffing_forced, symbolic regression, phi-dimension only, exactly at noise=5%:** 10/10 jobs (both degrees, all 5 seeds) fail `degree_ok` on dimension 2; 0/20 fail at noise=0% or 1%. Root cause traced to `simulators.py`: phi is an unwrapped, unbounded phase (φ̇ = ω = 1.2, a trivial constant) that drifts to ~240 rad over the trajectory; 5% noise is scaled by `state_std`, so on this large-range variable it becomes a large absolute perturbation relative to the near-zero true derivative, and finite-differencing the noisy near-linear ramp lets gplearn fit a spurious non-polynomial expression (its function set includes sin/cos, matched to the true forcing term elsewhere in the system). A genuine modeling-choice caveat (phi should arguably be wrapped mod 2π), not a bug — flagged for future work, does not affect any comparative conclusion since duffing_forced already fails on every other metric at every noise level. **Resolved (2026-08-18, NEXT_STEPS.md item #6, see DECISION_LOG.md "Duffing forced phi-noise-scaling fix"):** rerunning the identical 30-condition grid with the noise magnitude on phi computed from a wrapped (mod 2π) copy of phi, rather than the raw unbounded ramp, confirms this diagnosis exactly — `degree_ok` returns to 5/5 at noise=5% for both degrees (was 0/5), matching the 0%/1% cells, and Koopman one-step error at noise=1%/5% drops ~33x (0.00695→0.000211 at 1%; 0.0347→0.00105 at 5%), both previously inflated by the same artifact. The comparative conclusion is unchanged exactly as anticipated: `duffing_forced` remains 0/30 under the joint SR criterion and fails SINDy's `dynamically_distinct` gate in all 30/30 cells both before and after the fix — the artifact inflated intermediate diagnostic magnitudes, not the pass/fail outcome.
- **Metric disagreement replicates from Tier A at Tier B scale:** 60/210 (28.6%) SINDy jobs have `recovered=False` but `dynamically_distinct=False` — wrong coefficients that still pass the confirmation-trajectory VF-error gate. Same blind spot as `LIMITATIONS.md` #5, now quantified across the full Tier B grid.

**Off-attractor grid bounds fix and rerun (2026-08-16).** Same amplitude-derived-bounds fix as Tier A (`DECISION_LOG.md` "Off-attractor grid bounds derived from amplitude scale"), applied to Tier B's harmonic/duffing_unforced regimes and rerun in full (210/210, zero failures, 6456.0s wall-clock across 6 workers). Compared all 60 harmonic/duffing_unforced rows (both SINDy and symbolic-regression methods, every noise×degree×seed cell) old vs. new: zero `recovered`-flag flips, but the underlying `vf_l2_err_off_attractor_grid` values genuinely shifted in 117/120 method-cells (e.g. harmonic degree=3/noise=0%/seed=3: 2.772→4.445, matching Tier A's shift for the same cell/seed since both tiers share the same trajectory-generation code). No table or conclusion above changes.

**Multiplicity-corrected confidence intervals (PREREGISTRATION.md §9, resolved 2026-08-16):** exact Clopper-Pearson 99.17%-CI (Bonferroni-adjusted, k=6 fixed comparisons) on the pooled n=30 counts above — every control interval sits strictly below its matched chaotic/treatment interval (e.g. SINDy logistic: control [0.00,0.17] vs. chaotic [0.72,1.00]; tightest margin, symbolic-regression Lorenz: control [0.00,0.17] vs. chaotic [0.23,0.71]). Full table: DECISION_LOG.md "Multiplicity correction (PREREGISTRATION.md §9), resolved".

### Status
Original Tier B complete — 210/210 conditions, 630 total fits, zero job failures (`tier_b_run.log` shows only benign STLSQ/overflow warnings, no uncaught exceptions). SINDy replicates the Tier A chaos-aids-identifiability pattern cleanly. Symbolic regression replicates it too, once the raw `degree_ok` proxy is corrected against `dynamically_distinct` — but with a genuine reversal for the Duffing pair (forced-chaotic is worse than its non-chaotic control). **Koopman/EDMD one-step prediction error reverses the headline pattern outright in both regime pairs with a chaotic/non-chaotic match** — this is the most important single finding of Tier B and must be stated as a limit on the "chaos aids identifiability" claim, not folded silently into an average across methods. Any headline claim about chaos and identifiability going forward must be scoped to method and metric, not asserted as a universal effect. See below for the 2026-08-17 harmonic/Rössler extension, which brings Tier B to 240/240 conditions.

## Harmonic/Rössler matched pair (NEXT_STEPS.md item #1, 2026-08-17)

LIMITATIONS.md #7 (independent review) found that every chaotic-vs-non-chaotic contrast in the original 7-pair grid pitted chaos against a maximally *degenerate* non-chaotic control (a fixed point or short periodic cycle), never against the one genuinely rich, non-degenerate non-chaotic regime already in the study — the harmonic oscillator, whose trajectory is a continuum of states on a closed orbit, not a handful of support points. Added an 8th Tier B item, `rossler` (chaotic Rössler attractor, `a=b=0.2,c=5.7`, `DEFAULT_ROESSLER_PARAMS`), matched against the existing `harmonic` regime, run through the exact same 3-method pipeline at the full grid (noise {0%,1%,5%} × degree {2,3} × 5 seeds = 30 conditions). Checkpoint-key resumption worked as intended: 210/210 pre-existing jobs were skipped and only the 30 new `rossler` jobs ran (verified before launch, `experiments/main_study_tier_b.py`'s `_job_key` is keyed on `(family, regime_label, noise, degree, seed)`, which the new `rossler` family/label never collides with). Rössler has no known-parametric coefficient-recovery formula wired into this runner (`params=None`, same convention as `duffing_forced`) — its SINDy/SR arms are evaluated via the same VF-error/structural metrics as `duffing_forced`, not coefficient recovery.

**SINDy — primary (confirmation-trajectory VF error) pass counts, `n_pass/n` where pass = `dynamically_distinct=False`:**

| Regime | 0% | 1% | 5% | deg2 | deg3 | total |
|---|---|---|---|---|---|---|
| harmonic conservative (rich non-chaotic control) | 8/10 | 4/10 | 4/10 | 9/15 | 7/15 | 16/30 |
| rossler chaotic | 10/10 | 10/10 | 10/10 | 15/15 | 15/15 | 30/30 |

**Finding 1 (SINDy): the chaos-aids-identifiability pattern replicates, and strengthens, against a rich non-chaotic control.** Rössler passes the primary VF-error gate in every single cell (30/30), while the harmonic oscillator — despite recovering its (trivially low-degree) coefficients cleanly at degree=2 in Tier A/B's earlier reporting — fails the *primary* off-trajectory metric in 14/30 cells, concentrated at degree=3 (8/15) and at noise>=1% (6/20 at deg2, worse at deg3). This is the first direct test in this project of "chaos vs. a rich non-chaotic control" (as opposed to "chaos vs. a degenerate control"), and for SINDy's primary metric it comes out in the same direction as every prior chaos-vs-degenerate-control comparison: the chaotic regime is at least as identifiable, and here more robustly so at higher degree/noise. This directly narrows LIMITATIONS.md #7's gap for the SINDy method (see LIMITATIONS.md update).

**Symbolic regression — joint criterion (`degree_ok` all True AND `dynamically_distinct=False`) pass / 30:**

| Regime | joint-pass / 30 |
|---|---|
| harmonic conservative | 30/30 |
| rossler chaotic | 10/30 |

**Finding 2 (symbolic regression): the already-established SR reversal (chaos hurts gplearn's search) replicates on this pair too.** Harmonic passes the joint criterion in every cell; Rössler passes in only 10/30, consistent with the pattern already seen for Lorenz's cross-term (xz) dimension and Duffing forced's total SR failure — gplearn's genetic search on this project's matched `add/sub/mul/div` function set struggles specifically on 3-dimensional systems with genuine multiplicative cross-terms, independent of whether the regime is chaotic or a rich non-chaotic control.

**Koopman/EDMD — one-step relative RMS error, mean across seeds, by noise/degree:**

| noise | degree | harmonic (non-chaotic) | rossler (chaotic) | ratio (rossler/harmonic) |
|---|---|---|---|---|
| 0% | 2 | 5.9e-05 | 4.5e-05 | 0.75x |
| 0% | 3 | 7.6e-03 | 3.5e-07 | 0.00005x |
| 1% | 2 | 1.45e-02 | 1.36e-02 | 0.94x |
| 1% | 3 | 1.15e-01 | 1.35e-02 | 0.12x |
| 5% | 2 | 7.02e-02 | 6.77e-02 | 0.97x |
| 5% | 3 | 1.40e-01 | 6.67e-02 | 0.48x |

**Finding 3 (Koopman/EDMD): the established chaos-hurts-Koopman reversal does NOT replicate for this pair — the relationship is flat-to-inverted instead.** In every one of the 6 (noise, degree) cells, the chaotic Rössler regime's mean one-step error is *lower than or comparable to* the non-chaotic harmonic control's, the opposite direction from the logistic and Lorenz pairs' 6-11x chaos-is-worse reversal reported earlier in this section. Per-seed harmonic values are also far noisier (noise=0%/degree=2: 4.1e-6 to 1.08e-4 across 5 seeds, a ~26x spread) than Rössler's (4.1e-5 to 4.8e-5, a ~1.2x spread) at the same cell — the harmonic oscillator's EDMD fit is itself less stable across seeds than the chaotic regime's, opposite to what the "chaos hurts conditioning" story would predict. This is reported as a genuine non-replication, not smoothed into the earlier reversal claim: **the Koopman/EDMD reversal is confirmed method-and-pair-dependent, not a universal property of "chaotic vs. non-chaotic," and does not hold for the one rich-non-chaotic-vs-chaos pair in this study.**

**Net read on LIMITATIONS.md #7:** mixed by method, and genuinely informative in both directions — SINDy's primary metric now shows chaos aiding identifiability even against a rich, non-degenerate non-chaotic control (narrowing #7's gap for SINDy specifically), while Koopman/EDMD's reversal — the study's most novel prior finding — turns out NOT to replicate against this same rich control, meaning the Koopman reversal itself was partly (not wholly) a degenerate-control artifact. Symbolic regression's chaos-hurts pattern is unaffected either way (driven by dimensionality/cross-terms, not control degeneracy). Full raw data: `experiments/main_study_results/tier_b_results.json` (`family="rossler"`/`family="harmonic"` rows), `experiments/main_study_tier_b.py`.

## Tier C: partial-observation stress test, SINDy only, delay-embedded x(t) (`MAIN_STUDY_DESIGN.md` §2/§4 step 7)

140 conditions (7 regime pairs × noise {0%, 1%} × degree {2, 3} × 5 seeds), 140/140 checkpointed, zero job failures, 5.8s total wall-clock (SINDy on a delay-embedded scalar series is cheap — no symbolic regression or Koopman/EDMD arm, out of scope per `MAIN_STUDY_DESIGN.md` §5). Each job observes only x(t) from the identical Tier B trajectory (same RNG/noise/regime, `_generate_tier_b_data` reused directly), reconstructs an m-dimensional pseudo-state via Takens delay embedding (m = the family's true state dimension, delay τ chosen per-draw via first-1/e-autocorrelation-crossing — see `DECISION_LOG.md` "Tier C delay-embedding design"), and fits SINDy on the reconstructed state. Delay coordinates are not the true system's coordinates, so there is no coefficient-recovery ground truth here; identifiability is measured by predictive accuracy on a held-out delay-embedded confirmation trajectory instead.

**Noise-range extension (2026-08-16, `experiments/main_study_tier_c_noise5.py`, STATUS.md open item 6, resolved):** the original 140-condition grid only ran noise {0%, 1%}, leaving open whether "no gap detected" reflected genuine closure or just insufficient noise to expose one (Tier B's noise range goes up to 5%). Ran the identical 70-condition grid (7 regime pairs × degree {2, 3} × 5 seeds) at noise=5%, reusing Tier C's own job/fit code unmodified, writing to separate checkpoint/results files so the original 0%/1% data was untouched. Result: **0/70 dynamically_distinct at 5% too** — same outcome as 0% and 1%. This resolves the open question in favor of reading (a): the "no gap" finding holds across the full noise range Tier B itself uses, not an artifact of testing too little noise.

Three predictive diagnostics are reported per job. The **primary gate** is one-step-from-ground-truth state error (Euler-forward from each true confirmation-trajectory point, compared to the true next point) — chosen specifically because it isolates model correctness from a chaotic regime's inherent trajectory-divergence sensitivity. Two secondary diagnostics are also logged but NOT used as the gate: a derivative-based error (found in the pre-launch smoke test to be dominated by finite-differencing noise across highly-correlated delay columns) and a fixed-50-step rollout error (found to be confounded by Lyapunov-time sensitivity — a chaotic regime inflates rollout error over a fixed horizon even for a perfectly-identified model, which would misattribute normal chaotic divergence to a modeling failure if used as the primary metric).

| Regime | Noise | one-step err (mean) | rollout err (mean, diagnostic only) | distinct (gate) |
|---|---|---|---|---|
| logistic period_2 (control) | 0% | 2.6e-16 | 1.0e-15 | 0/10 |
| logistic period_2 (control) | 1% | 0.0026 | 0.0022 | 0/10 |
| logistic chaotic | 0% | 1.9e-15 | 0.062 | 0/10 |
| logistic chaotic | 1% | 0.0173 | 0.743 | 0/10 |
| lorenz stable_fixed_point (control) | 0% | 6.6e-08 | 7.9e-06 | 0/10 |
| lorenz stable_fixed_point (control) | 1% | 0.0011 | 0.0010 | 0/10 |
| lorenz classic_chaotic | 0% | 0.0088 | 0.195 | 0/10 |
| lorenz classic_chaotic | 1% | 0.0165 | 0.198 | 0/10 |
| harmonic conservative | 0% | 0.0003 | 0.0096 | 0/10 |
| harmonic conservative | 1% | 0.0141 | 0.034 | 0/10 |
| duffing_unforced conservative | 0% | 0.0001 | 0.0043 | 0/10 |
| duffing_unforced conservative | 1% | 0.0043 | 0.0074 | 0/10 |
| duffing_forced forced_chaotic | 0% | 3.1e-05 | 0.054 | 0/10 |
| duffing_forced forced_chaotic | 1% | 0.0135 | 0.067 | 0/10 |
| logistic period_2 (control) | 5% | 0.0128 | 0.0125 | 0/10 |
| logistic chaotic | 5% | 0.0860 | 0.767 | 0/10 |
| lorenz stable_fixed_point (control) | 5% | 0.0055 | 0.0052 | 0/10 |
| lorenz classic_chaotic | 5% | 0.0699 | 0.177 | 0/10 |
| harmonic conservative | 5% | 0.0698 | 0.0727 | 0/10 |
| duffing_unforced conservative | 5% | 0.0216 | 0.0299 | 0/10 |
| duffing_forced forced_chaotic | 5% | 0.0656 | 0.134 | 0/10 |

**Headline finding: on the primary (one-step-from-truth) gate, Tier C detects NO chaos-vs-non-chaos identifiability gap at all — every one of the 210 jobs (140 original + 70 noise=5% extension) passes (`dynamically_distinct=False`, threshold 0.10), chaotic and non-chaotic alike, across the full noise range {0%, 1%, 5%}.** This does not confirm the Tier A/B pattern, nor does it reverse it the way Koopman/EDMD did in Tier B — it is a genuinely different outcome: under delay-embedded single-coordinate observation, short-horizon predictive accuracy is uniformly high (worst case 0.086, well under the 0.10 threshold, at noise=5%) regardless of regime. The noise-range extension (above) rules out the "insufficient noise" reading that was open when only {0%, 1%} had been tested — the result now holds at Tier B's own top noise level too, so the better-supported reading is that a well-chosen delay embedding substantively closes the identifiability gap seen under full-state observation, at least for short-horizon one-step prediction.

The **rollout diagnostic** (fixed 50-step horizon, NOT the gate) does show a large, consistent gap in the expected direction — chaotic regimes' rollout error is 10-300x higher than their matched non-chaotic control (logistic: 0.74 vs 0.002 at noise=1%; Lorenz: 0.20 vs 0.001 at noise=1%; Duffing: 0.067 vs 0.007 at noise=1%) — but this is explicitly flagged, not reported as confirming evidence, because a fixed-horizon rollout on a chaotic attractor is expected to diverge faster than on a non-chaotic control even under a perfectly-identified model, purely from Lyapunov-time sensitivity to any infinitesimal perturbation (including ordinary floating-point roundoff). The one-step gate exists precisely to avoid this confound, and it shows no gap — so the rollout gap is presented as a distinct, unresolved observation about trajectory divergence, not treated as identifiability evidence.

The logistic-map regime pair is flagged as degenerate for this tier: the logistic map is already 1-dimensional, so single-coordinate observation is full observation, not a partial-observation stress test, for that family only. Its rows are included above for grid completeness but should not be read as informative about partial observation.

### Status
Tier C complete — 210/210 conditions (140 original + 70 noise=5% extension), zero job failures. On the primary metric, no chaos-vs-non-chaos gap is detected under delay-embedded partial observation across the full noise range {0%, 1%, 5%} — a third distinct outcome alongside Tier B's "pattern holds" (SINDy/symbolic regression) and "pattern reverses" (Koopman/EDMD) results. The secondary rollout diagnostic shows a large gap but is explicitly not used as evidence, since it is confounded by chaotic trajectory-divergence sensitivity rather than isolating model correctness. Per `MAIN_STUDY_DESIGN.md` §4, Tiers A-C are now complete; step 8 (unsealing the held-out Rössler confirmation family) is next.

## Held-out Rössler confirmation (MAIN_STUDY_DESIGN.md §3/§4 step 8)

Unblinded and run exactly once, per PREREGISTRATION.md §11 — the Rössler system (`src/simulators.py`) was never used during pilot/method development; only a literature-comparison Lyapunov-exponent check had touched it before this run (ruled non-violating of §11 in DECISION_LOG.md at freeze time). 16 conditions (chaotic `a=b=0.2,c=5.7` vs. matched non-chaotic control `c=3.0`, × noise {0%, 0.1%, 1%, 5%} × library degree {2, 3}), 5 seeds each = 80 fits, SINDy only, full-state observation, mirroring Tier A's exact metric stack (coefficient recovery, VF-confirmation error, off-attractor-grid VF error, Lyapunov-spectrum error, invariant-measure TV distance, all at Tier A's same COEFF_TOL/VF_ERR_TOL/STLSQ_THRESHOLD). 102.8s wall-clock, zero job failures.

| regime | noise | degree | coeff-recovered | off-attractor-grid-VF-recovered |
|---|---|---|---|---|
| chaotic | 0%/0.1%/1% | 2, 3 | 5/5 | 5/5 |
| chaotic | 5% | 2 | 3/5 | 5/5 |
| chaotic | 5% | 3 | 2/5 | 5/5 |
| non_chaotic_control | 0%/0.1% | 2, 3 | 5/5 | 5/5 |
| non_chaotic_control | 1% | 2 | 5/5 | 5/5 |
| non_chaotic_control | 1% | 3 | 3/5 | 3/5 |
| non_chaotic_control | 5% | 2, 3 | 0/5 | 0/5 |

**Headline finding: the chaos-aids-identifiability pattern REPLICATES on this genuinely untouched family.** At the noise levels where the two regimes separate (1-5%), the chaotic regime's coefficient recovery is consistently equal to or better than the matched non-chaotic control's — most starkly at noise=5%, where the chaotic regime still recovers 3/5 (degree=2) and 2/5 (degree=3) while the non-chaotic control recovers 0/5 at both degrees. The off-attractor-grid VF-error metric (the metric Tier A's own design doc flags as the more discriminating of the two VF-based metrics for point/periodic attractors) tracks the coefficient-recovery gap exactly at noise=5% (chaotic 5/5 vs. control 0/5) and partially at noise=1%/degree=3 (chaotic 5/5 vs. control 3/5). The confirmation-trajectory VF metric (`vf_l2_err_confirmation`) shows no gap anywhere (5/5 in every cell) — consistent with the same metric-forgiveness limitation already documented for the logistic-map period-2 control in the pilot (LIMITATIONS.md #5), not a new finding.

This is the first evidence in this project that the primary finding generalizes beyond the systems used to develop the method and choose hyperparameters — Rössler was never inspected, plotted, or fit before this run, and STLSQ_THRESHOLD/COEFF_TOL/VF_ERR_TOL were frozen well before its unblinding. Lyapunov-spectrum-error and invariant-measure-TV-distance (seed 0 only, matching Tier A's compute-scoping) both computed without error at every noise=0% condition, no `lyapunov_error_skipped`/`_failed` observed.

### Status
Held-out confirmation complete. Per `MAIN_STUDY_DESIGN.md` §4, Tiers A-C plus the frozen confirmation family are now all complete — PREREGISTRATION.md §11's blinded-regime requirement is discharged.

## Held-out parameter regime, Lorenz family (PREREGISTRATION.md §1 closing clause)

Closes a preregistration-completeness gap found post-hoc: §1's closing clause calls for "one held-out parameter regime per developed family," but Tiers A-C exhaustively used every one of §1's listed logistic (4) and Lorenz (5) regimes — none was ever actually reserved. Remedy (`experiments/main_study_confirmation_regime.py`, DECISION_LOG.md "Held-out parameter regime gap"): one new Lorenz value never referenced anywhere in this project before this run, rho=45.0 (outside the preregistered {14,22,24.5,28,100} set, deep chaotic band, chosen for being unremarkable rather than favorable), run once through Tier A's exact `fit_lorenz` code at Tier A's full noise × degree × seed grid (8 conditions, 40 fits). 65.3s wall-clock, zero job failures.

| degree | noise | coeff-recovered | grid-VF-recovered |
|---|---|---|---|
| 2 | 0%/0.1%/1% | 5/5 | 5/5 |
| 2 | 5% | 3/5 | 5/5 |
| 3 | 0%/0.1%/1% | 5/5 | 5/5 |
| 3 | 5% | 4/5 | 5/5 |

**Finding: chaos-aids-identifiability holds at a genuinely never-tested Lorenz parameter value, at a level comparable to the preregistered chaotic regime.** For comparison, Tier A's preregistered `classic_chaotic` (rho=28) recovers 5/5 at degree=2/noise=5% but drops to 3/5 at degree=3/noise=1% and 3/5 at degree=3/noise=5% (`tier_a_results.json`). rho=45.0's degradation pattern (3/5 at degree=2/noise=5%, 4/5 at degree=3/noise=5%) is of the same order, not qualitatively different — no cliff, no reversal, no new failure mode at an untested parameter value. This is a within-family robustness check, not a matched chaotic-vs-non-chaotic pair (no non-chaotic Lorenz value was held out alongside it — a single new regime was judged the minimal defensible fix, see DECISION_LOG.md).

### Status
PREREGISTRATION.md §1's closing clause is now discharged (Rössler for the new-family half, Lorenz rho=45.0 for the held-out-regime-per-family half — logistic/harmonic/Duffing were not given their own held-out regime; judged sufficient given Lorenz is the project's central backbone system, logged as a scope decision in DECISION_LOG.md). The frozen main study, its held-out confirmation family, and this held-out-regime check are now all complete.

## Independent adversarial review (PROMPT.md completion-gate item)

A dedicated adversarial-review pass (background agent, read-only, instructed to try to invalidate the central result) independently recomputed headline numbers from all four result JSON files, verified confound-mitigation code (not just comments) for Tier C's rollout/derivative-metric exclusions and Tier A's Lorenz transient-discard fix, and checked for threshold-gaming, cherry-picking, and cross-tier inconsistency. Full report preserved in the session transcript; summary of findings, most significant first:

- **Real, moderate — headline framing is narrower-supported than stated.** The matched-pair contrasts that carry the "chaos aids identifiability" claim (logistic period_2 vs. chaotic; Lorenz stable_fixed_point vs. classic_chaotic) pit chaos against maximally *degenerate* non-chaotic controls (fixed points, 2-point cycles — near rank-deficient by construction), not against the one rich-but-non-chaotic regime in the grid, the harmonic oscillator — which recovers 5/5 at degree=2 in every noise cell, as cleanly as the chaotic regimes, and was never given a chaotic partner (RESULTS.md's own harmonic section already flags "no pair"). A persistent-excitation/state-space-coverage mechanism is at least as well supported by this project's own data as a chaos-specific mechanism. This alternative-explanation test was never directly built into the matched-pair design. Flagged as an open gap below and in LIMITATIONS.md, not silently resolved.
- **Real, minor.** Tier A's Duffing forced-chaotic row is described as "0/5 on every metric" — but coefficient recovery there is structurally inapplicable (`library_mismatch_expected=true`, no ground-truth coefficients exist under the forced-embedding library), not a numeric failure; Tier B correctly reports the same regime as `n/a`. Tier A's prose should be read as "0/5 on both VF metrics; coefficient recovery not applicable."
- **Real, resolved same day.** PREREGISTRATION.md §9's promised Bonferroni-style multiplicity correction had never been applied to the frozen study's headline counts. Resolved by computing exact Clopper-Pearson binomial confidence intervals at a Bonferroni-adjusted level (k=6 fixed Tier B matched-pair comparisons, 99.17% CI per comparison) directly from `tier_b_results.json` — every control-vs-chaotic interval pair remains non-overlapping (closest margin: symbolic-regression Lorenz pair). No headline conclusion changes; see DECISION_LOG.md "Multiplicity correction (PREREGISTRATION.md §9), resolved" for the full table and method, and LIMITATIONS.md #6.
- **Real, minor — reproducibility gap, not evidence of an actual violation.** Unlike Tiers A-C, the Rössler confirmation's pre-launch blinding-preserving shape-only smoke test left no committed artifact (no script, no log) — the blinding claim rests on a contemporaneous self-report rather than an independently checkable trail, though the described mechanism (checking only dict keys/types, not values) is a legitimate way to satisfy §11 if it happened as described.
- **Real, minor.** Off-attractor-grid domain bounds (e.g., ±3 for harmonic, ±2 for Duffing unforced) are hand-picked round numbers, not derived from each system's natural amplitude scale — an unquantified degree of freedom in a metric that feeds headline claims.
- **False alarms, checked and ruled out:** threshold-gaming (COEFF_TOL/VF_ERR_TOL/STLSQ_THRESHOLD byte-identical and unmoved across every tier and the confirmation run — every post-hoc design correction in DECISION_LOG.md is self-reported as *weakening*, not strengthening, the headline); cherry-picking (every anomaly independently found in the raw JSON — period_3_window flip, Duffing total failure, SR y-dimension failures, Koopman reversal — was already surfaced in this project's own write-up); cross-tier inconsistency (Tier B/Tier C/Rössler's three different outcomes are each kept in their own explicitly-scoped lane, never averaged into a false consensus); confound handling (Tier C's `dynamically_distinct` gate and Tier A's Lorenz noise-then-discard fix verified directly in code, not just comments, to match their documentation).

**Verdict (reviewer's own words): "the frozen main study + confirmation family substantively satisfies the independent-review gate."** PROMPT.md's completion-gate item "independent review" is satisfied. The moderate framing finding above is the one item that should be corrected before any packaging/write-up claims "chaos aids identifiability" without qualification — see LIMITATIONS.md #7 (new).

## Lorenz-96 (N=6): higher-dimensional matched pair (EXTENSION_PLAN.md extension #4, 2026-08-17)

Deliberately the same system family Gallo, Anselmi, Lazzari (arXiv:2607.18490) used as their own held-out zero-refit check, chosen to invite direct comparison. `src/simulators.py` implements the standard cyclic Lorenz-96 ODE at N=6 (smallest N with well-supported chaos while staying inside this project's local-only compute ceiling), validated via a finite-difference Jacobian check (max abs diff ~1.2e-9) and a Lyapunov-spectrum F-scan: **F=1.0** is the matched non-chaotic control (stable fixed point, largest exponent -0.027, entire spectrum negative), **F=8.0** (the classic literature forcing value) is the chaotic regime (largest exponent +1.11).

`experiments/lorenz96_pipeline.py` ran the full grid — 2 regimes x 3 noise levels {0%,1%,5%} x 5 seeds = 30/30 conditions, zero failures, 27.7s wall-clock. Symbolic regression was explicitly **skipped** at N=6 (SR search cost scales badly with input dimension — 6 candidate operands per tree node vs. <=3 in every Tier B family — making a per-seed run prohibitively slow on this local-only-compute laptop); only SINDy (degree-2 polynomial library, 28 features) and Koopman/EDMD (degree-2 monomial dictionary) ran.

**SINDy coefficient recovery** (per-equation, against the 4 known nonzero true coefficients — const=F, linear=-1, two quadratic cross-terms — COEFF_TOL=0.05, recovered = all 6 equations pass):
| Regime | noise=0% | noise=1% | noise=5% |
|---|---|---|---|
| F=8.0 (chaotic) | 5/5 (mean max_rel_err 0.0019) | 5/5 (0.0046) | 5/5 (0.023) |
| F=1.0 (control) | 0/5 (mean max_rel_err 1.07) | 0/5 (153) | 0/5 (463) |

The F=1.0 control fails total coefficient recovery even at 0% noise — the state barely moves post-transient near the fixed point (std~0.076 vs. O(3-5) in the chaotic regime), leaving the 28-term quadratic feature library severely rank-deficient. This exactly matches the mechanism and outcome already on record for Tier A/B's own N<=3 Lorenz `stable_fixed_point` (rho=14) regime (also `recovered=False`, max_rel_err~1.0, every seed/noise level) — a replication of an already-established failure mode, not a new N=6 artifact.

**Koopman/EDMD one-step relative RMS error** (identical metric definition to Tier B):
| Regime | noise=0% | noise=1% | noise=5% |
|---|---|---|---|
| F=1.0 (control) | ~0 (machine precision) | 0.0028 | 0.0140 |
| F=8.0 (chaotic) | 0.0014 | 0.0119 | 0.0587 |

EDMD's one-step error is 4-9x worse in the chaotic regime than the matched control at every noise level.

**Finding: Tier B's (N<=3) reversal pattern replicates cleanly at N=6, and the SINDy side is more extreme.** Chaos again correlates with worse Koopman/EDMD one-step error (consistent with all prior Tier B/mechanistic-theory results above) but comparable-or-better SINDy identifiability — here a total 5/5-vs-0/5 contrast rather than Tier B's typical partial degradation, because the N=6 non-chaotic control's post-transient state variation is closer to zero than any Tier B control exhibited. Scope: single N=6 tested (no N-scan), SINDy/Koopman only (SR skipped by explicit compute-cost decision, not silently dropped), one seed count (5) per condition. Full detail: `experiments/main_study_results/lorenz96_results.json`, `DECISION_LOG.md` "Lorenz-96 (N=6) matched-pair extension".

## Neural-network Koopman arm: does the reversal generalize beyond EDMD's fixed dictionary? (EXTENSION_PLAN.md extension #3, 2026-08-17)

The Koopman/EDMD reversal has survived a Lyapunov-normalization check and a pykoopman independent-implementation cross-check, but both used EDMD's fixed, hand-specified monomial dictionary. `src/discovery_koopman_nn.py` implements a genuinely different Koopman surrogate: an encoder MLP into a learned latent space, a linear latent-dynamics matrix, and a decoder MLP, trained end-to-end by gradient descent (one-step prediction + latent-consistency + reconstruction loss) rather than least-squares onto a fixed basis. `experiments/nn_koopman_tier_b.py` ran it on the same two Tier B matched pairs (logistic period_2/chaotic; Lorenz stable_fixed_point/classic_chaotic) at noise {0,1%,5%} x 3 seeds, computing the identical `one_step_rel_rms_err` metric EDMD reports:

| family/regime | noise=1% (EDMD / NN) | noise=5% (EDMD / NN) |
|---|---|---|
| logistic period_2 (control) | 0.00299 / 0.00266 | 0.01495 / 0.01320 |
| logistic chaotic | 0.01733 / 0.04840 | 0.08595 / 0.08837 |
| lorenz stable_fixed_point (control) | 0.000659 / 0.00064 | 0.003295 / 0.00319 |
| lorenz classic_chaotic | 0.007342 / 0.03917 | 0.036398 / 0.05245 |

**Finding: the reversal generalizes beyond EDMD's fixed dictionary.** Every family/noise cell shows chaotic one-step error higher than its matched control for both EDMD and the NN arm; the NN arm's relative reversal magnitude (5.8-6.7x logistic, 11-61x lorenz) is comparable to or larger than EDMD's (5.7-11x), not smaller or absent. This is a third independent robustness axis for the reversal, alongside Lyapunov-normalization and the pykoopman cross-implementation — evidence this is a property of linear-in-representation Koopman methods on these systems generally, not an artifact of one fixed dictionary. Caveats: 3 seeds vs. Tier B's 5, degree-2-equivalent capacity only; the Lorenz control's training loss (not the reported error metric) behaved oddly at nonzero noise, flagged but did not corrupt the comparison. Full detail: `experiments/main_study_results/nn_koopman_results.json`, `DECISION_LOG.md` "Neural-network Koopman arm".

## PySR independent-implementation cross-check of the Tier B symbolic-regression Duffing reversal (EXTENSION_PLAN.md extension #2, 2026-08-17)

Tier B's symbolic-regression arm (gplearn) found duffing_forced (chaotic) 0/30 real recovery under the joint criterion (degree_ok all True AND dynamically_distinct=False) against its matched non-chaotic control duffing_unforced's 18/30 — the SR "reversal," opposite the headline chaos-aids-identifiability direction. `experiments/pysr_crosscheck_duffing.py` re-ran the same trajectory-generation/noise/methodology under PySR (Julia SymbolicRegression.jl backend, run in a separate `.venv_pysr311`), with the operator set matched exactly to gplearn's add/sub/mul/div (no trig), across 2 families x 3 noise levels {0%,1%,5%} x 3 seeds = 18 conditions. Because PySR searches expression complexity directly rather than a fixed-degree polynomial library, there is no `degree_ok` analog — the comparable per-implementation metric is `dynamically_distinct=False` alone (PySR's single recovery criterion) against gplearn's joint criterion:

| family | noise | PySR recovered/3 | gplearn recovered/10 (joint) |
|---|---|---|---|
| duffing_forced (chaotic) | 0% / 1% / 5% | 0/3, 0/3, 0/3 | 0/10, 0/10, 0/10 |
| duffing_unforced (control) | 0% / 1% / 5% | 3/3, 0/3, 0/3 | 7/10, 9/10, 2/10 |

PySR totals: duffing_forced 0/9, duffing_unforced 3/9. gplearn totals: duffing_forced 0/30, duffing_unforced 18/30.

**Finding: partial replication — the reversal's direction holds, its magnitude does not fully transfer.** The chaotic side replicates exactly: PySR fails to recover duffing_forced at every noise level, matching gplearn's total failure. The control side replicates only at zero noise (PySR 3/3, a clean recovery matching gplearn's own non-degenerate zero-noise result); at both nonzero noise levels PySR recovers *zero* of 6 conservative-control conditions, while gplearn still recovers most of them (9/10 at 1%, 2/10 at 5% — gplearn's noise sensitivity is itself non-monotonic but never zero). This confirms the reversal's qualitative direction (chaos makes SR strictly harder than its own non-chaotic control) is not a gplearn-specific artifact — an independent search algorithm shows the same asymmetry — but it also shows PySR is far more noise-brittle than gplearn on this specific control system, under this one operator-set/hyperparameter configuration (`niterations=40`, `population_size=50`, no batching). That noise-brittleness is reported as a property of this PySR configuration on this problem, not generalized into a claim about SR's noise-robustness in general. Full detail: `experiments/main_study_results/pysr_crosscheck_duffing_results.json`, `DECISION_LOG.md` "PySR independent-implementation cross-check".
