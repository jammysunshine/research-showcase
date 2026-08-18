# Preregistration

Frozen before any equation-discovery-method results are examined. Changes after this point must be logged in `DECISION_LOG.md` with a note that they occurred after seeing results (post-hoc), per PROMPT.md's claim-discipline requirement.

## 1. System families (pilot set; expand toward 50+ regimes only after pilot passes)

| System | Type | Parameter regimes | First integral? |
|---|---|---|---|
| Logistic map | 1D map | r=3.2 (period-2), r=3.5 (period-4), r=3.83 (period-3 window), r=4.0 (fully chaotic) | no |
| Lorenz | 3D ODE, dissipative | sigma=10, beta=8/3, rho in {14 (stable fixed point), 22 (transient chaos), 24.5 (near onset), 28 (classic chaotic), 100 (hyperchaotic)} | no |
| Harmonic oscillator | 2D ODE, conservative | fixed frequency, varying amplitude | yes (energy) — non-identifiable control case |
| Duffing oscillator | 2D/3D ODE (forced) | unforced (conservative) vs. forced-chaotic regimes | yes when unforced |

Held out for confirmation (untouched until frozen main study concludes): one system family not used during pilot/method-development (candidate: Rössler system or a coordinate-transformed Lorenz) plus one held-out parameter regime per developed family.

## 2. Function libraries / model classes

- **SINDy library:** polynomial terms up to degree 3 (matches true Lorenz/logistic/Duffing degree), optionally trigonometric terms for oscillators. Library size matched across compared methods where feasible.
- **Symbolic regression:** PySR/gplearn search space restricted to +, -, *, /, and the same low-degree polynomial primitives, to keep search coverage comparable to the SINDy library rather than open-ended.
- **Koopman/EDMD:** dictionary of monomials up to matched degree, or radial-basis-function dictionary as a secondary configuration.

## 3. Observation operators

- **Full-state, noise-free** (feasibility/upper-bound case).
- **Full-state, additive Gaussian noise** at sigma in {0.1%, 1%, 5%} of state standard deviation.
- **Partial observation:** single coordinate only (e.g., x(t) of Lorenz), forcing delay-embedding or hidden-variable reconstruction — a designed non-identifiability stress case.

## 4. Sample / noise regimes

- Trajectory lengths: short (N=500 samples), medium (N=5,000), long (N=50,000), at a fixed sampling interval chosen relative to each system's dominant timescale (documented per system in `experiments/`).
- Each (system, regime, noise, length) combination run across >=5 seeds (independent initial conditions / noise draws).

## 5. Train / confirmation split

- **Train trajectory:** used for equation discovery (fitting SINDy/symbolic-regression/Koopman models).
- **Confirmation trajectory:** independent initial condition, same system/parameters, untouched during method development; used only to evaluate off-trajectory generalization.
- **Confirmation system family:** entirely separate system (see held-out set above), touched only once per completed method, at the falsification/independent-replication gate.

## 6. Identifiability / equivalence definitions

A discovered model is **recovered within model class** if its active-term structure matches the true equation and its coefficients are within a preregistered tolerance (5% relative error, or absolute error < 0.05 for near-zero true coefficients).

Two models are **observationally equivalent** on a dataset if both achieve vector-field-error-on-observed-data below the noise floor, but they are **dynamically distinct** if any of the following exceeds threshold when evaluated off the observed trajectory / on the confirmation trajectory:
- Vector-field L2 error (normalized) > 10%
- Largest-Lyapunov-exponent estimate differs by > 0.1 (absolute)
- Invariant-measure (histogram/KDE over attractor) total-variation distance > 0.1
- Bifurcation location (e.g., period-doubling onset parameter) differs by > 5%

The **identifiability gap** for a (system, regime) is confirmed if an observationally-equivalent-but-dynamically-distinct alternative is exhibited (either found by a discovery method or constructed adversarially).

## 7. Baselines

- SINDy (PySINDy, sequential thresholded least squares) is the trusted primary comparator (see DECISION_LOG.md).
- Naive polynomial least-squares fit without sparsity (no thresholding) as a "no prior" baseline showing the sparsity assumption's contribution.

## 8. Metrics

Primary: vector-field error off-trajectory (normalized L2) on the confirmation trajectory. Secondary: coefficient recovery error, Lyapunov-spectrum error, invariant-measure divergence, bifurcation-structure error — all logged even when not the headline metric (PROMPT.md: "report all preregistered outcomes").

## 9. Seeds and statistical treatment

- Seeds fixed and recorded per experiment config in `experiments/`.
- Report mean +/- std across seeds; treat any single-seed "success" as anecdotal, not a claim.
- No multiple-comparison correction is applied for exploratory pilot sweeps; the frozen main study (post-pilot) will prespecify the exact comparator set and apply a Bonferroni-style correction across its fixed set of headline hypothesis tests before claiming significance.

## 10. Claim thresholds

- **Candidate contribution (chaos-helps-identifiability confirmed for a system):** matched chaotic regime achieves recovered-within-model-class on train AND confirmation trajectory across >=4/5 seeds, while matched non-chaotic regime (same system, same library, same noise) fails to do so on >=3/5 seeds.
- **Counterexample confirmed:** an explicit adversarial alternative is exhibited that is observationally equivalent on the train (and ideally confirmation) trajectory but dynamically distinct per §6, with the construction documented and independently checkable (exact analytic construction preferred; numerical construction with residual reported otherwise).

## 11. Blinded/reserved regimes

The held-out confirmation system family (§1) and its parameter regime are not inspected, plotted, or discussed until the frozen main study's falsification stage. Any accidental exposure will be logged in `DECISION_LOG.md` and the regime replaced.
