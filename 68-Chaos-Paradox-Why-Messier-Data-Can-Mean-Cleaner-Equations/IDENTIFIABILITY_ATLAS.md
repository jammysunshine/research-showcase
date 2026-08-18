# Identifiability Atlas

Map of identifiability outcomes across all systems, regimes, methods, and observation operators tested.

## Legend

- **✓** = Recovered (coefficients within tolerance / one-step error below threshold)
- **✗** = Not recovered (coefficients wrong / one-step error above threshold)
- **—** = Not applicable (no coefficient-recovery concept, or structural mismatch by design)
- **n/a** = Not tested at this method/observation combination

## Tier A: SINDy, Full-State Observation (132 conditions)

### Logistic Map (degree=2 / degree=3)

| Regime | Type | 0% | 0.1% | 1% | 5% |
|---|---|---|---|---|---|
| period_2 | non-chaotic | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ |
| period_4 | periodic | ✓ / ✓ | ✓ / ✓ | ✓ / ✗ | ✗ / ✗ |
| period_3_window | periodic (in chaotic band) | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ |
| chaotic | chaotic | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ (5/5 / 3/5) |

### Lorenz (degree=2 / degree=3)

| Regime | Type | 0% | 0.1% | 1% | 5% |
|---|---|---|---|---|---|
| stable_fixed_point (ρ=14) | non-chaotic | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ |
| pre_chaotic (ρ=22) | non-chaotic | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ |
| near_onset (ρ=24.5) | post-onset | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| classic_chaotic (ρ=28) | chaotic | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ (5/5 / 3/5) |
| high_rho_chaotic (ρ=100) | hyperchaotic | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |

### Other Systems

| System | Regime | degree=2, noise=0–5% | degree=3, noise=0–5% |
|---|---|---|---|
| Harmonic oscillator | conservative | ✓ / ✓ / ✓ / ✓ | ✗ / ✗ / ✗ / ✗ (degrades) |
| Duffing unforced | conservative | ✗ / ✗ / ✗ / ✗ | ✓ / ✗ / ✗ / ✗ |
| Duffing forced | forced chaotic | — / — / — / — (library mismatch) | — / — / — / — |

## Tier B: Cross-Method Comparison (7 matched pairs × 3 noise × 2 degrees × 5 seeds)

### SINDy Coefficient Recovery (n/15 per regime, collapsed across noise × degree)

| Regime | Recovered / 15 |
|---|---|
| logistic period_2 (control) | 0/15 |
| logistic chaotic | 28/15 (13/15 at deg3) |
| lorenz stable_fixed_point (control) | 0/15 |
| lorenz classic_chaotic | 26/15 (11/15 at deg3) |
| harmonic conservative | 16/15 (1/15 at deg3) |
| duffing_unforced | 5/15 |
| duffing_forced | n/a |

### Symbolic Regression (gplearn, joint criterion: degree_ok AND NOT dynamically_distinct)

| Regime | Recovered / 30 |
|---|---|
| logistic chaotic | 30/30 |
| logistic period_2 (control) | 4/30 |
| lorenz classic_chaotic | 14/30 |
| lorenz stable_fixed_point (control) | 0/30 |
| duffing_unforced | 18/30 |
| duffing_forced (chaotic) | **0/30** ← reversal |

### Koopman/EDMD (one-step relative RMS error, lower = better)

| Regime pair | Chaotic error / Control error (noise=5%) | Ratio |
|---|---|---|
| logistic chaotic vs period_2 | 0.086 / 0.011–0.015 | **6–8x worse** |
| lorenz classic_chaotic vs stable_fixed_point | 0.036 / 0.0033 | **11x worse** |

**Koopman reversal**: chaotic regimes have systematically *higher* (worse) one-step prediction error than their non-chaotic controls. Survives Lyapunov normalization, pykoopman cross-check, NN Koopman, and Gram-matrix analysis.

## Tier C: Partial Observation — Delay-Embedded SINDy (210 conditions)

| Regime pair | Noise range | Primary gate (one-step): distinct? |
|---|---|---|
| All 7 pairs | {0%, 1%, 5%} | **No** — 0/210 dynamically_distinct |

**No chaos-vs-non-chaos gap detected** under single-coordinate delay-embedded observation across the full noise range.

## Held-Out Confirmations

### Rössler (SINDy, full-state, 80 fits)

| Regime | noise=0/0.1% | noise=1% | noise=5% |
|---|---|---|---|
| chaotic | ✓ (5/5) | ✓ (5/5) | ✓ (3/5 deg2, 2/5 deg3) |
| non-chaotic control (c=3.0) | ✓ (5/5) | ✓/✗ (5/5 deg2, 3/5 deg3) | ✗ (0/5 both) |

**Pattern replicates** on genuinely untouched family.

### Lorenz ρ=45.0 (SINDy, full-state, 40 fits)

| degree | noise=0/0.1/1% | noise=5% |
|---|---|---|
| 2 | ✓ (5/5) | ✓ (3/5) |
| 3 | ✓ (5/5) | ✓ (4/5) |

Comparable degradation to preregistered ρ=28; no cliff at untested parameter.

## Lorenz-96 (N=6) Extension (30 conditions)

| Method | F=8.0 (chaotic) | F=1.0 (control) |
|---|---|---|
| SINDy | ✓ 5/5 at all noise | ✗ 0/5 at all noise |
| Koopman/EDMD (one-step error) | 4–9x worse than control | baseline |

**Tier B's N≤3 reversal replicates at N=6.**

## Summary: Method × Observation Regime Matrix

| Method | Full-state, matched pairs | Partial observation (delay embed) |
|---|---|---|
| SINDy | ✓ Chaos aids (universal across tested systems) | ✗ No gap detected |
| Symbolic regression | ~ Chaos aids for logistic/Lorenz; reverses for Duffing | n/a (not tested) |
| Koopman/EDMD | ✗ **Reversal** — chaos hurts (6–11x worse) | n/a (not tested) |
| NN Koopman | ✗ **Reversal** — same direction as EDMD | n/a (not tested) |

## Counterexamples

| Construction | System | Status | Observation class |
|---|---|---|---|
| First-integral gauge freedom | Harmonic oscillator | ✓ CONFIRMED | Same orbit, different off-orbit VF |
| Coordinate conjugacy (tent/logistic) | Logistic map | ✗ REFUTED (failed attempt) | — |
| Finite-trajectory interpolation | Logistic map | ✓ CONFIRMED (but outside frozen model class) | Generic finite-data effect |
