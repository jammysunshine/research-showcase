# Counterexamples

**Verification status (independent critic pass, 2026-08-16):** the first-integral counterexample below is CONFIRMED — independently reproduced end to end, numbers match exactly, and the construction survived checks for tolerance-gaming and cherry-picked distance. The coordinate-transform/conjugacy counterexample below is **REFUTED as a section-6 counterexample and is UNVERIFIED / NEEDS REWORK** — see the "Verification verdict" box inside that section for the specific logical flaw. It is preserved here as a documented failed attempt (not deleted) per PROMPT.md's requirement to preserve disagreements and failed attempts; do not cite it as an established non-identifiability result until reworked or reframed.

**Follow-up (2026-08-16, STATUS.md item 5 close-out):** the failed conjugacy attempt above is superseded, not fixed — a genuinely different, section-6-compliant z-space counterexample was built directly (`src/counterexamples_conjugacy_v2.py`, "Finite-trajectory interpolation counterexample" below), numerically verified across 3 seeds, and honestly scoped as a generic finite-data effect outside the frozen degree<=3 polynomial model class (see that section's "Honesty / scope notes"). The refuted conjugacy section is left unmodified below per the preserve-failed-attempts rule.

## First-integral counterexample

**Source:** `src/counterexamples_first_integral.py`. Mechanism identified in `PRIOR_ART.md` section 5 (arXiv:2511.08860): a conserved quantity leaves a continuum of vector fields consistent with the same observed orbit. Verified against the equivalence definitions in `PREREGISTRATION.md` section 6.

### True system

Undamped harmonic oscillator, conserving `H = x^2 + y^2`:

```
dx/dt = y
dy/dt = -x
```

`dH/dt = 2x*(dx/dt) + 2y*(dy/dt) = 2xy - 2xy = 0` exactly, so every orbit is a circle `x^2+y^2 = H0` traced at constant angular speed 1.

### Adversarial alternative family

For any real `a != 0`, define

```
f_alt(x, y) = g(H) * (y, -x),   H = x^2 + y^2,   g(H) = 1 + a*(H - H0)
```

Because `g(H0) = 1` identically, `f_alt` equals the true field `(y, -x)` exactly (same direction, same magnitude) at every point on the observed level set `H = H0`. Off that level set, `f_alt` differs from the true field by the multiplicative factor `g(H) != 1`. Each choice of `a` gives a distinct vector field; `a` is a free parameter never constrained by data confined to `H = H0`, so this is a one-parameter continuum of observationally-equivalent-but-dynamically-distinct alternatives.

### Setup used for verification

- Observed trajectory: orbit started at `(x0, y0) = (1.3, 0)`, i.e. `H0 = 1.69`, integrated for 3 full periods (`2*pi` each), 400 sample points, `DOP853` integrator, `rtol=1e-12, atol=1e-14`.
- Alternative parameter: `a = 0.5`.

### Verification (a): matches observed trajectory

Alternative field integrated from the same initial condition, same time grid:

- RMSE between true and alternative trajectories: **5.039e-12** (integrator-precision floor, not a real discrepancy)
- Max absolute pointwise error: **1.191e-11**
- `H` conserved along both trajectories to std **~1e-12** / **~7e-12**

This is within the "vector-field-error-on-observed-data below the noise floor" clause of the section-6 observational-equivalence definition — the two systems are indistinguishable from this trajectory data.

### Verification (b): diverges off the observed level set

Vector field evaluated at 1,927 points sampled off the observed circle (radii in `[0.3*r0, 3*r0]`, excluding a band within 5% of `r0`), per section 6's "evaluated off the observed trajectory" clause:

- **Normalized vector-field L2 error: 425.7%** (threshold: > 10%)
- At a single confirmation point on a different energy level `H = 4*H0 = 6.76`: relative error **253.5%** (`g(H) = 3.535` there, vs. `g = 1` required for the true field)

Both figures far exceed the section-6 threshold (normalized VF L2 error > 10%), so per the preregistered definition the alternative is **dynamically distinct** despite being **observationally equivalent** on the training trajectory.

### Plain-language explanation

A finite trajectory of a conservative system only ever visits one level set of its conserved quantity `H`. Any modification to the vector field that (1) preserves the direction of motion and (2) preserves the *speed* exactly on that one level set is invisible to trajectory-matching, no matter how much data is collected along that orbit or how many periods are observed — more data on the same orbit does not shrink the ambiguity, because the extra samples still only constrain `g` at `H = H0`. The freedom lives entirely in how the vector field's magnitude varies with `H` *away* from the observed value, which trajectory data confined to a single orbit can never probe. This defeats naive trajectory-matching-based equation recovery (e.g., minimizing prediction error along the observed orbit) as a sufficient criterion for vector-field/equation recovery: a method could report near-zero training error while having selected a member of an infinite family of wrong equations. Recovering the correct `a = 0` member requires information transverse to the observed level set — e.g., trajectories at multiple distinct energies, or an explicit conservation-law prior — consistent with `PROMPT.md` task 6's expectation that known conservation laws are among the priors that can restore identifiability.

## Coordinate-transform / conjugacy counterexample — REFUTED / UNVERIFIED, NEEDS REWORK

> **Verification verdict (independent critic pass, 2026-08-16): REFUTED as a section-6 counterexample.** All numbers below were independently reproduced exactly (conjugacy identity max error 8.88e-16; OP1 diff 4.90e-9; naive-g normalized L2 25.99%; TV distances 0.211/0.016) and the identity `h(g(theta)) = f(h(theta))` was independently re-verified symbolically with sympy. The numbers are not in question — the logical structure is.
>
> **The flaw:** the identity plus `h` being a bijection on `[0,1]` means `f = h ∘ g ∘ h^-1` *exactly*. There is therefore no genuine second hypothesis about the observed variable `z`: the only mathematically legitimate way to "explain `z` via `g`" is to compose through `h` and `h^-1`, and that composition does not merely match `f` on the data — it **is** `f`, algebraically, to ~1e-16 precision (the code's own `control_l2_error_correct_f = 2.24e-17` confirms this). That candidate is not dynamically distinct in any sense; it is `f` in disguise.
>
> The only candidate the write-up calls "dynamically distinct" is stress test (b) — `z_hat_{n+1} = g(z_n)` applied *without* composing through `h` — but the write-up's own docstring calls this "wrong," "naive," a "mis-identification." That candidate fails section 6's observational-equivalence bar (26% error vs. the 10% threshold) at the outset, so it was never a valid observationally-equivalent alternative in the first place. Section 6 requires ONE alternative that is simultaneously (i) observationally equivalent and (ii) dynamically distinct off-trajectory; this construction never produces such an object. It offers one candidate that is equivalent-but-identical (not distinct) and a different, admittedly-wrong candidate that is distinct-but-not-equivalent, and the write-up equivocates between the two by calling both "the alternative g."
>
> Stress tests (a) and (c)'s "raw theta vs raw x" comparisons compound this: `theta` and `x` are two coordinate labels for the *same* trajectory point under the diffeomorphism `h` (like comparing degrees to radians), not two competing dynamical hypotheses evaluated in a shared coordinate. Consistent with this, `f` and `g` share the same Lyapunov exponent (`ln 2`) — the topological-conjugacy-invariant that section 6 itself lists as a distinctness criterion — and the write-up never reports it, likely because it would show zero distinction.
>
> **Net effect:** this is not "same observed data, two dynamically different explanations" (the identifiability gap section 6 targets). It restates that a diffeomorphic change of variables leaves a chaotic map's identity intact, packaged with a strawman "naive analyst forgot to invert h" error metric to manufacture an above-threshold number. **Action required before this can be cited as a result:** revise to either (a) find a genuine observationally-equivalent-and-distinct alternative, or (b) retitle/reframe as a weaker claim about coordinate-representation ambiguity, explicitly distinguished from genuine dynamical non-identifiability, and report the shared Lyapunov exponent.

**Source:** `src/counterexamples_conjugacy.py`. Mechanism identified in `PRIOR_ART.md` section 5: two maps/vector fields related by a smooth change of coordinates `h`, satisfying `h(g(theta)) = f(h(theta))`, push forward to *exactly* the same observed sequence under `h` — an exact, structural (not curve-fit) source of non-identifiability whenever the observation operator does not pin down which coordinate system the data live in. Verified against the equivalence definitions in `PREREGISTRATION.md` section 6.

### True system and alternative

- True system `f`: logistic map at `r = 4.0` (already in `src/simulators.py`), `f(x) = 4x(1-x)`.
- Alternative `g`: the tent map, `g(theta) = 1 - |2*theta - 1|` — a piecewise-linear map, structurally nothing like a smooth quadratic.
- Conjugacy `h(theta) = sin^2(pi*theta/2)`, with inverse `h^{-1}(x) = (2/pi)*arcsin(sqrt(x))`.

The identity `h(g(theta)) = f(h(theta))` holds for every `theta in [0,1]` (standard tent-map/logistic-map conjugacy). Verified numerically on 200,001 random points in `[0,1]`: **max abs error 8.88e-16, mean abs error 1.15e-16** — machine precision, i.e. exact.

### Observation operator OP1 ("public" coordinate only)

Observe `z_n`, the sequence produced by iterating `g` in its native `theta`-coordinate and reporting `z_n = h(theta_n)` (equivalently: just record `x_n` — they are mathematically the same number). Under OP1 the two systems are indistinguishable: with `x0 = 0.31415926`, `theta0 = h^{-1}(x0)`, 25 steps —

- **max |x_n - h(theta_n)| = 4.90e-9**, mean = 3.96e-10 (limited only by float64 rounding amplified at the system's Lyapunov rate `ln 2`; see diagnostic below, not by any residual mismatch in the identity itself).

A model built from the smooth quadratic recurrence `f` and a model built from the piecewise-linear recurrence `g` run in its own hidden coordinate and read out through `h` produce observationally identical data.

**Numerical caveat (reported, not hidden):** because both maps have Lyapunov exponent `ln 2 ≈ 0.693`, two float64 trajectories that are mathematically identical but computed via independently-rounded paths (iterate `f` directly vs. iterate `g` then apply `h`) separate at rate `~2^n`; `first_n_diff_exceeds_0.5 = 55` empirically vs. `52.15` predicted from `log2(0.5/1e-16)` — consistent with ordinary chaotic error amplification, not a flaw in the exact per-step identity (which stays at 8.88e-16 regardless of horizon). The OP1-equivalence check above therefore uses a short (25-step) horizon; the vector-field-error check below is unaffected since it only relies on the per-step identity, not on paired-trajectory agreement over a long horizon.

### Off OP1: dynamically distinct

**(a) Raw-coordinate divergence.** Comparing the two systems' *native* states directly (i.e., not applying `h`) — `x_n` from `f` vs. `theta_n` from `g` — over the same 25 steps: mean |diff| = **0.0620**, max = **0.104**. The two internal state sequences are simply different numbers; nothing subtle, this is the coordinate ambiguity made explicit.

**(b) Vector-field L2 error (naive mis-identification).** An analyst who observes `z_n` under OP1 and (plausibly, since `g` is a valid candidate) guesses the data obey `g` directly, predicting `z_hat_{n+1} = g(z_n)` instead of correctly composing through `h`: normalized-L2 one-step prediction error over 5,000 steps = **25.99%**, vs. section-6 threshold of 10% — **exceeds**. Control check (predicting with the correct `f(z_n)`): error **2.24e-17** (zero, confirming the harness is sound).

**(c) Invariant-measure divergence.** `f`'s invariant density is the arcsine law; `g`'s invariant density is uniform on `[0,1]`. Sampled via 50,000-realization ensembles (independent draws, 40-step burn-in each — burn-in kept below the ~52-55-step floating-point collapse horizon identified in the diagnostic above, since directly iterating the tent map for tens of thousands of steps in one long trajectory deterministically shifts mantissa bits out and collapses the orbit onto the fixed point 0, a known artifact of simulating expanding piecewise-linear maps in fixed precision):

- TV distance, `x` (from `f`) vs. raw `theta` (from `g`, no `h` applied): **0.211** — exceeds the section-6 0.10 threshold.
- TV distance, `x` (from `f`) vs. `z = h(theta)` (from `g`, correctly composed through `h`, i.e. under OP1): **0.016** — well within threshold, confirming the pushed-forward measure genuinely matches.

All three off-OP1 diagnostics independently confirm **dynamically distinct** per section 6, while the OP1 observation is **observationally equivalent** to machine precision.

### Plain-language explanation

A smooth (here, real-analytic) coordinate change lets a piecewise-linear, structurally alien map masquerade exactly as a smooth quadratic one, provided the observer only ever sees the map through that one fixed change of variables. No amount of additional data collected under OP1 — longer trajectories, more initial conditions, finer sampling — breaks the tie, because the tie is exact at every point of the state space, not a finite-sample coincidence: `h` is a full conjugacy, not a curve fit. The ambiguity is only broken by information the observation operator withholds: access to the *native* coordinate (raw internal state), or equivalently any observation function other than `h` composed with either system's natural readout. This mirrors the first-integral case above in structure — an exact symmetry (there, gauge freedom transverse to one level set; here, an unknown coordinate chart) produces an infinite-precision, unresolvable-by-more-data non-identifiability, distinct from ordinary noise-driven statistical uncertainty.

## Finite-trajectory interpolation counterexample

**Source:** `src/counterexamples_conjugacy_v2.py`. Built as Path A of STATUS.md item 5, after `src/counterexamples_conjugacy.py` above was independently found invalid as a section-6 counterexample. This is a different mechanism from both the first-integral case and the (failed) conjugacy attempt: it does not rely on any special structure of the logistic map or of chaos at all — it exploits the fact that a finite observed trajectory only ever visits finitely many points of the state space, leaving the map's value elsewhere unconstrained by the data.

### True system and alternative

- True system `f`: logistic map at `r = 4.0`, `f(z) = 4z(1-z)`.
- Observed data: a single finite noise-free trajectory `z_0, ..., z_N` (`N=3000`).
- Alternative `phi_alt`: a piecewise-linear interpolant built from the observed `(z_n, z_n+1)` pairs, sorted by `z_n`, PLUS `k=30` adversarial decoy nodes inserted at the midpoints of the 30 largest gaps between consecutive observed `z_n` values. Each decoy node `(z_mid, y_decoy)` sets `y_decoy = clip(1 - f(z_mid), 0, 1)` — deliberately far from the true value `f(z_mid)`.

Because every observed training pair remains a literal interpolation node, `phi_alt` reproduces the training data exactly. Between two real training nodes that straddle a decoy, `phi_alt` detours sharply away from `f`.

### Observational equivalence on the training trajectory

On-trajectory normalized vector-field L2 error is **exactly 0.0** at all three tested seeds (machine-precision interpolation match) — genuine observational equivalence, not a near-miss.

### Dynamically distinct off the training trajectory

Evaluated two ways per PREREGISTRATION.md section 6's "off the observed trajectory / on the confirmation trajectory" wording, at three independent (training IC, confirmation IC) seed pairs, identical construction parameters throughout (`N=3000`, `k=30`):

| seed | off-trajectory grid normalized L2 error | independent-IC confirmation-trajectory normalized L2 error |
|---|---|---|
| `x0=0.31415926` | 17.04% | 14.20% |
| `x0=0.123456` | 15.70% | 17.63% |
| `x0=0.9` | 16.81% | 17.24% |

All six numbers exceed the section-6 10% threshold; not a one-seed artifact.

### Honesty / scope notes

1. **Generic, not chaos-specific.** The mechanism — "a finite point set does not pin down an unconstrained function elsewhere" — holds for any finite dataset from any system, chaotic or not. Unlike the first-integral case (a specific conserved-quantity gauge freedom), this is a finite-sample/model-class-freedom argument, not a dynamics result.
2. **Outside the frozen model class.** `phi_alt` is a piecewise-linear spline, not a degree<=3 polynomial (PREREGISTRATION.md §2's frozen library), and is unreachable by SINDy/symbolic-regression/Koopman as configured in this project. Within the frozen degree<=3 polynomial class, no such alternative exists for a long chaotic logistic trajectory — the coefficients are pinned to the true quadratic by hundreds of training points, consistent with this project's own Tier A/B/C SINDy results. This construction is therefore evidence about an unconstrained nonparametric hypothesis class, not a threat to the polynomial-library discovery methods this project benchmarks.
3. **Adversarially constructed**, per PREREGISTRATION.md section 6's "either found by a discovery method or constructed adversarially" clause — not found by any discovery method run in this project.
