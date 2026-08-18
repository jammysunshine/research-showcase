"""Tier A main-study runner (MAIN_STUDY_DESIGN.md SS2/SS4 step 4/5).

Full 12-regime grid, SINDy only (the trusted primary comparator per
PREREGISTRATION.md SS7), at full-state noise levels {0%, 0.1%, 1%, 5%},
medium length (N=5000, t_end/n_steps chosen per system to give ~5000
post-transient samples), 5 seeds, both degree-2 and degree-3 polynomial
libraries. A short (N=500) / long (N=50000) sensitivity sub-sweep runs
SINDy-only at {0%, 1%} noise, per MAIN_STUDY_DESIGN.md SS2 Tier A.

Regimes (12): logistic {r=3.2, 3.5, 3.83, 4.0}; Lorenz {rho=14, 22, 24.5,
28, 100}; harmonic oscillator (1 conservative regime); Duffing {unforced
(conservative), forced-chaotic}.

Every run wires in, alongside the existing coefficient-recovery and
off-trajectory VF-error metrics carried over from
experiments/pilot_chaos_vs_periodic.py (MAIN_STUDY_DESIGN.md SS4 steps
2a/2b/3's completion criteria):

  - Naive (non-sparse) least-squares baseline comparator
    (src/discovery_naive_baseline.py, PREREGISTRATION.md SS7) --
    n_nonzero coefficients on the identical polynomial library, showing
    what SINDy's sparsity assumption buys over "no prior."
  - Largest-Lyapunov-exponent error between the true system and the
    fitted SINDy model (src/metrics_dynamical.py, PREREGISTRATION.md SS6/SS8),
    via a finite-difference Jacobian of the fitted polynomial model (no
    fitted model has an analytic Jacobian available).
  - Invariant-measure (per-dimension marginal-histogram) TV distance
    between the true and a model-simulated trajectory, same module.
  - Off-attractor evaluation-grid VF error (DECISION_LOG.md "Off-attractor
    evaluation grid", pilot-validated fix for periodic/point-attractor
    regimes).

NOT wired into Tier A (logged here, not silently dropped):
  - `bifurcation_location_error` (src/metrics_dynamical.py) requires a
    PARAMETRIC FAMILY of fitted models swept across a parameter range
    (discovered_system_family(param) -> map). Tier A fits one regime
    (one fixed parameter value) at a time; producing a fitted-model
    family across the logistic map's r would require its own dedicated
    sweep, out of scope for this pass. Flagged as future work, consistent
    with MAIN_STUDY_DESIGN.md SS5's "not silently dropped" standard.

Duffing forced-chaotic library-mismatch note: the true forced term is
`gamma*cos(phi)`, which is NOT representable in a degree-3 polynomial
library (PREREGISTRATION.md SS2's frozen library). SINDy is EXPECTED to
fail exact coefficient recovery on this regime regardless of chaos --
this is a genuine, informative library-mismatch test case, not a bug.
Its VF-error/Lyapunov metrics (computed on the x,v subsystem only) are
the informative outputs for this regime, not coefficient recovery.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pysindy as ps

N_WORKERS = 6  # leave headroom on this 10-core/16GB Mac (PROJECT_CHARTER.md local-only ceiling)

from src.discovery_naive_baseline import fit_naive_polynomial
from src.metrics_dynamical import (
    DynamicalSystem,
    invariant_measure_tv_distance,
    lyapunov_spectrum_error,
)
from src.simulators import (
    DEFAULT_DUFFING_FORCED_PARAMS,
    DEFAULT_DUFFING_UNFORCED_PARAMS,
    DEFAULT_LORENZ_PARAMS,
    duffing_forced_trajectory,
    duffing_unforced_trajectory,
    harmonic_trajectory,
    logistic_map,
    logistic_trajectory,
    lorenz_rhs,
    lorenz_trajectory,
)

SEEDS = [0, 1, 2, 3, 4]
COEFF_TOL = 0.05  # relative, per PREREGISTRATION.md SS6
NOISE_LEVELS_FULL = [0.0, 0.001, 0.01, 0.05]  # fraction of state std, PREREGISTRATION.md SS4
NOISE_LEVELS_SENSITIVITY = [0.0, 0.01]
LIBRARY_DEGREES = [2, 3]
VF_ERR_TOL = 0.10  # normalized L2 off-trajectory VF error, PREREGISTRATION.md SS6
N_OFF_ATTRACTOR_GRID_POINTS = 500
# Multiplier applied to a trajectory's own per-dimension max |state| to build
# an off-attractor sampling box for systems whose single trajectory only
# covers one energy shell/orbit (harmonic, duffing_unforced) -- replaces the
# previously hand-picked round numbers (+/-3.0, +/-2.0) flagged in
# LIMITATIONS.md #7 as not derived from each system's natural amplitude
# scale. See DECISION_LOG.md "Off-attractor grid bounds derived from
# amplitude scale".
OFF_ATTRACTOR_GRID_SCALE = 3.0
STLSQ_THRESHOLD = 0.1  # DECISION_LOG.md "Discovery-method hyperparameters"

# Medium length per system: chosen so the post-transient tail has ~5000 samples.
MEDIUM = dict(logistic_n_steps=5500, lorenz_n_points=25000, lorenz_t_end=50.0,
              harmonic_n_points=5000, harmonic_t_end=50.0,
              duffing_n_points=25000, duffing_t_end=200.0)
SHORT = dict(logistic_n_steps=1000, lorenz_n_points=2500, lorenz_t_end=5.0)
LONG = dict(logistic_n_steps=55000, lorenz_n_points=250000, lorenz_t_end=500.0)


def _confirmation_offset(seed):
    return seed + 10_000


def _grid_offset(seed):
    return seed + 20_000


def _fd_jacobian(f, dim, eps=1e-6):
    """Finite-difference Jacobian wrapper for a fitted model with no analytic derivative."""
    def jac(state):
        state = np.asarray(state, dtype=float)
        f0 = f(state)
        J = np.zeros((dim, dim))
        for i in range(dim):
            perturbed = state.copy()
            perturbed[i] += eps
            J[:, i] = (f(perturbed) - f0) / eps
        return J
    return jac


def _lyapunov_error_ode(model, true_rhs, true_jac, dim, x0, n_steps=400, transient=150, dt=0.01):
    """Largest-LE error between the true ODE and a fitted SINDy model, via
    a finite-difference Jacobian of model.predict (no analytic Jacobian
    exists for an arbitrary fitted polynomial).

    src/simulators.py's generic_ode_lyapunov_spectrum calls solve_ivp once
    PER STEP at hardcoded rtol=1e-10/atol=1e-11 (frozen trusted-simulator
    tolerances, not overridden here since that module is the trusted
    comparator per PREREGISTRATION.md SS7) -- a single dim=3 call at
    n_steps=1500/transient=500 measured ~50s locally. At PROJECT_CHARTER.md's
    local-only compute ceiling this does not scale to Tier A's ~420 ODE
    conditions. n_steps/transient are cut well below
    src/metrics_dynamical.py's own defaults (5000/2000) accordingly; this
    trades exponent precision for tractability on what is a SECONDARY,
    confirmatory metric (PREREGISTRATION.md SS6/SS8) -- coefficient recovery
    and VF error, the primary identifiability metrics, are unaffected and
    still computed at full length/seed count. See DECISION_LOG.md "Tier A
    Lyapunov-error compute scoping".
    """
    def disc_rhs(t, state):
        return model.predict(state.reshape(1, -1))[0]

    disc_jac = _fd_jacobian(lambda s: disc_rhs(0.0, s), dim)
    true_sys = DynamicalSystem(kind="ode", rhs=true_rhs, jacobian=true_jac, dim=dim)
    disc_sys = DynamicalSystem(kind="ode", rhs=disc_rhs, jacobian=disc_jac, dim=dim)

    # generic_ode_lyapunov_spectrum (src/simulators.py, frozen trusted
    # comparator) integrates the discovered model at rtol=1e-10/atol=1e-11
    # with no step-size floor. A norm-based blow-up guard (as used in
    # _simulate_model_trajectory) does NOT catch this: a badly-fit model can
    # stay bounded yet be locally stiff/near-singular, which makes the
    # adaptive per-step solver crawl with vanishing step sizes even though
    # the trajectory never exceeds any norm threshold. This is a distinct
    # failure mode from the blow-up hang fixed earlier (see DECISION_LOG.md
    # "Tier A divergent-model blow-up guard") and stalled the second Tier A
    # run for 26+ min on a bounded-but-stiff Lorenz model (see DECISION_LOG.md
    # "Tier A Lyapunov-error stiffness timeout"). A hard wall-clock timeout
    # on just this secondary/confirmatory metric (PREREGISTRATION.md SS6/SS8)
    # guarantees no hang regardless of root cause; coefficient recovery and
    # VF error (the primary metrics) are computed before this call and are
    # unaffected by a timeout here.
    import signal

    def _on_timeout(signum, frame):
        raise TimeoutError("lyapunov_error computation exceeded 60s wall-clock budget")

    old_handler = signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(60)
    try:
        return lyapunov_spectrum_error(true_sys, disc_sys, x0, n_steps=n_steps,
                                        transient=transient, dt=dt)
    except TimeoutError as e:
        return f"lyapunov_error_skipped: {e}"
    except Exception as e:  # noqa: BLE001 - report, don't fabricate a number
        return f"lyapunov_error_failed: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _blowup_event(t, y):
    return 1e4 - np.linalg.norm(y)


_blowup_event.terminal = True
_blowup_event.direction = -1


def _simulate_model_trajectory(model, x0, t_span, n_points):
    """Roll out the fitted SINDy model's own vector field for the invariant-measure check.

    A poorly-fit polynomial model can have a finite-time blow-up (e.g. an
    unconstrained cubic term with the wrong sign) — solve_ivp's adaptive
    stepper crawls with vanishing step sizes trying to resolve that
    singularity rather than failing fast, which stalled the first Tier A
    run for 40+ minutes on a single condition (see DECISION_LOG.md "Tier A
    divergent-model blow-up guard"). A terminal event that stops
    integration once the state norm exceeds 1e4 turns that hang into an
    immediate, informative `sol.success=False` (rolled up as
    `invariant_measure_tv=None`) instead.
    """
    from scipy.integrate import solve_ivp

    def rhs(t, state):
        return model.predict(state.reshape(1, -1))[0]

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8,
                     events=_blowup_event, max_step=(t_span[1] - t_span[0]) / 20.0)
    if not sol.success or sol.t[-1] < t_span[1]:
        return None
    return sol.y.T


# ---------------------------------------------------------------------------
# Logistic map (1D discrete)
# ---------------------------------------------------------------------------

def fit_logistic(r, seed, noise_frac, degree, n_steps):
    rng = np.random.default_rng(seed)
    x0 = 0.1 + 0.3 * rng.random()
    traj = logistic_trajectory(x0, r=r, n_steps=n_steps, transient=500)
    if noise_frac > 0:
        traj = traj + rng.normal(0, noise_frac * traj.std(), size=traj.shape)

    x_n = traj[:-1]
    x_np1 = traj[1:]
    A = np.stack([x_n ** k for k in range(degree + 1)], axis=1)
    coef, *_ = np.linalg.lstsq(A, x_np1, rcond=None)
    c0 = coef[0]
    r_hat_lin = coef[1]
    r_hat_quad = -coef[2]
    err_terms = [abs(c0), abs(r_hat_lin - r) / r, abs(r_hat_quad - r) / r]
    if degree >= 3:
        err_terms.append(abs(coef[3]))
    err = max(err_terms)

    naive = fit_naive_polynomial(traj, degree=degree, discrete=True, feature_names=["x"])

    conf_rng = np.random.default_rng(_confirmation_offset(seed))
    x0_conf = 0.1 + 0.3 * conf_rng.random()
    traj_conf = logistic_trajectory(x0_conf, r=r, n_steps=n_steps, transient=500)
    if noise_frac > 0:
        traj_conf = traj_conf + conf_rng.normal(0, noise_frac * traj_conf.std(), size=traj_conf.shape)
    x_conf = traj_conf[:-1]
    true_vf = np.array([logistic_map(x, r) for x in x_conf])
    A_conf = np.stack([x_conf ** k for k in range(degree + 1)], axis=1)
    pred_vf = A_conf @ coef
    vf_err = np.linalg.norm(pred_vf - true_vf) / max(np.linalg.norm(true_vf), 1e-300)

    grid_rng = np.random.default_rng(_grid_offset(seed))
    x_grid = grid_rng.uniform(0.0, 1.0, size=N_OFF_ATTRACTOR_GRID_POINTS)
    true_vf_grid = np.array([logistic_map(x, r) for x in x_grid])
    A_grid = np.stack([x_grid ** k for k in range(degree + 1)], axis=1)
    pred_vf_grid = A_grid @ coef
    vf_err_grid = np.linalg.norm(pred_vf_grid - true_vf_grid) / max(np.linalg.norm(true_vf_grid), 1e-300)

    def disc_map(x):
        return float(np.array([x ** k for k in range(degree + 1)]) @ coef)

    def disc_jac(x):
        return float(sum(k * coef[k] * x ** (k - 1) for k in range(1, degree + 1)))

    true_sys = DynamicalSystem(kind="map", rhs=lambda x: logistic_map(x, r),
                                jacobian=lambda x: r * (1.0 - 2.0 * x), dim=1)
    disc_sys = DynamicalSystem(kind="map", rhs=disc_map, jacobian=disc_jac, dim=1)
    try:
        lyap_err = lyapunov_spectrum_error(true_sys, disc_sys, x0=0.4, n_steps=5000, transient=1000)
    except Exception as e:  # noqa: BLE001
        lyap_err = f"lyapunov_error_failed: {e}"

    disc_traj = [x0]
    x = x0
    diverged = False
    for _ in range(len(traj) - 1):
        x = disc_map(x)
        if not np.isfinite(x) or abs(x) > 1e6:
            diverged = True
            break
        disc_traj.append(x)
    tv_dist = None if diverged else invariant_measure_tv_distance(traj, np.array(disc_traj))

    return dict(seed=seed, r=r, noise_frac=noise_frac, degree=degree,
                c0=float(c0), c1=float(r_hat_lin), c2=float(r_hat_quad), max_rel_err=float(err),
                recovered=bool(err < COEFF_TOL),
                vf_l2_err_confirmation=float(vf_err), dynamically_distinct=bool(vf_err > VF_ERR_TOL),
                vf_l2_err_off_attractor_grid=float(vf_err_grid),
                dynamically_distinct_off_attractor=bool(vf_err_grid > VF_ERR_TOL),
                naive_n_nonzero=naive["n_nonzero"], naive_n_features=len(naive["feature_names"]),
                lyapunov_error=lyap_err, invariant_measure_tv=tv_dist)


# ---------------------------------------------------------------------------
# Lorenz (3D ODE)
# ---------------------------------------------------------------------------

def fit_lorenz(rho, seed, noise_frac, degree, n_points, t_end, transient_frac=0.5):
    rng = np.random.default_rng(seed)
    x0 = np.array([-8.0, 8.0, 27.0]) + rng.normal(0, 0.5, size=3)
    params = dict(sigma=10.0, rho=rho, beta=8.0 / 3.0)
    t, states = lorenz_trajectory(x0, t_span=(0, t_end), n_points=n_points, params=params)
    if noise_frac > 0:
        state_std = states.std(axis=0)
        states = states + rng.normal(0, noise_frac * state_std, size=states.shape)

    n_discard = int(len(t) * transient_frac)
    t_tail, states_tail = t[n_discard:], states[n_discard:]
    dt = t_tail[1] - t_tail[0]

    model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=degree),
                      optimizer=ps.STLSQ(threshold=STLSQ_THRESHOLD))
    model.fit(states_tail, t=dt, feature_names=["x", "y", "z"])
    coeffs = model.coefficients()
    names = model.get_feature_names()
    idx = {n: i for i, n in enumerate(names)}
    sigma_hat, rho_hat, beta_hat = -coeffs[0][idx["x"]], coeffs[1][idx["x"]], -coeffs[2][idx["z"]]
    err = max(abs(sigma_hat - params["sigma"]) / params["sigma"],
              abs(rho_hat - params["rho"]) / max(abs(params["rho"]), 1e-9),
              abs(beta_hat - params["beta"]) / params["beta"])

    naive = fit_naive_polynomial(states_tail, dt=dt, degree=degree, feature_names=["x", "y", "z"])

    conf_rng = np.random.default_rng(_confirmation_offset(seed))
    x0_conf = np.array([-8.0, 8.0, 27.0]) + conf_rng.normal(0, 0.5, size=3)
    t_conf, states_conf = lorenz_trajectory(x0_conf, t_span=(0, t_end), n_points=n_points, params=params)
    if noise_frac > 0:
        states_conf = states_conf + conf_rng.normal(0, noise_frac * states_conf.std(axis=0), size=states_conf.shape)
    states_conf_tail = states_conf[n_discard:]
    true_vf = np.array([lorenz_rhs(0.0, s, params["sigma"], params["rho"], params["beta"]) for s in states_conf_tail])
    pred_vf = model.predict(states_conf_tail)
    vf_err = np.linalg.norm(pred_vf - true_vf) / max(np.linalg.norm(true_vf), 1e-300)

    grid_rng = np.random.default_rng(_grid_offset(seed))
    lo, hi = states.min(axis=0), states.max(axis=0)
    grid_pts = grid_rng.uniform(lo, hi, size=(N_OFF_ATTRACTOR_GRID_POINTS, 3))
    true_vf_grid = np.array([lorenz_rhs(0.0, s, params["sigma"], params["rho"], params["beta"]) for s in grid_pts])
    pred_vf_grid = model.predict(grid_pts)
    vf_err_grid = np.linalg.norm(pred_vf_grid - true_vf_grid) / max(np.linalg.norm(true_vf_grid), 1e-300)

    # Lyapunov-error / invariant-measure TV computed on seed 0 only, per the
    # compute-scoping decision in _lyapunov_error_ode's docstring / DECISION_LOG.md
    # "Tier A Lyapunov-error compute scoping" -- coefficient recovery and VF
    # error (the primary metrics) still run on all 5 seeds above.
    if seed == 0:
        true_jac = lambda s: np.array([  # noqa: E731
            [-params["sigma"], params["sigma"], 0.0],
            [params["rho"] - s[2], -1.0, -s[0]],
            [s[1], s[0], -params["beta"]],
        ])
        lyap_err = _lyapunov_error_ode(model, lambda t, s: lorenz_rhs(t, s, **params), true_jac, 3, states_tail[0])
        model_traj = _simulate_model_trajectory(model, states_tail[0], (0, min(t_end, 20.0)), 2000)
        tv_dist = (invariant_measure_tv_distance(states_tail, model_traj)
                   if model_traj is not None else None)
    else:
        lyap_err = "skipped_non_representative_seed"
        tv_dist = None

    return dict(seed=seed, rho=rho, noise_frac=noise_frac, degree=degree,
                sigma_hat=float(sigma_hat), rho_hat=float(rho_hat), beta_hat=float(beta_hat),
                max_rel_err=float(err), recovered=bool(err < COEFF_TOL),
                vf_l2_err_confirmation=float(vf_err), dynamically_distinct=bool(vf_err > VF_ERR_TOL),
                vf_l2_err_off_attractor_grid=float(vf_err_grid),
                dynamically_distinct_off_attractor=bool(vf_err_grid > VF_ERR_TOL),
                naive_n_nonzero=naive["n_nonzero"], naive_n_features=len(naive["feature_names"]) * 3,
                lyapunov_error=lyap_err, invariant_measure_tv=tv_dist)


# ---------------------------------------------------------------------------
# Harmonic oscillator (2D ODE, conservative)
# ---------------------------------------------------------------------------

def fit_harmonic(seed, noise_frac, degree, n_points, t_end):
    rng = np.random.default_rng(seed)
    x0 = np.array([1.0, 0.0]) + rng.normal(0, 0.1, size=2)
    t, states = harmonic_trajectory(x0, t_span=(0, t_end), n_points=n_points, omega=1.0)
    if noise_frac > 0:
        states = states + rng.normal(0, noise_frac * states.std(axis=0), size=states.shape)
    dt = t[1] - t[0]

    model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=degree),
                      optimizer=ps.STLSQ(threshold=STLSQ_THRESHOLD))
    model.fit(states, t=dt, feature_names=["x", "v"])
    coeffs = model.coefficients()
    names = model.get_feature_names()
    idx = {n: i for i, n in enumerate(names)}
    # True: dx/dt = v (coeff 1 on "v" in row 0); dv/dt = -x (coeff -1 on "x" in row 1).
    v_coef_hat = coeffs[0][idx["v"]]
    x_coef_hat = coeffs[1][idx["x"]]
    err = max(abs(v_coef_hat - 1.0), abs(x_coef_hat - (-1.0)))

    naive = fit_naive_polynomial(states, dt=dt, degree=degree, feature_names=["x", "v"])

    conf_rng = np.random.default_rng(_confirmation_offset(seed))
    x0_conf = np.array([1.0, 0.0]) + conf_rng.normal(0, 0.1, size=2)
    t_conf, states_conf = harmonic_trajectory(x0_conf, t_span=(0, t_end), n_points=n_points, omega=1.0)
    if noise_frac > 0:
        states_conf = states_conf + conf_rng.normal(0, noise_frac * states_conf.std(axis=0), size=states_conf.shape)
    true_vf = np.array([[s[1], -s[0]] for s in states_conf])
    pred_vf = model.predict(states_conf)
    vf_err = np.linalg.norm(pred_vf - true_vf) / max(np.linalg.norm(true_vf), 1e-300)

    # Off-attractor grid: harmonic orbits fill a family of nested ellipses
    # (one per energy level), so a wider-amplitude annulus genuinely extends
    # beyond this specific trajectory's energy shell.
    grid_rng = np.random.default_rng(_grid_offset(seed))
    amp_scale = np.abs(states).max(axis=0)
    grid_pts = grid_rng.uniform(-OFF_ATTRACTOR_GRID_SCALE * amp_scale,
                                 OFF_ATTRACTOR_GRID_SCALE * amp_scale,
                                 size=(N_OFF_ATTRACTOR_GRID_POINTS, 2))
    true_vf_grid = np.array([[s[1], -s[0]] for s in grid_pts])
    pred_vf_grid = model.predict(grid_pts)
    vf_err_grid = np.linalg.norm(pred_vf_grid - true_vf_grid) / max(np.linalg.norm(true_vf_grid), 1e-300)

    if seed == 0:
        true_jac = lambda s: np.array([[0.0, 1.0], [-1.0, 0.0]])  # noqa: E731
        lyap_err = _lyapunov_error_ode(model, lambda t, s: np.array([s[1], -s[0]]), true_jac, 2, states[0])
        model_traj = _simulate_model_trajectory(model, states[0], (0, t_end), n_points)
        tv_dist = invariant_measure_tv_distance(states, model_traj) if model_traj is not None else None
    else:
        lyap_err = "skipped_non_representative_seed"
        tv_dist = None

    return dict(seed=seed, noise_frac=noise_frac, degree=degree,
                v_coef_hat=float(v_coef_hat), x_coef_hat=float(x_coef_hat), max_rel_err=float(err),
                recovered=bool(err < COEFF_TOL),
                vf_l2_err_confirmation=float(vf_err), dynamically_distinct=bool(vf_err > VF_ERR_TOL),
                vf_l2_err_off_attractor_grid=float(vf_err_grid),
                dynamically_distinct_off_attractor=bool(vf_err_grid > VF_ERR_TOL),
                naive_n_nonzero=naive["n_nonzero"], naive_n_features=len(naive["feature_names"]) * 2,
                lyapunov_error=lyap_err, invariant_measure_tv=tv_dist)


# ---------------------------------------------------------------------------
# Duffing unforced (2D ODE, conservative)
# ---------------------------------------------------------------------------

def fit_duffing_unforced(seed, noise_frac, degree, n_points, t_end):
    alpha = DEFAULT_DUFFING_UNFORCED_PARAMS["alpha"]
    beta = DEFAULT_DUFFING_UNFORCED_PARAMS["beta"]
    rng = np.random.default_rng(seed)
    x0 = np.array([0.5, 0.0]) + rng.normal(0, 0.05, size=2)
    t, states = duffing_unforced_trajectory(x0, t_span=(0, t_end), n_points=n_points)
    if noise_frac > 0:
        states = states + rng.normal(0, noise_frac * states.std(axis=0), size=states.shape)
    dt = t[1] - t[0]

    model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=degree),
                      optimizer=ps.STLSQ(threshold=STLSQ_THRESHOLD))
    model.fit(states, t=dt, feature_names=["x", "v"])
    coeffs = model.coefficients()
    names = model.get_feature_names()
    idx = {n: i for i, n in enumerate(names)}
    v_coef_hat = coeffs[0][idx["v"]]
    x_coef_hat = coeffs[1][idx["x"]]
    x3_coef_hat = coeffs[1][idx["x^3"]] if "x^3" in idx else 0.0
    err = max(abs(v_coef_hat - 1.0), abs(x_coef_hat - (-alpha)), abs(x3_coef_hat - (-beta)))

    naive = fit_naive_polynomial(states, dt=dt, degree=degree, feature_names=["x", "v"])

    conf_rng = np.random.default_rng(_confirmation_offset(seed))
    x0_conf = np.array([0.5, 0.0]) + conf_rng.normal(0, 0.05, size=2)
    t_conf, states_conf = duffing_unforced_trajectory(x0_conf, t_span=(0, t_end), n_points=n_points)
    if noise_frac > 0:
        states_conf = states_conf + conf_rng.normal(0, noise_frac * states_conf.std(axis=0), size=states_conf.shape)
    true_vf = np.array([[s[1], -alpha * s[0] - beta * s[0] ** 3] for s in states_conf])
    pred_vf = model.predict(states_conf)
    vf_err = np.linalg.norm(pred_vf - true_vf) / max(np.linalg.norm(true_vf), 1e-300)

    grid_rng = np.random.default_rng(_grid_offset(seed))
    amp_scale = np.abs(states).max(axis=0)
    grid_pts = grid_rng.uniform(-OFF_ATTRACTOR_GRID_SCALE * amp_scale,
                                 OFF_ATTRACTOR_GRID_SCALE * amp_scale,
                                 size=(N_OFF_ATTRACTOR_GRID_POINTS, 2))
    true_vf_grid = np.array([[s[1], -alpha * s[0] - beta * s[0] ** 3] for s in grid_pts])
    pred_vf_grid = model.predict(grid_pts)
    vf_err_grid = np.linalg.norm(pred_vf_grid - true_vf_grid) / max(np.linalg.norm(true_vf_grid), 1e-300)

    if seed == 0:
        true_jac = lambda s: np.array([[0.0, 1.0], [-alpha - 3 * beta * s[0] ** 2, 0.0]])  # noqa: E731
        lyap_err = _lyapunov_error_ode(
            model, lambda t, s: np.array([s[1], -alpha * s[0] - beta * s[0] ** 3]), true_jac, 2, states[0])
        model_traj = _simulate_model_trajectory(model, states[0], (0, t_end), n_points)
        tv_dist = invariant_measure_tv_distance(states, model_traj) if model_traj is not None else None
    else:
        lyap_err = "skipped_non_representative_seed"
        tv_dist = None

    return dict(seed=seed, noise_frac=noise_frac, degree=degree,
                v_coef_hat=float(v_coef_hat), x_coef_hat=float(x_coef_hat), x3_coef_hat=float(x3_coef_hat),
                max_rel_err=float(err), recovered=bool(err < COEFF_TOL),
                vf_l2_err_confirmation=float(vf_err), dynamically_distinct=bool(vf_err > VF_ERR_TOL),
                vf_l2_err_off_attractor_grid=float(vf_err_grid),
                dynamically_distinct_off_attractor=bool(vf_err_grid > VF_ERR_TOL),
                naive_n_nonzero=naive["n_nonzero"], naive_n_features=len(naive["feature_names"]) * 2,
                lyapunov_error=lyap_err, invariant_measure_tv=tv_dist)


# ---------------------------------------------------------------------------
# Duffing forced-chaotic (3D autonomous-embedded ODE: x, v, phi)
# ---------------------------------------------------------------------------

def fit_duffing_forced(seed, noise_frac, degree, n_points, t_end):
    p = DEFAULT_DUFFING_FORCED_PARAMS
    rng = np.random.default_rng(seed)
    x0 = np.array([0.5, 0.0, 0.0]) + rng.normal(0, 0.05, size=3)
    x0[2] = x0[2] % (2 * np.pi)
    t, states = duffing_forced_trajectory(x0, t_span=(0, t_end), n_points=n_points, params=p)
    if noise_frac > 0:
        states = states + rng.normal(0, noise_frac * states.std(axis=0), size=states.shape)
    dt = t[1] - t[0]

    # LIBRARY MISMATCH BY DESIGN (see module docstring): gamma*cos(phi) is not
    # representable in a polynomial library. This regime intentionally probes
    # that mismatch, not just chaos-vs-non-chaos identifiability.
    model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=degree),
                      optimizer=ps.STLSQ(threshold=STLSQ_THRESHOLD))
    model.fit(states, t=dt, feature_names=["x", "v", "phi"])

    naive = fit_naive_polynomial(states, dt=dt, degree=degree, feature_names=["x", "v", "phi"])

    def true_rhs(t_, s):
        return np.array([s[1], -p["delta"] * s[1] - p["alpha"] * s[0] - p["beta"] * s[0] ** 3
                          + p["gamma"] * np.cos(s[2]), p["omega"]])

    conf_rng = np.random.default_rng(_confirmation_offset(seed))
    x0_conf = np.array([0.5, 0.0, 0.0]) + conf_rng.normal(0, 0.05, size=3)
    x0_conf[2] = x0_conf[2] % (2 * np.pi)
    t_conf, states_conf = duffing_forced_trajectory(x0_conf, t_span=(0, t_end), n_points=n_points, params=p)
    if noise_frac > 0:
        states_conf = states_conf + conf_rng.normal(0, noise_frac * states_conf.std(axis=0), size=states_conf.shape)
    true_vf = np.array([true_rhs(0.0, s) for s in states_conf])
    pred_vf = model.predict(states_conf)
    vf_err = np.linalg.norm(pred_vf - true_vf) / max(np.linalg.norm(true_vf), 1e-300)

    grid_rng = np.random.default_rng(_grid_offset(seed))
    lo, hi = states.min(axis=0), states.max(axis=0)
    grid_pts = grid_rng.uniform(lo, hi, size=(N_OFF_ATTRACTOR_GRID_POINTS, 3))
    true_vf_grid = np.array([true_rhs(0.0, s) for s in grid_pts])
    pred_vf_grid = model.predict(grid_pts)
    vf_err_grid = np.linalg.norm(pred_vf_grid - true_vf_grid) / max(np.linalg.norm(true_vf_grid), 1e-300)

    def true_jac(s):
        return np.array([
            [0.0, 1.0, 0.0],
            [-p["alpha"] - 3 * p["beta"] * s[0] ** 2, -p["delta"], -p["gamma"] * np.sin(s[2])],
            [0.0, 0.0, 0.0],
        ])
    if seed == 0:
        lyap_err = _lyapunov_error_ode(model, true_rhs, true_jac, 3, states[0])
        model_traj = _simulate_model_trajectory(model, states[0], (0, min(t_end, 50.0)), 2000)
        tv_dist = invariant_measure_tv_distance(states, model_traj) if model_traj is not None else None
    else:
        lyap_err = "skipped_non_representative_seed"
        tv_dist = None

    return dict(seed=seed, noise_frac=noise_frac, degree=degree,
                library_mismatch_expected=True,
                vf_l2_err_confirmation=float(vf_err), dynamically_distinct=bool(vf_err > VF_ERR_TOL),
                vf_l2_err_off_attractor_grid=float(vf_err_grid),
                dynamically_distinct_off_attractor=bool(vf_err_grid > VF_ERR_TOL),
                naive_n_nonzero=naive["n_nonzero"], naive_n_features=len(naive["feature_names"]) * 3,
                lyapunov_error=lyap_err, invariant_measure_tv=tv_dist)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

LOGISTIC_REGIMES = [(3.2, "period_2"), (3.5, "period_4"), (3.83, "period_3_window"), (4.0, "chaotic")]
LORENZ_REGIMES = [(14.0, "stable_fixed_point"), (22.0, "pre_chaotic"), (24.5, "near_onset"),
                   (28.0, "classic_chaotic"), (100.0, "high_rho_chaotic")]

# Dispatch by name (not lambda) so jobs are picklable for ProcessPoolExecutor
# (macOS default start method is 'spawn', which requires picklable callables).
_FIT_FNS = dict(
    logistic=lambda regime_args, seed, noise_frac, degree, **kw: fit_logistic(regime_args, seed, noise_frac, degree, **kw),
    lorenz=lambda regime_args, seed, noise_frac, degree, **kw: fit_lorenz(regime_args, seed, noise_frac, degree, **kw),
    harmonic=lambda regime_args, seed, noise_frac, degree, **kw: fit_harmonic(seed, noise_frac, degree, **kw),
    duffing_unforced=lambda regime_args, seed, noise_frac, degree, **kw: fit_duffing_unforced(seed, noise_frac, degree, **kw),
    duffing_forced=lambda regime_args, seed, noise_frac, degree, **kw: fit_duffing_forced(seed, noise_frac, degree, **kw),
)


def _run_condition(job):
    """One (family, regime, noise, degree) condition: fit all SEEDS, summarize. Runs in a worker process."""
    fit_kind, results_key, regime_args, label, noise_frac, degree, extra_kwargs = job
    fit_fn = _FIT_FNS[fit_kind]
    t_start = time.time()
    runs = [fit_fn(regime_args, seed, noise_frac, degree, **extra_kwargs) for seed in SEEDS]
    wall = time.time() - t_start
    n_ok = sum(r.get("recovered", None) is True for r in runs)
    n_recoverable = sum(1 for r in runs if "recovered" in r)
    n_vf_ok = sum(not r["dynamically_distinct"] for r in runs)
    n_vf_grid_ok = sum(not r["dynamically_distinct_off_attractor"] for r in runs)
    summary = dict(regime_args=regime_args, noise_frac=noise_frac, degree=degree,
                    n_recovered=n_ok, n_total=len(SEEDS), n_coeff_recovery_applicable=n_recoverable,
                    n_vf_recovered=n_vf_ok, n_vf_grid_recovered=n_vf_grid_ok, runs=runs)
    manifest_entry = dict(family=results_key, regime=label, degree=degree,
                           noise_frac=noise_frac, n_seeds=len(SEEDS), wall_clock_s=wall)
    coeff_str = f"{n_ok}/{n_recoverable} coeff-recovered" if n_recoverable else "coeff-recovery n/a"
    progress_line = (f"[Tier A][degree={degree}] {results_key} {label} noise={noise_frac:.1%}: "
                      f"{coeff_str}; {n_vf_ok}/{len(SEEDS)} confirmation-VF; "
                      f"{n_vf_grid_ok}/{len(SEEDS)} grid-VF; wall={wall:.1f}s")
    return results_key, degree, label, noise_frac, summary, manifest_entry, progress_line


def _build_jobs(fit_kind, results_key, regime_list, noise_levels, degrees, extra_kwargs_fn):
    jobs = []
    for degree in degrees:
        for regime_args, label in regime_list:
            for noise_frac in noise_levels:
                jobs.append((fit_kind, results_key, regime_args, label, noise_frac, degree, extra_kwargs_fn()))
    return jobs


def main():
    manifest = []
    results = {"logistic": {}, "lorenz": {}, "harmonic": {}, "duffing_unforced": {}, "duffing_forced": {}}
    sensitivity = {"logistic": {"short": {}, "long": {}}, "lorenz": {"short": {}, "long": {}}}
    results_by_key = dict(results)

    jobs = []
    jobs += _build_jobs("logistic", "logistic", LOGISTIC_REGIMES, NOISE_LEVELS_FULL, LIBRARY_DEGREES,
                         lambda: dict(n_steps=MEDIUM["logistic_n_steps"]))
    jobs += _build_jobs("lorenz", "lorenz", LORENZ_REGIMES, NOISE_LEVELS_FULL, LIBRARY_DEGREES,
                         lambda: dict(n_points=MEDIUM["lorenz_n_points"], t_end=MEDIUM["lorenz_t_end"]))
    jobs += _build_jobs("harmonic", "harmonic", [(None, "conservative")], NOISE_LEVELS_FULL, LIBRARY_DEGREES,
                         lambda: dict(n_points=MEDIUM["harmonic_n_points"], t_end=MEDIUM["harmonic_t_end"]))
    jobs += _build_jobs("duffing_unforced", "duffing_unforced", [(None, "conservative")], NOISE_LEVELS_FULL,
                         LIBRARY_DEGREES,
                         lambda: dict(n_points=MEDIUM["duffing_n_points"], t_end=MEDIUM["duffing_t_end"]))
    jobs += _build_jobs("duffing_forced", "duffing_forced", [(None, "forced_chaotic")], NOISE_LEVELS_FULL,
                         LIBRARY_DEGREES,
                         lambda: dict(n_points=MEDIUM["duffing_n_points"], t_end=MEDIUM["duffing_t_end"]))

    # --- Short/long length sensitivity sub-sweep: SINDy-only, logistic + Lorenz,
    # {0%, 1%} noise (MAIN_STUDY_DESIGN.md SS2 Tier A). ---
    sensitivity_key_map = {}  # results_key -> (family, length_label)
    for length_label, length_cfg in [("short", SHORT), ("long", LONG)]:
        logistic_key = f"logistic_{length_label}"
        lorenz_key = f"lorenz_{length_label}"
        sensitivity_key_map[logistic_key] = ("logistic", length_label)
        sensitivity_key_map[lorenz_key] = ("lorenz", length_label)
        jobs += _build_jobs("logistic", logistic_key, LOGISTIC_REGIMES, NOISE_LEVELS_SENSITIVITY, [3],
                             lambda cfg=length_cfg: dict(n_steps=cfg["logistic_n_steps"]))
        jobs += _build_jobs("lorenz", lorenz_key, LORENZ_REGIMES, NOISE_LEVELS_SENSITIVITY, [3],
                             lambda cfg=length_cfg: dict(n_points=cfg["lorenz_n_points"], t_end=cfg["lorenz_t_end"]))

    print(f"Tier A: {len(jobs)} conditions queued across {N_WORKERS} worker processes.", flush=True)
    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_run_condition, job): job for job in jobs}
        for future in as_completed(futures):
            results_key, degree, label, noise_frac, summary, manifest_entry, progress_line = future.result()
            deg_key = f"degree{degree}"
            if results_key in sensitivity_key_map:
                family, length_label = sensitivity_key_map[results_key]
                sensitivity[family][length_label].setdefault(deg_key, {}).setdefault(label, {})[noise_frac] = summary
            else:
                results_by_key[results_key].setdefault(deg_key, {}).setdefault(label, {})[noise_frac] = summary
            manifest.append(manifest_entry)
            n_done += 1
            print(f"[{n_done}/{len(jobs)}] {progress_line}", flush=True)

    for key in results:
        results[key] = results_by_key[key]
    results["length_sensitivity"] = sensitivity

    with open("experiments/main_study_results/tier_a_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    with open("experiments/main_study_results/tier_a_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=float)

    total_cpu = sum(m["wall_clock_s"] for m in manifest)
    total_wall = time.time() - t0
    print(f"\nTier A complete. Wall-clock: {total_wall:.1f}s (summed per-condition CPU time: {total_cpu:.1f}s) "
          f"across {len(manifest)} conditions, {N_WORKERS} workers.", flush=True)
    return results


if __name__ == "__main__":
    main()
