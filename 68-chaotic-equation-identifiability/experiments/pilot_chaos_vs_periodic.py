"""Pilot experiment (PREREGISTRATION.md-scoped): does SINDy recover the true
equations more reliably in a chaotic regime than a matched non-chaotic regime
of the *same* underlying system?

This is exploratory/pilot, not the frozen main study - thresholds match
PREREGISTRATION.md SS6/SS10 but the comparator set here (SINDy only, 2 systems)
is a cheap decisive first look, per PROMPT.md's "cheapest decisive feasibility
test" guidance. Results feed the frozen main study design, they do not
themselves constitute a confirmed claim.

Non-chaotic Lorenz control (rho=14, stable fixed point): the pilot originally
compared this against rho=28 using the *whole* trajectory including the
pre-convergence transient. That transient still excites the regression even
though the attractor itself (a fixed point) carries no information, so the
comparison was confounded (see DECISION_LOG.md, "Pilot revealed
regime-selection confound"). Fix applied here: every condition (chaotic and
non-chaotic alike) discards the first `transient_frac` of the simulated
trajectory and fits SINDy only on the post-transient tail. For rho=14 this
tail sits on (or extremely close to) the fixed point -> genuinely
non-exciting, matching the logistic period-2 control's rank-deficiency
mechanism. For rho=28 the tail is still on the chaotic attractor -> still
exciting. This is option (a) from the decision log (discard transient), kept
symmetric across both regimes rather than switching to a limit-cycle system,
since it requires no new system and directly targets the confound's cause.

Noise regimes (PREREGISTRATION.md SS4): additive Gaussian noise at 0%, 0.1%,
1%, 5% of state standard deviation, added to the full (pre-transient-discard)
trajectory so the noise scale is tied to the system's natural amplitude and
is identical between the chaotic and non-chaotic conditions of a system
(matched-noise design) rather than to the (near-zero) variance of the
non-chaotic tail alone.

LIBRARY DEGREE (LIMITATIONS.md #2): PREREGISTRATION.md SS2 freezes the SINDy
library at "polynomial terms up to degree 3." The original version of this
pilot used degree=2 (matching the true systems' active-term degree exactly),
which mechanically inflates apparent recovery cleanliness by never giving
STLSQ a spurious higher-degree term to wrongly keep or correctly zero out.
This version runs BOTH degree=2 (kept for continuity/comparison, and because
it was the version originally reported in RESULTS.md) and degree=3 (the
actually-preregistered value), and reports both side by side. Degree-2
results below are numerically identical to the prior session's
`pilot_chaos_vs_periodic_results_degree2.json` (same code path, same seeds).

PRIMARY METRIC (LIMITATIONS.md #5): PREREGISTRATION.md SS8 designates
vector-field error off-trajectory on an independent confirmation trajectory
as the PRIMARY identifiability metric; in-sample coefficient recovery is
secondary. This version adds that check: for every fitted model, an
independent confirmation trajectory (different initial condition, same
system/parameters/regime, same noise_frac drawn with a different seed
offset) is generated, and the fitted model's vector field (Lorenz:
`model.predict`; logistic map: the fitted polynomial map) is compared
against the true analytic vector field (`src/simulators.py`'s
`lorenz_rhs`/`logistic_map`) at every point along that confirmation
trajectory. The normalized L2 vector-field error
(||f_hat - f_true||_2 / ||f_true||_2, per PREREGISTRATION.md SS6) is reported
per run alongside the existing in-sample coefficient-recovery metric.
Threshold: >10% off-trajectory normalized L2 error is "dynamically distinct"
/ non-recovery per SS6; this is reported but does NOT override the
preregistered 5%-relative-coefficient pass/fail already in use for the
headline gap, since both metrics are logged per PROMPT.md ("report all
preregistered outcomes") rather than one silently replacing the other.

OFF-ATTRACTOR GRID METRIC (LIMITATIONS.md #5, decided 2026-08-16, see
DECISION_LOG.md "Off-attractor evaluation grid"): the confirmation-trajectory
VF-error above is NOT a genuine extrapolation stress test for periodic/point
attractors, because an independent-IC confirmation trajectory for such a
regime converges onto the SAME attractor (e.g. the same 2-cycle) as the
training trajectory - it never actually visits states outside the training
support. Fix: alongside the confirmation-trajectory metric, also evaluate the fitted
model's vector field at a fixed grid of points drawn from a domain that
genuinely extends beyond the attractor itself: for Lorenz, uniformly from the
FULL (pre-transient-discard) trajectory's bounding box, so the fixed point's
transient approach is exercised; for the logistic map (whose
`logistic_trajectory` discards its transient before returning, so the
trajectory's own range cannot be reused for this purpose), uniformly from the
map's actual state-space domain [0, 1]. Reported as
`vf_l2_err_off_attractor_grid`; both this and the confirmation-trajectory
metric are logged per-run per PROMPT.md, neither silently replacing the other.
This does not fully resolve LIMITATIONS.md #5 for regimes whose off-attractor
domain is unbounded or not well-defined a priori (e.g. an unstable regime with
no natural bounding box) - flagged as a residual limitation for the frozen
main study, not claimed as a complete general-purpose fix.
"""
import json

import numpy as np
import pysindy as ps

from src.simulators import lorenz_trajectory, logistic_trajectory, lorenz_rhs, logistic_map

SEEDS = [0, 1, 2, 3, 4]
COEFF_TOL = 0.05  # relative, per PREREGISTRATION.md SS6
NOISE_LEVELS = [0.0, 0.001, 0.01, 0.05]  # fraction of state std, per PREREGISTRATION.md SS4
LIBRARY_DEGREES = [2, 3]  # degree=2: prior-session value (kept for comparison);
                           # degree=3: PREREGISTRATION.md SS2 frozen value
VF_ERR_TOL = 0.10  # normalized L2 off-trajectory vector-field error, PREREGISTRATION.md SS6


def _confirmation_offset(seed):
    # Distinct RNG stream from the training run's seed, same system/params/regime/noise.
    return seed + 10_000


def _off_attractor_grid_offset(seed):
    # A third, distinct RNG stream (training=seed, confirmation=+10_000, grid=+20_000).
    return seed + 20_000


N_OFF_ATTRACTOR_GRID_POINTS = 500


def fit_sindy_lorenz(rho, seed, noise_frac=0.0, degree=2, n_points=25000, t_end=50.0,
                      transient_frac=0.5):
    rng = np.random.default_rng(seed)
    x0 = np.array([-8.0, 8.0, 27.0]) + rng.normal(0, 0.5, size=3)
    params = dict(sigma=10.0, rho=rho, beta=8.0 / 3.0)
    t, states = lorenz_trajectory(x0, t_span=(0, t_end), n_points=n_points, params=params)

    if noise_frac > 0:
        state_std = states.std(axis=0)
        states = states + rng.normal(0, noise_frac * state_std, size=states.shape)

    n_discard = int(len(t) * transient_frac)
    t_tail = t[n_discard:]
    states_tail = states[n_discard:]
    dt = t_tail[1] - t_tail[0]

    model = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=degree),
        optimizer=ps.STLSQ(threshold=0.1),
    )
    model.fit(states_tail, t=dt, feature_names=["x", "y", "z"])
    coeffs = model.coefficients()
    names = model.get_feature_names()
    idx = {n: i for i, n in enumerate(names)}
    sigma_hat = -coeffs[0][idx["x"]]
    rho_hat = coeffs[1][idx["x"]]
    beta_hat = -coeffs[2][idx["z"]]
    err = max(
        abs(sigma_hat - params["sigma"]) / params["sigma"],
        abs(rho_hat - params["rho"]) / max(abs(params["rho"]), 1e-9),
        abs(beta_hat - params["beta"]) / params["beta"],
    )

    # --- Primary metric: off-trajectory vector-field error on an independent
    # confirmation trajectory (PREREGISTRATION.md SS5/SS8). ---
    conf_seed = _confirmation_offset(seed)
    conf_rng = np.random.default_rng(conf_seed)
    x0_conf = np.array([-8.0, 8.0, 27.0]) + conf_rng.normal(0, 0.5, size=3)
    t_conf, states_conf = lorenz_trajectory(x0_conf, t_span=(0, t_end), n_points=n_points,
                                             params=params)
    if noise_frac > 0:
        state_std_conf = states_conf.std(axis=0)
        states_conf = states_conf + conf_rng.normal(0, noise_frac * state_std_conf,
                                                      size=states_conf.shape)
    states_conf_tail = states_conf[n_discard:]

    true_vf = np.array([
        lorenz_rhs(0.0, s, params["sigma"], params["rho"], params["beta"])
        for s in states_conf_tail
    ])
    pred_vf = model.predict(states_conf_tail)
    vf_l2_err = np.linalg.norm(pred_vf - true_vf) / max(np.linalg.norm(true_vf), 1e-300)

    # --- Off-attractor grid metric (LIMITATIONS.md #5, DECISION_LOG.md
    # "Off-attractor evaluation grid"): points drawn uniformly from the FULL
    # (pre-transient-discard) trajectory's bounding box, so a fixed-point/
    # periodic regime's transient approach is genuinely exercised. ---
    grid_rng = np.random.default_rng(_off_attractor_grid_offset(seed))
    lo, hi = states.min(axis=0), states.max(axis=0)
    grid_pts = grid_rng.uniform(lo, hi, size=(N_OFF_ATTRACTOR_GRID_POINTS, 3))
    true_vf_grid = np.array([
        lorenz_rhs(0.0, s, params["sigma"], params["rho"], params["beta"]) for s in grid_pts
    ])
    pred_vf_grid = model.predict(grid_pts)
    vf_l2_err_grid = (np.linalg.norm(pred_vf_grid - true_vf_grid)
                       / max(np.linalg.norm(true_vf_grid), 1e-300))

    return dict(seed=seed, rho=rho, noise_frac=noise_frac, degree=degree,
                sigma_hat=sigma_hat, rho_hat=rho_hat, beta_hat=beta_hat,
                max_rel_err=err, recovered=err < COEFF_TOL,
                vf_l2_err_confirmation=vf_l2_err,
                dynamically_distinct=vf_l2_err > VF_ERR_TOL,
                vf_l2_err_off_attractor_grid=vf_l2_err_grid,
                dynamically_distinct_off_attractor=vf_l2_err_grid > VF_ERR_TOL)


def fit_sindy_logistic(r, seed, noise_frac=0.0, degree=2, n_steps=2000):
    rng = np.random.default_rng(seed)
    x0 = 0.1 + 0.3 * rng.random()
    traj = logistic_trajectory(x0, r=r, n_steps=n_steps, transient=500)

    if noise_frac > 0:
        traj = traj + rng.normal(0, noise_frac * traj.std(), size=traj.shape)

    x_n = traj[:-1].reshape(-1, 1)
    x_np1 = traj[1:]
    # Discrete map: fit x_{n+1} as polynomial in x_n via plain least squares
    # over [1, x, x^2, ..., x^degree] (SINDy's discrete-time / library-regression
    # equivalent for a 1D map is a direct sparse regression, done explicitly here
    # since pysindy's discrete-time API targets multivariate/control settings).
    A = np.stack([x_n[:, 0] ** k for k in range(degree + 1)], axis=1)
    coef, *_ = np.linalg.lstsq(A, x_np1, rcond=None)
    # True model: x_{n+1} = r*x - r*x^2  -> c0=0, c1=r, c2=-r, and (if degree=3) c3=0
    c0 = coef[0]
    r_hat_from_linear = coef[1]
    r_hat_from_quad = -coef[2]
    err_terms = [abs(c0), abs(r_hat_from_linear - r) / r, abs(r_hat_from_quad - r) / r]
    if degree >= 3:
        err_terms.append(abs(coef[3]))  # true c3 = 0 -> absolute error
    err = max(err_terms)

    # --- Primary metric: off-trajectory vector-field error on an independent
    # confirmation trajectory (PREREGISTRATION.md SS5/SS8). For a 1D map the
    # "vector field" is the map itself: f_hat(x) vs f_true(x) = r*x*(1-x). ---
    conf_seed = _confirmation_offset(seed)
    conf_rng = np.random.default_rng(conf_seed)
    x0_conf = 0.1 + 0.3 * conf_rng.random()
    traj_conf = logistic_trajectory(x0_conf, r=r, n_steps=n_steps, transient=500)
    if noise_frac > 0:
        traj_conf = traj_conf + conf_rng.normal(0, noise_frac * traj_conf.std(),
                                                  size=traj_conf.shape)
    x_conf = traj_conf[:-1]
    true_vf = np.array([logistic_map(x, r) for x in x_conf])
    A_conf = np.stack([x_conf ** k for k in range(degree + 1)], axis=1)
    pred_vf = A_conf @ coef
    vf_l2_err = np.linalg.norm(pred_vf - true_vf) / max(np.linalg.norm(true_vf), 1e-300)

    # --- Off-attractor grid metric (LIMITATIONS.md #5, DECISION_LOG.md
    # "Off-attractor evaluation grid"). NOTE: logistic_trajectory() discards
    # its transient before returning (src/simulators.py), so `traj`'s own
    # range is just the attractor itself and cannot be used here (that would
    # reproduce the exact collapse this metric exists to fix). Instead grid
    # points are drawn from the logistic map's actual state-space domain
    # [0, 1] (r*x*(1-x) maps [0,1]->[0,1] for r in [0,4]) - the map's full
    # domain, not merely the training trajectory's observed support -
    # genuinely exercising off-attractor states for a periodic regime. ---
    grid_rng = np.random.default_rng(_off_attractor_grid_offset(seed))
    x_grid = grid_rng.uniform(0.0, 1.0, size=N_OFF_ATTRACTOR_GRID_POINTS)
    true_vf_grid = np.array([logistic_map(x, r) for x in x_grid])
    A_grid = np.stack([x_grid ** k for k in range(degree + 1)], axis=1)
    pred_vf_grid = A_grid @ coef
    vf_l2_err_grid = (np.linalg.norm(pred_vf_grid - true_vf_grid)
                       / max(np.linalg.norm(true_vf_grid), 1e-300))

    return dict(seed=seed, r=r, noise_frac=noise_frac, degree=degree,
                c0=c0, c1=r_hat_from_linear, c2=r_hat_from_quad, max_rel_err=err,
                recovered=err < COEFF_TOL,
                vf_l2_err_confirmation=vf_l2_err,
                dynamically_distinct=vf_l2_err > VF_ERR_TOL,
                vf_l2_err_off_attractor_grid=vf_l2_err_grid,
                dynamically_distinct_off_attractor=vf_l2_err_grid > VF_ERR_TOL)


def main():
    results = {"lorenz": {}, "logistic": {}}

    for degree in LIBRARY_DEGREES:
        deg_key = f"degree{degree}"
        results["lorenz"][deg_key] = {}
        for rho, label in [(14.0, "stable_fixed_point"), (28.0, "classic_chaotic")]:
            results["lorenz"][deg_key][label] = {}
            for noise_frac in NOISE_LEVELS:
                runs = [fit_sindy_lorenz(rho, seed, noise_frac=noise_frac, degree=degree)
                        for seed in SEEDS]
                n_ok = sum(run["recovered"] for run in runs)
                n_vf_ok = sum(not run["dynamically_distinct"] for run in runs)
                n_vf_grid_ok = sum(not run["dynamically_distinct_off_attractor"] for run in runs)
                results["lorenz"][deg_key][label][noise_frac] = dict(
                    rho=rho, noise_frac=noise_frac, degree=degree,
                    n_recovered=n_ok, n_total=len(SEEDS),
                    n_vf_recovered=n_vf_ok, n_vf_grid_recovered=n_vf_grid_ok, runs=runs)
                print(f"[degree={degree}] Lorenz rho={rho} ({label}), noise={noise_frac:.1%}: "
                      f"{n_ok}/{len(SEEDS)} coeff-recovered; "
                      f"{n_vf_ok}/{len(SEEDS)} confirmation-VF-recovered; "
                      f"{n_vf_grid_ok}/{len(SEEDS)} off-attractor-grid-VF-recovered (<=10% err)")

        results["logistic"][deg_key] = {}
        for r, label in [(3.2, "period_2"), (4.0, "chaotic")]:
            results["logistic"][deg_key][label] = {}
            for noise_frac in NOISE_LEVELS:
                runs = [fit_sindy_logistic(r, seed, noise_frac=noise_frac, degree=degree)
                        for seed in SEEDS]
                n_ok = sum(run["recovered"] for run in runs)
                n_vf_ok = sum(not run["dynamically_distinct"] for run in runs)
                n_vf_grid_ok = sum(not run["dynamically_distinct_off_attractor"] for run in runs)
                results["logistic"][deg_key][label][noise_frac] = dict(
                    r=r, noise_frac=noise_frac, degree=degree,
                    n_recovered=n_ok, n_total=len(SEEDS),
                    n_vf_recovered=n_vf_ok, n_vf_grid_recovered=n_vf_grid_ok, runs=runs)
                print(f"[degree={degree}] Logistic r={r} ({label}), noise={noise_frac:.1%}: "
                      f"{n_ok}/{len(SEEDS)} coeff-recovered; "
                      f"{n_vf_ok}/{len(SEEDS)} confirmation-VF-recovered; "
                      f"{n_vf_grid_ok}/{len(SEEDS)} off-attractor-grid-VF-recovered (<=10% err)")

    with open("experiments/pilot_chaos_vs_periodic_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    return results


if __name__ == "__main__":
    main()
