"""Held-out Rössler confirmation family (MAIN_STUDY_DESIGN.md SS3/SS4 step 8).

Run exactly once, per PREREGISTRATION.md SS11: Rössler trajectories are
unblinded and fit here for the first time in this project. This mirrors
Tier A's fitting/metric machinery as closely as possible (same SEEDS,
NOISE_LEVELS_FULL, LIBRARY_DEGREES, coefficient-recovery + VF-confirmation +
off-attractor-grid + Lyapunov-error + invariant-measure-TV metric stack,
same COEFF_TOL/VF_ERR_TOL/STLSQ_THRESHOLD) so the confirmation result is
directly comparable to Tier A's frozen chaos-vs-non-chaos backbone finding,
NOT a new ad hoc metric design.

Scope is deliberately narrower than Tier A/B: SINDy only (the trusted
primary comparator, PREREGISTRATION.md SS7), full-state observation only
(the delay-embedded partial-observation axis is Tier C's separate question,
already answered there -- not re-asked here), one chaotic regime
(a=b=0.2, c=5.7, literature-matched, MAIN_STUDY_DESIGN.md step 1) and one
matched non-chaotic control (c=3.0, same a/b, pre-chaotic limit cycle,
verified negative largest-LE in tests/test_simulators.py at freeze time).
Keeping this pass narrow, rather than replicating Tier B's full 3-method
matrix, is deliberate: PREREGISTRATION.md SS11 permits running the held-out
family only once, so there is no opportunity to iterate on scope after
seeing results -- a small, exactly-Tier-A-shaped pass is the lowest-risk
design for a single-shot confirmatory test.

This script was smoke-tested for shape/crash-freedom only (return-dict keys
and types checked; the actual identifiability numbers were not read or
reasoned about) before this run, to keep the diligence this project applies
to every other tier without violating SS11's blinding on the substantive
result. See DECISION_LOG.md "Rössler confirmation run".
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pysindy as ps

from experiments.main_study import (
    COEFF_TOL,
    N_OFF_ATTRACTOR_GRID_POINTS,
    NOISE_LEVELS_FULL,
    LIBRARY_DEGREES,
    SEEDS,
    STLSQ_THRESHOLD,
    VF_ERR_TOL,
    _confirmation_offset,
    _grid_offset,
    _lyapunov_error_ode,
    _simulate_model_trajectory,
)
from src.discovery_naive_baseline import fit_naive_polynomial
from src.metrics_dynamical import DynamicalSystem, invariant_measure_tv_distance
from src.simulators import rossler_jacobian, rossler_rhs, rossler_trajectory

N_WORKERS = 6  # PROJECT_CHARTER.md local-only compute ceiling, matches Tier A/B/C

ROSSLER_REGIMES = [
    (dict(a=0.2, b=0.2, c=5.7), "chaotic"),
    (dict(a=0.2, b=0.2, c=3.0), "non_chaotic_control"),
]
MEDIUM = dict(n_points=25000, t_end=200.0)  # matches Tier A Lorenz medium-length scale


def fit_rossler(params, seed, noise_frac, degree, n_points, t_end, transient_frac=0.5):
    a, b, c = params["a"], params["b"], params["c"]
    rng = np.random.default_rng(seed)
    x0 = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.1, size=3)
    t, states = rossler_trajectory(x0, t_span=(0, t_end), n_points=n_points, params=params)
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
    # True: dx/dt = -y-z ; dy/dt = x+a*y ; dz/dt = b + x*z - c*z
    a_hat = coeffs[1][idx["y"]] if "y" in idx else 0.0
    b_hat = coeffs[2][idx["1"]] if "1" in idx else 0.0
    c_hat = -coeffs[2][idx["z"]] if "z" in idx else 0.0
    err = max(abs(a_hat - a) / max(abs(a), 1e-9),
              abs(b_hat - b) / max(abs(b), 1e-9),
              abs(c_hat - c) / max(abs(c), 1e-9))

    naive = fit_naive_polynomial(states_tail, dt=dt, degree=degree, feature_names=["x", "y", "z"])

    conf_rng = np.random.default_rng(_confirmation_offset(seed))
    x0_conf = np.array([1.0, 1.0, 1.0]) + conf_rng.normal(0, 0.1, size=3)
    t_conf, states_conf = rossler_trajectory(x0_conf, t_span=(0, t_end), n_points=n_points, params=params)
    if noise_frac > 0:
        states_conf = states_conf + conf_rng.normal(0, noise_frac * states_conf.std(axis=0), size=states_conf.shape)
    states_conf_tail = states_conf[n_discard:]
    true_vf = np.array([rossler_rhs(0.0, s, a, b, c) for s in states_conf_tail])
    pred_vf = model.predict(states_conf_tail)
    vf_err = np.linalg.norm(pred_vf - true_vf) / max(np.linalg.norm(true_vf), 1e-300)

    grid_rng = np.random.default_rng(_grid_offset(seed))
    lo, hi = states.min(axis=0), states.max(axis=0)
    grid_pts = grid_rng.uniform(lo, hi, size=(N_OFF_ATTRACTOR_GRID_POINTS, 3))
    true_vf_grid = np.array([rossler_rhs(0.0, s, a, b, c) for s in grid_pts])
    pred_vf_grid = model.predict(grid_pts)
    vf_err_grid = np.linalg.norm(pred_vf_grid - true_vf_grid) / max(np.linalg.norm(true_vf_grid), 1e-300)

    # Lyapunov-error / invariant-measure TV computed on seed 0 only, matching
    # Tier A's compute-scoping decision (DECISION_LOG.md "Tier A Lyapunov-error
    # compute scoping") -- coefficient recovery and VF error, the primary
    # metrics, still run on all 5 seeds above.
    if seed == 0:
        true_jac = lambda s: rossler_jacobian(s, a, b, c)  # noqa: E731
        lyap_err = _lyapunov_error_ode(model, lambda t, s: rossler_rhs(t, s, a, b, c), true_jac, 3, states_tail[0])
        model_traj = _simulate_model_trajectory(model, states_tail[0], (0, min(t_end, 50.0)), 2000)
        tv_dist = (invariant_measure_tv_distance(states_tail, model_traj)
                   if model_traj is not None else None)
    else:
        lyap_err = "skipped_non_representative_seed"
        tv_dist = None

    return dict(seed=seed, a=a, b=b, c=c, noise_frac=noise_frac, degree=degree,
                a_hat=float(a_hat), b_hat=float(b_hat), c_hat=float(c_hat),
                max_rel_err=float(err), recovered=bool(err < COEFF_TOL),
                vf_l2_err_confirmation=float(vf_err), dynamically_distinct=bool(vf_err > VF_ERR_TOL),
                vf_l2_err_off_attractor_grid=float(vf_err_grid),
                dynamically_distinct_off_attractor=bool(vf_err_grid > VF_ERR_TOL),
                naive_n_nonzero=naive["n_nonzero"], naive_n_features=len(naive["feature_names"]) * 3,
                lyapunov_error=lyap_err, invariant_measure_tv=tv_dist)


def _run_condition(job):
    params, label, noise_frac, degree = job
    t_start = time.time()
    runs = [fit_rossler(params, seed, noise_frac, degree, MEDIUM["n_points"], MEDIUM["t_end"]) for seed in SEEDS]
    wall = time.time() - t_start
    n_ok = sum(r["recovered"] for r in runs)
    n_vf_ok = sum(not r["dynamically_distinct"] for r in runs)
    n_vf_grid_ok = sum(not r["dynamically_distinct_off_attractor"] for r in runs)
    summary = dict(params=params, noise_frac=noise_frac, degree=degree,
                    n_recovered=n_ok, n_total=len(SEEDS),
                    n_vf_recovered=n_vf_ok, n_vf_grid_recovered=n_vf_grid_ok, runs=runs)
    manifest_entry = dict(regime=label, degree=degree, noise_frac=noise_frac,
                           n_seeds=len(SEEDS), wall_clock_s=wall)
    progress_line = (f"[Confirmation][degree={degree}] rossler {label} noise={noise_frac:.1%}: "
                      f"{n_ok}/{len(SEEDS)} coeff-recovered; {n_vf_ok}/{len(SEEDS)} confirmation-VF; "
                      f"{n_vf_grid_ok}/{len(SEEDS)} grid-VF; wall={wall:.1f}s")
    return label, degree, noise_frac, summary, manifest_entry, progress_line


def main():
    jobs = [(params, label, noise_frac, degree)
            for params, label in ROSSLER_REGIMES
            for noise_frac in NOISE_LEVELS_FULL
            for degree in LIBRARY_DEGREES]

    print(f"Rössler confirmation: {len(jobs)} conditions queued across {N_WORKERS} worker processes. "
          f"UNBLINDING NOW, per PREREGISTRATION.md SS11 -- this is the single confirmatory run.", flush=True)

    results = {}
    manifest = []
    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_run_condition, job): job for job in jobs}
        for future in as_completed(futures):
            label, degree, noise_frac, summary, manifest_entry, progress_line = future.result()
            deg_key = f"degree{degree}"
            results.setdefault(deg_key, {}).setdefault(label, {})[noise_frac] = summary
            manifest.append(manifest_entry)
            n_done += 1
            print(f"[{n_done}/{len(jobs)}] {progress_line}", flush=True)

    with open("experiments/main_study_results/confirmation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    with open("experiments/main_study_results/confirmation_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=float)

    total_wall = time.time() - t0
    print(f"\nRössler confirmation complete. Wall-clock: {total_wall:.1f}s across {len(manifest)} conditions, "
          f"{N_WORKERS} workers.", flush=True)
    return results


if __name__ == "__main__":
    main()
