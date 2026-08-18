"""Tier B main-study runner (MAIN_STUDY_DESIGN.md SS2/SS4 step 6).

One matched chaotic/non-chaotic (or conservative-control) pair per system
family, chosen as the clearest contrast from Tier A, compared across all
THREE discovery methods (SINDy, symbolic regression, Koopman/EDMD) on
IDENTICAL per-condition trajectories (shared trajectory generation below),
so the cross-method comparison is not confounded by each method seeing
different training data.

Regime-count note (DECISION_LOG.md "Tier B regime-count discrepancy"):
MAIN_STUDY_DESIGN.md SS2 states "8 regimes x 3 noise x 5 seeds x 2 degrees
x 3 methods = 720 fits", but its own enumerated regime list sums to 7, not
8: logistic {period_2, chaotic} (2) + Lorenz {stable_fixed_point,
classic_chaotic} (2) + harmonic {conservative} (1, no pair -- the design
doc itself says harmonic has "1 conservative regime, no pair") + Duffing
{unforced conservative, forced chaotic} (2) = 7. This runner uses the
actual enumerated 7 regimes (7 x 3 x 5 x 2 x 3 = 630 fits), not a
fabricated 8th regime invented to match the doc's arithmetic. Reported
here rather than silently reconciled, per PROMPT.md claim discipline.

Noise levels: {0%, 1%, 5%} (0.1% dropped per Tier A's own finding that
0%/1% already bracket the interesting behavior -- MAIN_STUDY_DESIGN.md SS2).

Symbolic-regression hyperparameters are PINNED per MAIN_STUDY_DESIGN.md SS2
(resolves LIMITATIONS.md #4): population_size=3000, generations=25,
max_degree_retries=5 -- NOT fit_symbolic_regression()'s undocumented
defaults (population_size=2000, generations=20).

Checkpointing (MAIN_STUDY_DESIGN.md SS4 step 6): each (family, regime,
noise, degree, seed) job is the checkpoint unit (finer-grained than Tier
A's 5-seeds-per-condition, since symbolic regression's per-seed cost on a
3-dim system dominates Tier B's wall-clock). Completed jobs are appended
to a JSONL file; a resumed run skips any job key already present there
rather than restarting from zero.

Discovery-method output schemas are NOT unified into one coefficient-
recovery metric, because only SINDy (and the map-specific lstsq fit used
for the logistic map, matching Tier A) has a well-defined notion of
"coefficient recovery" against a known parametric truth. Symbolic
regression is evaluated via structural degree/is_polynomial checks plus
off-trajectory VF error (matching this project's established convention,
EVIDENCE_INDEX.md rows 12-13). Koopman/EDMD is evaluated via one-step and
rollout prediction error, K-matrix spectral error vs. the true Jacobian,
and K condition number -- explicitly NOT equation/coefficient recovery
(EVIDENCE_INDEX.md row 15). Reporting genuinely different metrics per
method, rather than forcing a single number, is itself the honest choice
here.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pysindy as ps
from gplearn.genetic import SymbolicRegressor

from src.discovery_koopman import fit_edmd
from src.discovery_symbolic_regression import (
    MATCHED_FUNCTION_SET,
    estimate_derivatives,
    program_polynomial_degree,
)
from src.simulators import (
    DEFAULT_DUFFING_FORCED_PARAMS,
    DEFAULT_DUFFING_UNFORCED_PARAMS,
    DEFAULT_ROESSLER_PARAMS,
    duffing_forced_trajectory,
    duffing_unforced_trajectory,
    harmonic_trajectory,
    logistic_map,
    logistic_trajectory,
    lorenz_rhs,
    lorenz_trajectory,
    rossler_jacobian,
    rossler_rhs,
    rossler_trajectory,
)

N_WORKERS = 6  # PROJECT_CHARTER.md local-only compute ceiling, matches Tier A
SEEDS = [0, 1, 2, 3, 4]
NOISE_LEVELS_B = [0.0, 0.01, 0.05]
LIBRARY_DEGREES = [2, 3]
COEFF_TOL = 0.05
VF_ERR_TOL = 0.10
N_OFF_ATTRACTOR_GRID_POINTS = 500
OFF_ATTRACTOR_GRID_SCALE = 3.0  # see main_study.py's constant of the same name / DECISION_LOG.md
STLSQ_THRESHOLD = 0.1
SR_POPULATION_SIZE = 3000  # MAIN_STUDY_DESIGN.md SS2, pinned (resolves LIMITATIONS.md #4)
SR_GENERATIONS = 25
SR_MAX_DEGREE_RETRIES = 5
KOOPMAN_ROLLOUT_HORIZON = 50

MEDIUM = dict(logistic_n_steps=5500, lorenz_n_points=25000, lorenz_t_end=50.0,
              harmonic_n_points=5000, harmonic_t_end=50.0,
              duffing_n_points=25000, duffing_t_end=200.0,
              rossler_n_points=25000, rossler_t_end=250.0)

TIER_B_ITEMS = [
    ("logistic", 3.2, "period_2"),
    ("logistic", 4.0, "chaotic"),
    ("lorenz", 14.0, "stable_fixed_point"),
    ("lorenz", 28.0, "classic_chaotic"),
    ("harmonic", None, "conservative"),
    ("duffing_unforced", None, "conservative"),
    ("duffing_forced", None, "forced_chaotic"),
    ("rossler", None, "chaotic"),
]

CHECKPOINT_PATH = "experiments/main_study_results/tier_b_checkpoint.jsonl"
RESULTS_PATH = "experiments/main_study_results/tier_b_results.json"
MANIFEST_PATH = "experiments/main_study_results/tier_b_manifest.json"


def _confirmation_offset(seed):
    return seed + 10_000


def _grid_offset(seed):
    return seed + 20_000


# ---------------------------------------------------------------------------
# Shared per-condition trajectory generation (identical training/confirmation/
# grid data across all 3 discovery methods for a given family/regime/noise/seed).
# ---------------------------------------------------------------------------

def _generate_tier_b_data(family, regime_args, seed, noise_frac):
    rng = np.random.default_rng(seed)
    conf_rng = np.random.default_rng(_confirmation_offset(seed))
    grid_rng = np.random.default_rng(_grid_offset(seed))

    if family == "logistic":
        r = regime_args
        x0 = 0.1 + 0.3 * rng.random()
        n_steps = MEDIUM["logistic_n_steps"]
        traj = logistic_trajectory(x0, r=r, n_steps=n_steps, transient=500)
        if noise_frac > 0:
            traj = traj + rng.normal(0, noise_frac * traj.std(), size=traj.shape)
        x0_conf = 0.1 + 0.3 * conf_rng.random()
        traj_conf = logistic_trajectory(x0_conf, r=r, n_steps=n_steps, transient=500)
        if noise_frac > 0:
            traj_conf = traj_conf + conf_rng.normal(0, noise_frac * traj_conf.std(), size=traj_conf.shape)
        x_grid = grid_rng.uniform(0.0, 1.0, size=N_OFF_ATTRACTOR_GRID_POINTS)
        return dict(
            kind="map", dim=1, feature_names=["x"], dt=None,
            states=traj.reshape(-1, 1), states_conf=traj_conf.reshape(-1, 1),
            grid_pts=x_grid.reshape(-1, 1),
            true_rhs=lambda s: np.array([logistic_map(float(s[0]), r)]),
            true_jac=lambda s: np.array([[r * (1.0 - 2.0 * float(s[0]))]]),
            params=dict(r=r), x0_for_lyap=np.array([0.4]),
        )

    if family == "lorenz":
        rho = regime_args
        params = dict(sigma=10.0, rho=rho, beta=8.0 / 3.0)
        x0 = np.array([-8.0, 8.0, 27.0]) + rng.normal(0, 0.5, size=3)
        t, states = lorenz_trajectory(x0, t_span=(0, MEDIUM["lorenz_t_end"]),
                                       n_points=MEDIUM["lorenz_n_points"], params=params)
        if noise_frac > 0:
            states = states + rng.normal(0, noise_frac * states.std(axis=0), size=states.shape)
        n_discard = int(len(t) * 0.5)
        t_tail, states_tail = t[n_discard:], states[n_discard:]
        dt = t_tail[1] - t_tail[0]

        x0_conf = np.array([-8.0, 8.0, 27.0]) + conf_rng.normal(0, 0.5, size=3)
        t_conf, states_conf = lorenz_trajectory(x0_conf, t_span=(0, MEDIUM["lorenz_t_end"]),
                                                  n_points=MEDIUM["lorenz_n_points"], params=params)
        if noise_frac > 0:
            states_conf = states_conf + conf_rng.normal(0, noise_frac * states_conf.std(axis=0), size=states_conf.shape)
        states_conf_tail = states_conf[n_discard:]

        lo, hi = states.min(axis=0), states.max(axis=0)
        grid_pts = grid_rng.uniform(lo, hi, size=(N_OFF_ATTRACTOR_GRID_POINTS, 3))

        def true_rhs(s):
            return lorenz_rhs(0.0, s, params["sigma"], params["rho"], params["beta"])

        def true_jac(s):
            return np.array([
                [-params["sigma"], params["sigma"], 0.0],
                [params["rho"] - s[2], -1.0, -s[0]],
                [s[1], s[0], -params["beta"]],
            ])

        return dict(kind="ode", dim=3, feature_names=["x", "y", "z"], dt=dt,
                    states=states_tail, states_conf=states_conf_tail, grid_pts=grid_pts,
                    true_rhs=true_rhs, true_jac=true_jac, params=params,
                    x0_for_lyap=states_tail[0])

    if family == "harmonic":
        x0 = np.array([1.0, 0.0]) + rng.normal(0, 0.1, size=2)
        t, states = harmonic_trajectory(x0, t_span=(0, MEDIUM["harmonic_t_end"]),
                                         n_points=MEDIUM["harmonic_n_points"], omega=1.0)
        if noise_frac > 0:
            states = states + rng.normal(0, noise_frac * states.std(axis=0), size=states.shape)
        dt = t[1] - t[0]

        x0_conf = np.array([1.0, 0.0]) + conf_rng.normal(0, 0.1, size=2)
        t_conf, states_conf = harmonic_trajectory(x0_conf, t_span=(0, MEDIUM["harmonic_t_end"]),
                                                    n_points=MEDIUM["harmonic_n_points"], omega=1.0)
        if noise_frac > 0:
            states_conf = states_conf + conf_rng.normal(0, noise_frac * states_conf.std(axis=0), size=states_conf.shape)
        amp_scale = np.abs(states).max(axis=0)
        grid_pts = grid_rng.uniform(-OFF_ATTRACTOR_GRID_SCALE * amp_scale,
                                     OFF_ATTRACTOR_GRID_SCALE * amp_scale,
                                     size=(N_OFF_ATTRACTOR_GRID_POINTS, 2))

        return dict(kind="ode", dim=2, feature_names=["x", "v"], dt=dt,
                    states=states, states_conf=states_conf, grid_pts=grid_pts,
                    true_rhs=lambda s: np.array([s[1], -s[0]]),
                    true_jac=lambda s: np.array([[0.0, 1.0], [-1.0, 0.0]]),
                    params=dict(v_coef=1.0, x_coef=-1.0), x0_for_lyap=states[0])

    if family == "duffing_unforced":
        alpha = DEFAULT_DUFFING_UNFORCED_PARAMS["alpha"]
        beta = DEFAULT_DUFFING_UNFORCED_PARAMS["beta"]
        x0 = np.array([0.5, 0.0]) + rng.normal(0, 0.05, size=2)
        t, states = duffing_unforced_trajectory(x0, t_span=(0, MEDIUM["duffing_t_end"]),
                                                  n_points=MEDIUM["duffing_n_points"])
        if noise_frac > 0:
            states = states + rng.normal(0, noise_frac * states.std(axis=0), size=states.shape)
        dt = t[1] - t[0]

        x0_conf = np.array([0.5, 0.0]) + conf_rng.normal(0, 0.05, size=2)
        t_conf, states_conf = duffing_unforced_trajectory(x0_conf, t_span=(0, MEDIUM["duffing_t_end"]),
                                                            n_points=MEDIUM["duffing_n_points"])
        if noise_frac > 0:
            states_conf = states_conf + conf_rng.normal(0, noise_frac * states_conf.std(axis=0), size=states_conf.shape)
        amp_scale = np.abs(states).max(axis=0)
        grid_pts = grid_rng.uniform(-OFF_ATTRACTOR_GRID_SCALE * amp_scale,
                                     OFF_ATTRACTOR_GRID_SCALE * amp_scale,
                                     size=(N_OFF_ATTRACTOR_GRID_POINTS, 2))

        return dict(kind="ode", dim=2, feature_names=["x", "v"], dt=dt,
                    states=states, states_conf=states_conf, grid_pts=grid_pts,
                    true_rhs=lambda s: np.array([s[1], -alpha * s[0] - beta * s[0] ** 3]),
                    true_jac=lambda s: np.array([[0.0, 1.0], [-alpha - 3 * beta * s[0] ** 2, 0.0]]),
                    params=dict(v_coef=1.0, x_coef=-alpha, x3_coef=-beta), x0_for_lyap=states[0])

    if family == "duffing_forced":
        p = DEFAULT_DUFFING_FORCED_PARAMS
        x0 = np.array([0.5, 0.0, 0.0]) + rng.normal(0, 0.05, size=3)
        x0[2] = x0[2] % (2 * np.pi)
        t, states = duffing_forced_trajectory(x0, t_span=(0, MEDIUM["duffing_t_end"]),
                                                n_points=MEDIUM["duffing_n_points"], params=p)
        if noise_frac > 0:
            states = states + rng.normal(0, noise_frac * states.std(axis=0), size=states.shape)
        dt = t[1] - t[0]

        x0_conf = np.array([0.5, 0.0, 0.0]) + conf_rng.normal(0, 0.05, size=3)
        x0_conf[2] = x0_conf[2] % (2 * np.pi)
        t_conf, states_conf = duffing_forced_trajectory(x0_conf, t_span=(0, MEDIUM["duffing_t_end"]),
                                                          n_points=MEDIUM["duffing_n_points"], params=p)
        if noise_frac > 0:
            states_conf = states_conf + conf_rng.normal(0, noise_frac * states_conf.std(axis=0), size=states_conf.shape)
        lo, hi = states.min(axis=0), states.max(axis=0)
        grid_pts = grid_rng.uniform(lo, hi, size=(N_OFF_ATTRACTOR_GRID_POINTS, 3))

        def true_rhs(s):
            return np.array([s[1], -p["delta"] * s[1] - p["alpha"] * s[0] - p["beta"] * s[0] ** 3
                              + p["gamma"] * np.cos(s[2]), p["omega"]])

        def true_jac(s):
            return np.array([
                [0.0, 1.0, 0.0],
                [-p["alpha"] - 3 * p["beta"] * s[0] ** 2, -p["delta"], -p["gamma"] * np.sin(s[2])],
                [0.0, 0.0, 0.0],
            ])

        return dict(kind="ode", dim=3, feature_names=["x", "v", "phi"], dt=dt,
                    states=states, states_conf=states_conf, grid_pts=grid_pts,
                    true_rhs=true_rhs, true_jac=true_jac,
                    params=None,  # library-mismatch by design, no coeff recovery (see fit_duffing_forced note)
                    x0_for_lyap=states[0])

    if family == "rossler":
        p = DEFAULT_ROESSLER_PARAMS
        x0 = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.5, size=3)
        t, states = rossler_trajectory(x0, t_span=(0, MEDIUM["rossler_t_end"]),
                                        n_points=MEDIUM["rossler_n_points"], params=p)
        if noise_frac > 0:
            states = states + rng.normal(0, noise_frac * states.std(axis=0), size=states.shape)
        n_discard = int(len(t) * 0.5)
        t_tail, states_tail = t[n_discard:], states[n_discard:]
        dt = t_tail[1] - t_tail[0]

        x0_conf = np.array([1.0, 1.0, 1.0]) + conf_rng.normal(0, 0.5, size=3)
        t_conf, states_conf = rossler_trajectory(x0_conf, t_span=(0, MEDIUM["rossler_t_end"]),
                                                   n_points=MEDIUM["rossler_n_points"], params=p)
        if noise_frac > 0:
            states_conf = states_conf + conf_rng.normal(0, noise_frac * states_conf.std(axis=0), size=states_conf.shape)
        states_conf_tail = states_conf[n_discard:]

        lo, hi = states_tail.min(axis=0), states_tail.max(axis=0)
        grid_pts = grid_rng.uniform(lo, hi, size=(N_OFF_ATTRACTOR_GRID_POINTS, 3))

        def true_rhs(s):
            return rossler_rhs(0.0, s, p["a"], p["b"], p["c"])

        def true_jac(s):
            return rossler_jacobian(s, p["a"], p["b"], p["c"])

        return dict(kind="ode", dim=3, feature_names=["x", "y", "z"], dt=dt,
                    states=states_tail, states_conf=states_conf_tail, grid_pts=grid_pts,
                    true_rhs=true_rhs, true_jac=true_jac,
                    params=None,  # no known-parametric coeff-recovery formula wired for Rossler in this runner
                    x0_for_lyap=states_tail[0])

    raise ValueError(f"unknown family {family!r}")


def _vf_err(pred, true):
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-300))


# ---------------------------------------------------------------------------
# SINDy arm
# ---------------------------------------------------------------------------

def _run_sindy(family, data, degree):
    if data["kind"] == "map":
        traj = data["states"][:, 0]
        r = data["params"]["r"]
        x_n, x_np1 = traj[:-1], traj[1:]
        A = np.stack([x_n ** k for k in range(degree + 1)], axis=1)
        coef, *_ = np.linalg.lstsq(A, x_np1, rcond=None)
        c0, r_hat_lin, r_hat_quad = coef[0], coef[1], -coef[2]
        err_terms = [abs(c0), abs(r_hat_lin - r) / r, abs(r_hat_quad - r) / r]
        if degree >= 3:
            err_terms.append(abs(coef[3]))
        max_rel_err = max(err_terms)
        recovered = bool(max_rel_err < COEFF_TOL)

        def predict(X):
            x = X[:, 0]
            Amat = np.stack([x ** k for k in range(degree + 1)], axis=1)
            return (Amat @ coef).reshape(-1, 1)

        pred_conf = predict(data["states_conf"])
        true_conf = np.array([data["true_rhs"](s) for s in data["states_conf"]])
        pred_grid = predict(data["grid_pts"])
        true_grid = np.array([data["true_rhs"](s) for s in data["grid_pts"]])
        return dict(method="sindy", degree=degree, max_rel_err=float(max_rel_err), recovered=recovered,
                    vf_l2_err_confirmation=_vf_err(pred_conf, true_conf),
                    vf_l2_err_off_attractor_grid=_vf_err(pred_grid, true_grid),
                    dynamically_distinct=bool(_vf_err(pred_conf, true_conf) > VF_ERR_TOL),
                    dynamically_distinct_off_attractor=bool(_vf_err(pred_grid, true_grid) > VF_ERR_TOL))

    model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=degree),
                      optimizer=ps.STLSQ(threshold=STLSQ_THRESHOLD))
    model.fit(data["states"], t=data["dt"], feature_names=data["feature_names"])

    max_rel_err = None
    recovered = None
    if family == "lorenz":
        coeffs = model.coefficients()
        names = model.get_feature_names()
        idx = {n: i for i, n in enumerate(names)}
        sigma_hat, rho_hat, beta_hat = -coeffs[0][idx["x"]], coeffs[1][idx["x"]], -coeffs[2][idx["z"]]
        p = data["params"]
        max_rel_err = max(abs(sigma_hat - p["sigma"]) / p["sigma"],
                           abs(rho_hat - p["rho"]) / max(abs(p["rho"]), 1e-9),
                           abs(beta_hat - p["beta"]) / p["beta"])
        recovered = bool(max_rel_err < COEFF_TOL)
    elif family in ("harmonic", "duffing_unforced"):
        coeffs = model.coefficients()
        names = model.get_feature_names()
        idx = {n: i for i, n in enumerate(names)}
        v_coef_hat = coeffs[0][idx["v"]]
        x_coef_hat = coeffs[1][idx["x"]]
        p = data["params"]
        if family == "harmonic":
            max_rel_err = max(abs(v_coef_hat - p["v_coef"]), abs(x_coef_hat - p["x_coef"]))
        else:
            x3_coef_hat = coeffs[1][idx["x^3"]] if "x^3" in idx else 0.0
            max_rel_err = max(abs(v_coef_hat - p["v_coef"]), abs(x_coef_hat - p["x_coef"]),
                               abs(x3_coef_hat - p["x3_coef"]))
        recovered = bool(max_rel_err < COEFF_TOL)
    # duffing_forced: no coefficient recovery, library mismatch by design (params is None)

    true_conf = np.array([data["true_rhs"](s) for s in data["states_conf"]])
    pred_conf = model.predict(data["states_conf"])
    true_grid = np.array([data["true_rhs"](s) for s in data["grid_pts"]])
    pred_grid = model.predict(data["grid_pts"])
    vf_conf = _vf_err(pred_conf, true_conf)
    vf_grid = _vf_err(pred_grid, true_grid)

    out = dict(method="sindy", degree=degree,
               vf_l2_err_confirmation=vf_conf, vf_l2_err_off_attractor_grid=vf_grid,
               dynamically_distinct=bool(vf_conf > VF_ERR_TOL),
               dynamically_distinct_off_attractor=bool(vf_grid > VF_ERR_TOL))
    if max_rel_err is not None:
        out["max_rel_err"] = float(max_rel_err)
        out["recovered"] = recovered
    else:
        out["library_mismatch_expected"] = True
    return out


# ---------------------------------------------------------------------------
# Symbolic-regression arm
# ---------------------------------------------------------------------------

def _fit_symbolic_regression_map(x_n, x_np1, max_degree, seed):
    """1D discrete-map variant of fit_symbolic_regression's retry loop: fits
    x_{n+1} = f(x_n) directly (no derivative estimation), since the logistic
    map is discrete, not an ODE. Mirrors the ODE version's degree-enforcement
    retry logic (src/discovery_symbolic_regression.py) so the discrete and
    continuous arms use the same acceptance criterion.
    """
    X = x_n.reshape(-1, 1)
    min_depth, max_depth = 2, 2
    while (2 ** (max_depth - 1)) < max_degree and max_depth < 4:
        max_depth += 1
    init_depth = (min_depth, max_depth)

    best = None
    for attempt in range(max(1, SR_MAX_DEGREE_RETRIES)):
        est = SymbolicRegressor(
            population_size=SR_POPULATION_SIZE, generations=SR_GENERATIONS,
            function_set=MATCHED_FUNCTION_SET, init_depth=init_depth,
            parsimony_coefficient=0.001, stopping_criteria=1e-9,
            random_state=seed + attempt, n_jobs=1, feature_names=["x"], verbose=0,
        )
        est.fit(X, x_np1)
        deg, is_poly = program_polynomial_degree(est._program.program)
        satisfies = is_poly and deg is not None and deg <= max_degree
        rank = (0 if satisfies else 1, 0 if is_poly else 1, deg if deg is not None else float("inf"))
        if best is None or rank < best[0]:
            best = (rank, est, deg, is_poly)
        if satisfies:
            break
    _, est, deg, is_poly = best
    return dict(regressor=est, degree=deg, is_polynomial=is_poly,
                degree_ok=bool(is_poly and deg is not None and deg <= max_degree))


def _run_symbolic_regression(family, data, degree, seed):
    if data["kind"] == "map":
        traj = data["states"][:, 0]
        x_n, x_np1 = traj[:-1], traj[1:]
        fit = _fit_symbolic_regression_map(x_n, x_np1, max_degree=degree, seed=seed)

        def predict(X):
            return fit["regressor"].predict(X[:, :1]).reshape(-1, 1)

        pred_conf = predict(data["states_conf"])
        true_conf = np.array([data["true_rhs"](s) for s in data["states_conf"]])
        pred_grid = predict(data["grid_pts"])
        true_grid = np.array([data["true_rhs"](s) for s in data["grid_pts"]])
        return dict(method="symbolic_regression", degree=degree,
                    degrees=[fit["degree"]], is_polynomial=[fit["is_polynomial"]],
                    degree_ok=[fit["degree_ok"]],
                    vf_l2_err_confirmation=_vf_err(pred_conf, true_conf),
                    vf_l2_err_off_attractor_grid=_vf_err(pred_grid, true_grid),
                    dynamically_distinct=bool(_vf_err(pred_conf, true_conf) > VF_ERR_TOL),
                    dynamically_distinct_off_attractor=bool(_vf_err(pred_grid, true_grid) > VF_ERR_TOL))

    from src.discovery_symbolic_regression import fit_symbolic_regression
    result = fit_symbolic_regression(
        data["states"], data["dt"], feature_names=data["feature_names"],
        population_size=SR_POPULATION_SIZE, generations=SR_GENERATIONS,
        max_degree=degree, random_state=seed, n_jobs=1,
        max_degree_retries=SR_MAX_DEGREE_RETRIES,
    )
    regressors = result["regressors"]

    def predict(X):
        return np.column_stack([reg.predict(X) for reg in regressors])

    true_conf = np.array([data["true_rhs"](s) for s in data["states_conf"]])
    pred_conf = predict(data["states_conf"])
    true_grid = np.array([data["true_rhs"](s) for s in data["grid_pts"]])
    pred_grid = predict(data["grid_pts"])
    vf_conf = _vf_err(pred_conf, true_conf)
    vf_grid = _vf_err(pred_grid, true_grid)

    return dict(method="symbolic_regression", degree=degree,
                degrees=result["degrees"], is_polynomial=result["is_polynomial"],
                degree_ok=result["degree_ok"],
                vf_l2_err_confirmation=vf_conf, vf_l2_err_off_attractor_grid=vf_grid,
                dynamically_distinct=bool(vf_conf > VF_ERR_TOL),
                dynamically_distinct_off_attractor=bool(vf_grid > VF_ERR_TOL))


# ---------------------------------------------------------------------------
# Koopman / EDMD arm
# ---------------------------------------------------------------------------

def _run_koopman(family, data, degree):
    dt = data["dt"] if data["dt"] is not None else 1.0
    model = fit_edmd(data["states"], dt=dt, degree=degree, var_names=data["feature_names"])

    states_conf = data["states_conf"]
    X0, X1 = states_conf[:-1], states_conf[1:]
    pred1 = model.predict_state(X0)
    one_step_err = _vf_err(pred1, X1)

    horizon = min(KOOPMAN_ROLLOUT_HORIZON, len(states_conf) - 1)
    rollout = model.simulate(states_conf[0], horizon)
    rollout_err = _vf_err(rollout, states_conf[:horizon + 1])

    true_jac_at_x0 = data["true_jac"](data["x0_for_lyap"])
    true_eigs = np.sort_complex(np.linalg.eigvals(true_jac_at_x0))
    approx_lin = model.linearization()
    model_eigs = np.sort_complex(np.linalg.eigvals(approx_lin))
    spectral_err = float(np.linalg.norm(model_eigs - true_eigs) / max(np.linalg.norm(true_eigs), 1e-300))

    k_cond = float(np.linalg.cond(model.K))

    return dict(method="koopman", degree=degree,
                one_step_rel_rms_err=one_step_err, rollout_horizon=horizon,
                rollout_rel_err=rollout_err, spectral_rel_err=spectral_err,
                k_condition_number=k_cond, residual_rms=model.residual_rms,
                # No coefficient/equation-recovery concept for EDMD (EVIDENCE_INDEX.md row 15).
                )


# ---------------------------------------------------------------------------
# Job dispatch
# ---------------------------------------------------------------------------

def _job_key(family, label, noise_frac, degree, seed):
    return f"{family}|{label}|{noise_frac}|{degree}|{seed}"


def _run_job(job):
    family, regime_args, label, noise_frac, degree, seed = job
    t_start = time.time()
    data = _generate_tier_b_data(family, regime_args, seed, noise_frac)
    sindy_out = _run_sindy(family, data, degree)
    sr_out = _run_symbolic_regression(family, data, degree, seed)
    koopman_out = _run_koopman(family, data, degree)
    wall = time.time() - t_start
    key = _job_key(family, label, noise_frac, degree, seed)
    return dict(key=key, family=family, regime=label, noise_frac=noise_frac, degree=degree,
                seed=seed, wall_clock_s=wall, sindy=sindy_out,
                symbolic_regression=sr_out, koopman=koopman_out)


def _build_jobs():
    jobs = []
    for family, regime_args, label in TIER_B_ITEMS:
        for noise_frac in NOISE_LEVELS_B:
            for degree in LIBRARY_DEGREES:
                for seed in SEEDS:
                    jobs.append((family, regime_args, label, noise_frac, degree, seed))
    return jobs


def _load_checkpoint():
    done = {}
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["key"]] = rec
    return done


def main():
    os.makedirs("experiments/main_study_results", exist_ok=True)
    all_jobs = _build_jobs()
    done = _load_checkpoint()
    pending = [j for j in all_jobs if _job_key(j[0], j[2], j[3], j[4], j[5]) not in done]
    print(f"Tier B: {len(all_jobs)} total jobs, {len(done)} already checkpointed, "
          f"{len(pending)} pending across {N_WORKERS} worker processes.", flush=True)

    t0 = time.time()
    n_done = 0
    with open(CHECKPOINT_PATH, "a") as ckpt_f:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(_run_job, job): job for job in pending}
            for future in as_completed(futures):
                rec = future.result()
                ckpt_f.write(json.dumps(rec, default=float) + "\n")
                ckpt_f.flush()
                done[rec["key"]] = rec
                n_done += 1
                print(f"[{n_done}/{len(pending)}] {rec['family']} {rec['regime']} "
                      f"noise={rec['noise_frac']:.1%} degree={rec['degree']} seed={rec['seed']}: "
                      f"sindy_recovered={rec['sindy'].get('recovered')} "
                      f"sr_degree_ok={rec['symbolic_regression'].get('degree_ok')} "
                      f"koopman_one_step_err={rec['koopman']['one_step_rel_rms_err']:.4g} "
                      f"wall={rec['wall_clock_s']:.1f}s", flush=True)

    results = list(done.values())
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    manifest = [dict(key=r["key"], family=r["family"], regime=r["regime"], noise_frac=r["noise_frac"],
                      degree=r["degree"], seed=r["seed"], wall_clock_s=r["wall_clock_s"]) for r in results]
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=float)

    total_wall = time.time() - t0
    print(f"\nTier B complete. {len(results)}/{len(all_jobs)} jobs checkpointed. "
          f"This run's wall-clock: {total_wall:.1f}s across {N_WORKERS} workers.", flush=True)
    return results


if __name__ == "__main__":
    main()
