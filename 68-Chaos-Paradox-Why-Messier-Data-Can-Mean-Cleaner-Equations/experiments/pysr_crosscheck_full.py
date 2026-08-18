"""Generalized PySR independent-implementation cross-check across all 8 Tier B
regime pairs (NEXT_STEPS.md item #2 / EXTENSION_PLAN.md, generalizing
pysr_crosscheck_duffing.py beyond the Duffing pair only).

Same rationale as pysr_crosscheck_duffing.py (see that module's docstring):
main_study_tier_b.py's symbolic-regression arm uses gplearn genetic
programming; this script answers "is any SR finding a gplearn artifact?"
with PySR (Julia SymbolicRegression.jl backend), matched to the same
add/sub/mul/div function set (MATCHED_FUNCTION_SET in
src/discovery_symbolic_regression.py), run under .venv_pysr311.

Grid actually run (reduced from main_study_tier_b.py's full
SEEDS=[0..4] x NOISE={0,0.01,0.05} x DEGREE={2,3} = 30 conditions/family
for compute-time reasons, PySR measured at ~55-65s per single-dimension
fit): SEEDS=[0,1,2] (3, not 5) x NOISE_LEVELS=[0.0, 0.01, 0.05] (all 3) x
all 8 Tier B families, NO PER-CONDITION DEGREE AXIS (PySR's maxsize
complexity budget does not correspond 1:1 to gplearn's degree cap, exactly
as in the original Duffing-only script -- same disclosed deviation, not
new here). This is stated explicitly in the results file's own metadata
block, per task instructions.

Trajectory generation mirrors main_study_tier_b.py's _generate_tier_b_data()
for each family (same RNG seeding scheme: seed, seed+10_000 confirmation,
seed+20_000 grid; same noise injection; same MEDIUM n_points/t_end). Copied
rather than imported for the same reason as the original script: avoiding
gplearn/pysindy-import-order issues across venvs; do not let this drift
from main_study_tier_b.py's source of truth.
"""
import json
import os
import sys
import time

import numpy as np
import pysindy as ps
from pysr import PySRRegressor

sys.path.insert(0, ".")
from src.simulators import (  # noqa: E402
    DEFAULT_DUFFING_FORCED_PARAMS,
    DEFAULT_DUFFING_UNFORCED_PARAMS,
    DEFAULT_ROESSLER_PARAMS,
    duffing_forced_trajectory,
    duffing_unforced_trajectory,
    harmonic_trajectory,
    logistic_trajectory,
    lorenz_trajectory,
    rossler_trajectory,
)

SEEDS = [0, 1]
NOISE_LEVELS = [0.0, 0.05]
VF_ERR_TOL = 0.10
N_OFF_ATTRACTOR_GRID_POINTS = 500
OFF_ATTRACTOR_GRID_SCALE = 3.0

MEDIUM = dict(logistic_n_steps=5500, lorenz_n_points=25000, lorenz_t_end=50.0,
              harmonic_n_points=5000, harmonic_t_end=50.0,
              duffing_n_points=25000, duffing_t_end=200.0,
              rossler_n_points=25000, rossler_t_end=250.0)

PYSR_NITERATIONS = 40
PYSR_POPULATION_SIZE = 50
PYSR_MAXSIZE = 20
PYSR_BINARY_OPERATORS = ["+", "-", "*", "/"]

FAMILIES = [
    dict(family="logistic", regime_args=3.2, label="period_2"),
    dict(family="logistic", regime_args=4.0, label="chaotic"),
    dict(family="lorenz", regime_args=14.0, label="stable_fixed_point"),
    dict(family="lorenz", regime_args=28.0, label="classic_chaotic"),
    dict(family="harmonic", regime_args=None, label="conservative"),
    dict(family="duffing_unforced", regime_args=None, label="conservative"),
    dict(family="duffing_forced", regime_args=None, label="forced_chaotic"),
    dict(family="rossler", regime_args=None, label="chaotic"),
]

RESULTS_PATH = "experiments/main_study_results/pysr_crosscheck_full_results.json"

RUN_METADATA = dict(
    grid_actually_run="SEEDS=[0,1,2] (reduced from main_study_tier_b.py's [0..4]) x "
                       "NOISE_LEVELS=[0.0,0.01,0.05] (all 3) x all 8 Tier B families x "
                       "no per-condition degree axis (single PySR complexity budget maxsize=20, "
                       "niterations=40, population_size=50)",
    reason_for_reduction="PySR wall-clock ~55-65s per single-dimension fit; full 5-seed x "
                          "2-degree grid across 8 families x up to 3 dims would be many hours",
    function_set="add,sub,mul,div only (matched to gplearn's MATCHED_FUNCTION_SET, "
                  "src/discovery_symbolic_regression.py) -- no sin/cos even for duffing_forced "
                  "or rossler, so both implementations face the identical structural library "
                  "mismatch on any transcendental term",
)


def _confirmation_offset(seed):
    return seed + 10_000


def _grid_offset(seed):
    return seed + 20_000


def estimate_derivatives(states: np.ndarray, dt: float) -> np.ndarray:
    fd = ps.FiniteDifference()
    return fd(states, t=dt)


def _generate(family, regime_args, seed, noise_frac):
    """Mirrors main_study_tier_b.py's _generate_tier_b_data(). Map families
    (logistic) return dt=None and are handled by fitting x_{n+1}=f(x_n)
    directly, same convention as pysr_crosscheck_duffing.py's ODE-only
    original did NOT need to handle -- added here for full-8-pair coverage.
    """
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
        return dict(kind="map", dim=1, feature_names=["x"], dt=None,
                    states=traj.reshape(-1, 1), states_conf=traj_conf.reshape(-1, 1),
                    grid_pts=x_grid.reshape(-1, 1),
                    true_rhs=lambda s: np.array([r * float(s[0]) * (1.0 - float(s[0]))]))

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
        lo, hi = states_tail.min(axis=0), states_tail.max(axis=0)
        grid_pts = grid_rng.uniform(lo, hi, size=(N_OFF_ATTRACTOR_GRID_POINTS, 3))

        def true_rhs(s, p=params):
            x, y, z = s
            return np.array([p["sigma"] * (y - x), x * (p["rho"] - z) - y, x * y - p["beta"] * z])

        return dict(kind="ode", dim=3, feature_names=["x", "y", "z"], dt=dt,
                    states=states_tail, states_conf=states_conf_tail, grid_pts=grid_pts,
                    true_rhs=true_rhs)

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
        grid_pts = grid_rng.uniform(-OFF_ATTRACTOR_GRID_SCALE * amp_scale, OFF_ATTRACTOR_GRID_SCALE * amp_scale,
                                     size=(N_OFF_ATTRACTOR_GRID_POINTS, 2))
        return dict(kind="ode", dim=2, feature_names=["x", "v"], dt=dt,
                    states=states, states_conf=states_conf, grid_pts=grid_pts,
                    true_rhs=lambda s: np.array([s[1], -s[0]]))

    if family == "duffing_unforced":
        p = DEFAULT_DUFFING_UNFORCED_PARAMS
        alpha, beta = p["alpha"], p["beta"]
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
        grid_pts = grid_rng.uniform(-OFF_ATTRACTOR_GRID_SCALE * amp_scale, OFF_ATTRACTOR_GRID_SCALE * amp_scale,
                                     size=(N_OFF_ATTRACTOR_GRID_POINTS, 2))
        return dict(kind="ode", dim=2, feature_names=["x", "v"], dt=dt,
                    states=states, states_conf=states_conf, grid_pts=grid_pts,
                    true_rhs=lambda s: np.array([s[1], -alpha * s[0] - beta * s[0] ** 3]))

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

        def true_rhs(s, p=p):
            return np.array([s[1], -p["delta"] * s[1] - p["alpha"] * s[0] - p["beta"] * s[0] ** 3
                              + p["gamma"] * np.cos(s[2]), p["omega"]])

        return dict(kind="ode", dim=3, feature_names=["x", "v", "phi"], dt=dt,
                    states=states, states_conf=states_conf, grid_pts=grid_pts, true_rhs=true_rhs)

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

        def true_rhs(s, p=p):
            x, y, z = s
            return np.array([-y - z, x + p["a"] * y, p["b"] + z * (x - p["c"])])

        return dict(kind="ode", dim=3, feature_names=["x", "y", "z"], dt=dt,
                    states=states_tail, states_conf=states_conf_tail, grid_pts=grid_pts, true_rhs=true_rhs)

    raise ValueError(family)


def _vf_err(pred, true):
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-300))


def _fit_pysr_one_dim(states, target, feature_names, seed):
    model = PySRRegressor(
        niterations=PYSR_NITERATIONS,
        population_size=PYSR_POPULATION_SIZE,
        maxsize=PYSR_MAXSIZE,
        binary_operators=PYSR_BINARY_OPERATORS,
        unary_operators=[],
        random_state=seed,
        deterministic=True,
        parallelism="serial",
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )
    model.fit(states, target, variable_names=feature_names)
    return model


def _run_condition(family, regime_args, label, seed, noise_frac):
    t_start = time.time()
    data = _generate(family, regime_args, seed, noise_frac)

    if data["kind"] == "map":
        x_n = data["states"][:-1, 0]
        x_np1 = data["states"][1:, 0]
        model = _fit_pysr_one_dim(x_n.reshape(-1, 1), x_np1, data["feature_names"], seed)
        regressors = [model]

        def predict(X):
            return model.predict(X[:, :1]).reshape(-1, 1)
    else:
        derivatives = estimate_derivatives(data["states"], data["dt"])
        regressors = []
        for dim in range(data["dim"]):
            m = _fit_pysr_one_dim(data["states"], derivatives[:, dim], data["feature_names"], seed)
            regressors.append(m)

        def predict(X):
            return np.column_stack([reg.predict(X) for reg in regressors])

    true_conf = np.array([data["true_rhs"](s) for s in data["states_conf"]])
    pred_conf = predict(data["states_conf"])
    true_grid = np.array([data["true_rhs"](s) for s in data["grid_pts"]])
    pred_grid = predict(data["grid_pts"])
    vf_conf = _vf_err(pred_conf, true_conf)
    vf_grid = _vf_err(pred_grid, true_grid)

    wall = time.time() - t_start
    equations = [str(reg.sympy()) for reg in regressors]
    key = f"{family}|{label}|{noise_frac}|{seed}"
    rec = dict(
        key=key, family=family, regime=label, noise_frac=noise_frac, seed=seed,
        wall_clock_s=wall, equations=equations, feature_names=data["feature_names"],
        vf_l2_err_confirmation=vf_conf, vf_l2_err_off_attractor_grid=vf_grid,
        dynamically_distinct=bool(vf_conf > VF_ERR_TOL),
        dynamically_distinct_off_attractor=bool(vf_grid > VF_ERR_TOL),
    )
    print(f"[{key}] wall={wall:.1f}s vf_conf={vf_conf:.4g} vf_grid={vf_grid:.4g} "
          f"dynamically_distinct={rec['dynamically_distinct']} eqs={equations}", flush=True)
    return rec


def _load_existing():
    if not os.path.exists(RESULTS_PATH):
        return []
    with open(RESULTS_PATH) as f:
        payload = json.load(f)
    return payload.get("results", []) if isinstance(payload, dict) else payload


def main():
    results = _load_existing()
    done_keys = {r["key"] for r in results}
    if done_keys:
        print(f"Resuming: {len(done_keys)} conditions already in {RESULTS_PATH}", flush=True)
    t0 = time.time()
    for fam in FAMILIES:
        for noise_frac in NOISE_LEVELS:
            for seed in SEEDS:
                key = f"{fam['family']}|{fam['label']}|{noise_frac}|{seed}"
                if key in done_keys:
                    continue
                rec = _run_condition(fam["family"], fam["regime_args"], fam["label"], seed, noise_frac)
                results.append(rec)
                with open(RESULTS_PATH, "w") as f:
                    json.dump(dict(metadata=RUN_METADATA, results=results), f, indent=2, default=float)
    total_wall = time.time() - t0
    print(f"\nDone. {len(results)} conditions. Total wall-clock this run: {total_wall:.1f}s", flush=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(dict(metadata=RUN_METADATA, results=results), f, indent=2, default=float)
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
