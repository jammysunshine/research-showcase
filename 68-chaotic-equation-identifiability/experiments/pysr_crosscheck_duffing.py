"""Independent-implementation cross-check for the Tier B duffing_forced
symbolic-regression "reversal" (EXTENSION_PLAN.md next-highest-priority item
after the pykoopman Koopman cross-check, same rationale).

main_study_tier_b.py's symbolic-regression arm (gplearn genetic programming,
src/discovery_symbolic_regression.py) finds duffing_forced|forced_chaotic to
be 0/30 "real recovery" under the joint criterion (degree_ok all True AND
dynamically_distinct=False, VF_ERR_TOL=0.10) -- every single seed/noise/
degree job has vf_l2_err_confirmation >> 0.10 -- while its matched
non-chaotic control duffing_unforced|conservative scores 18/30 (RESULTS.md
"Symbolic regression" section, EVIDENCE_INDEX.md). This is the paper's
Tier-B SR "reversal": chaos makes SR *harder*, opposite of the headline
chaos-aids-identifiability pattern for SINDy. Because it rests entirely on
gplearn's genetic-programming search, a reviewer's first objection is "is
this a gplearn artifact?" -- this script answers that with PySR (wraps
Julia's SymbolicRegression.jl, a genuinely different search algorithm/
internals), matching the pykoopman cross-check pattern
(experiments/koopman_crosscheck_pykoopman.py / koopman_crosscheck_generate.py).

Run under .venv_pysr311 (Python 3.11; pysr + Julia backend pre-bootstrapped),
NOT the main project venv. pysindy is also installed there (pip-installed
into .venv_pysr311 for this script only) purely for its FiniteDifference
derivative estimator, to exactly match src/discovery_symbolic_regression.py's
estimate_derivatives() convention -- gplearn/pysr are otherwise independent.

Trajectory generation is a faithful copy of main_study_tier_b.py's
_generate_tier_b_data() duffing_forced/duffing_unforced branches (same RNG
seeding scheme: seed, seed+10_000 for confirmation, seed+20_000 for the
off-attractor grid; same noise injection; same MEDIUM n_points/t_end). It is
copied rather than imported because main_study_tier_b.py imports pysindy
AND gplearn at module scope and gplearn is not installed in .venv_pysr311 --
importing it would require polluting this venv with a dependency this
script doesn't use. Do not let this copy drift from the original; if
_generate_tier_b_data changes, this must change to match.

Function set: gplearn's MATCHED_FUNCTION_SET (src/discovery_symbolic_
regression.py) is exactly ("add", "sub", "mul", "div") -- no trig/exp/log,
per PREREGISTRATION.md section 2. This matters here specifically: the true
duffing_forced RHS contains gamma*cos(phi), a term gplearn's function set
cannot represent even in principle (RESULTS.md's Tier B anomaly note that
gplearn "has sin/cos" is not accurate for this codebase's actual
MATCHED_FUNCTION_SET -- checked directly against src/discovery_symbolic_
regression.py, which defines only add/sub/mul/div). To keep this an
apples-to-apples algorithm comparison (same search *problem*, different
search *engine*) rather than a fairer-library comparison, PySR is matched
to the same operator set: binary_operators=["+","-","*","/"], no unary
operators, no sin/cos for either implementation. This means both
implementations face the identical structural library-mismatch on v_dot's
forcing term; that mismatch is a known, pre-declared confound in the
original finding, not something this cross-check can or should paper over.

Search budget: PySR has no gplearn-style "population_size/generations x
degree cap with retries" structure. Comparable settings are used instead:
niterations=40, population_size=50, maxsize=20 (single fixed complexity
budget, no per-condition "degree" axis -- PySR's complexity-budget dial does
not correspond 1:1 to gplearn's polynomial-degree cap, so no such axis is
invented here; this is a deliberate, disclosed deviation, not an oversight).
Measured wall-clock: ~55-65s per single-dimension PySR fit at this budget
(timed directly on this machine before committing to the run), so a full
5-seed x 3-noise x 2-family x 3-dims grid would be ~30 fits x 3 dims x 60s
~= 90 minutes; a REDUCED grid of 3 seeds x 3 noise x 2 families (both
duffing_forced/forced_chaotic and its matched control duffing_unforced/
conservative, to preserve the chaotic-vs-control contrast that defines the
"reversal") x 3 dims = 54 fits, ~55-60 minutes, was run instead. n=3 seeds
per noise level is enough to report variance, not a single point estimate.
"""
import json
import sys
import time

import numpy as np
import pysindy as ps
from pysr import PySRRegressor

sys.path.insert(0, ".")
from src.simulators import (  # noqa: E402
    DEFAULT_DUFFING_FORCED_PARAMS,
    DEFAULT_DUFFING_UNFORCED_PARAMS,
    duffing_forced_trajectory,
    duffing_unforced_trajectory,
)

SEEDS = [0, 1, 2]
NOISE_LEVELS = [0.0, 0.01, 0.05]
VF_ERR_TOL = 0.10  # matches main_study_tier_b.py
N_OFF_ATTRACTOR_GRID_POINTS = 500
MEDIUM_DUFFING_N_POINTS = 25000
MEDIUM_DUFFING_T_END = 200.0

PYSR_NITERATIONS = 40
PYSR_POPULATION_SIZE = 50
PYSR_MAXSIZE = 20
# Matched to gplearn's MATCHED_FUNCTION_SET (add, sub, mul, div) -- no
# unary/trig operators, see module docstring.
PYSR_BINARY_OPERATORS = ["+", "-", "*", "/"]

FAMILIES = [
    dict(family="duffing_forced", label="forced_chaotic"),
    dict(family="duffing_unforced", label="conservative"),
]

RESULTS_PATH = "experiments/main_study_results/pysr_crosscheck_duffing_results.json"


def _confirmation_offset(seed):
    return seed + 10_000


def _grid_offset(seed):
    return seed + 20_000


def estimate_derivatives(states: np.ndarray, dt: float) -> np.ndarray:
    """Exact copy of src/discovery_symbolic_regression.py's
    estimate_derivatives(): pysindy's default FiniteDifference operator,
    the same one both the original SINDy and gplearn-SR arms use."""
    fd = ps.FiniteDifference()
    return fd(states, t=dt)


def _generate(family, seed, noise_frac):
    """Faithful copy of main_study_tier_b.py's _generate_tier_b_data()
    duffing_forced / duffing_unforced branches. Do not let this drift from
    the original -- see module docstring."""
    rng = np.random.default_rng(seed)
    conf_rng = np.random.default_rng(_confirmation_offset(seed))
    grid_rng = np.random.default_rng(_grid_offset(seed))

    if family == "duffing_unforced":
        p = DEFAULT_DUFFING_UNFORCED_PARAMS
        alpha, beta = p["alpha"], p["beta"]
        x0 = np.array([0.5, 0.0]) + rng.normal(0, 0.05, size=2)
        t, states = duffing_unforced_trajectory(
            x0, t_span=(0, MEDIUM_DUFFING_T_END), n_points=MEDIUM_DUFFING_N_POINTS)
        if noise_frac > 0:
            states = states + rng.normal(0, noise_frac * states.std(axis=0), size=states.shape)
        dt = t[1] - t[0]

        x0_conf = np.array([0.5, 0.0]) + conf_rng.normal(0, 0.05, size=2)
        t_conf, states_conf = duffing_unforced_trajectory(
            x0_conf, t_span=(0, MEDIUM_DUFFING_T_END), n_points=MEDIUM_DUFFING_N_POINTS)
        if noise_frac > 0:
            states_conf = states_conf + conf_rng.normal(0, noise_frac * states_conf.std(axis=0), size=states_conf.shape)
        amp_scale = np.abs(states).max(axis=0)
        grid_pts = grid_rng.uniform(-3.0 * amp_scale, 3.0 * amp_scale,
                                     size=(N_OFF_ATTRACTOR_GRID_POINTS, 2))

        return dict(dim=2, feature_names=["x", "v"], dt=dt,
                    states=states, states_conf=states_conf, grid_pts=grid_pts,
                    true_rhs=lambda s: np.array([s[1], -alpha * s[0] - beta * s[0] ** 3]))

    if family == "duffing_forced":
        p = DEFAULT_DUFFING_FORCED_PARAMS
        x0 = np.array([0.5, 0.0, 0.0]) + rng.normal(0, 0.05, size=3)
        x0[2] = x0[2] % (2 * np.pi)
        t, states = duffing_forced_trajectory(
            x0, t_span=(0, MEDIUM_DUFFING_T_END), n_points=MEDIUM_DUFFING_N_POINTS, params=p)
        if noise_frac > 0:
            states = states + rng.normal(0, noise_frac * states.std(axis=0), size=states.shape)
        dt = t[1] - t[0]

        x0_conf = np.array([0.5, 0.0, 0.0]) + conf_rng.normal(0, 0.05, size=3)
        x0_conf[2] = x0_conf[2] % (2 * np.pi)
        t_conf, states_conf = duffing_forced_trajectory(
            x0_conf, t_span=(0, MEDIUM_DUFFING_T_END), n_points=MEDIUM_DUFFING_N_POINTS, params=p)
        if noise_frac > 0:
            states_conf = states_conf + conf_rng.normal(0, noise_frac * states_conf.std(axis=0), size=states_conf.shape)
        lo, hi = states.min(axis=0), states.max(axis=0)
        grid_pts = grid_rng.uniform(lo, hi, size=(N_OFF_ATTRACTOR_GRID_POINTS, 3))

        def true_rhs(s):
            return np.array([s[1], -p["delta"] * s[1] - p["alpha"] * s[0] - p["beta"] * s[0] ** 3
                              + p["gamma"] * np.cos(s[2]), p["omega"]])

        return dict(dim=3, feature_names=["x", "v", "phi"], dt=dt,
                    states=states, states_conf=states_conf, grid_pts=grid_pts,
                    true_rhs=true_rhs)

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


def _run_condition(family, label, seed, noise_frac):
    t_start = time.time()
    data = _generate(family, seed, noise_frac)
    derivatives = estimate_derivatives(data["states"], data["dt"])

    regressors = []
    for dim in range(data["dim"]):
        model = _fit_pysr_one_dim(data["states"], derivatives[:, dim], data["feature_names"], seed)
        regressors.append(model)

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
    import os
    if not os.path.exists(RESULTS_PATH):
        return []
    with open(RESULTS_PATH) as f:
        return json.load(f)


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
                rec = _run_condition(fam["family"], fam["label"], seed, noise_frac)
                results.append(rec)
                with open(RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2, default=float)
    total_wall = time.time() - t0
    print(f"\nDone. {len(results)} conditions. Total wall-clock: {total_wall:.1f}s", flush=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
