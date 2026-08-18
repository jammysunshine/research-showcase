"""Generalized stage-1 pykoopman cross-check data generator across all 8
Tier B regime pairs (NEXT_STEPS.md item #3, generalizing
koopman_crosscheck_generate.py beyond the 4-condition logistic/lorenz-only
original). See that module's docstring for full rationale.

Stage 1 (this script, main .venv): regenerates representative trajectories
for all 8 Tier B families/regimes, fits this project's own fit_edmd() on
each at degree 2 and 3, dumps raw states + our own fitted-model metrics.
Clean signal only (noise_frac=0.0), single seed (0) -- same convention as
the original 4-condition script: isolating implementation agreement, not
re-running the noise/seed grid.

Stage 2: koopman_crosscheck_pykoopman_full.py (run under .venv_pykoopman311).
"""
import json

import numpy as np

from src.discovery_koopman import fit_edmd
from src.simulators import (
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

SEED = 0
DEGREES = [2, 3]
LORENZ_N_POINTS = 25000
LORENZ_T_END = 50.0
LOGISTIC_N_STEPS = 5500
HARMONIC_N_POINTS = 5000
HARMONIC_T_END = 50.0
DUFFING_N_POINTS = 25000
DUFFING_T_END = 200.0
ROSSLER_N_POINTS = 25000
ROSSLER_T_END = 250.0

CONDITIONS = [
    dict(family="logistic", regime="period_2", r=3.2),
    dict(family="logistic", regime="chaotic", r=4.0),
    dict(family="lorenz", regime="stable_fixed_point", rho=14.0),
    dict(family="lorenz", regime="classic_chaotic", rho=28.0),
    dict(family="harmonic", regime="conservative"),
    dict(family="duffing_unforced", regime="conservative"),
    dict(family="duffing_forced", regime="forced_chaotic"),
    dict(family="rossler", regime="chaotic"),
]


def _generate(cond):
    rng = np.random.default_rng(SEED)
    conf_rng = np.random.default_rng(SEED + 10_000)

    if cond["family"] == "logistic":
        r = cond["r"]
        x0 = 0.1 + 0.3 * rng.random()
        traj = logistic_trajectory(x0, r=r, n_steps=LOGISTIC_N_STEPS, transient=500)
        x0_conf = 0.1 + 0.3 * conf_rng.random()
        traj_conf = logistic_trajectory(x0_conf, r=r, n_steps=LOGISTIC_N_STEPS, transient=500)
        return dict(states=traj.reshape(-1, 1), states_conf=traj_conf.reshape(-1, 1),
                    dt=1.0, feature_names=["x"])

    if cond["family"] == "lorenz":
        rho = cond["rho"]
        params = dict(sigma=10.0, rho=rho, beta=8.0 / 3.0)
        x0 = np.array([-8.0, 8.0, 27.0]) + rng.normal(0, 0.5, size=3)
        t, states = lorenz_trajectory(x0, t_span=(0, LORENZ_T_END), n_points=LORENZ_N_POINTS, params=params)
        n_discard = int(len(t) * 0.5)
        t_tail, states_tail = t[n_discard:], states[n_discard:]
        dt = float(t_tail[1] - t_tail[0])
        x0_conf = np.array([-8.0, 8.0, 27.0]) + conf_rng.normal(0, 0.5, size=3)
        t_conf, states_conf = lorenz_trajectory(x0_conf, t_span=(0, LORENZ_T_END), n_points=LORENZ_N_POINTS, params=params)
        states_conf_tail = states_conf[n_discard:]
        return dict(states=states_tail, states_conf=states_conf_tail, dt=dt,
                    feature_names=["x", "y", "z"])

    if cond["family"] == "harmonic":
        x0 = np.array([1.0, 0.0]) + rng.normal(0, 0.1, size=2)
        t, states = harmonic_trajectory(x0, t_span=(0, HARMONIC_T_END), n_points=HARMONIC_N_POINTS, omega=1.0)
        dt = float(t[1] - t[0])
        x0_conf = np.array([1.0, 0.0]) + conf_rng.normal(0, 0.1, size=2)
        t_conf, states_conf = harmonic_trajectory(x0_conf, t_span=(0, HARMONIC_T_END), n_points=HARMONIC_N_POINTS, omega=1.0)
        return dict(states=states, states_conf=states_conf, dt=dt, feature_names=["x", "v"])

    if cond["family"] == "duffing_unforced":
        x0 = np.array([0.5, 0.0]) + rng.normal(0, 0.05, size=2)
        t, states = duffing_unforced_trajectory(x0, t_span=(0, DUFFING_T_END), n_points=DUFFING_N_POINTS)
        dt = float(t[1] - t[0])
        x0_conf = np.array([0.5, 0.0]) + conf_rng.normal(0, 0.05, size=2)
        t_conf, states_conf = duffing_unforced_trajectory(x0_conf, t_span=(0, DUFFING_T_END), n_points=DUFFING_N_POINTS)
        return dict(states=states, states_conf=states_conf, dt=dt, feature_names=["x", "v"])

    if cond["family"] == "duffing_forced":
        p = DEFAULT_DUFFING_FORCED_PARAMS
        x0 = np.array([0.5, 0.0, 0.0]) + rng.normal(0, 0.05, size=3)
        x0[2] = x0[2] % (2 * np.pi)
        t, states = duffing_forced_trajectory(x0, t_span=(0, DUFFING_T_END), n_points=DUFFING_N_POINTS, params=p)
        dt = float(t[1] - t[0])
        x0_conf = np.array([0.5, 0.0, 0.0]) + conf_rng.normal(0, 0.05, size=3)
        x0_conf[2] = x0_conf[2] % (2 * np.pi)
        t_conf, states_conf = duffing_forced_trajectory(x0_conf, t_span=(0, DUFFING_T_END), n_points=DUFFING_N_POINTS, params=p)
        return dict(states=states, states_conf=states_conf, dt=dt, feature_names=["x", "v", "phi"])

    if cond["family"] == "rossler":
        p = DEFAULT_ROESSLER_PARAMS
        x0 = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.5, size=3)
        t, states = rossler_trajectory(x0, t_span=(0, ROSSLER_T_END), n_points=ROSSLER_N_POINTS, params=p)
        n_discard = int(len(t) * 0.5)
        t_tail, states_tail = t[n_discard:], states[n_discard:]
        dt = float(t_tail[1] - t_tail[0])
        x0_conf = np.array([1.0, 1.0, 1.0]) + conf_rng.normal(0, 0.5, size=3)
        t_conf, states_conf = rossler_trajectory(x0_conf, t_span=(0, ROSSLER_T_END), n_points=ROSSLER_N_POINTS, params=p)
        states_conf_tail = states_conf[n_discard:]
        return dict(states=states_tail, states_conf=states_conf_tail, dt=dt, feature_names=["x", "y", "z"])

    raise ValueError(cond["family"])


def _vf_err(pred, true):
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-300))


def main():
    data_out = {}
    ours_out = []

    for cond in CONDITIONS:
        key = f"{cond['family']}|{cond['regime']}"
        data = _generate(cond)
        data_out[key] = dict(
            states=data["states"].tolist(),
            states_conf=data["states_conf"].tolist(),
            dt=data["dt"],
            feature_names=data["feature_names"],
        )

        for degree in DEGREES:
            model = fit_edmd(data["states"], dt=data["dt"], degree=degree, var_names=data["feature_names"])
            X0, X1 = data["states_conf"][:-1], data["states_conf"][1:]
            pred1 = model.predict_state(X0)
            one_step_err = _vf_err(pred1, X1)
            eigs = np.sort_complex(model.eigenvalues)
            ours_out.append(dict(
                key=key, degree=degree,
                one_step_rel_rms_err=one_step_err,
                residual_rms=model.residual_rms,
                n_snapshot_pairs=model.n_snapshot_pairs,
                k_shape=list(model.K.shape),
                eigenvalues_real=eigs.real.tolist(),
                eigenvalues_imag=eigs.imag.tolist(),
            ))
            print(f"[ours] {key} degree={degree}: one_step_err={one_step_err:.6g} "
                  f"residual_rms={model.residual_rms:.6g} n_pairs={model.n_snapshot_pairs}")

    out_dir = "experiments/main_study_results"
    with open(f"{out_dir}/koopman_crosscheck_full_data.json", "w") as f:
        json.dump(data_out, f)
    with open(f"{out_dir}/koopman_crosscheck_full_ours.json", "w") as f:
        json.dump(ours_out, f, indent=2)
    print("Wrote koopman_crosscheck_full_data.json and koopman_crosscheck_full_ours.json")


if __name__ == "__main__":
    main()
