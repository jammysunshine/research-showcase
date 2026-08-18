"""Independent-implementation cross-check for src/discovery_koopman.py's
self-written EDMD (EXTENSION_PLAN.md next-highest-priority item, resolved
2026-08-17).

This project's Koopman/EDMD arm is a self-implemented fallback (pykoopman
originally failed to install under the project's Python 3.14 environment --
see src/discovery_koopman.py's module docstring). Tier B's headline Koopman
finding (one-step error REVERSES chaotic vs. non-chaotic relative to
SINDy/SR) is the paper's most novel/surprising result, so a reviewer's first
objection would be "is this a bug in your homemade EDMD?". This script
answers that by fitting the same snapshot data with the independent
third-party `pykoopman` library (installed separately under .venv_pykoopman311,
a Python 3.11 venv, since pykoopman's dependency chain is incompatible with
this project's main Python 3.14 venv) and comparing fitted-model outputs.

Stage 1 (this script, run under the main .venv): regenerates representative
trajectories for the two matched chaotic/non-chaotic pairs used in Tier B
(logistic period_2/chaotic, Lorenz stable_fixed_point/classic_chaotic),
fits this project's own fit_edmd() on each at degree 2 and 3, and dumps
both the raw states and this project's own fitted-model metrics to
koopman_crosscheck_data.json / koopman_crosscheck_ours.json. Clean signal
only (noise_frac=0.0), single seed (0): the goal is isolating implementation
agreement, not re-running the noise/seed grid.

Stage 2 (experiments/koopman_crosscheck_pykoopman.py, run under
.venv_pykoopman311) loads the dumped states and fits pykoopman's own
Koopman(observables=Polynomial(degree=d, include_bias=True),
regressor=EDMD()) -- the closest API-level match to this project's monomial
dictionary + lstsq EDMD -- then writes comparable metrics to
koopman_crosscheck_pykoopman.json for side-by-side comparison.
"""
import json

import numpy as np

from src.discovery_koopman import fit_edmd
from src.simulators import logistic_trajectory, lorenz_trajectory

SEED = 0
DEGREES = [2, 3]
LORENZ_N_POINTS = 25000
LORENZ_T_END = 50.0
LOGISTIC_N_STEPS = 5500

CONDITIONS = [
    dict(family="logistic", regime="period_2", r=3.2),
    dict(family="logistic", regime="chaotic", r=4.0),
    dict(family="lorenz", regime="stable_fixed_point", rho=14.0),
    dict(family="lorenz", regime="classic_chaotic", rho=28.0),
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
    with open(f"{out_dir}/koopman_crosscheck_data.json", "w") as f:
        json.dump(data_out, f)
    with open(f"{out_dir}/koopman_crosscheck_ours.json", "w") as f:
        json.dump(ours_out, f, indent=2)
    print("Wrote koopman_crosscheck_data.json and koopman_crosscheck_ours.json")


if __name__ == "__main__":
    main()
