"""Stage 2 of the pykoopman cross-check (see koopman_crosscheck_generate.py's
module docstring for full context). Run under .venv_pykoopman311, NOT the
main project venv -- pykoopman's dependency chain requires Python 3.11 and
is incompatible with this project's main Python 3.14 venv.

Loads the states dumped by koopman_crosscheck_generate.py and fits
pykoopman's own EDMD (Koopman(observables=Polynomial(degree=d,
include_bias=True), regressor=EDMD())), the closest API-level match to this
project's monomial-dictionary + lstsq EDMD (src/discovery_koopman.py).
Writes comparable one-step-error / spectrum metrics to
koopman_crosscheck_pykoopman.json for side-by-side comparison against
koopman_crosscheck_ours.json.
"""
import json

import numpy as np
import pykoopman as pk

DEGREES = [2, 3]
DATA_PATH = "experiments/main_study_results/koopman_crosscheck_data.json"
OUT_PATH = "experiments/main_study_results/koopman_crosscheck_pykoopman.json"


def _vf_err(pred, true):
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-300))


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    out = []
    for key, d in data.items():
        states = np.array(d["states"])
        states_conf = np.array(d["states_conf"])
        dt = d["dt"]

        for degree in DEGREES:
            model = pk.Koopman(
                observables=pk.observables.Polynomial(degree=degree, include_bias=True),
                regressor=pk.regression.EDMD(),
            )
            model.fit(states, dt=dt)

            X0, X1 = states_conf[:-1], states_conf[1:]
            pred1 = model.predict(X0)
            one_step_err = _vf_err(pred1, X1)

            resid = model.psi(states[1:].T) - model.A @ model.psi(states[:-1].T)
            residual_rms = float(np.sqrt(np.mean(resid ** 2)))

            eigs = np.sort_complex(np.linalg.eigvals(model.A))

            out.append(dict(
                key=key, degree=degree,
                one_step_rel_rms_err=one_step_err,
                residual_rms=residual_rms,
                a_shape=list(model.A.shape),
                eigenvalues_real=eigs.real.tolist(),
                eigenvalues_imag=eigs.imag.tolist(),
            ))
            print(f"[pykoopman] {key} degree={degree}: one_step_err={one_step_err:.6g} "
                  f"residual_rms={residual_rms:.6g} A_shape={model.A.shape}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
