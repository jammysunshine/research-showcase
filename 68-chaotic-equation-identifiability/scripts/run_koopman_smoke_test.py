"""Koopman/EDMD smoke test: fit the self-implemented EDMD fallback (see
src/discovery_koopman.py -- pykoopman would not install cleanly on this
Python 3.14 environment) on a clean, densely sampled Lorenz trajectory and
report what identifiability-relevant evidence IS available from a Koopman
approach.

Runs the SAME comparison at TWO dictionary degrees -- degree=2 and degree=3
-- on the same trajectory/seed/reference point, side by side. PREREGISTRATION.md
section 2 freezes the SINDy library at "polynomial terms up to degree 3" and
notes Koopman/EDMD should use a monomial dictionary "degree-matched" to it;
this script reports both degrees honestly rather than only the degree that
looks best (LIMITATIONS.md item 2).

Unlike SINDy (scripts/run_sindy_smoke_test.py), EDMD does not hand back
symbolic vector-field coefficients: K is a linear operator on a dictionary
of observables, not a sparse ODE right-hand side. So this script reports
the comparisons that ARE meaningful for a Koopman method, for both degrees:
  1. One-step prediction accuracy on a held-out (independent-initial-
     condition) confirmation trajectory.
  2. Short-horizon trajectory (rollout) error vs the true confirmation
     trajectory, before/after the two diverge past the Lyapunov horizon.
  3. Spectral comparison: eigenvalues of the linear-in-state block of K
     (an approximate local one-step-map linearization) vs eigenvalues of
     exp(J_true * dt) using the analytic Lorenz Jacobian at the same
     reference point.

This is an honest report, not a pass/fail gate against symbolic-recovery
tolerances -- EDMD with a low-degree monomial dictionary is not expected to
recover Lorenz's dynamics exactly (the polynomial dictionary is not
Koopman-invariant for Lorenz), and the smoke test says so explicitly. A
larger (degree-3) dictionary could improve the fit (more expressive) or
degrade it (more observables/parameters fit from the same amount of data,
worse conditioning of the least-squares problem) -- this script reports
whichever actually happens, it does not pick the more flattering degree.
"""
import numpy as np

from src.discovery_koopman import fit_edmd
from src.simulators import DEFAULT_LORENZ_PARAMS, lorenz_jacobian, lorenz_trajectory


def run_for_degree(degree, states_train, dt, states_conf, params, ref_point):
    model = fit_edmd(states_train, dt=dt, degree=degree, var_names=["x", "y", "z"])

    print(f"\n=== Degree = {degree} ===")
    print(f"Dictionary: {model.dictionary.n_features} monomial observables up to degree "
          f"{model.dictionary.degree}: {model.dictionary.names}")
    print(f"Training snapshot pairs: {model.n_snapshot_pairs}, dt={dt:.5f}")
    print(f"Training-set dictionary-space residual RMS: {model.residual_rms:.6e}")
    cond_K = np.linalg.cond(model.K)
    print(f"Condition number of fitted K: {cond_K:.4e}")

    # 1) One-step prediction error across the whole confirmation trajectory.
    X0 = states_conf[:-1]
    X1_true = states_conf[1:]
    X1_pred = model.predict_state(X0)
    one_step_err = np.linalg.norm(X1_pred - X1_true, axis=1)
    state_scale = np.linalg.norm(states_conf, axis=1).mean()
    one_step_rel_rms = np.sqrt(np.mean(one_step_err ** 2)) / state_scale
    print(f"One-step prediction (confirmation trajectory): "
          f"relative RMS error = {one_step_rel_rms:.4%}")

    # 2) Short-horizon rollout error vs the true confirmation trajectory,
    # from several starting points along it, tracking how error grows with
    # horizon length (the chaotic system's Lyapunov time sets an inherent
    # ceiling on any method's rollout accuracy).
    horizons = [1, 10, 50, 100, 300]
    n_starts = 20
    rng = np.random.default_rng(0)
    max_start = states_conf.shape[0] - max(horizons) - 1
    start_idxs = rng.choice(max_start, size=n_starts, replace=False)

    rollout_errs = {}
    print(f"Short-horizon rollout error (relative L2, mean over "
          f"{n_starts} starting points on confirmation trajectory):")
    for h in horizons:
        rel_errs = []
        for s in start_idxs:
            x0 = states_conf[s]
            true_seg = states_conf[s: s + h + 1]
            pred_seg = model.simulate(x0, h)
            err = np.linalg.norm(pred_seg[-1] - true_seg[-1])
            rel_errs.append(err / state_scale)
        mean_rel = float(np.mean(rel_errs))
        rollout_errs[h] = mean_rel
        print(f"  horizon={h:>4d} steps ({h * dt:6.3f} time units): "
              f"mean relative error = {mean_rel:.4f}")

    # 3) Spectral comparison: eigenvalues of K's linear-in-state block
    # (approximate local one-step-map linearization) vs eigenvalues of
    # exp(J_true * dt) from the analytic Lorenz Jacobian, evaluated at the
    # same reference point used for both degrees.
    J_true = lorenz_jacobian(ref_point, params["sigma"], params["rho"], params["beta"])
    true_discrete_eigs = np.linalg.eigvals(np.eye(3) + J_true * dt)  # 1st-order exp(J dt)

    K_lin = model.linearization()
    edmd_eigs = np.linalg.eigvals(K_lin)

    def sorted_by_real(eigs):
        return eigs[np.argsort(-eigs.real)]

    true_sorted = sorted_by_real(true_discrete_eigs)
    edmd_sorted = sorted_by_real(edmd_eigs)

    print(f"Spectral comparison at reference point {ref_point} "
          f"(local linearization, first-order exp(J*dt)):")
    print(f"  True local discrete eigenvalues:  {true_sorted}")
    print(f"  EDMD K linear-block eigenvalues:  {edmd_sorted}")
    spectral_err = np.linalg.norm(edmd_sorted - true_sorted) / np.linalg.norm(true_sorted)
    print(f"  Relative eigenvalue-vector error: {spectral_err:.4%}")

    print(f"Full EDMD dictionary-space eigenvalues (|.|, {len(model.eigenvalues)} total): "
          f"{np.round(np.abs(model.eigenvalues), 4)}")

    return {
        "degree": degree,
        "n_features": model.dictionary.n_features,
        "residual_rms": model.residual_rms,
        "cond_K": cond_K,
        "one_step_rel_rms": one_step_rel_rms,
        "rollout_errs": rollout_errs,
        "spectral_err": spectral_err,
    }


def main():
    params = DEFAULT_LORENZ_PARAMS

    # Training trajectory: clean, densely sampled, on-attractor. Same
    # trajectory/seed used for both degree=2 and degree=3 fits below.
    t_train, states_train = lorenz_trajectory(
        np.array([-8.0, 8.0, 27.0]), t_span=(0, 50), n_points=25000
    )
    dt = t_train[1] - t_train[0]

    # --- Confirmation trajectory: independent initial condition, untouched
    # during fitting (PREREGISTRATION.md section 5 train/confirmation split).
    t_conf, states_conf = lorenz_trajectory(
        np.array([12.0, -5.0, 19.0]), t_span=(0, 50), n_points=25000
    )
    assert abs((t_conf[1] - t_conf[0]) - dt) < 1e-9, "dt mismatch between train/confirmation"

    ref_point = states_train.mean(axis=0)

    results = {}
    for degree in (2, 3):
        results[degree] = run_for_degree(degree, states_train, dt, states_conf, params, ref_point)

    r2, r3 = results[2], results[3]

    # --- Honest side-by-side summary ------------------------------------
    print("\n--- Summary: degree=2 vs degree=3 ---")
    print("EDMD does NOT recover Lorenz's symbolic vector field (unlike SINDy) at "
          "either degree: K is a linear operator on observables, and no finite "
          "monomial dictionary is exactly Koopman-invariant for Lorenz's cubic "
          "nonlinearity (x*z, x*y terms propagate outside the dictionary), so "
          "both fits are lossy finite-dimensional truncations.")
    print(f"Dictionary size: degree=2 -> {r2['n_features']} observables; "
          f"degree=3 -> {r3['n_features']} observables.")
    print(f"Condition number of K: degree=2 -> {r2['cond_K']:.4e}; "
          f"degree=3 -> {r3['cond_K']:.4e}.")
    print(f"One-step relative RMS error: degree=2 -> {r2['one_step_rel_rms']:.4%}; "
          f"degree=3 -> {r3['one_step_rel_rms']:.4%}.")
    one_step_better = "degree=3" if r3['one_step_rel_rms'] < r2['one_step_rel_rms'] else "degree=2"
    print(f"  -> {one_step_better} has lower one-step error.")

    h_ref = 100
    print(f"Rollout error at horizon={h_ref}: degree=2 -> {r2['rollout_errs'][h_ref]:.4f}; "
          f"degree=3 -> {r3['rollout_errs'][h_ref]:.4f}.")
    rollout_better = "degree=3" if r3['rollout_errs'][h_ref] < r2['rollout_errs'][h_ref] else "degree=2"
    print(f"  -> {rollout_better} has lower rollout error at horizon={h_ref}.")

    print(f"Local spectral (eigenvalue-vector) relative error: degree=2 -> "
          f"{r2['spectral_err']:.4%}; degree=3 -> {r3['spectral_err']:.4%}.")
    spectral_better = "degree=3" if r3['spectral_err'] < r2['spectral_err'] else "degree=2"
    print(f"  -> {spectral_better} has lower spectral error.")

    n_metrics_favoring_3 = sum([
        r3['one_step_rel_rms'] < r2['one_step_rel_rms'],
        r3['rollout_errs'][h_ref] < r2['rollout_errs'][h_ref],
        r3['spectral_err'] < r2['spectral_err'],
    ])
    print(f"\nOf 3 metrics compared (one-step, rollout@{h_ref}, spectral), "
          f"{n_metrics_favoring_3}/3 favor degree=3 over degree=2.")
    if r3['cond_K'] > r2['cond_K']:
        print(f"Note: condition number of K INCREASED from degree=2 to degree=3 "
              f"({r2['cond_K']:.4e} -> {r3['cond_K']:.4e}) -- the larger dictionary "
              "is worse-conditioned for the least-squares fit on this amount of data, "
              "consistent with the expected overfitting/conditioning risk of a bigger "
              "dictionary, independent of whether the accuracy metrics above happened "
              "to improve or degrade.")
    else:
        print(f"Note: condition number of K did not increase from degree=2 to degree=3 "
              f"({r2['cond_K']:.4e} -> {r3['cond_K']:.4e}) on this trajectory/seed.")
    print("This comparison is reported as observed on this single trajectory/seed; "
          "it is not evidence that a larger dictionary is generally better or worse -- "
          "see LIMITATIONS.md item 2 for the repo-wide degree-matching status.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
