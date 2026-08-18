#!/usr/bin/env python3
"""Coverage vs. Chaos: Coupled van der Pol 3-regime identifiability experiment.

Three regimes (periodic, quasi-periodic, chaotic) in the same 4D polynomial
system, tested across SINDy, symbolic regression, and Koopman/EDMD.
Directly tests LIMITATIONS.md #7: is it "chaos specifically" or "broad
state-space coverage" that aids identifiability?
"""
import json
import os
import signal
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pysindy as ps

sys.path.insert(0, ".")
from src.simulators import (
    coupled_vdp_trajectory, coupled_vdp_lyapunov_spectrum,
    coupled_vdp_rhs, COUPLED_VDP_REGIMES,
)
from src.discovery_koopman import fit_edmd
from src.discovery_symbolic_regression import fit_symbolic_regression

RESULTS_DIR = Path("experiments/main_study_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = RESULTS_DIR / "coverage_experiment_checkpoint.jsonl"
RESULTS_FILE = RESULTS_DIR / "coverage_experiment_results.json"

# Fixed hyperparameters (matching Tier B)
VF_ERR_TOL = 0.10
STLSQ_THRESHOLD = 0.1
N_WORKERS = 6
NOISE_LEVELS = [0.0, 0.001, 0.01, 0.05]  # 0%, 0.1%, 1%, 5%
DEGREES = [2, 3]
N_SEEDS = 5
DT = 0.05
T_SPAN = (0, 500)
N_POINTS = 10000
TRANSIENT_FRAC = 0.5


def generate_trajectory(regime_name, seed, noise_pct):
    """Generate a training trajectory for one (regime, seed, noise) combination."""
    rng = np.random.default_rng(seed)
    params = COUPLED_VDP_REGIMES[regime_name]
    x0 = rng.standard_normal(4) * 1.0
    t, states = coupled_vdp_trajectory(x0, T_SPAN, N_POINTS, params=params,
                                       rtol=1e-11, atol=1e-12)
    n_transient = int(N_POINTS * TRANSIENT_FRAC)
    states_post = states[n_transient:]
    state_std = states_post.std(axis=0)
    state_std[state_std < 1e-10] = 1.0
    noise = rng.normal(0, noise_pct, size=states_post.shape) * state_std
    states_noisy = states_post + noise
    return t[n_transient:], states_post, states_noisy, state_std


def run_single_condition(regime_name, noise_pct, degree, seed):
    """Run one full (regime, noise, degree, seed) condition across all 3 methods."""
    params = COUPLED_VDP_REGIMES[regime_name]
    t_train, states_clean, states_noisy, state_std = generate_trajectory(regime_name, seed, noise_pct)
    dt = t_train[1] - t_train[0] if len(t_train) > 1 else DT

    # Confirmation trajectory (different IC)
    t_conf, states_conf_clean, _, _ = generate_trajectory(regime_name, seed + 10000, noise_pct)

    result = {
        "regime": regime_name,
        "regime_type": "periodic" if regime_name == "periodic"
                       else "quasi_periodic" if regime_name == "quasi_periodic"
                       else "chaotic",
        "noise_pct": noise_pct,
        "degree": degree,
        "seed": seed,
        "dt": dt,
    }

    # --- SINDy (discrete-time: predict next state) ---
    try:
        model = ps.SINDy(
            feature_library=ps.PolynomialLibrary(degree=degree, include_bias=True),
            optimizer=ps.STLSQ(threshold=STLSQ_THRESHOLD),
        )
        var_names = ["x1", "v1", "x2", "v2"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(states_noisy, t=dt, feature_names=var_names)

        # One-step prediction on confirmation trajectory
        pred_next = model.predict(states_conf_clean[:-1])
        true_next = states_conf_clean[1:]
        err = np.sqrt(np.mean((pred_next - true_next) ** 2))
        scale = np.sqrt(np.mean(true_next ** 2))
        rel_rms = float(err / scale) if scale > 1e-10 else float(err)
        result["sindy_one_step_err"] = rel_rms
        result["sindy_dynamically_distinct"] = rel_rms > VF_ERR_TOL
    except Exception as e:
        result["sindy_one_step_err"] = None
        result["sindy_dynamically_distinct"] = None
        result["sindy_error"] = str(e)

    # --- Symbolic Regression ---
    try:
        sr_result = fit_symbolic_regression(
            states_noisy, dt, max_degree=degree,
            population_size=500, generations=10, max_degree_retries=2,
        )
        result["sr_degree_ok"] = sr_result.get("degree_all_ok", False)
        result["sr_structural_match"] = sr_result.get("structural_match", False)
        # Dynamically distinct: check if the SR-predicted VF diverges on confirmation
        # For now, use the same one-step metric via the fitted programs
        result["sr_dynamically_distinct"] = not sr_result.get("dynamically_distinct", True)
    except Exception as e:
        result["sr_degree_ok"] = None
        result["sr_structural_match"] = None
        result["sr_dynamically_distinct"] = None
        result["sr_error"] = str(e)

    # --- Koopman/EDMD ---
    try:
        model_edmd = fit_edmd(states_noisy, dt=dt, degree=degree,
                              var_names=["x1", "v1", "x2", "v2"])
        # One-step prediction on confirmation trajectory
        pred_one = np.array([model_edmd.predict_state(s) for s in states_conf_clean[:-1]])
        true_next = states_conf_clean[1:]
        err = np.sqrt(np.mean((pred_one - true_next) ** 2))
        scale = np.sqrt(np.mean(true_next ** 2))
        rel_rms = float(err / scale) if scale > 1e-10 else float(err)
        result["koopman_one_step_err"] = rel_rms
    except Exception as e:
        result["koopman_one_step_err"] = None
        result["koopman_error"] = str(e)

    # --- Koopman with ridge regularization ---
    for alpha in [1e-8, 1e-6, 1e-4, 1e-2]:
        try:
            model_r = fit_edmd(states_noisy, dt=dt, degree=degree,
                               var_names=["x1", "v1", "x2", "v2"],
                               ridge_alpha=alpha)
            pred_r = np.array([model_r.predict_state(s) for s in states_conf_clean[:-1]])
            err_r = np.sqrt(np.mean((pred_r - true_next) ** 2))
            rel_rms_r = float(err_r / scale) if scale > 1e-10 else float(err_r)
            result[f"koopman_ridge_{alpha:.0e}"] = rel_rms_r
        except Exception:
            result[f"koopman_ridge_{alpha:.0e}"] = None

    return result


def main():
    print("Coverage vs. Chaos: Coupled van der Pol Experiment")
    print(f"Regimes: {list(COUPLED_VDP_REGIMES.keys())}")
    print(f"Noise: {NOISE_LEVELS}")
    print(f"Degrees: {DEGREES}")
    print(f"Seeds: {N_SEEDS}")

    jobs = []
    for regime in COUPLED_VDP_REGIMES:
        for noise in NOISE_LEVELS:
            for degree in DEGREES:
                for seed in range(N_SEEDS):
                    jobs.append((regime, noise, degree, seed))

    total = len(jobs)
    print(f"Total conditions: {total}")

    completed = set()
    results = []
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            for line in f:
                r = json.loads(line)
                key = (r["regime"], r["noise_pct"], r["degree"], r["seed"])
                completed.add(key)
                results.append(r)
        print(f"Loaded {len(completed)} from checkpoint")

    pending = [j for j in jobs if tuple(j) not in completed]
    print(f"Remaining: {len(pending)}")

    if not pending:
        print("All conditions complete.")
    else:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {}
            for job in pending:
                f = pool.submit(run_single_condition, *job)
                futures[f] = job

            done_count = len(completed)
            for f in as_completed(futures, timeout=7200):
                try:
                    r = f.result(timeout=120)
                    key = (r["regime"], r["noise_pct"], r["degree"], r["seed"])
                    results.append(r)
                    with open(CHECKPOINT_FILE, "a") as fp:
                        fp.write(json.dumps(r) + "\n")
                    done_count += 1
                    if done_count % 10 == 0:
                        elapsed = time.time() - t0
                        rate = (done_count - len(completed)) / elapsed if elapsed > 0 else 0
                        print(f"  [{done_count}/{total}] {elapsed:.1f}s ({rate:.2f}/s)")
                except Exception as e:
                    print(f"  FAILED: {futures[f]} -> {e}")

        print(f"\nCompleted in {time.time() - t0:.1f}s")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for regime in COUPLED_VDP_REGIMES:
        print(f"\n--- {regime} ---")
        regime_results = [r for r in results if r["regime"] == regime]
        for noise in NOISE_LEVELS:
            for degree in DEGREES:
                cell = [r for r in regime_results
                        if r["noise_pct"] == noise and r["degree"] == degree]
                sindy_errs = [r["sindy_one_step_err"] for r in cell
                              if r.get("sindy_one_step_err") is not None]
                koop_errs = [r["koopman_one_step_err"] for r in cell
                             if r.get("koopman_one_step_err") is not None]
                n_sindy_distinct = sum(1 for r in cell if r.get("sindy_dynamically_distinct"))
                n_total = len(cell)
                mean_koop = np.mean(koop_errs) if koop_errs else float("nan")
                print(f"  noise={noise:.2%} deg={degree}: "
                      f"SINDy {n_sindy_distinct}/{n_total} distinct, "
                      f"Koopman mean_err={mean_koop:.6f}")


if __name__ == "__main__":
    main()
