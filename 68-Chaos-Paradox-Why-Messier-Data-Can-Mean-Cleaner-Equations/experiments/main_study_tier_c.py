"""Tier C main-study runner (MAIN_STUDY_DESIGN.md SS2/SS4 step 7): partial-
observation stress test.

Single-coordinate observation only: instead of fitting on the full state
vector, each job observes x(t) alone (the first component of the same
per-family trajectories generated for Tier B) and reconstructs an
m-dimensional pseudo-state via Takens delay embedding
(src/delay_embedding.py) before fitting. SINDy only -- symbolic regression
and Koopman/EDMD on delay-embedded partial observations are explicitly out
of scope for this pass (MAIN_STUDY_DESIGN.md SS5), left as future work.

Reuses Tier B's exact trajectory-generation code (_generate_tier_b_data)
so the underlying dynamics, noise injection, and RNG seeding are identical
to Tier B -- only the *observation* changes (full state -> x(t) only ->
delay-embedded reconstruction). Same 7 regime pairs as Tier B
(TIER_B_ITEMS), same 7-vs-8 regime-count discrepancy noted in
MAIN_STUDY_DESIGN.md SS2 and already resolved for Tier B
(DECISION_LOG.md "Tier B regime-count discrepancy") -- not re-litigated
here.

Embedding design (logged in DECISION_LOG.md "Tier C delay-embedding
design"): embedding dimension m = the family's true state dimension
(logistic=1, harmonic/duffing_unforced=2, lorenz/duffing_forced=3) -- a
pragmatic choice, NOT a Cao/false-nearest-neighbors-estimated minimal
embedding dimension; delay tau chosen per (family, noise, seed) draw via
the standard first-1/e-autocorrelation-crossing heuristic
(choose_delay_by_autocorrelation). The logistic map is 1-dimensional, so
delay embedding is a no-op and "partial observation" is actually FULL
observation for that family -- included for grid completeness only, and
explicitly flagged as degenerate in the results, not silently treated as
an informative partial-observation test.

Metric (necessarily different from Tiers A/B): delay coordinates are not
the true system's coordinates, so there is no ground-truth coefficient
vector to compare against. Identifiability is instead measured purely by
one-step and rollout predictive accuracy of the fitted delay-embedded
SINDy model against a held-out delay-embedded confirmation trajectory
(same vf_l2_err_confirmation / dynamically_distinct convention used for
Tier B's Koopman/EDMD and duffing_forced arms, EVIDENCE_INDEX.md rows
15/22), NOT the recovered/max_rel_err coefficient-recovery metric used
elsewhere. logistic ("map") uses discrete_time=True SINDy (map iteration);
all other families use continuous SINDy with finite-difference derivative
estimation on the reconstructed trajectory.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pysindy as ps

from experiments.main_study_tier_b import (
    MEDIUM,
    N_OFF_ATTRACTOR_GRID_POINTS,
    SEEDS,
    STLSQ_THRESHOLD,
    TIER_B_ITEMS,
    VF_ERR_TOL,
    _generate_tier_b_data,
    _vf_err,
)
from src.delay_embedding import choose_delay_by_autocorrelation, delay_embed

N_WORKERS = 6  # PROJECT_CHARTER.md local-only compute ceiling, matches Tier A/B
NOISE_LEVELS_C = [0.0, 0.01]  # MAIN_STUDY_DESIGN.md SS2 Tier C: {0%, 1%} only
LIBRARY_DEGREES = [2, 3]

EMBEDDING_DIM = {
    "logistic": 1, "lorenz": 3, "harmonic": 2,
    "duffing_unforced": 2, "duffing_forced": 3,
}

CHECKPOINT_PATH = "experiments/main_study_results/tier_c_checkpoint.jsonl"
RESULTS_PATH = "experiments/main_study_results/tier_c_results.json"
MANIFEST_PATH = "experiments/main_study_results/tier_c_manifest.json"


def _delay_embed_pair(x_train, x_conf, dim):
    """Choose tau from the training series, embed both series with it."""
    tau = choose_delay_by_autocorrelation(x_train) if dim > 1 else 1
    Y_train = delay_embed(x_train, dim, tau)
    Y_conf = delay_embed(x_conf, dim, tau)
    return Y_train, Y_conf, tau


def _run_tier_c_sindy(family, data, degree):
    dim = EMBEDDING_DIM[family]
    x_train = data["states"][:, 0]
    x_conf = data["states_conf"][:, 0]
    Y_train, Y_conf, tau = _delay_embed_pair(x_train, x_conf, dim)
    feature_names = [f"y{i}" for i in range(dim)]

    if data["kind"] == "map":
        # dim=1 (logistic map): delay embedding is a no-op, this is the
        # degenerate full-observation case (see module docstring). Fit the
        # 1D map directly via polynomial lstsq, matching Tier B's map arm
        # (pysindy's discrete-time mode targets multi-trajectory library
        # data, not a plain scalar map iteration).
        x_n, x_np1 = Y_train[:-1, 0], Y_train[1:, 0]
        A = np.stack([x_n ** k for k in range(degree + 1)], axis=1)
        coef, *_ = np.linalg.lstsq(A, x_np1, rcond=None)

        def _predict1(x):
            Amat = np.stack([x ** k for k in range(degree + 1)], axis=1)
            return Amat @ coef

        pred = _predict1(Y_conf[:-1, 0]).reshape(-1, 1)
        true = Y_conf[1:, :]
        vf_conf = _vf_err(pred, true)
        horizon = min(50, len(Y_conf) - 1)
        rollout = np.zeros((horizon + 1, 1))
        rollout[0, 0] = Y_conf[0, 0]
        for i in range(horizon):
            rollout[i + 1, 0] = _predict1(rollout[i:i + 1, 0])[0]
        rollout_err = _vf_err(rollout, Y_conf[:horizon + 1])
    else:
        dt = data["dt"]
        model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=degree),
                          optimizer=ps.STLSQ(threshold=STLSQ_THRESHOLD))
        model.fit(Y_train, t=dt, feature_names=feature_names)
        # Derivative-based error is a noisy diagnostic here: adjacent delay
        # columns are the same underlying signal shifted by tau, so
        # np.gradient across them amplifies noise far more than
        # differentiating true independent state components (Tier A/B).
        true_deriv = np.gradient(Y_conf, dt, axis=0)
        pred_deriv = model.predict(Y_conf)
        vf_conf = _vf_err(pred_deriv, true_deriv)
        # One-step-from-ground-truth state error (Euler-forward from each
        # true Y_conf[t], compared against the true Y_conf[t+1]) mirrors
        # Tier B's Koopman one_step_rel_rms_err convention: every step
        # starts from the true state, so it measures model correctness
        # without letting a chaotic regime's Lyapunov-time sensitivity
        # compound the error the way a multi-step rollout would.
        pred_next = Y_conf[:-1] + dt * model.predict(Y_conf[:-1])
        one_step_err = _vf_err(pred_next, Y_conf[1:])
        horizon = min(50, len(Y_conf) - 1)
        t_horizon = np.arange(horizon + 1) * dt
        try:
            rollout = model.simulate(Y_conf[0], t_horizon)
            rollout_err = _vf_err(rollout, Y_conf[:horizon + 1])
        except Exception:
            rollout_err = None

    # Primary identifiability gate is the one-step-from-ground-truth error,
    # not rollout_rel_err or the derivative-based vf_l2_err_confirmation:
    # a fixed-horizon rollout on a chaotic attractor diverges faster than
    # on a non-chaotic control even for a PERFECTLY identified model
    # (Lyapunov-time sensitivity to any tiny perturbation), which would
    # confound "bad model" with "chaotic dynamics" if used as the gate.
    # One-step-from-truth error avoids that confound. Both alternatives
    # are kept as diagnostics. See DECISION_LOG.md "Tier C delay-embedding
    # design".
    gate_err = one_step_err if data["kind"] != "map" else vf_conf
    return dict(method="sindy_delay_embedded", degree=degree, embedding_dim=dim, delay_tau=int(tau),
                one_step_rel_rms_err=(vf_conf if data["kind"] == "map" else one_step_err),
                vf_l2_err_confirmation=vf_conf,
                rollout_rel_err=rollout_err,
                dynamically_distinct=bool(gate_err > VF_ERR_TOL),
                degenerate_full_observation=bool(dim == 1))


def _job_key(family, label, noise_frac, degree, seed):
    return f"{family}|{label}|{noise_frac}|{degree}|{seed}"


def _run_job(job):
    family, regime_args, label, noise_frac, degree, seed = job
    t_start = time.time()
    data = _generate_tier_b_data(family, regime_args, seed, noise_frac)
    sindy_out = _run_tier_c_sindy(family, data, degree)
    wall = time.time() - t_start
    key = _job_key(family, label, noise_frac, degree, seed)
    return dict(key=key, family=family, regime=label, noise_frac=noise_frac, degree=degree,
                seed=seed, wall_clock_s=wall, sindy_delay_embedded=sindy_out)


def _build_jobs():
    jobs = []
    for family, regime_args, label in TIER_B_ITEMS:
        for noise_frac in NOISE_LEVELS_C:
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
    print(f"Tier C: {len(all_jobs)} total jobs, {len(done)} already checkpointed, "
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
                sr = rec["sindy_delay_embedded"]
                print(f"[{n_done}/{len(pending)}] {rec['family']} {rec['regime']} "
                      f"noise={rec['noise_frac']:.1%} degree={rec['degree']} seed={rec['seed']}: "
                      f"tau={sr['delay_tau']} vf_err={sr['vf_l2_err_confirmation']:.4g} "
                      f"distinct={sr['dynamically_distinct']} wall={rec['wall_clock_s']:.1f}s", flush=True)

    results = list(done.values())
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    manifest = [dict(key=r["key"], family=r["family"], regime=r["regime"], noise_frac=r["noise_frac"],
                      degree=r["degree"], seed=r["seed"], wall_clock_s=r["wall_clock_s"]) for r in results]
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=float)

    total_wall = time.time() - t0
    print(f"\nTier C complete. {len(results)}/{len(all_jobs)} jobs checkpointed. "
          f"This run's wall-clock: {total_wall:.1f}s across {N_WORKERS} workers.", flush=True)
    return results


if __name__ == "__main__":
    main()
