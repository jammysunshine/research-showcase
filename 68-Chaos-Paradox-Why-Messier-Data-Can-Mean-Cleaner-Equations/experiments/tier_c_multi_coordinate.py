"""NEXT_STEPS.md Tier-2 item #7: does observing a SECOND coordinate reopen
the chaos-vs-control identifiability gap that Tier C found closed under
single-coordinate (x-only) delay embedding?

Tier C (`main_study_tier_c.py`) observes only x(t), delay-embeds it into an
m-dimensional pseudo-state (m = the family's true state dimension), fits
SINDy, and gates on one-step-from-truth prediction error. Headline result
(RESULTS.md "Tier C"): 0/210 `dynamically_distinct` across the full noise
range {0%, 1%, 5%} -- chaotic and non-chaotic regimes are EQUALLY well
predicted one step ahead under delay embedding; the gap Tiers A/B show
under full-state observation is not visible here at all. NEXT_STEPS.md item
#7 asks whether that closure is an artifact of observing only ONE
coordinate (an unusually harsh partial-observation scenario) or whether it
holds even under the more realistic case of observing 2 of the Lorenz
system's 3 coordinates (x and y; z hidden).

SCOPE: Lorenz only (the project's central backbone system, already Tier B/
Tier C's most-tested 3D system), matched pair `stable_fixed_point`
(rho=14.0, non-chaotic control) vs. `classic_chaotic` (rho=28.0), reusing
`_generate_tier_b_data` from main_study_tier_b.py unmodified -- identical
underlying trajectories/noise/RNG seeding to both Tier B and Tier C, only
the OBSERVATION changes.

METHODOLOGICAL DECISION (full reasoning in DECISION_LOG.md "Tier C
multi-coordinate partial observation"): of the two options NEXT_STEPS.md
raised --
  (a) fit SINDy directly on (x, y) as a naively-closed 2D system, or
  (b) reconstruct the missing z dimension via a GENERALIZED (multi-channel)
      Takens embedding using both observed channels,
this script implements (b) only. (a) is not run: dz/dt genuinely depends on
x, y, AND z (Lorenz's z-equation is dz/dt = x*y - beta*z), so any 2D
closed-system fit on (x, y) alone is wrong by construction regardless of
identifiability -- it would not answer NEXT_STEPS.md's question, it would
just demonstrate a known structural fact (a system is not autonomous in a
strict subset of its own coordinates), so running it would not be
informative and was skipped rather than run as a token comparison.

The generalized embedding used: Y(t) = [x(t), y(t), x(t-tau)]. Two raw
observed channels (x, y) supply 2 of the 3 needed coordinates directly (no
reconstruction error on those two, unlike Tier C's x-only case where even
the "directly observed" coordinate is really x(t) alone); a single delayed
copy of x supplies the third, exactly mirroring how Tier C already picks
tau (first 1/e-autocorrelation crossing of x, `choose_delay_by_autocorrelation`
from src/delay_embedding.py, applied to the SAME channel). This keeps the
total embedding dimension identical to Tier C's Lorenz case (3), so the
comparison to Tier C's 0/210 result is apples-to-apples on embedding
dimension, changing only how many of those 3 coordinates are genuine
observations vs. reconstructed delays (1-of-3 genuine here vs. Tier C's
0-of-3). Using y(t-tau) instead of x(t-tau) as the third coordinate would
have been an equally defensible, algebraically symmetric alternative;
x(t-tau) was chosen only for consistency with tau itself being computed
from x, not for any principled reason favoring x over y -- disclosed here
rather than presented as a uniquely correct choice.

Grid: 2 Lorenz regimes x noise {0%, 1%, 5%} (Tier C's own full range,
including its noise=5% extension) x degree {2, 3} x 5 seeds (Tier C's own
seed count) = 60 conditions. Metric, gate, and pass/fail convention are an
exact copy of Tier C's ode-family branch (`_run_tier_c_sindy`'s non-map
path): one-step-from-ground-truth Euler error is the primary gate
(`dynamically_distinct = one_step_err > VF_ERR_TOL`, VF_ERR_TOL=0.10,
avoiding the chaotic-rollout-divergence confound Tier C's own design log
already diagnosed), derivative-based error and fixed-50-step rollout error
kept as secondary diagnostics only.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pysindy as ps

from experiments.main_study_tier_b import (
    N_WORKERS,
    SEEDS,
    STLSQ_THRESHOLD,
    TIER_B_ITEMS,
    VF_ERR_TOL,
    _generate_tier_b_data,
    _vf_err,
)
from src.delay_embedding import choose_delay_by_autocorrelation

NOISE_LEVELS_MC = [0.0, 0.01, 0.05]  # Tier C's full range, including the noise=5% extension
LIBRARY_DEGREES = [2, 3]
EMBEDDING_DIM = 3  # matches Tier C's Lorenz embedding dim, see module docstring

LORENZ_ITEMS = [item for item in TIER_B_ITEMS if item[0] == "lorenz"]
assert len(LORENZ_ITEMS) == 2, f"expected exactly 2 Lorenz regimes, got {LORENZ_ITEMS}"

CHECKPOINT_PATH = "experiments/main_study_results/tier_c_multi_coordinate_checkpoint.jsonl"
RESULTS_PATH = "experiments/main_study_results/tier_c_multi_coordinate_results.json"
MANIFEST_PATH = "experiments/main_study_results/tier_c_multi_coordinate_manifest.json"


def _multi_delay_embed(x, y, tau):
    """Generalized (multi-channel) Takens embedding: Y(t) = [x(t), y(t), x(t-tau)].
    See module docstring for the reasoning. Returns shape (len(x)-tau, 3),
    forward-time-ordered so Y[i+1] is one sample after Y[i]."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x) - tau
    if n <= 1:
        raise ValueError(f"series too short for tau={tau}, len={len(x)}")
    return np.column_stack([x[tau:], y[tau:], x[:n]])


def _run_multi_coordinate_sindy(data, degree):
    x_train, y_train = data["states"][:, 0], data["states"][:, 1]
    x_conf, y_conf = data["states_conf"][:, 0], data["states_conf"][:, 1]
    tau = choose_delay_by_autocorrelation(x_train)
    Y_train = _multi_delay_embed(x_train, y_train, tau)
    Y_conf = _multi_delay_embed(x_conf, y_conf, tau)
    feature_names = ["y0", "y1", "y2"]

    dt = data["dt"]
    model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=degree),
                      optimizer=ps.STLSQ(threshold=STLSQ_THRESHOLD))
    model.fit(Y_train, t=dt, feature_names=feature_names)

    # Same three diagnostics as Tier C's _run_tier_c_sindy ode branch.
    true_deriv = np.gradient(Y_conf, dt, axis=0)
    pred_deriv = model.predict(Y_conf)
    vf_conf = _vf_err(pred_deriv, true_deriv)

    pred_next = Y_conf[:-1] + dt * model.predict(Y_conf[:-1])
    one_step_err = _vf_err(pred_next, Y_conf[1:])

    horizon = min(50, len(Y_conf) - 1)
    t_horizon = np.arange(horizon + 1) * dt
    try:
        rollout = model.simulate(Y_conf[0], t_horizon)
        rollout_err = _vf_err(rollout, Y_conf[:horizon + 1])
    except Exception:
        rollout_err = None

    return dict(method="sindy_multi_coordinate_delay_embedded", degree=degree,
                embedding_dim=EMBEDDING_DIM, delay_tau=int(tau), observed_coords=["x", "y"],
                one_step_rel_rms_err=one_step_err, vf_l2_err_confirmation=vf_conf,
                rollout_rel_err=rollout_err,
                dynamically_distinct=bool(one_step_err > VF_ERR_TOL))


def _job_key(label, noise_frac, degree, seed):
    return f"lorenz|{label}|{noise_frac}|{degree}|{seed}"


def _run_job(job):
    regime_args, label, noise_frac, degree, seed = job
    t_start = time.time()
    data = _generate_tier_b_data("lorenz", regime_args, seed, noise_frac)
    sindy_out = _run_multi_coordinate_sindy(data, degree)
    wall = time.time() - t_start
    key = _job_key(label, noise_frac, degree, seed)
    return dict(key=key, family="lorenz", regime=label, noise_frac=noise_frac, degree=degree,
                seed=seed, wall_clock_s=wall, sindy_multi_coordinate=sindy_out)


def _build_jobs():
    jobs = []
    for family, regime_args, label in LORENZ_ITEMS:
        for noise_frac in NOISE_LEVELS_MC:
            for degree in LIBRARY_DEGREES:
                for seed in SEEDS:
                    jobs.append((regime_args, label, noise_frac, degree, seed))
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
    pending = [j for j in all_jobs if _job_key(j[1], j[2], j[3], j[4]) not in done]
    print(f"Tier C multi-coordinate (x,y observed, z hidden): {len(all_jobs)} total jobs, "
          f"{len(done)} already checkpointed, {len(pending)} pending across {N_WORKERS} "
          f"worker processes.", flush=True)

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
                sr = rec["sindy_multi_coordinate"]
                print(f"[{n_done}/{len(pending)}] {rec['regime']} noise={rec['noise_frac']:.1%} "
                      f"degree={rec['degree']} seed={rec['seed']}: tau={sr['delay_tau']} "
                      f"one_step_err={sr['one_step_rel_rms_err']:.4g} "
                      f"distinct={sr['dynamically_distinct']} wall={rec['wall_clock_s']:.1f}s", flush=True)

    results = list(done.values())
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    manifest = [dict(key=r["key"], family=r["family"], regime=r["regime"], noise_frac=r["noise_frac"],
                      degree=r["degree"], seed=r["seed"], wall_clock_s=r["wall_clock_s"]) for r in results]
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=float)

    total_wall = time.time() - t0
    print(f"\nTier C multi-coordinate complete. {len(results)}/{len(all_jobs)} jobs checkpointed. "
          f"This run's wall-clock: {total_wall:.1f}s across {N_WORKERS} workers.", flush=True)
    return results


if __name__ == "__main__":
    main()
