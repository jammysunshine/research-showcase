"""NEXT_STEPS.md Tier-2 item #6: fix a noise-injection scaling artifact on
the Tier B duffing_forced (forced-chaotic Duffing) regime's phase variable
phi, then rerun the affected 30 conditions (all 3 methods x noise
{0%,1%,5%} x degree {2,3} x seed 0-4) for comparison against the original
tier_b_results.json duffing_forced records.

THE ARTIFACT (DECISION_LOG.md has the full writeup): duffing_forced's state
is (x, v, phi), where phi is carried as an autonomous third state with
dphi/dt = omega = const (src/simulators.py duffing_forced_rhs), so the
integrated trajectory's phi column is a straight, UNBOUNDED ramp over the
full t_end=200 window (phi0 -> phi0 + omega*200 ~= phi0 + 240 radians).
main_study_tier_b.py's _generate_tier_b_data() injects per-dimension noise
as `noise_frac * states.std(axis=0)` (one std per column). For x and v this
is a sensible bounded amplitude scale. For phi, `states[:, 2].std()` is the
std of a near-uniform ramp over ~38 forcing cycles -- large (order tens of
radians) and, critically, GROWING with t_end/n_points, not a property of
the phase variable's actual dynamical range. This means the absolute noise
magnitude added to phi at noise_frac=5% is a modeling artifact of trajectory
length, not a real "5% noise" in any physically meaningful sense on that
channel.

THE FIX (approach (a) from NEXT_STEPS.md item #6's two options, chosen and
justified in DECISION_LOG.md): compute the phi-channel noise MAGNITUDE from
a wrapped copy of phi (`states[:, 2] % (2*pi)`, whose std is bounded at
~2*pi/sqrt(12) ~= 1.81 regardless of trajectory length -- the std of a
variable uniformly covering one cycle), while adding that noise to the
RAW/UNWRAPPED phi column exactly as before. This is deliberately NOT
approach (b) (replacing phi with (sin(phi), cos(phi)) in the observed
state): that would change the SINDy/SR feature basis and Koopman
observable dictionary for this one regime, conflating "did the noise-scale
artifact affect prior findings" with "does a different observation basis
change the story" -- two different questions. This script isolates only the
former. The state phi itself remains raw/unwrapped/continuous going into
every discovery method exactly as in the original pipeline -- only the
NOISE MAGNITUDE computation changes. No 2*pi->0 discontinuity is introduced
anywhere (unlike wrapping the state itself would produce).

Everything else (RNG seeding scheme, x0 sampling, dt, off-attractor grid,
confirmation trajectory, degree/noise/seed grid, SINDy/SR/Koopman fitting
code) is reused UNCHANGED by importing directly from main_study_tier_b.py
(_run_sindy, _run_symbolic_regression, _run_koopman, _job_key, N_WORKERS,
SEEDS, NOISE_LEVELS_B, LIBRARY_DEGREES, MEDIUM, N_OFF_ATTRACTOR_GRID_POINTS)
-- only the duffing_forced branch of _generate_tier_b_data is copied
(faithfully, same pattern as experiments/pysr_crosscheck_duffing.py) and
modified for the phi-noise fix. main_study_tier_b.py itself is NOT edited:
its existing tier_b_checkpoint.jsonl/tier_b_results.json (the frozen
30 original duffing_forced records) are left completely untouched. Results
from this script go to a new file, duffing_forced_phi_wrapped_results.json.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from experiments.main_study_tier_b import (
    LIBRARY_DEGREES,
    MEDIUM,
    N_OFF_ATTRACTOR_GRID_POINTS,
    N_WORKERS,
    NOISE_LEVELS_B,
    SEEDS,
    _run_koopman,
    _run_sindy,
    _run_symbolic_regression,
)
from src.simulators import DEFAULT_DUFFING_FORCED_PARAMS, duffing_forced_trajectory

FAMILY = "duffing_forced"
LABEL = "forced_chaotic"

CHECKPOINT_PATH = "experiments/main_study_results/duffing_forced_phi_wrapped_checkpoint.jsonl"
RESULTS_PATH = "experiments/main_study_results/duffing_forced_phi_wrapped_results.json"


def _confirmation_offset(seed):
    return seed + 10_000


def _grid_offset(seed):
    return seed + 20_000


def _generate_phi_wrapped(seed, noise_frac):
    """Faithful copy of main_study_tier_b.py's _generate_tier_b_data()
    duffing_forced branch, with ONE change: the noise magnitude for the phi
    column (index 2) is computed from a wrapped (mod 2*pi) copy of phi
    rather than the raw unbounded ramp. See module docstring."""
    rng = np.random.default_rng(seed)
    conf_rng = np.random.default_rng(_confirmation_offset(seed))
    grid_rng = np.random.default_rng(_grid_offset(seed))

    p = DEFAULT_DUFFING_FORCED_PARAMS
    x0 = np.array([0.5, 0.0, 0.0]) + rng.normal(0, 0.05, size=3)
    x0[2] = x0[2] % (2 * np.pi)
    t, states = duffing_forced_trajectory(x0, t_span=(0, MEDIUM["duffing_t_end"]),
                                            n_points=MEDIUM["duffing_n_points"], params=p)
    if noise_frac > 0:
        noise_std = states.std(axis=0).copy()
        noise_std[2] = (states[:, 2] % (2 * np.pi)).std()  # bounded phi noise scale (THE FIX)
        states = states + rng.normal(0, noise_frac * noise_std, size=states.shape)
    dt = t[1] - t[0]

    x0_conf = np.array([0.5, 0.0, 0.0]) + conf_rng.normal(0, 0.05, size=3)
    x0_conf[2] = x0_conf[2] % (2 * np.pi)
    t_conf, states_conf = duffing_forced_trajectory(x0_conf, t_span=(0, MEDIUM["duffing_t_end"]),
                                                      n_points=MEDIUM["duffing_n_points"], params=p)
    if noise_frac > 0:
        noise_std_conf = states_conf.std(axis=0).copy()
        noise_std_conf[2] = (states_conf[:, 2] % (2 * np.pi)).std()  # THE FIX (confirmation traj)
        states_conf = states_conf + conf_rng.normal(0, noise_frac * noise_std_conf, size=states_conf.shape)
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
                params=None,  # library-mismatch by design, no coeff recovery (matches original)
                x0_for_lyap=states[0])


def _job_key(noise_frac, degree, seed):
    return f"{FAMILY}|{LABEL}|{noise_frac}|{degree}|{seed}"


def _run_job(job):
    noise_frac, degree, seed = job
    t_start = time.time()
    data = _generate_phi_wrapped(seed, noise_frac)
    sindy_out = _run_sindy(FAMILY, data, degree)
    sr_out = _run_symbolic_regression(FAMILY, data, degree, seed)
    koopman_out = _run_koopman(FAMILY, data, degree)
    wall = time.time() - t_start
    key = _job_key(noise_frac, degree, seed)
    return dict(key=key, family=FAMILY, regime=LABEL, noise_frac=noise_frac, degree=degree,
                seed=seed, wall_clock_s=wall, sindy=sindy_out,
                symbolic_regression=sr_out, koopman=koopman_out)


def _build_jobs():
    jobs = []
    for noise_frac in NOISE_LEVELS_B:
        for degree in LIBRARY_DEGREES:
            for seed in SEEDS:
                jobs.append((noise_frac, degree, seed))
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
    pending = [j for j in all_jobs if _job_key(j[0], j[1], j[2]) not in done]
    print(f"duffing_forced phi-wrapped rerun: {len(all_jobs)} total jobs, {len(done)} already "
          f"checkpointed, {len(pending)} pending across {N_WORKERS} worker processes.", flush=True)

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
                print(f"[{n_done}/{len(pending)}] noise={rec['noise_frac']:.1%} "
                      f"degree={rec['degree']} seed={rec['seed']}: "
                      f"sindy_recovered={rec['sindy'].get('recovered')} "
                      f"sr_dyn_distinct={rec['symbolic_regression'].get('dynamically_distinct')} "
                      f"koopman_one_step_err={rec['koopman']['one_step_rel_rms_err']:.4g} "
                      f"wall={rec['wall_clock_s']:.1f}s", flush=True)

    results = list(done.values())
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)

    total_wall = time.time() - t0
    print(f"\nDone. {len(results)}/{len(all_jobs)} jobs checkpointed. "
          f"This run's wall-clock: {total_wall:.1f}s across {N_WORKERS} workers.", flush=True)
    return results


if __name__ == "__main__":
    main()
