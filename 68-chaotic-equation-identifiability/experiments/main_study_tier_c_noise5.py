"""Tier C noise-range extension (STATUS.md open item 6): adds the 5% noise
level to Tier C's existing {0%, 1%} grid (NOISE_LEVELS_C in
experiments/main_study_tier_c.py), to distinguish the two open readings of
Tier C's "no gap detected" finding -- genuine closure of the partial-
observation identifiability gap vs. simply insufficient noise to expose it.

Reuses Tier C's own job/run/checkpoint machinery unmodified (_run_tier_c_sindy,
_job_key, _run_job, _load_checkpoint) rather than reimplementing it. Runs
ONLY the new noise=0.05 cells across the same 7 regime pairs x 2 degrees x 5
seeds grid, writing to separate checkpoint/results/manifest paths so the
existing tier_c_checkpoint.jsonl / tier_c_results.json (0%/1% data) are not
touched or re-run.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments.main_study_tier_b import SEEDS, TIER_B_ITEMS
from experiments.main_study_tier_c import LIBRARY_DEGREES, N_WORKERS, _job_key, _run_job

NOISE_LEVELS_C5 = [0.05]

CHECKPOINT_PATH = "experiments/main_study_results/tier_c_noise5_checkpoint.jsonl"
RESULTS_PATH = "experiments/main_study_results/tier_c_noise5_results.json"
MANIFEST_PATH = "experiments/main_study_results/tier_c_noise5_manifest.json"


def _build_jobs():
    jobs = []
    for family, regime_args, label in TIER_B_ITEMS:
        for noise_frac in NOISE_LEVELS_C5:
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
    print(f"Tier C noise=5% extension: {len(all_jobs)} total jobs, {len(done)} already "
          f"checkpointed, {len(pending)} pending across {N_WORKERS} worker processes.", flush=True)

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
                      degree=r["degree"], seed=r["seed"]) for r in results]
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nTier C noise=5% extension DONE: {len(results)} total jobs checkpointed.", flush=True)


if __name__ == "__main__":
    main()
