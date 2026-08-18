"""Higher-seed-count rerun of the hyperparameter-sensitivity SINDy arm
(NEXT_STEPS.md Tier-3 item #11).

`hyperparameter_sensitivity.py` (NEXT_STEPS.md Tier-2 item #5) ran its
STLSQ-threshold sweep at seeds=[0,1,2] (3 seeds, disclosed reduction from
Tier B's 5). Item #11 asks to rerun the CHEAPEST viable extension at 10+
seeds to see whether the chaos-vs-control finding tightens, stays the same,
or changes. The SINDy/STLSQ-threshold arm is the cheapest candidate (plain
`np.linalg.lstsq`/PySINDy STLSQ fits, no gplearn genetic search) -- the
gplearn population_size arm and the PySR full-grid extension are explicitly
OUT OF SCOPE for a 10-seed rerun per the item's own compute-cost reasoning
(gplearn ~10-60s/fit x 3 pop sizes x 4 regimes x 2 noise x 10 seeds is still
material; PySR at ~200-220s/fit was explicitly called out as infeasible at
10 seeds in NEXT_STEPS.md item 11's own instructions). Only Phase 1 (SINDy,
STLSQ threshold sweep) is rerun here, at seeds=[0..9].

Reuses the exact same copy-adapted `_run_sindy` logic as
`hyperparameter_sensitivity.py` (verified byte-identical below by import
diffing against that module is not attempted -- instead this file imports
`_run_sindy` directly from that module to guarantee identical logic, rather
than re-copy-adapting it a second time).
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments.hyperparameter_sensitivity import (
    NOISE_LEVELS,
    PAIR_ITEMS,
    STLSQ_THRESHOLDS,
    DEGREE,
    N_WORKERS,
    _generate_tier_b_data,
    _job_key,
    _run_sindy,
)

SEEDS_N10 = list(range(10))  # 0..9, extended from hyperparameter_sensitivity.py's [0,1,2]

CHECKPOINT_PATH = "experiments/main_study_results/hyperparameter_sensitivity_n10_checkpoint.jsonl"
RESULTS_PATH = "experiments/main_study_results/hyperparameter_sensitivity_n10_results.json"


def _run_sindy_job(job):
    family, regime_args, label, noise_frac, seed, threshold = job
    t0 = time.time()
    data = _generate_tier_b_data(family, regime_args, seed, noise_frac)
    sindy_out = _run_sindy(family, data, DEGREE, threshold)
    wall = time.time() - t0
    key = _job_key(family, label, noise_frac, seed, "stlsq_threshold", threshold)
    return dict(key=key, sweep="stlsq_threshold", sweep_value=threshold,
                family=family, regime=label, noise_frac=noise_frac, seed=seed,
                degree=DEGREE, wall_clock_s=wall, sindy=sindy_out)


def _build_jobs():
    jobs = []
    for threshold in STLSQ_THRESHOLDS:
        for family, regime_args, label in PAIR_ITEMS:
            for noise_frac in NOISE_LEVELS:
                for seed in SEEDS_N10:
                    jobs.append((family, regime_args, label, noise_frac, seed, threshold))
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
    done = _load_checkpoint()
    jobs = _build_jobs()
    pending = [j for j in jobs if _job_key(j[0], j[2], j[3], j[4], "stlsq_threshold", j[5]) not in done]
    print(f"{len(jobs)} total, {len(done)} checkpointed, {len(pending)} pending, {N_WORKERS} workers.", flush=True)

    t0 = time.time()
    with open(CHECKPOINT_PATH, "a") as ckpt_f:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(_run_sindy_job, job): job for job in pending}
            for i, future in enumerate(as_completed(futures), 1):
                rec = future.result()
                ckpt_f.write(json.dumps(rec, default=float) + "\n")
                ckpt_f.flush()
                done[rec["key"]] = rec
                print(f"[{i}/{len(pending)}] {rec['family']} {rec['regime']} "
                      f"noise={rec['noise_frac']:.1%} threshold={rec['sweep_value']} "
                      f"seed={rec['seed']} wall={rec['wall_clock_s']:.2f}s", flush=True)
    print(f"Complete: {time.time() - t0:.1f}s wall-clock.", flush=True)

    records = list(done.values())
    metadata = dict(
        reason_for_reduction=(
            "Extends hyperparameter_sensitivity.py's Phase 1 (SINDy, STLSQ "
            "threshold sweep) ONLY -- from seeds=[0,1,2] to seeds=0..9 (10 "
            "seeds), per NEXT_STEPS.md item 11's own compute-cost scoping. "
            "Same 2 matched pairs (logistic, lorenz), same noise_frac "
            "{0.0,0.05}, same degree=2, same STLSQ_THRESHOLDS "
            "{0.01,0.05,0.1,0.2}. Phase 2 (gplearn population_size sweep) "
            "and the PySR full-grid extension are explicitly NOT rerun at "
            "10 seeds -- out of scope per item 11's stated reasoning."
        ),
        pair_items=[list(x) for x in PAIR_ITEMS],
        seeds=SEEDS_N10, noise_levels=NOISE_LEVELS, degree=DEGREE,
        stlsq_thresholds=STLSQ_THRESHOLDS,
        baseline_run="experiments/main_study_results/hyperparameter_sensitivity_results.json (seeds=[0,1,2])",
    )
    with open(RESULTS_PATH, "w") as f:
        json.dump(dict(metadata=metadata, records=records), f, indent=2, default=float)
    print(f"\nWrote {len(records)} records to {RESULTS_PATH}.", flush=True)
    return records


if __name__ == "__main__":
    main()
