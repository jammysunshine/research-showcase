"""Held-out parameter regime, one already-developed family (PREREGISTRATION.md
SS1's "plus one held-out parameter regime per developed family" clause,
distinct from and in addition to the Rössler held-out family in
experiments/main_study_confirmation.py).

Gap this closes: PREREGISTRATION.md SS1 lists exactly 4 logistic-map and 5
Lorenz parameter regimes, and Tier A/B/C (experiments/main_study.py,
main_study_tier_b.py, main_study_tier_c.py) exhaustively used every single
one of them -- no parameter value was ever actually reserved untouched for
a family-level confirmation test, even though SS1's prose calls for one.
Logged as a preregistration-completeness gap in DECISION_LOG.md ("Held-out
parameter regime gap"), remedied here with the minimal defensible fix: one
new Lorenz rho value never referenced anywhere in this repo before this
file was written, run once through the identical SINDy fitting/metric code
Tier A already uses (fit_lorenz, imported directly -- not reimplemented),
at Tier A's full noise/degree/seed grid.

rho=45.0 chosen: strictly outside PREREGISTRATION.md SS1's listed set
{14, 22, 24.5, 28, 100}, deep in the classic Lorenz chaotic band (well past
rho~24.74 onset, well short of the rho=100 hyperchaotic endpoint already
tested), and not a value that appears in any textbook Lorenz exposition
this project has cited (SOURCES.json) -- chosen for being an unremarkable,
non-"lucky" chaotic value, not for a favorable result.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments.main_study import LIBRARY_DEGREES, MEDIUM, NOISE_LEVELS_FULL, SEEDS, fit_lorenz

HELD_OUT_RHO = 45.0


def _run_condition(job):
    rho, noise_frac, degree = job
    t_start = time.time()
    runs = [fit_lorenz(rho, seed, noise_frac, degree, MEDIUM["lorenz_n_points"], MEDIUM["lorenz_t_end"])
            for seed in SEEDS]
    wall = time.time() - t_start
    n_ok = sum(r["recovered"] for r in runs)
    n_vf_ok = sum(not r["dynamically_distinct"] for r in runs)
    n_vf_grid_ok = sum(not r["dynamically_distinct_off_attractor"] for r in runs)
    summary = dict(rho=rho, noise_frac=noise_frac, degree=degree,
                    n_recovered=n_ok, n_total=len(SEEDS),
                    n_vf_recovered=n_vf_ok, n_vf_grid_recovered=n_vf_grid_ok, runs=runs)
    manifest_entry = dict(rho=rho, degree=degree, noise_frac=noise_frac, n_seeds=len(SEEDS), wall_clock_s=wall)
    progress_line = (f"[HeldOutRegime][degree={degree}] lorenz rho={rho} noise={noise_frac:.1%}: "
                      f"{n_ok}/{len(SEEDS)} coeff-recovered; {n_vf_ok}/{len(SEEDS)} confirmation-VF; "
                      f"{n_vf_grid_ok}/{len(SEEDS)} grid-VF; wall={wall:.1f}s")
    return degree, noise_frac, summary, manifest_entry, progress_line


def main():
    jobs = [(HELD_OUT_RHO, noise_frac, degree) for noise_frac in NOISE_LEVELS_FULL for degree in LIBRARY_DEGREES]
    print(f"Held-out Lorenz regime rho={HELD_OUT_RHO}: {len(jobs)} conditions queued.", flush=True)

    results = {}
    manifest = []
    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_run_condition, job): job for job in jobs}
        for future in as_completed(futures):
            degree, noise_frac, summary, manifest_entry, progress_line = future.result()
            results.setdefault(f"degree{degree}", {})[noise_frac] = summary
            manifest.append(manifest_entry)
            n_done += 1
            print(f"[{n_done}/{len(jobs)}] {progress_line}", flush=True)

    with open("experiments/main_study_results/confirmation_regime_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    with open("experiments/main_study_results/confirmation_regime_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=float)

    total_wall = time.time() - t0
    print(f"\nHeld-out regime run complete. Wall-clock: {total_wall:.1f}s across {len(manifest)} conditions.",
          flush=True)
    return results


if __name__ == "__main__":
    main()
