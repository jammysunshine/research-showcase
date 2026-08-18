"""Hyperparameter-sensitivity sweep (NEXT_STEPS.md Tier-2 item #5,
LIMITATIONS.md #4 "Undocumented hyperparameter choices").

DECISION_LOG.md's 2026-08-16 "Discovery-method hyperparameters" entry logged
STLSQ_THRESHOLD=0.1 and gplearn's population_size/generations as untuned
library defaults, never varied against this repo's own results. This script
empirically closes that gap: does main_study_tier_b.py's chaos-vs-control
identifiability gap survive across a range of these two hyperparameters, or
is Tier B's headline finding an artifact of the one pinned value tested?

Reuse strategy: `_generate_tier_b_data`, `_vf_err`, `TIER_B_ITEMS`,
`COEFF_TOL`, `VF_ERR_TOL`, `SR_MAX_DEGREE_RETRIES` are imported UNMODIFIED
from experiments/main_study_tier_b.py (frozen, never edited by this script).
`fit_symbolic_regression` (src/discovery_symbolic_regression.py) already
takes `population_size`/`generations` as explicit parameters, so it is
imported and called directly, unmodified, for the ODE-family SR arm.

However, Tier B's `_run_sindy` and `_fit_symbolic_regression_map` hardcode
STLSQ_THRESHOLD / SR_POPULATION_SIZE / SR_GENERATIONS as MODULE-LEVEL
GLOBALS inside their bodies -- since this script's entire purpose is
varying those two hyperparameters per run, they cannot be imported and used
as-is. They are copy-adapted below with threshold/population_size/
generations promoted to explicit function parameters (logic otherwise
identical, verified against tier_b_results.json at the shared baseline
values -- see DECISION_LOG.md entry for this item). This mirrors an
existing precedent in this project for independent per-file constant
duplication: OFF_ATTRACTOR_GRID_SCALE is defined separately in both
main_study.py and main_study_tier_b.py rather than shared (DECISION_LOG.md
"Off-attractor grid bounds derived from amplitude scale").

Grid (disclosed reductions from Tier B's own grid, per PROMPT.md claim
discipline):
  - 2 matched pairs only (not all 8 Tier B regimes): logistic
    (period_2 r=3.2 vs chaotic r=4.0) and lorenz (stable_fixed_point
    rho=14.0 vs classic_chaotic rho=28.0) -- regime_args taken verbatim
    from TIER_B_ITEMS.
  - noise_frac in {0.0, 0.05} only (Tier B also tests 0.01).
  - degree=2 only (Tier B also tests degree=3).
  - seeds=[0,1,2] (REDUCED from Tier B's 5 seeds). Disclosed explicitly,
    same convention as the PySR-full-grid seed reduction disclosed in
    DECISION_LOG.md's "2026-08-18 -- PySR independent-implementation
    cross-check generalized to all 8 Tier B pairs" entry (that entry's
    `metadata.reason_for_reduction` field is the precedent for this
    script's own `metadata.reason_for_reduction` field, below).

Two phases:
  Phase 1 (STLSQ threshold sweep, SINDy-only, cheap): threshold in
  {0.01, 0.05, 0.1, 0.2}, all 4 family/regime items x 2 noise x 3 seed = 24
  conditions per threshold value, 96 SINDy fits total. gplearn/SR is not
  run in this phase at all (SINDy-only sweep, per instructions).

  Phase 2 (gplearn population_size sweep, SR, the expensive one):
  population_size in {1000, 3000, 10000}, generations=25 fixed, same 24
  conditions per value, 72 SR fits total. The "SINDy arm run alongside for
  reference" at STLSQ_THRESHOLD=0.1 is NOT recomputed 3x in this phase --
  `_generate_tier_b_data` is deterministic in (family, regime_args, seed,
  noise_frac) alone, independent of any SR hyperparameter, so Phase 1's
  threshold=0.1 SINDy records for the same (family, label, noise_frac,
  seed) are byte-identical to what a fresh Phase-2 SINDy fit would produce;
  they are reused by lookup instead of re-run, avoiding pure waste.

SR joint-pass criterion: this project's established authoritative
criterion is `degree_ok` (all output dims) AND NOT `dynamically_distinct`
(confirmation trajectory) -- see instructions. This script additionally
includes AND NOT `dynamically_distinct_off_attractor` (the off-attractor
grid gate) in its joint-pass criterion, since `_run_symbolic_regression`
already computes that quantity at zero extra cost, and DECISION_LOG.md's
PySR cross-check entries show the off-attractor gate is needed to catch
vacuous near-degenerate fits (e.g. logistic period_2 fitting a secant
through a 2-point cycle) that the confirmation-trajectory gate alone
misses. This is a strengthening of the minimum-required criterion, not a
reduction, and is disclosed here rather than silently assumed.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pysindy as ps

from experiments.main_study_tier_b import (
    COEFF_TOL,
    SR_MAX_DEGREE_RETRIES,
    TIER_B_ITEMS,
    VF_ERR_TOL,
    _generate_tier_b_data,
    _vf_err,
)
from src.discovery_symbolic_regression import MATCHED_FUNCTION_SET, fit_symbolic_regression

N_WORKERS = 6  # PROJECT_CHARTER.md local-only compute ceiling, matches Tier B
SEEDS = [0, 1, 2]  # reduced from Tier B's [0,1,2,3,4], disclosed in module docstring
NOISE_LEVELS = [0.0, 0.05]  # reduced from Tier B's [0.0, 0.01, 0.05]
DEGREE = 2  # reduced from Tier B's [2, 3]

STLSQ_THRESHOLDS = [0.01, 0.05, 0.1, 0.2]
SR_POPULATION_SIZES = [1000, 3000, 10000]
SR_GENERATIONS = 25  # fixed, matches Tier B's frozen SR_GENERATIONS
BASELINE_STLSQ_THRESHOLD = 0.1  # Tier B's frozen default
BASELINE_SR_POPULATION_SIZE = 3000  # Tier B's frozen default

PAIR_ITEMS = [item for item in TIER_B_ITEMS if item[0] in ("logistic", "lorenz")]
assert len(PAIR_ITEMS) == 4, f"expected 4 logistic/lorenz TIER_B_ITEMS, got {len(PAIR_ITEMS)}"

CHECKPOINT_PATH = "experiments/main_study_results/hyperparameter_sensitivity_checkpoint.jsonl"
RESULTS_PATH = "experiments/main_study_results/hyperparameter_sensitivity_results.json"


# ---------------------------------------------------------------------------
# Copy-adapted from main_study_tier_b.py: threshold/population_size/
# generations promoted from module globals to explicit parameters.
# ---------------------------------------------------------------------------

def _run_sindy(family, data, degree, threshold):
    if data["kind"] == "map":
        traj = data["states"][:, 0]
        r = data["params"]["r"]
        x_n, x_np1 = traj[:-1], traj[1:]
        A = np.stack([x_n ** k for k in range(degree + 1)], axis=1)
        coef, *_ = np.linalg.lstsq(A, x_np1, rcond=None)
        c0, r_hat_lin, r_hat_quad = coef[0], coef[1], -coef[2]
        err_terms = [abs(c0), abs(r_hat_lin - r) / r, abs(r_hat_quad - r) / r]
        if degree >= 3:
            err_terms.append(abs(coef[3]))
        max_rel_err = max(err_terms)
        recovered = bool(max_rel_err < COEFF_TOL)

        def predict(X):
            x = X[:, 0]
            Amat = np.stack([x ** k for k in range(degree + 1)], axis=1)
            return (Amat @ coef).reshape(-1, 1)

        pred_conf = predict(data["states_conf"])
        true_conf = np.array([data["true_rhs"](s) for s in data["states_conf"]])
        pred_grid = predict(data["grid_pts"])
        true_grid = np.array([data["true_rhs"](s) for s in data["grid_pts"]])
        return dict(method="sindy", degree=degree, threshold=threshold,
                    max_rel_err=float(max_rel_err), recovered=recovered,
                    vf_l2_err_confirmation=_vf_err(pred_conf, true_conf),
                    vf_l2_err_off_attractor_grid=_vf_err(pred_grid, true_grid),
                    dynamically_distinct=bool(_vf_err(pred_conf, true_conf) > VF_ERR_TOL),
                    dynamically_distinct_off_attractor=bool(_vf_err(pred_grid, true_grid) > VF_ERR_TOL))

    model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=degree),
                      optimizer=ps.STLSQ(threshold=threshold))
    model.fit(data["states"], t=data["dt"], feature_names=data["feature_names"])

    max_rel_err = None
    recovered = None
    if family == "lorenz":
        coeffs = model.coefficients()
        names = model.get_feature_names()
        idx = {n: i for i, n in enumerate(names)}
        sigma_hat, rho_hat, beta_hat = -coeffs[0][idx["x"]], coeffs[1][idx["x"]], -coeffs[2][idx["z"]]
        p = data["params"]
        max_rel_err = max(abs(sigma_hat - p["sigma"]) / p["sigma"],
                           abs(rho_hat - p["rho"]) / max(abs(p["rho"]), 1e-9),
                           abs(beta_hat - p["beta"]) / p["beta"])
        recovered = bool(max_rel_err < COEFF_TOL)

    true_conf = np.array([data["true_rhs"](s) for s in data["states_conf"]])
    pred_conf = model.predict(data["states_conf"])
    true_grid = np.array([data["true_rhs"](s) for s in data["grid_pts"]])
    pred_grid = model.predict(data["grid_pts"])
    vf_conf = _vf_err(pred_conf, true_conf)
    vf_grid = _vf_err(pred_grid, true_grid)

    out = dict(method="sindy", degree=degree, threshold=threshold,
               vf_l2_err_confirmation=vf_conf, vf_l2_err_off_attractor_grid=vf_grid,
               dynamically_distinct=bool(vf_conf > VF_ERR_TOL),
               dynamically_distinct_off_attractor=bool(vf_grid > VF_ERR_TOL))
    if max_rel_err is not None:
        out["max_rel_err"] = float(max_rel_err)
        out["recovered"] = recovered
    else:
        out["library_mismatch_expected"] = True
    return out


def _fit_symbolic_regression_map(x_n, x_np1, max_degree, seed, population_size, generations):
    """1D discrete-map SR fit, copy-adapted from
    main_study_tier_b._fit_symbolic_regression_map with population_size/
    generations promoted to parameters (see module docstring)."""
    from gplearn.genetic import SymbolicRegressor
    from src.discovery_symbolic_regression import program_polynomial_degree

    X = x_n.reshape(-1, 1)
    min_depth, max_depth = 2, 2
    while (2 ** (max_depth - 1)) < max_degree and max_depth < 4:
        max_depth += 1
    init_depth = (min_depth, max_depth)

    best = None
    for attempt in range(max(1, SR_MAX_DEGREE_RETRIES)):
        est = SymbolicRegressor(
            population_size=population_size, generations=generations,
            function_set=MATCHED_FUNCTION_SET, init_depth=init_depth,
            parsimony_coefficient=0.001, stopping_criteria=1e-9,
            random_state=seed + attempt, n_jobs=1, feature_names=["x"], verbose=0,
        )
        est.fit(X, x_np1)
        deg, is_poly = program_polynomial_degree(est._program.program)
        satisfies = is_poly and deg is not None and deg <= max_degree
        rank = (0 if satisfies else 1, 0 if is_poly else 1, deg if deg is not None else float("inf"))
        if best is None or rank < best[0]:
            best = (rank, est, deg, is_poly)
        if satisfies:
            break
    _, est, deg, is_poly = best
    return dict(regressor=est, degree=deg, is_polynomial=is_poly,
                degree_ok=bool(is_poly and deg is not None and deg <= max_degree))


def _run_symbolic_regression(family, data, degree, seed, population_size, generations):
    if data["kind"] == "map":
        traj = data["states"][:, 0]
        x_n, x_np1 = traj[:-1], traj[1:]
        fit = _fit_symbolic_regression_map(x_n, x_np1, max_degree=degree, seed=seed,
                                            population_size=population_size, generations=generations)

        def predict(X):
            return fit["regressor"].predict(X[:, :1]).reshape(-1, 1)

        pred_conf = predict(data["states_conf"])
        true_conf = np.array([data["true_rhs"](s) for s in data["states_conf"]])
        pred_grid = predict(data["grid_pts"])
        true_grid = np.array([data["true_rhs"](s) for s in data["grid_pts"]])
        return dict(method="symbolic_regression", degree=degree,
                    population_size=population_size, generations=generations,
                    degrees=[fit["degree"]], is_polynomial=[fit["is_polynomial"]],
                    degree_ok=[fit["degree_ok"]],
                    vf_l2_err_confirmation=_vf_err(pred_conf, true_conf),
                    vf_l2_err_off_attractor_grid=_vf_err(pred_grid, true_grid),
                    dynamically_distinct=bool(_vf_err(pred_conf, true_conf) > VF_ERR_TOL),
                    dynamically_distinct_off_attractor=bool(_vf_err(pred_grid, true_grid) > VF_ERR_TOL))

    # ODE families: fit_symbolic_regression already takes population_size/
    # generations as explicit parameters, imported and called unmodified.
    result = fit_symbolic_regression(
        data["states"], data["dt"], feature_names=data["feature_names"],
        population_size=population_size, generations=generations,
        max_degree=degree, random_state=seed, n_jobs=1,
        max_degree_retries=SR_MAX_DEGREE_RETRIES,
    )
    regressors = result["regressors"]

    def predict(X):
        return np.column_stack([reg.predict(X) for reg in regressors])

    true_conf = np.array([data["true_rhs"](s) for s in data["states_conf"]])
    pred_conf = predict(data["states_conf"])
    true_grid = np.array([data["true_rhs"](s) for s in data["grid_pts"]])
    pred_grid = predict(data["grid_pts"])
    vf_conf = _vf_err(pred_conf, true_conf)
    vf_grid = _vf_err(pred_grid, true_grid)

    return dict(method="symbolic_regression", degree=degree,
                population_size=population_size, generations=generations,
                degrees=result["degrees"], is_polynomial=result["is_polynomial"],
                degree_ok=result["degree_ok"],
                vf_l2_err_confirmation=vf_conf, vf_l2_err_off_attractor_grid=vf_grid,
                dynamically_distinct=bool(vf_conf > VF_ERR_TOL),
                dynamically_distinct_off_attractor=bool(vf_grid > VF_ERR_TOL))


# ---------------------------------------------------------------------------
# Job dispatch
# ---------------------------------------------------------------------------

def _job_key(family, label, noise_frac, seed, sweep_name, sweep_value):
    return f"{family}|{label}|{noise_frac}|{seed}|{sweep_name}|{sweep_value}"


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


def _run_sr_job(job):
    family, regime_args, label, noise_frac, seed, population_size = job
    t0 = time.time()
    data = _generate_tier_b_data(family, regime_args, seed, noise_frac)
    sr_out = _run_symbolic_regression(family, data, DEGREE, seed, population_size, SR_GENERATIONS)
    wall = time.time() - t0
    key = _job_key(family, label, noise_frac, seed, "sr_population_size", population_size)
    return dict(key=key, sweep="sr_population_size", sweep_value=population_size,
                family=family, regime=label, noise_frac=noise_frac, seed=seed,
                degree=DEGREE, wall_clock_s=wall, symbolic_regression=sr_out)


def _build_sindy_jobs():
    jobs = []
    for threshold in STLSQ_THRESHOLDS:
        for family, regime_args, label in PAIR_ITEMS:
            for noise_frac in NOISE_LEVELS:
                for seed in SEEDS:
                    jobs.append((family, regime_args, label, noise_frac, seed, threshold))
    return jobs


def _build_sr_jobs():
    jobs = []
    for population_size in SR_POPULATION_SIZES:
        for family, regime_args, label in PAIR_ITEMS:
            for noise_frac in NOISE_LEVELS:
                for seed in SEEDS:
                    jobs.append((family, regime_args, label, noise_frac, seed, population_size))
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


def _run_phase(jobs, run_fn, done, ckpt_f, phase_name):
    pending = [j for j in jobs if _job_key(j[0], j[2], j[3], j[4], *(
        ("stlsq_threshold", j[5]) if run_fn is _run_sindy_job else ("sr_population_size", j[5])
    )) not in done]
    print(f"{phase_name}: {len(jobs)} total, {len(done)} checkpointed so far (all phases), "
          f"{len(pending)} pending this phase, {N_WORKERS} workers.", flush=True)
    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_fn, job): job for job in pending}
        for future in as_completed(futures):
            rec = future.result()
            ckpt_f.write(json.dumps(rec, default=float) + "\n")
            ckpt_f.flush()
            done[rec["key"]] = rec
            n_done += 1
            print(f"[{phase_name} {n_done}/{len(pending)}] {rec['family']} {rec['regime']} "
                  f"noise={rec['noise_frac']:.1%} sweep={rec['sweep']}={rec['sweep_value']} "
                  f"seed={rec['seed']} wall={rec['wall_clock_s']:.1f}s", flush=True)
    print(f"{phase_name} complete: {n_done} newly run this session, "
          f"{time.time() - t0:.1f}s wall-clock.", flush=True)


def main():
    os.makedirs("experiments/main_study_results", exist_ok=True)
    done = _load_checkpoint()

    with open(CHECKPOINT_PATH, "a") as ckpt_f:
        _run_phase(_build_sindy_jobs(), _run_sindy_job, done, ckpt_f, "Phase1-STLSQ")
        _run_phase(_build_sr_jobs(), _run_sr_job, done, ckpt_f, "Phase2-SR-population")

    records = list(done.values())
    metadata = dict(
        reason_for_reduction=(
            "2 of Tier B's 8 matched pairs (logistic, lorenz); noise_frac "
            "{0.0,0.05} of Tier B's {0.0,0.01,0.05}; degree=2 only of Tier "
            "B's {2,3}; seeds=[0,1,2] of Tier B's [0,1,2,3,4] -- all "
            "reductions from NEXT_STEPS.md item 5's own scoped instructions, "
            "not a post-hoc compute-driven cut. Phase 2 SR-arm SINDy-at-"
            "threshold=0.1 'reference' values are NOT separately stored: "
            "they are byte-identical to Phase 1's stlsq_threshold=0.1 "
            "records for the same (family,label,noise_frac,seed), reused by "
            "key lookup in analysis rather than recomputed 3x."
        ),
        pair_items=[list(x) for x in PAIR_ITEMS],
        seeds=SEEDS, noise_levels=NOISE_LEVELS, degree=DEGREE,
        stlsq_thresholds=STLSQ_THRESHOLDS, sr_population_sizes=SR_POPULATION_SIZES,
        sr_generations=SR_GENERATIONS,
        baseline_stlsq_threshold=BASELINE_STLSQ_THRESHOLD,
        baseline_sr_population_size=BASELINE_SR_POPULATION_SIZE,
        sr_joint_criterion="degree_ok (all dims) AND NOT dynamically_distinct AND NOT dynamically_distinct_off_attractor",
    )
    with open(RESULTS_PATH, "w") as f:
        json.dump(dict(metadata=metadata, records=records), f, indent=2, default=float)
    print(f"\nWrote {len(records)} records to {RESULTS_PATH}.", flush=True)
    return records


if __name__ == "__main__":
    main()
