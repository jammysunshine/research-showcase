"""Active experimental design vs. passive random-IC baseline (NEXT_STEPS.md
Tier-3 item #10).

Tier B (`main_study_tier_b.py`) always draws the training-trajectory initial
condition passively: `x0 = 0.1 + 0.3 * rng.random()`, independent of any
persistent-excitation criterion. This script asks whether ACTIVELY choosing
the training IC -- from a pool of candidates, scored and selected by the
same lambda_min(G) Gram-matrix persistent-excitation quantity this project's
Koopman mechanistic theory (`koopman_gram_matrix_analysis.py`) and Gallo et
al. (arXiv:2607.18490) both use to predict SINDy/PySR's identifiability
ceiling -- measurably improves SINDy coefficient recovery relative to Tier
B's passive baseline, at matched trajectory length/noise/seed count.

Scope: logistic map only (cheap, matches NEXT_STEPS.md's own suggestion),
matched pair period_2 (r=3.2, control) vs. chaotic (r=4.0), the same pair
Tier B uses. noise_frac in {0.0, 0.05} (2 of Tier B's 3 levels, disclosed
reduction), degree=2 only (of Tier B's {2,3}), 5 "active-selection process"
seeds (matching Tier B's 5 seeds, so run counts are directly comparable).

Active-selection heuristic (simple pool-based greedy design, per the item's
own scoping -- not a sophisticated Bayesian design):
  1. For each process seed s, draw a pool of N_CANDIDATES=50 candidate ICs
     x0 from Uniform(0.02, 0.98) -- WIDER than Tier B's own baseline draw
     range (0.1, 0.4) -- deliberately, so the greedy search has a genuinely
     different candidate set to choose from than Tier B's passive draw;
     using the same narrow (0.1, 0.4) range would make "active selection"
     trivially identical to "passive selection" by construction (any point
     in that whole narrow range converges to the same post-transient
     support for this map, see below). Disclosed as a deviation, not a
     matched-range design.
  2. For each candidate, simulate the FULL trajectory Tier B would use
     (same n_steps=5500, transient=500 -- identical to
     `main_study_tier_b.MEDIUM["logistic_n_steps"]` and the map's transient
     convention), noise-free, and compute the degree-2 monomial Gram matrix
     G = Phi0^T Phi0 / n (Phi0 columns [1, x, x^2], matching the SINDy
     lstsq design matrix `_run_sindy`'s map branch builds) and its smallest
     eigenvalue lambda_min(G).
  3. Select the candidate x0 maximizing lambda_min(G) (greedy, pool-based
     optimal design -- ties broken by candidate order).
  4. Generate the actual training trajectory from that x0 (same
     logistic_trajectory call, same n_steps/transient), inject noise at the
     given noise_frac (same convention as Tier B: rng.normal(0,
     noise_frac*traj.std())), fit SINDy (degree=2, plain lstsq -- identical
     code path to `main_study_tier_b._run_sindy`'s map branch / degree=2),
     and evaluate against an INDEPENDENT confirmation trajectory and
     off-attractor grid.

Confirmation trajectory and off-attractor grid IC are NOT actively
selected -- they remain simple random draws (same range/mechanism as Tier
B) precisely because they represent held-out generalization checks; only
the TRAINING trajectory's IC is a design choice under this experiment's
control, matching the item's framing ("choosing initial conditions ... to
maximize identifiability" of the OBSERVED/training data).
"""
import json
import os

import numpy as np

from src.simulators import logistic_map, logistic_trajectory

N_WORKERS_NOTE = "sequential -- logistic-map trajectories are numpy-cheap, no multiprocessing needed"

REGIMES = [(3.2, "period_2"), (4.0, "chaotic")]
NOISE_LEVELS = [0.0, 0.05]
DEGREE = 2
SEEDS = [0, 1, 2, 3, 4]  # process seeds, matched count to Tier B's 5 seeds
N_CANDIDATES = 50
CANDIDATE_LOW, CANDIDATE_HIGH = 0.02, 0.98
N_STEPS = 5500  # matches main_study_tier_b.MEDIUM["logistic_n_steps"]
TRANSIENT = 500
COEFF_TOL = 0.05  # matches main_study_tier_b.COEFF_TOL
VF_ERR_TOL = 0.10  # matches main_study_tier_b.VF_ERR_TOL
N_GRID_POINTS = 500

# Baseline draw range Tier B uses for its passive IC (main_study_tier_b.py:135),
# reused unchanged for the confirmation trajectory and off-attractor grid.
BASELINE_LOW, BASELINE_HIGH = 0.1, 0.4

RESULTS_PATH = "experiments/main_study_results/active_design_logistic_results.json"


def _gram_lambda_min(traj, degree):
    n = traj.shape[0] - 1
    Phi0 = np.stack([traj[:-1] ** k for k in range(degree + 1)], axis=1)
    G = (Phi0.T @ Phi0) / n
    eigs = np.linalg.eigvalsh(G)
    return float(np.clip(eigs[0], 0.0, None))


def _select_active_ic(r, pool_rng):
    candidates = pool_rng.uniform(CANDIDATE_LOW, CANDIDATE_HIGH, size=N_CANDIDATES)
    best_x0, best_lam = None, -np.inf
    scored = []
    for x0 in candidates:
        traj = logistic_trajectory(float(x0), r=r, n_steps=N_STEPS, transient=TRANSIENT)
        lam = _gram_lambda_min(traj, DEGREE)
        scored.append(lam)
        if lam > best_lam:
            best_lam, best_x0 = lam, float(x0)
    return best_x0, best_lam, dict(min=float(np.min(scored)), max=float(np.max(scored)),
                                    mean=float(np.mean(scored)), std=float(np.std(scored)))


def _vf_err(pred, true):
    scale = max(float(np.abs(true).max()), 1e-12)
    return float(np.sqrt(np.mean((pred - true) ** 2)) / scale)


def _fit_sindy_map(traj, r, degree):
    x_n, x_np1 = traj[:-1], traj[1:]
    A = np.stack([x_n ** k for k in range(degree + 1)], axis=1)
    coef, *_ = np.linalg.lstsq(A, x_np1, rcond=None)
    c0, r_hat_lin, r_hat_quad = coef[0], coef[1], -coef[2]
    err_terms = [abs(c0), abs(r_hat_lin - r) / r, abs(r_hat_quad - r) / r]
    max_rel_err = max(err_terms)
    recovered = bool(max_rel_err < COEFF_TOL)

    def predict(x):
        Amat = np.stack([x ** k for k in range(degree + 1)], axis=1)
        return Amat @ coef

    return dict(coef=coef.tolist(), max_rel_err=float(max_rel_err), recovered=recovered), predict


def run_condition(r, label, noise_frac, seed):
    pool_rng = np.random.default_rng(100_000 + seed)
    noise_rng = np.random.default_rng(200_000 + seed)
    conf_rng = np.random.default_rng(300_000 + seed)
    grid_rng = np.random.default_rng(400_000 + seed)

    active_x0, active_lam, pool_stats = _select_active_ic(r, pool_rng)

    traj = logistic_trajectory(active_x0, r=r, n_steps=N_STEPS, transient=TRANSIENT)
    if noise_frac > 0:
        traj = traj + noise_rng.normal(0, noise_frac * traj.std(), size=traj.shape)

    x0_conf = BASELINE_LOW + (BASELINE_HIGH - BASELINE_LOW) * conf_rng.random()
    traj_conf = logistic_trajectory(x0_conf, r=r, n_steps=N_STEPS, transient=TRANSIENT)
    if noise_frac > 0:
        traj_conf = traj_conf + conf_rng.normal(0, noise_frac * traj_conf.std(), size=traj_conf.shape)

    x_grid = grid_rng.uniform(0.0, 1.0, size=N_GRID_POINTS)

    fit, predict = _fit_sindy_map(traj, r, DEGREE)

    true_conf = np.array([logistic_map(float(x), r) for x in traj_conf])
    pred_conf = predict(traj_conf)
    vf_conf = _vf_err(pred_conf, true_conf)

    true_grid = np.array([logistic_map(float(x), r) for x in x_grid])
    pred_grid = predict(x_grid)
    vf_grid = _vf_err(pred_grid, true_grid)

    # Also record the realized lambda_min(G) of the ACTUAL (possibly noised)
    # training trajectory actually fit, for direct comparison to the
    # pre-noise candidate score used for selection.
    realized_lambda_min = _gram_lambda_min(traj, DEGREE)

    return dict(
        family="logistic", regime=label, r=r, noise_frac=noise_frac, degree=DEGREE, seed=seed,
        active_x0=active_x0, active_candidate_lambda_min=active_lam,
        pool_lambda_min_stats=pool_stats, realized_lambda_min=realized_lambda_min,
        max_rel_err=fit["max_rel_err"], recovered=fit["recovered"],
        vf_l2_err_confirmation=vf_conf, vf_l2_err_off_attractor_grid=vf_grid,
        dynamically_distinct=bool(vf_conf > VF_ERR_TOL),
        dynamically_distinct_off_attractor=bool(vf_grid > VF_ERR_TOL),
    )


def main():
    os.makedirs("experiments/main_study_results", exist_ok=True)
    records = []
    for r, label in REGIMES:
        for noise_frac in NOISE_LEVELS:
            for seed in SEEDS:
                rec = run_condition(r, label, noise_frac, seed)
                records.append(rec)
                print(f"{label} noise={noise_frac:.0%} seed={seed}: active_x0={rec['active_x0']:.4f} "
                      f"pool_lambda_min[min={rec['pool_lambda_min_stats']['min']:.4g},"
                      f"max={rec['pool_lambda_min_stats']['max']:.4g}] "
                      f"recovered={rec['recovered']} max_rel_err={rec['max_rel_err']:.4g}",
                      flush=True)

    metadata = dict(
        reason_for_reduction=(
            "logistic map only (of Tier B's 8 families); noise_frac {0.0,0.05} "
            "of Tier B's {0.0,0.01,0.05}; degree=2 only of Tier B's {2,3} -- "
            "reductions matching NEXT_STEPS.md item 10's own scoped "
            "instructions ('one matched pair', 'noise {0%,5%}', 'degree=2')."
        ),
        n_candidates=N_CANDIDATES, candidate_range=[CANDIDATE_LOW, CANDIDATE_HIGH],
        baseline_range_for_confirmation_and_grid=[BASELINE_LOW, BASELINE_HIGH],
        n_steps=N_STEPS, transient=TRANSIENT, seeds=SEEDS, noise_levels=NOISE_LEVELS,
        degree=DEGREE, coeff_tol=COEFF_TOL, vf_err_tol=VF_ERR_TOL,
        baseline_comparison_source="experiments/main_study_results/tier_b_results.json "
                                    "(family=logistic, degree=2, noise_frac in {0.0,0.05}, seeds 0-4, sindy arm)",
    )
    with open(RESULTS_PATH, "w") as f:
        json.dump(dict(metadata=metadata, records=records), f, indent=2, default=float)
    print(f"\nWrote {len(records)} records to {RESULTS_PATH}.")


if __name__ == "__main__":
    main()
