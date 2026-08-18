"""Lorenz-96 matched-pair extension, generalized across state dimension N
(EXTENSION_PLAN.md extension #4; NEXT_STEPS.md Tier-2 item #4, "Lorenz-96
N-scan").

Purpose: Tier B (experiments/main_study_tier_b.py) established a chaos-vs-
Koopman/EDMD reversal pattern (chaos correlates with WORSE EDMD one-step
error but comparable-or-better SINDy coefficient recovery) across 7 matched
chaotic/non-chaotic pairs, all with state dimension N<=3. The original
version of this script tested whether that pattern survives at N=6 (result:
yes, see DECISION_LOG.md "2026-08-17 -- Lorenz-96 (N=6) matched-pair
extension"). This version generalizes the same pipeline to run at ANY N,
so the N=6 finding can be checked for how it scales with dimension.

N=6 results (experiments/main_study_results/lorenz96_results.json /
lorenz96_checkpoint.jsonl, job keys of the unprefixed form
"lorenz96|<label>|<noise>|<seed>") are NOT rerun by this script: `_job_key`
below preserves that exact unprefixed format only for N==6 (matching the
already-checkpointed keys byte-for-byte), so a checkpoint-driven run at
N=6 is a guaranteed no-op (all 30 jobs found already done) rather than a
silent duplicate run under a different key. All other N use a
"lorenz96|N{N}|..." prefixed key so results at different N never collide.

Both N=6 and other-N results are appended into the SAME
lorenz96_results.json / lorenz96_checkpoint.jsonl files (decision: one file,
N encoded in the job key, rather than per-N files -- keeps a single
resumable checkpoint and a single results file to load for cross-N
analysis, consistent with how noise_frac/seed already coexist as job-key
fields in the same file rather than separate files per condition). See
DECISION_LOG.md "2026-08-18 -- Lorenz-96 N-scan" for the disclosure of this
choice and of any grid-reduction decisions made per N.

Regimes: F=1.0 (non-chaotic control) vs. F=8.0 (chaotic), the same forcing
values used at N=6, confirmed to remain a valid matched pair by a Lyapunov-
spectrum check at each target N before committing compute to the full grid
(see module-level docstring notes below and DECISION_LOG.md for the actual
per-N confirmation numbers, including two N values where the pair is
NOT cleanly separated -- disclosed, not hidden).

Scoping decision -- SYMBOLIC REGRESSION IS SKIPPED AT EVERY N (stated here
and in the task instructions, not hidden): gplearn/PySR symbolic-regression
search cost scales badly with input dimension (N candidate operands per
node vs. <=3 in every Tier B family), and running it per-seed per-condition
within this project's local-only-compute budget (PROJECT_CHARTER.md) would
be prohibitively slow, worse still at higher N. Only SINDy and Koopman/EDMD
are run here. This is a narrower method comparison than Tier B, reported
honestly as such.

Trajectory-generation conventions mirror Tier B's Lorenz arm as closely as
makes sense, with initial condition and dt choices matching the ALREADY
VALIDATED lorenz96 test conventions in tests/test_simulators.py (x0 =
[8,0,...,0] + 0.01, dt=0.01) generalized to N dimensions (first coordinate
8.0, remaining N-1 coordinates 0.0, uniform +0.01 offset -- identical to the
N=6 convention when N=6).

SINDy feature library: PolynomialLibrary(degree=2) is the exact match for
Lorenz-96's RHS dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F, which is
degree-2 (one constant term F, one linear term -x_i, two quadratic
cross-terms +x_{i+1}*x_{i-1} and -x_{i-2}*x_{i-1}) regardless of N. Genuine
per-term coefficient-recovery checking against these four known nonzero
coefficients per equation (N equations) is implemented below, matching
Tier B's COEFF_TOL=0.05 convention -- not just off-trajectory VF error.
Feature-name lookup (pysindy's PolynomialLibrary naming, e.g. "x0 x1",
"x0^2") was verified correct at N in {4,6,8,12,20} before trusting any
max_rel_err numbers (see DECISION_LOG.md) -- a KeyError or silent
wrong-index bug here would corrupt every result silently, so this was
checked, not assumed.

Cost/memory note: PolynomialLibrary(degree=2) feature count is
C(N+2,2) -- 15 at N=4, 28 at N=6, 45 at N=8, 91 at N=12, 231 at N=20. The
EDMD Koopman matrix K is (that count) x (that count), i.e. 231x231 at
N=20 -- this is the dimension driving cost/memory concerns at high N.

Koopman/EDMD: degree-2 monomial dictionary (src/discovery_koopman.py
fit_edmd), one-step prediction error defined identically to Tier B's
`one_step_rel_rms_err` (relative L2 norm of predicted vs. true next-state,
_vf_err below, copied verbatim from main_study_tier_b.py), plus the same
rollout and spectral-consistency checks Tier B's EDMD arm reports.

Checkpointing: identical JSONL-append, resumable-by-job-key pattern as
Tier B (experiments/main_study_results/lorenz96_checkpoint.jsonl).

CLI: `python -m experiments.lorenz96_pipeline --n 4 8 12` runs (or resumes)
the full grid at each listed N in turn, appending to the shared checkpoint/
results files. `--seeds` optionally overrides the default 5-seed SEEDS list
for a given invocation (used, if needed, for a disclosed reduced-grid run
at large N -- see DECISION_LOG.md for whether/where this was used).
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pysindy as ps

from src.discovery_koopman import fit_edmd
from src.simulators import (
    lorenz96_jacobian,
    lorenz96_rhs,
    lorenz96_trajectory,
)

N_WORKERS = 6  # PROJECT_CHARTER.md local-only compute ceiling, matches Tier A/B
SEEDS = [0, 1, 2, 3, 4]
NOISE_LEVELS = [0.0, 0.01, 0.05]
DEGREE = 2  # SINDy library degree and EDMD dictionary degree; exact match
COEFF_TOL = 0.05
VF_ERR_TOL = 0.10
N_OFF_ATTRACTOR_GRID_POINTS = 500
STLSQ_THRESHOLD = 0.1
KOOPMAN_ROLLOUT_HORIZON = 50

# dt=0.01 matches the already-validated lorenz96 test convention (see
# tests/test_simulators.py test_lorenz96_F8_chaotic_regime_positive_largest_lyapunov,
# which uses dt=0.01). t_end=100 -> n_points=10001 gives dt exactly 0.01.
# Kept identical across all N (task instruction: only N becomes a variable).
T_END = 100.0
N_POINTS = 10001

REGIMES = [
    ("control", 1.0),
    ("chaotic", 8.0),
]

CHECKPOINT_PATH = "experiments/main_study_results/lorenz96_checkpoint.jsonl"
RESULTS_PATH = "experiments/main_study_results/lorenz96_results.json"


def _confirmation_offset(seed):
    return seed + 10_000


def _grid_offset(seed):
    return seed + 20_000


def _vf_err(pred, true):
    # Copied verbatim from main_study_tier_b.py for metric consistency.
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-300))


def _base_x0(N):
    # Generalizes the validated N=6 convention x0=[8,0,0,0,0,0]+0.01: first
    # coordinate 8.0, rest 0.0, uniform +0.01 offset. Identical to the N=6
    # convention when N=6.
    x0 = np.zeros(N)
    x0[0] = 8.0
    return x0 + 0.01


# ---------------------------------------------------------------------------
# Shared per-condition trajectory generation (train + confirmation +
# off-attractor grid), mirroring _generate_tier_b_data's Lorenz arm,
# generalized to N dimensions.
# ---------------------------------------------------------------------------

def _generate_data(F, seed, noise_frac, N):
    rng = np.random.default_rng(seed)
    conf_rng = np.random.default_rng(_confirmation_offset(seed))
    grid_rng = np.random.default_rng(_grid_offset(seed))

    base_x0 = _base_x0(N)
    params = dict(F=F)

    x0 = base_x0 + rng.normal(0, 0.5, size=N)
    t, states = lorenz96_trajectory(x0, t_span=(0, T_END), n_points=N_POINTS, params=params)
    if noise_frac > 0:
        states = states + rng.normal(0, noise_frac * states.std(axis=0), size=states.shape)
    n_discard = int(len(t) * 0.5)
    t_tail, states_tail = t[n_discard:], states[n_discard:]
    dt = t_tail[1] - t_tail[0]

    x0_conf = base_x0 + conf_rng.normal(0, 0.5, size=N)
    t_conf, states_conf = lorenz96_trajectory(x0_conf, t_span=(0, T_END), n_points=N_POINTS, params=params)
    if noise_frac > 0:
        states_conf = states_conf + conf_rng.normal(0, noise_frac * states_conf.std(axis=0), size=states_conf.shape)
    states_conf_tail = states_conf[n_discard:]

    lo, hi = states_tail.min(axis=0), states_tail.max(axis=0)
    grid_pts = grid_rng.uniform(lo, hi, size=(N_OFF_ATTRACTOR_GRID_POINTS, N))

    def true_rhs(s):
        return lorenz96_rhs(0.0, s, F)

    def true_jac(s):
        return lorenz96_jacobian(s, F)

    return dict(
        states=states_tail, states_conf=states_conf_tail, grid_pts=grid_pts, dt=dt,
        true_rhs=true_rhs, true_jac=true_jac, F=F, N=N,
        feature_names=[f"x{i}" for i in range(N)],
        x0_for_lyap=states_tail[0],
    )


# ---------------------------------------------------------------------------
# SINDy arm: genuine per-term coefficient-recovery checking against
# Lorenz-96's known analytic RHS, generalized to N dimensions.
# ---------------------------------------------------------------------------

def _prod_name(a, b):
    lo, hi = min(a, b), max(a, b)
    return f"x{lo}^2" if lo == hi else f"x{lo} x{hi}"


def _run_sindy(data, F, N):
    n = N
    model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=DEGREE),
                      optimizer=ps.STLSQ(threshold=STLSQ_THRESHOLD))
    model.fit(data["states"], t=data["dt"], feature_names=data["feature_names"])

    coeffs = model.coefficients()  # (n, n_features)
    names = model.get_feature_names()
    idx = {name: i for i, name in enumerate(names)}

    per_equation = []
    for i in range(n):
        im1 = (i - 1) % n
        im2 = (i - 2) % n
        ip1 = (i + 1) % n
        const_hat = float(coeffs[i][idx["1"]])
        lin_hat = float(coeffs[i][idx[f"x{i}"]])
        term1_hat = float(coeffs[i][idx[_prod_name(ip1, im1)]])  # true coef +1: x_{i+1} x_{i-1}
        term2_hat = float(coeffs[i][idx[_prod_name(im2, im1)]])  # true coef -1: x_{i-2} x_{i-1}
        errs = {
            "const": abs(const_hat - F) / max(abs(F), 1e-9),
            "linear": abs(lin_hat - (-1.0)) / max(1.0, 1e-9),
            "term1_xip1_xim1": abs(term1_hat - 1.0) / max(1.0, 1e-9),
            "term2_xim2_xim1": abs(term2_hat - (-1.0)) / max(1.0, 1e-9),
        }
        max_rel_err_i = max(errs.values())
        per_equation.append(dict(
            eq=i, max_rel_err=float(max_rel_err_i), recovered=bool(max_rel_err_i < COEFF_TOL),
            const_hat=const_hat, lin_hat=lin_hat, term1_hat=term1_hat, term2_hat=term2_hat,
            errs=errs,
        ))

    max_rel_err = max(e["max_rel_err"] for e in per_equation)
    n_recovered = sum(e["recovered"] for e in per_equation)
    recovered_all = bool(n_recovered == n)

    true_conf = np.array([data["true_rhs"](s) for s in data["states_conf"]])
    pred_conf = model.predict(data["states_conf"])
    true_grid = np.array([data["true_rhs"](s) for s in data["grid_pts"]])
    pred_grid = model.predict(data["grid_pts"])
    vf_conf = _vf_err(pred_conf, true_conf)
    vf_grid = _vf_err(pred_grid, true_grid)

    return dict(
        method="sindy", degree=DEGREE, per_equation=per_equation,
        max_rel_err=float(max_rel_err), n_equations_recovered=n_recovered,
        n_equations_total=n, recovered=recovered_all,
        vf_l2_err_confirmation=vf_conf, vf_l2_err_off_attractor_grid=vf_grid,
        dynamically_distinct=bool(vf_conf > VF_ERR_TOL),
        dynamically_distinct_off_attractor=bool(vf_grid > VF_ERR_TOL),
    )


# ---------------------------------------------------------------------------
# Koopman / EDMD arm -- identical metric definitions to Tier B's EDMD arm.
# Already N-generic (fit_edmd infers n_vars from states.shape[1]).
# ---------------------------------------------------------------------------

def _run_koopman(data):
    dt = data["dt"]
    model = fit_edmd(data["states"], dt=dt, degree=DEGREE, var_names=data["feature_names"])

    states_conf = data["states_conf"]
    X0, X1 = states_conf[:-1], states_conf[1:]
    pred1 = model.predict_state(X0)
    one_step_err = _vf_err(pred1, X1)

    horizon = min(KOOPMAN_ROLLOUT_HORIZON, len(states_conf) - 1)
    rollout = model.simulate(states_conf[0], horizon)
    rollout_err = _vf_err(rollout, states_conf[:horizon + 1])

    true_jac_at_x0 = data["true_jac"](data["x0_for_lyap"])
    true_eigs = np.sort_complex(np.linalg.eigvals(true_jac_at_x0))
    approx_lin = model.linearization()
    model_eigs = np.sort_complex(np.linalg.eigvals(approx_lin))
    spectral_err = float(np.linalg.norm(model_eigs - true_eigs) / max(np.linalg.norm(true_eigs), 1e-300))

    k_cond = float(np.linalg.cond(model.K))

    return dict(
        method="koopman", degree=DEGREE,
        one_step_rel_rms_err=one_step_err, rollout_horizon=horizon,
        rollout_rel_err=rollout_err, spectral_rel_err=spectral_err,
        k_condition_number=k_cond, residual_rms=model.residual_rms,
        # No coefficient/equation-recovery concept for EDMD (EVIDENCE_INDEX.md row 15).
    )


# ---------------------------------------------------------------------------
# Job dispatch (checkpoint-and-resume, mirrors main_study_tier_b.py exactly).
# ---------------------------------------------------------------------------

def _job_key(N, label, noise_frac, seed):
    # N==6 preserves the exact unprefixed key format already checkpointed
    # by the original N=6-only version of this script, so a run at N=6
    # (accidental or otherwise) is a guaranteed no-op against the existing
    # 30 checkpointed records rather than a silent duplicate under a new
    # key. All other N get an "N{N}|" prefix so they never collide.
    if N == 6:
        return f"lorenz96|{label}|{noise_frac}|{seed}"
    return f"lorenz96|N{N}|{label}|{noise_frac}|{seed}"


def _run_job(job):
    N, label, F, noise_frac, seed = job
    t_start = time.time()
    data = _generate_data(F, seed, noise_frac, N)
    sindy_out = _run_sindy(data, F, N)
    koopman_out = _run_koopman(data)
    wall = time.time() - t_start
    key = _job_key(N, label, noise_frac, seed)
    return dict(key=key, family="lorenz96", N=N, regime=label, F=F, noise_frac=noise_frac,
                seed=seed, wall_clock_s=wall, sindy=sindy_out, koopman=koopman_out)


def _build_jobs(N, seeds):
    jobs = []
    for label, F in REGIMES:
        for noise_frac in NOISE_LEVELS:
            for seed in seeds:
                jobs.append((N, label, F, noise_frac, seed))
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


def run_for_n(N, seeds=None):
    """Run (or resume) the full REGIMES x NOISE_LEVELS x seeds grid at a
    single N, appending results into the shared checkpoint/results files.
    """
    seeds = list(seeds) if seeds is not None else SEEDS
    os.makedirs("experiments/main_study_results", exist_ok=True)
    all_jobs = _build_jobs(N, seeds)
    done = _load_checkpoint()
    pending = [j for j in all_jobs if _job_key(j[0], j[1], j[3], j[4]) not in done]
    print(f"Lorenz-96 (N={N}, seeds={seeds}): {len(all_jobs)} total jobs, "
          f"{len(all_jobs) - len(pending)} already checkpointed, "
          f"{len(pending)} pending across {N_WORKERS} worker processes. "
          f"SR SKIPPED (scoping decision, see module docstring).", flush=True)

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
                print(f"[{n_done}/{len(pending)}] N={N} {rec['regime']} (F={rec['F']}) "
                      f"noise={rec['noise_frac']:.1%} seed={rec['seed']}: "
                      f"sindy_recovered={rec['sindy']['recovered']} "
                      f"({rec['sindy']['n_equations_recovered']}/{rec['sindy']['n_equations_total']} eqs) "
                      f"koopman_one_step_err={rec['koopman']['one_step_rel_rms_err']:.4g} "
                      f"wall={rec['wall_clock_s']:.1f}s", flush=True)

    results = list(done.values())
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)

    total_wall = time.time() - t0
    print(f"\nLorenz-96 N={N} complete. {len(pending)} jobs run this call "
          f"({len(results)} total records across all N in the shared results file). "
          f"This call's wall-clock: {total_wall:.1f}s across {N_WORKERS} workers.", flush=True)
    return results, total_wall


def main():
    """Back-compat entry point: runs the default N=6 grid. Guaranteed to be
    a no-op against the already-checkpointed N=6 records (see _job_key).
    """
    return run_for_n(6, seeds=SEEDS)[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[6],
                         help="State dimension(s) N to run (default: [6], a no-op).")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Override SEEDS for this invocation (default: SEEDS = [0,1,2,3,4]).")
    args = parser.parse_args()
    for n_val in args.n:
        run_for_n(n_val, seeds=args.seeds)
