"""Genuine z-space counterexample for the logistic map: finite-trajectory
interpolation degeneracy.

`src/counterexamples_conjugacy.py` (preserved, not deleted) tried to build a
section-6 counterexample out of the logistic/tent-map topological conjugacy
and was independently found INVALID (CLAIM_LEDGER.md, DECISION_LOG.md
"2026-08-16 -- Independent critic pass", COUNTEREXAMPLES.md): composing the
tent map through the conjugacy just reproduces the logistic map itself
(not a distinct hypothesis), and the only candidate offered as "distinct"
(the tent map misapplied directly to the observed z-coordinate) fails the
observational-equivalence bar outright (26% one-step error vs the 10%
threshold) -- a category error, not a near-miss.

This module instead builds a section-6-compliant counterexample directly in
the OBSERVED z-coordinate, using a completely different mechanism from
either the tent-map attempt above or the first-integral construction
(`src/counterexamples_first_integral.py`):

  True system f: logistic map at r=4,  f(z) = 4 z (1-z).
  Observed data: a single finite trajectory z_0, ..., z_N (no noise).

  Because the trajectory is finite, it visits only N+1 of the uncountably
  many points in [0,1]. Take the (z_n, z_{n+1}) pairs, sort by z_n, and
  build phi_alt as a piecewise-linear interpolant through those exact
  pairs, but with a handful of extra "decoy" nodes inserted into the
  largest gaps between consecutive observed z_n values. Each decoy node
  sits at the gap's midpoint zmid and is assigned the value
  clip(1 - f(zmid), 0, 1) -- deliberately far from the true f(zmid).

  phi_alt matches f EXACTLY at every observed training point (they remain
  literal interpolation nodes), so on-trajectory vector-field error is
  exactly 0. Between two real training nodes that happen to straddle a
  decoy, phi_alt detours sharply away from f, so vector-field error
  evaluated off the training support (a dense grid over [0,1], or an
  independent-IC confirmation trajectory of the same length) is large.

Honesty / scope notes (read before citing this as a general result):

1. Mechanism is generic, not chaos-specific. This construction exploits
   "a finite point set does not pin down an unconstrained function
   elsewhere" -- true for ANY finite dataset from ANY system, chaotic or
   not, discrete map or continuous ODE observed pointwise. Unlike the
   first-integral counterexample (a specific conserved-quantity gauge
   freedom) or the tent-map attempt (a specific, if flawed, conjugacy
   argument), this is not a chaos-dynamics mechanism -- it is a
   finite-sample/model-class-freedom argument. Report it as such.

2. Out of the frozen model class. PREREGISTRATION.md section 2 restricts
   the discovery methods under study (SINDy et al.) to a degree<=3
   polynomial library. phi_alt is a piecewise-linear spline, not a
   degree<=3 polynomial, and is NOT reachable by any discovery method
   this project actually runs. Within the frozen degree<=3 polynomial
   class, no such freedom exists for a long chaotic logistic trajectory:
   the true map is already degree-2, and fitting degree<=2/3 polynomials
   to hundreds of trajectory points pins the coefficients down to the
   true quadratic (consistent with this project's own Tier A/B/C SINDy
   results, RESULTS.md). So this construction demonstrates non-uniqueness
   for an UNCONSTRAINED nonparametric hypothesis class, not a threat to
   the polynomial-library discovery methods this project benchmarks.

3. Adversarially constructed, not found by any discovery method, per
   PREREGISTRATION.md section 6's "either found by a discovery method or
   constructed adversarially" clause -- constructed here, explicitly.

Verified numerically below (three independent initial conditions, not
cherry-picked to one lucky seed): on-trajectory normalized VF L2 error is
exactly 0.0 in all three cases; off-trajectory grid error is 15.7-17.0%;
independent-IC confirmation-trajectory error is 14.2-17.6% -- all well
past the PREREGISTRATION.md section 6 10% threshold, and all obtained with
identical construction parameters (k=30 decoys, N=3000 training points)
across the three seeds.
"""
from __future__ import annotations

import numpy as np

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from simulators import logistic_map  # trusted comparator, r=4 => f(x)=4x(1-x)


R_CHAOTIC = 4.0
N_TRAIN_DEFAULT = 3000
K_DECOYS_DEFAULT = 30
SECTION6_THRESHOLD = 0.10


def f_true(x):
    """True system: logistic map at r=4."""
    return logistic_map(x, R_CHAOTIC)


def generate_trajectory(x0: float, n_steps: int) -> np.ndarray:
    traj = np.empty(n_steps + 1)
    traj[0] = x0
    for i in range(n_steps):
        traj[i + 1] = f_true(traj[i])
    return traj


def build_alternative(
    x0: float, n_steps: int = N_TRAIN_DEFAULT, k_decoys: int = K_DECOYS_DEFAULT
):
    """Build phi_alt: a piecewise-linear interpolant through the observed
    (z_n, z_{n+1}) pairs plus k_decoys adversarial decoy nodes inserted at
    the midpoints of the largest gaps in the training support.

    Returns (phi_alt, diagnostics) where diagnostics records the training
    trajectory, the decoy locations/values, and the node arrays used.
    """
    traj = generate_trajectory(x0, n_steps)
    z_n, z_next = traj[:-1], traj[1:]

    # sort + dedupe by z_n (should already be unique almost surely for a
    # continuous chaotic orbit, but dedupe defensively)
    z_sorted, idx = np.unique(z_n, return_index=True)
    y_sorted = z_next[idx]

    gaps = np.diff(z_sorted)
    gap_idx = np.argsort(gaps)[-k_decoys:]  # largest k_decoys gaps

    decoy_z, decoy_y = [], []
    for gi in gap_idx:
        zmid = 0.5 * (z_sorted[gi] + z_sorted[gi + 1])
        y_true_here = float(f_true(zmid))
        y_decoy = float(np.clip(1.0 - y_true_here, 0.0, 1.0))
        decoy_z.append(zmid)
        decoy_y.append(y_decoy)

    all_z = np.concatenate([z_sorted, decoy_z])
    all_y = np.concatenate([y_sorted, decoy_y])
    order = np.argsort(all_z)
    all_z, all_y = all_z[order], all_y[order]

    def phi_alt(x):
        return np.interp(np.asarray(x, dtype=float), all_z, all_y)

    diagnostics = {
        "x0": x0,
        "n_steps": n_steps,
        "k_decoys": k_decoys,
        "n_train_nodes": int(z_sorted.size),
        "decoy_z": [float(v) for v in decoy_z],
        "decoy_y": [float(v) for v in decoy_y],
        "decoy_f_true": [float(f_true(v)) for v in decoy_z],
        "z_n": z_n,
        "z_next": z_next,
    }
    return phi_alt, diagnostics


# ---------------------------------------------------------------------------
# Verification (a): exact match on the observed training trajectory
# ---------------------------------------------------------------------------

def verify_on_trajectory_match(phi_alt, diagnostics) -> dict:
    pred = phi_alt(diagnostics["z_n"])
    err = pred - diagnostics["z_next"]
    l2_err = float(np.sqrt(np.mean(err**2)))
    normalizer = float(np.sqrt(np.mean(diagnostics["z_next"] ** 2)))
    normalized_l2 = l2_err / normalizer if normalizer > 0 else float("inf")
    return {
        "l2_error": l2_err,
        "normalized_l2_error": normalized_l2,
        "max_abs_error": float(np.max(np.abs(err))),
    }


# ---------------------------------------------------------------------------
# Verification (b): vector-field divergence off the observed trajectory
# (dense grid over the full state space [0,1], excluding a small
# neighborhood of every training node so we are not just re-measuring
# on-support agreement)
# ---------------------------------------------------------------------------

def verify_off_trajectory_grid_divergence(
    phi_alt, diagnostics, n_grid: int = 5000, near_node_eps: float = 1e-3
) -> dict:
    z_sorted = np.unique(diagnostics["z_n"])
    grid = np.linspace(0.001, 0.999, n_grid)

    f_true_grid = f_true(grid)
    f_alt_grid = phi_alt(grid)
    err = f_alt_grid - f_true_grid

    l2_err_all = float(np.sqrt(np.mean(err**2)))
    normalizer_all = float(np.sqrt(np.mean(f_true_grid**2)))
    normalized_l2_all = l2_err_all / normalizer_all

    dist_to_nearest_node = np.min(
        np.abs(grid[:, None] - z_sorted[None, :]), axis=1
    )
    far_mask = dist_to_nearest_node > near_node_eps
    err_far = err[far_mask]
    f_true_far = f_true_grid[far_mask]
    l2_err_far = float(np.sqrt(np.mean(err_far**2)))
    normalizer_far = float(np.sqrt(np.mean(f_true_far**2)))
    normalized_l2_far = l2_err_far / normalizer_far

    return {
        "n_grid": n_grid,
        "normalized_l2_error_full_grid": normalized_l2_all,
        "normalized_l2_error_excluding_near_training_nodes": normalized_l2_far,
        "n_far_from_training_points": int(far_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Verification (c): vector-field divergence on an independent-IC
# confirmation trajectory, per PREREGISTRATION.md section 6's "... or on
# the confirmation trajectory" wording
# ---------------------------------------------------------------------------

def verify_confirmation_trajectory_divergence(
    phi_alt, confirmation_x0: float, n_steps: int = N_TRAIN_DEFAULT
) -> dict:
    traj = generate_trajectory(confirmation_x0, n_steps)
    z_n, z_next = traj[:-1], traj[1:]
    pred = phi_alt(z_n)
    err = pred - z_next
    l2_err = float(np.sqrt(np.mean(err**2)))
    normalizer = float(np.sqrt(np.mean(z_next**2)))
    normalized_l2 = l2_err / normalizer if normalizer > 0 else float("inf")
    return {
        "confirmation_x0": confirmation_x0,
        "n_steps": n_steps,
        "normalized_l2_error": normalized_l2,
    }


# ---------------------------------------------------------------------------
# Full report, run at multiple (training IC, confirmation IC) pairs to show
# the result is not a one-seed artifact
# ---------------------------------------------------------------------------

SEED_PAIRS = [
    (0.31415926, 0.777, "seed_orig"),
    (0.123456, 0.654321, "seed_alt1"),
    (0.9, 0.05, "seed_alt2"),
]


def run(seed_pairs=SEED_PAIRS, n_steps: int = N_TRAIN_DEFAULT, k_decoys: int = K_DECOYS_DEFAULT):
    print("True system: logistic map at r=4, f(z) = 4z(1-z)")
    print(
        f"Alternative phi_alt: piecewise-linear interpolant through the "
        f"observed (z_n, z_n+1) pairs plus {k_decoys} adversarial decoy "
        f"nodes at the largest training-support gaps."
    )
    print()

    results = []
    for x0, conf_x0, label in seed_pairs:
        phi_alt, diag = build_alternative(x0, n_steps=n_steps, k_decoys=k_decoys)

        on_traj = verify_on_trajectory_match(phi_alt, diag)
        off_grid = verify_off_trajectory_grid_divergence(phi_alt, diag)
        confirmation = verify_confirmation_trajectory_divergence(phi_alt, conf_x0, n_steps)

        passed_on = on_traj["normalized_l2_error"] < SECTION6_THRESHOLD
        passed_off_grid = off_grid["normalized_l2_error_full_grid"] > SECTION6_THRESHOLD
        passed_confirmation = confirmation["normalized_l2_error"] > SECTION6_THRESHOLD

        print(f"-- {label}: x0={x0}, confirmation_x0={conf_x0} --")
        print(f"  on-trajectory normalized L2 error:            {on_traj['normalized_l2_error']:.6e}  "
              f"(observationally equivalent: {passed_on})")
        print(f"  off-trajectory grid normalized L2 error:       {off_grid['normalized_l2_error_full_grid']:.4f}  "
              f"(exceeds {SECTION6_THRESHOLD:.0%}: {passed_off_grid})")
        print(f"  confirmation-trajectory normalized L2 error:   {confirmation['normalized_l2_error']:.4f}  "
              f"(exceeds {SECTION6_THRESHOLD:.0%}: {passed_confirmation})")
        print()

        results.append({
            "label": label,
            "on_trajectory": on_traj,
            "off_trajectory_grid": off_grid,
            "confirmation_trajectory": confirmation,
            "section6_verdict": bool(passed_on and passed_off_grid and passed_confirmation),
        })

    all_passed = all(r["section6_verdict"] for r in results)
    print(f"Section-6 counterexample criteria satisfied at all {len(results)} tested seeds: {all_passed}")
    print(
        "Caveats (see module docstring): mechanism is a generic finite-data "
        "interpolation degeneracy, not chaos-specific, and phi_alt is a "
        "piecewise-linear spline OUTSIDE the frozen degree<=3 polynomial "
        "model class used by this project's discovery methods."
    )
    return results


if __name__ == "__main__":
    run()
