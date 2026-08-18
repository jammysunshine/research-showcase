"""Explicit topological/smooth-conjugacy counterexample.

Mechanism (PRIOR_ART.md section 5): two vector fields/maps related by a
smooth change of coordinates h, satisfying h(g(theta)) = f(h(theta)), push
forward to *exactly* the same observed sequence under h. If an analyst
observes only h(theta_n) and does not know h, the smooth quadratic map f and
the piecewise-linear map g (composed with the unknown coordinate change h)
are perfectly observationally equivalent -- yet they are trivially,
massively dynamically distinct the moment either (a) the raw internal
coordinate theta is compared against the raw internal coordinate x, or
(b) an analyst who suspects "maybe it's a tent map" naively applies g
directly to the observed sequence instead of composing through h.

Concretely:
  - True system f: logistic map at r=4,      f(x)     = 4 x (1-x)
  - Alternative g: tent map,                 g(theta) = 1 - |2 theta - 1|
  - Conjugacy h:                             h(theta) = sin^2(pi theta / 2)
    with h(g(theta)) == f(h(theta)) for all theta in [0,1] (exact identity,
    verified both symbolically-by-construction and numerically below).

Observation operator OP1 ("public" observation): record z_n = h(theta_n)
(equivalently, record x_n directly from f -- they are the same sequence).
Under OP1 the two systems are observationally indistinguishable to machine
precision.

Off OP1 (section 6 stress tests):
  (a) Raw-coordinate divergence: theta_n (system g's native state) vs
      x_n (system f's native state) starting from "the same" observed point
      -- these differ by O(1), not by numerical noise.
  (b) Vector-field L2 error: an analyst who (wrongly) assumes the observed
      sequence obeys g directly, i.e. predicts z_hat_{n+1} = g(z_n) instead
      of f(z_n), incurs a normalized-L2 one-step prediction error that
      blows past the section-6 10% threshold.
  (c) Invariant-measure divergence: the invariant density of x under f is
      the arcsine law rho(x) = 1/(pi sqrt(x(1-x))); the invariant density of
      theta under g is uniform on [0,1]. Confusing theta_n for x_n gives a
      histogram total-variation distance far past the section-6 0.1
      threshold, while z_n = h(theta_n) matches x_n's histogram almost
      exactly (TV distance ~ sampling noise only).
"""
from __future__ import annotations

import numpy as np

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from simulators import logistic_map  # trusted comparator, r=4 => f(x)=4x(1-x)


R_CHAOTIC = 4.0


# ---------------------------------------------------------------------------
# The two maps and the conjugacy between them
# ---------------------------------------------------------------------------

def f_logistic(x: np.ndarray | float) -> np.ndarray | float:
    """True system: logistic map at r=4."""
    return logistic_map(x, R_CHAOTIC)


def g_tent(theta: np.ndarray | float) -> np.ndarray | float:
    """Alternative, structurally different-looking map: the tent map on [0,1]."""
    theta = np.asarray(theta, dtype=float)
    return 1.0 - np.abs(2.0 * theta - 1.0)


def h_conjugacy(theta: np.ndarray | float) -> np.ndarray | float:
    """Smooth conjugacy h: theta-coordinate -> x-coordinate."""
    theta = np.asarray(theta, dtype=float)
    return np.sin(np.pi * theta / 2.0) ** 2


def h_inv(x: np.ndarray | float) -> np.ndarray | float:
    """Inverse conjugacy x -> theta, principal branch theta in [0,1]."""
    x = np.asarray(x, dtype=float)
    x_clipped = np.clip(x, 0.0, 1.0)
    return (2.0 / np.pi) * np.arcsin(np.sqrt(x_clipped))


# ---------------------------------------------------------------------------
# Trajectory generators (native coordinates for each system)
# ---------------------------------------------------------------------------

def trajectory_f(x0: float, n_steps: int) -> np.ndarray:
    xs = np.empty(n_steps + 1)
    xs[0] = x0
    for i in range(n_steps):
        xs[i + 1] = f_logistic(xs[i])
    return xs


def trajectory_g(theta0: float, n_steps: int) -> np.ndarray:
    thetas = np.empty(n_steps + 1)
    thetas[0] = theta0
    for i in range(n_steps):
        thetas[i + 1] = g_tent(thetas[i])
    return thetas


# ---------------------------------------------------------------------------
# Verification 1: exact conjugacy identity h(g(theta)) == f(h(theta))
# ---------------------------------------------------------------------------

def verify_conjugacy_identity(n_points: int = 200_001, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 1.0, size=n_points)
    lhs = h_conjugacy(g_tent(theta))
    rhs = f_logistic(h_conjugacy(theta))
    err = np.abs(lhs - rhs)
    return {
        "max_abs_error": float(err.max()),
        "mean_abs_error": float(err.mean()),
        "n_points": n_points,
    }


# ---------------------------------------------------------------------------
# Verification 2: observational indistinguishability under OP1
# (observe z_n = h(theta_n); compare against x_n from f directly)
# ---------------------------------------------------------------------------

def verify_observational_equivalence(x0: float = 0.31415926, n_steps: int = 25) -> dict:
    """Demonstrate x_n == h(theta_n) to near machine precision.

    n_steps is kept modest (default 25) by design: both maps have Lyapunov
    exponent ln(2), so two float64 trajectories that are mathematically
    identical but computed via independently-rounded paths (iterate f
    directly, vs iterate g then apply h) separate at rate ~2^n even though
    the underlying identity h(g(theta))=f(h(theta)) is exact at every single
    step (see verify_conjugacy_identity, error ~1e-16 regardless of n). This
    is the ordinary chaos/floating-point-amplification effect, not a flaw in
    the conjugacy; see `floating_point_amplification_diagnostic` below for
    the explicit growth curve.
    """
    theta0 = float(h_inv(x0))
    x_traj = trajectory_f(x0, n_steps)          # system f, native coordinate
    theta_traj = trajectory_g(theta0, n_steps)  # system g, native coordinate
    z_observed = h_conjugacy(theta_traj)        # system g pushed through h -> OP1 observation

    diff = np.abs(x_traj - z_observed)
    return {
        "x0": x0,
        "theta0": theta0,
        "n_steps": n_steps,
        "max_abs_diff_under_OP1": float(diff.max()),
        "mean_abs_diff_under_OP1": float(diff.mean()),
        "x_traj": x_traj,
        "theta_traj": theta_traj,
        "z_observed": z_observed,
    }


def floating_point_amplification_diagnostic(
    x0: float = 0.31415926, n_max: int = 60
) -> dict:
    """Track |x_n - h(theta_n)| vs n to show the divergence is driven by
    rounding-error amplification at rate ~2^n (Lyapunov exponent ln 2), not
    by any failure of the exact conjugacy identity."""
    theta0 = float(h_inv(x0))
    x_traj = trajectory_f(x0, n_max)
    theta_traj = trajectory_g(theta0, n_max)
    z_observed = h_conjugacy(theta_traj)
    diff = np.abs(x_traj - z_observed)
    n_at_O1 = int(np.argmax(diff > 0.5)) if np.any(diff > 0.5) else None
    return {
        "n_max": n_max,
        "diff_by_step": diff.tolist(),
        "first_n_diff_exceeds_0.5": n_at_O1,
        "predicted_n_from_lyapunov": float(np.log(1e-16 ** -1 * 0.5) / np.log(2.0)),
    }


# ---------------------------------------------------------------------------
# Stress test (a): raw-coordinate divergence (off OP1)
# ---------------------------------------------------------------------------

def raw_coordinate_divergence(x_traj: np.ndarray, theta_traj: np.ndarray) -> dict:
    diff = np.abs(x_traj - theta_traj)
    return {
        "max_abs_diff_raw": float(diff.max()),
        "mean_abs_diff_raw": float(diff.mean()),
        "rms_diff_raw": float(np.sqrt(np.mean(diff ** 2))),
    }


# ---------------------------------------------------------------------------
# Stress test (b): vector-field L2 error under the naive "apply g directly
# to observed data" mis-identification (section 6 metric: normalized L2 > 10%)
# ---------------------------------------------------------------------------

def naive_vector_field_error(z_observed: np.ndarray) -> dict:
    """An analyst observes z_n = z_observed (== x_n under the true system)
    and, suspecting a tent map, predicts z_hat_{n+1} = g(z_n) instead of the
    correct f(z_n). Report normalized-L2 one-step prediction error."""
    z_n = z_observed[:-1]
    z_next_true = z_observed[1:]  # == f(z_n), since z_observed == x_traj under OP1
    z_next_naive_g = g_tent(z_n)  # wrong: applying g without composing through h

    err = z_next_naive_g - z_next_true
    l2_error = np.sqrt(np.mean(err ** 2))
    normalizer = np.sqrt(np.mean(z_next_true ** 2))
    normalized_l2 = l2_error / normalizer if normalizer > 0 else float("inf")

    # control: correctly composing through h and h_inv recovers zero error
    z_next_correct = f_logistic(z_n)
    control_err = np.sqrt(np.mean((z_next_correct - z_next_true) ** 2))

    return {
        "normalized_l2_error_naive_g": float(normalized_l2),
        "l2_error_naive_g": float(l2_error),
        "control_l2_error_correct_f": float(control_err),
        "section6_threshold": 0.10,
        "exceeds_threshold": bool(normalized_l2 > 0.10),
    }


# ---------------------------------------------------------------------------
# Stress test (c): invariant-measure total-variation divergence
# ---------------------------------------------------------------------------

def _histogram_density(samples: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(samples, bins=bins, density=False)
    probs = counts / counts.sum()
    return probs


def invariant_measure_tv_distance(
    x_long: np.ndarray, theta_long: np.ndarray, z_long: np.ndarray, n_bins: int = 50
) -> dict:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    p_x = _histogram_density(x_long, bins)
    p_theta = _histogram_density(theta_long, bins)
    p_z = _histogram_density(z_long, bins)

    tv_x_vs_theta_raw = 0.5 * np.sum(np.abs(p_x - p_theta))
    tv_x_vs_z_observed = 0.5 * np.sum(np.abs(p_x - p_z))

    return {
        "n_bins": n_bins,
        "tv_distance_x_vs_theta_raw_offOP1": float(tv_x_vs_theta_raw),
        "tv_distance_x_vs_z_observed_underOP1": float(tv_x_vs_z_observed),
        "section6_threshold": 0.10,
        "raw_exceeds_threshold": bool(tv_x_vs_theta_raw > 0.10),
        "observed_within_threshold": bool(tv_x_vs_z_observed <= 0.10),
    }


def _ensemble_invariant_samples(step_func, n_burn: int, n_realizations: int, rng: np.random.Generator) -> np.ndarray:
    """Sample the invariant measure via many short independent trajectories
    rather than one very long trajectory.

    This sidesteps a real numerical artifact: the tent map is (up to the
    conjugacy) a bit-shift/doubling map, so iterating float64 orbits of it
    for tens of thousands of steps deterministically shifts mantissa bits
    out and the orbit collapses onto the exact fixed point 0 within ~50-60
    steps -- a floating-point-precision artifact of expanding piecewise-
    linear maps, not a statement about the true (uniform) invariant measure.
    Many independent short (n_burn-step) trajectories, well below the
    collapse horizon identified by `floating_point_amplification_diagnostic`,
    avoid the artifact while still sampling the invariant measure (the tent
    map mixes exponentially fast, so n_burn steps is ample burn-in).
    """
    x0 = rng.uniform(0.0, 1.0, size=n_realizations)
    x = x0.copy()
    for _ in range(n_burn):
        x = step_func(x)
    return x


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def run_full_verification(seed: int = 0) -> dict:
    conjugacy = verify_conjugacy_identity(n_points=200_001, seed=seed)

    obs = verify_observational_equivalence(x0=0.31415926, n_steps=25)
    raw_div = raw_coordinate_divergence(obs["x_traj"], obs["theta_traj"])

    # naive vector-field error is computed on a longer trajectory: it only
    # relies on the per-step identity (exact to ~1e-16 at every step
    # regardless of horizon), not on paired-trajectory agreement, so it is
    # unaffected by the chaos/floating-point amplification discussed above.
    obs_long = verify_observational_equivalence(x0=0.31415926, n_steps=5000)
    vf_err = naive_vector_field_error(obs_long["z_observed"])
    fp_diag = floating_point_amplification_diagnostic(x0=0.31415926, n_max=60)

    # Invariant-measure comparison: many independent short trajectories
    # (ensemble sampling), not one very long trajectory -- see
    # `_ensemble_invariant_samples` docstring for why (tent-map mantissa
    # collapse under long single-trajectory float64 iteration).
    rng = np.random.default_rng(seed)
    n_burn = 40  # well below the ~52-55 step collapse horizon
    n_realizations = 50_000

    x_long = _ensemble_invariant_samples(f_logistic, n_burn, n_realizations, rng)
    theta_long = _ensemble_invariant_samples(
        g_tent, n_burn, n_realizations, np.random.default_rng(seed + 1)
    )
    z_long = h_conjugacy(theta_long)

    tv = invariant_measure_tv_distance(x_long, theta_long, z_long, n_bins=50)

    return {
        "conjugacy_identity": conjugacy,
        "observational_equivalence_OP1": {
            k: v for k, v in obs.items() if k not in ("x_traj", "theta_traj", "z_observed")
        },
        "raw_coordinate_divergence_offOP1": raw_div,
        "naive_vector_field_error_offOP1": vf_err,
        "invariant_measure_tv_distance": tv,
        "floating_point_amplification_diagnostic": {
            k: v for k, v in fp_diag.items() if k != "diff_by_step"
        },
    }


if __name__ == "__main__":
    import json

    report = run_full_verification(seed=0)
    print(json.dumps(report, indent=2))

    print("\n--- Summary ---")
    print(f"Conjugacy identity max error: {report['conjugacy_identity']['max_abs_error']:.3e}")
    print(
        "OP1 observational-equivalence max diff: "
        f"{report['observational_equivalence_OP1']['max_abs_diff_under_OP1']:.3e}"
    )
    print(
        "Off-OP1 raw-coordinate mean |diff|: "
        f"{report['raw_coordinate_divergence_offOP1']['mean_abs_diff_raw']:.4f}"
    )
    print(
        "Off-OP1 naive vector-field normalized L2 error: "
        f"{report['naive_vector_field_error_offOP1']['normalized_l2_error_naive_g']:.4f} "
        f"(threshold 0.10, exceeds="
        f"{report['naive_vector_field_error_offOP1']['exceeds_threshold']})"
    )
    print(
        "Invariant-measure TV distance, x vs theta_raw (off OP1): "
        f"{report['invariant_measure_tv_distance']['tv_distance_x_vs_theta_raw_offOP1']:.4f} "
        f"(threshold 0.10, exceeds="
        f"{report['invariant_measure_tv_distance']['raw_exceeds_threshold']})"
    )
    print(
        "Invariant-measure TV distance, x vs z_observed (under OP1): "
        f"{report['invariant_measure_tv_distance']['tv_distance_x_vs_z_observed_underOP1']:.4f} "
        f"(within threshold="
        f"{report['invariant_measure_tv_distance']['observed_within_threshold']})"
    )
