"""Explicit first-integral counterexample: non-identifiability of the
undamped harmonic oscillator from trajectory data alone.

True system (conservative, first integral H = x^2 + y^2):
    dx/dt = y
    dy/dt = -x

Every orbit is a circle of radius r0 = sqrt(H0), traversed at constant
angular speed 1. H is conserved exactly: dH/dt = 2x*dx/dt + 2y*dy/dt
= 2xy - 2xy = 0.

Adversarial alternative family (indexed by a real parameter a != 0):
    f_alt_a(x, y) = g_a(H) * (y, -x),   where H = x^2 + y^2
    g_a(H) = 1 + a * (H - H0)

g_a(H0) = 1 identically, so on the exact observed level set H = H0 the
alternative vector field is IDENTICAL to the true one (same direction,
same speed) at every point of the orbit -- not just close, but equal in
direction and magnitude, because the scalar factor evaluates to exactly
1 wherever H = H0. Consequently the alternative reproduces the observed
trajectory to numerical-integration precision.

Off that single level set (H != H0) the two vector fields differ by the
multiplicative factor g_a(H), which is exactly the mechanism described in
PRIOR_ART.md section 5 / arXiv:2511.08860: a conserved quantity leaves a
whole continuum of vector fields (one per choice of a, or more generally
one per choice of g with g(H0)=1) consistent with the same orbit, because
trajectory data can only ever probe H = H0 and is blind to how the vector
field's magnitude varies with H away from that surface.

This directly instantiates PREREGISTRATION.md section 6's definition of
"observationally equivalent but dynamically distinct": vector-field L2
error on the observed trajectory is ~0 (within integrator tolerance),
while the normalized vector-field L2 error evaluated off the observed
level set exceeds the 10% threshold.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# True system: undamped harmonic oscillator
# ---------------------------------------------------------------------------

def true_field(t, state):
    x, y = state
    return [y, -x]


def true_field_vec(xy: np.ndarray) -> np.ndarray:
    """Vectorized true field. xy shape (N, 2) -> (N, 2)."""
    x, y = xy[:, 0], xy[:, 1]
    return np.stack([y, -x], axis=1)


# ---------------------------------------------------------------------------
# Adversarial alternative family
# ---------------------------------------------------------------------------

def make_alt_field(H0: float, a: float):
    """Return (scalar-ODE rhs, vectorized field) for g_a(H) = 1 + a*(H - H0)."""

    def g(H):
        return 1.0 + a * (H - H0)

    def alt_field(t, state):
        x, y = state
        H = x * x + y * y
        factor = g(H)
        return [factor * y, -factor * x]

    def alt_field_vec(xy: np.ndarray) -> np.ndarray:
        x, y = xy[:, 0], xy[:, 1]
        H = x * x + y * y
        factor = g(H)
        return np.stack([factor * y, -factor * x], axis=1)

    return alt_field, alt_field_vec, g


# ---------------------------------------------------------------------------
# Generate observed trajectory (finite orbit at fixed energy H0)
# ---------------------------------------------------------------------------

def generate_observed_trajectory(r0: float = 1.3, n_periods: float = 3.0,
                                  n_points: int = 400):
    """Integrate the true oscillator started on the circle of radius r0
    for n_periods full revolutions (period = 2*pi for this system)."""
    x0, y0 = r0, 0.0
    t_end = n_periods * 2 * np.pi
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(true_field, (0.0, t_end), [x0, y0], t_eval=t_eval,
                     method="DOP853", rtol=1e-12, atol=1e-14)
    assert sol.success, "reference integration failed"
    return sol.t, sol.y.T  # shape (n_points, 2)


# ---------------------------------------------------------------------------
# Verification (a): alternative reproduces the observed trajectory
# ---------------------------------------------------------------------------

def verify_trajectory_match(r0: float, a: float, n_periods: float = 3.0,
                             n_points: int = 400):
    H0 = r0 * r0
    t, true_traj = generate_observed_trajectory(r0, n_periods, n_points)

    alt_field, _, _ = make_alt_field(H0, a)
    x0, y0 = r0, 0.0
    t_end = t[-1]
    sol_alt = solve_ivp(alt_field, (0.0, t_end), [x0, y0], t_eval=t,
                         method="DOP853", rtol=1e-12, atol=1e-14)
    assert sol_alt.success, "alternative integration failed"
    alt_traj = sol_alt.y.T

    diff = alt_traj - true_traj
    rmse = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    max_abs_err = np.max(np.abs(diff))

    # Also confirm H is exactly conserved by both trajectories (sanity
    # check that we are indeed comparing two curves on the same circle).
    H_true = np.sum(true_traj ** 2, axis=1)
    H_alt = np.sum(alt_traj ** 2, axis=1)

    return {
        "rmse": float(rmse),
        "max_abs_err": float(max_abs_err),
        "H0": H0,
        "H_true_std": float(np.std(H_true)),
        "H_alt_std": float(np.std(H_alt)),
    }


# ---------------------------------------------------------------------------
# Verification (b): vector-field divergence off the observed level set
# ---------------------------------------------------------------------------

def verify_offset_vector_field_divergence(r0: float, a: float,
                                           n_points: int = 2000,
                                           seed: int = 0):
    """Normalized vector-field L2 error between true and alternative
    fields, evaluated at points sampled OFF the observed level set
    H = H0 (per PREREGISTRATION.md section 6 definition)."""
    H0 = r0 * r0
    _, alt_field_vec, g = make_alt_field(H0, a)

    rng = np.random.default_rng(seed)
    # Sample points on an annulus [0.3*r0, 3*r0] in radius, excluding a
    # thin band immediately around r0 (the exact observed level set) so
    # we are honestly measuring off-orbit divergence, not on-orbit noise.
    radii = rng.uniform(0.3 * r0, 3.0 * r0, size=n_points)
    radii = radii[np.abs(radii - r0) > 0.05 * r0]
    angles = rng.uniform(0.0, 2 * np.pi, size=radii.size)
    xy = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

    f_true = true_field_vec(xy)
    f_alt = alt_field_vec(xy)

    err = f_alt - f_true
    l2_err = np.linalg.norm(err, axis=1)
    l2_true = np.linalg.norm(f_true, axis=1)
    normalized_l2 = np.sqrt(np.sum(l2_err ** 2) / np.sum(l2_true ** 2))

    # Also report divergence at a single clean "different energy level"
    # confirmation point, per the prereg wording ("or on the confirmation
    # trajectory").
    H_far = 4.0 * H0
    r_far = np.sqrt(H_far)
    far_point = np.array([[r_far, 0.0]])
    f_true_far = true_field_vec(far_point)[0]
    f_alt_far = alt_field_vec(far_point)[0]
    far_rel_err = np.linalg.norm(f_alt_far - f_true_far) / np.linalg.norm(f_true_far)

    return {
        "n_offset_points": int(radii.size),
        "normalized_l2_vector_field_error": float(normalized_l2),
        "far_level_H": float(H_far),
        "far_level_relative_error": float(far_rel_err),
        "g_at_far_level": float(g(H_far)),
    }


# ---------------------------------------------------------------------------
# Main: run both verifications and print the numbers
# ---------------------------------------------------------------------------

def run(r0: float = 1.3, a: float = 0.5):
    print(f"True system: dx/dt=y, dy/dt=-x  (conserves H = x^2 + y^2)")
    print(f"Observed orbit: r0 = {r0}  (H0 = {r0**2:.6f})")
    print(f"Alternative family: f_alt(x,y) = g(H)*(y,-x), g(H) = 1 + a*(H-H0), a = {a}")
    print()

    traj_result = verify_trajectory_match(r0, a)
    print("-- (a) Trajectory match on observed level set --")
    for k, v in traj_result.items():
        print(f"  {k}: {v:.3e}" if isinstance(v, float) else f"  {k}: {v}")
    print()

    field_result = verify_offset_vector_field_divergence(r0, a)
    print("-- (b) Vector-field divergence off observed level set --")
    for k, v in field_result.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print()

    threshold = 0.10
    passed = field_result["normalized_l2_vector_field_error"] > threshold
    print(f"PREREGISTRATION.md section 6 threshold (normalized VF L2 error > {threshold:.0%}): "
          f"{'EXCEEDED -> dynamically distinct' if passed else 'NOT exceeded'}")
    print(f"Trajectory RMSE on observed set: {traj_result['rmse']:.3e} "
          f"(effectively zero -> observationally equivalent on training data)")

    return traj_result, field_result


if __name__ == "__main__":
    run()
