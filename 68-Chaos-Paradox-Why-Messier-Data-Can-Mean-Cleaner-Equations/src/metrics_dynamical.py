"""Secondary identifiability metrics from PREREGISTRATION.md section 6:

  - Largest-Lyapunov-exponent error between a TRUE and a DISCOVERED system.
  - Invariant-measure total-variation distance between two trajectories.
  - Bifurcation-location (period-doubling onset) error for the logistic map.

Each metric compares a TRUE system against a DISCOVERED one (e.g. a fitted
SINDy/symbolic-regression/Koopman model) on the same regime, per the
"dynamically distinct" thresholds in PREREGISTRATION.md section 6.

Reuses the Benettin-QR machinery in src/simulators.py (generic_map_lyapunov_exponent,
generic_ode_lyapunov_spectrum) rather than reimplementing it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from src.simulators import generic_map_lyapunov_exponent, generic_ode_lyapunov_spectrum


@dataclass
class DynamicalSystem:
    """Minimal system spec consumed by lyapunov_spectrum_error.

    kind="map": rhs(x) -> x_next, jacobian(x) -> scalar derivative, dim=1.
    kind="ode": rhs(t, state) -> dstate, jacobian(state) -> dim x dim matrix.
    params is unused by the metric itself; carried for caller bookkeeping only.
    """
    kind: str  # "map" | "ode"
    rhs: Callable
    jacobian: Callable
    dim: int = 1
    params: Optional[dict] = None


# ---------------------------------------------------------------------------
# 1. Lyapunov-spectrum (largest exponent) error
# ---------------------------------------------------------------------------

def lyapunov_spectrum_error(
    true_system: DynamicalSystem,
    discovered_system: DynamicalSystem,
    x0,
    n_steps: int = 5000,
    transient: int = 2000,
    dt: float = 0.01,
) -> float:
    """Absolute difference between largest-Lyapunov-exponent estimates.

    Both systems must be evaluated on the SAME regime (same x0, same
    n_steps/transient/dt) for the comparison to be meaningful.
    """
    if true_system.kind != discovered_system.kind:
        raise ValueError(f"kind mismatch: true={true_system.kind!r} discovered={discovered_system.kind!r}")
    if true_system.dim != discovered_system.dim:
        raise ValueError(f"dim mismatch: true={true_system.dim} discovered={discovered_system.dim}")

    if true_system.kind == "map":
        lam_true = generic_map_lyapunov_exponent(
            true_system.rhs, true_system.jacobian, x0, n_steps, transient,
        )
        lam_disc = generic_map_lyapunov_exponent(
            discovered_system.rhs, discovered_system.jacobian, x0, n_steps, transient,
        )
    elif true_system.kind == "ode":
        spec_true = generic_ode_lyapunov_spectrum(
            true_system.rhs, true_system.jacobian, x0, true_system.dim,
            dt=dt, n_steps=n_steps, transient_steps=transient,
        )
        spec_disc = generic_ode_lyapunov_spectrum(
            discovered_system.rhs, discovered_system.jacobian, x0, discovered_system.dim,
            dt=dt, n_steps=n_steps, transient_steps=transient,
        )
        lam_true = float(np.max(spec_true))
        lam_disc = float(np.max(spec_disc))
    else:
        raise ValueError(f"unknown kind {true_system.kind!r}, expected 'map' or 'ode'")

    return abs(lam_true - lam_disc)


# ---------------------------------------------------------------------------
# 2. Invariant-measure total-variation distance
# ---------------------------------------------------------------------------

def invariant_measure_tv_distance(
    true_trajectory: np.ndarray,
    discovered_trajectory: np.ndarray,
    bins: int = 50,
) -> float:
    """Histogram-based total-variation distance between two empirical invariant
    measures (attractor occupation densities).

    LIMITATION: this computes per-dimension MARGINAL histograms and averages
    their TV distances across dimensions -- it does NOT estimate the full
    joint-density TV distance (which would require a dim-dimensional
    histogram/KDE and is far more sample-hungry, especially for 3D Lorenz
    trajectories of realistic length). Two attractors can share all marginals
    while differing in joint structure (e.g. correlation/shape), so this is a
    necessary-but-not-sufficient check, not the literal quantity in
    PREREGISTRATION.md section 6. Documented here rather than silently
    treated as equivalent to joint-density TV distance.
    """
    true_traj = np.atleast_1d(np.asarray(true_trajectory, dtype=float))
    disc_traj = np.atleast_1d(np.asarray(discovered_trajectory, dtype=float))
    if true_traj.ndim == 1:
        true_traj = true_traj[:, None]
    if disc_traj.ndim == 1:
        disc_traj = disc_traj[:, None]
    if true_traj.shape[1] != disc_traj.shape[1]:
        raise ValueError(f"dimension mismatch: true has {true_traj.shape[1]} dims, discovered has {disc_traj.shape[1]}")

    dim = true_traj.shape[1]
    tv_per_dim = np.empty(dim)
    for d in range(dim):
        col_true = true_traj[:, d]
        col_disc = disc_traj[:, d]
        lo = min(col_true.min(), col_disc.min())
        hi = max(col_true.max(), col_disc.max())
        if hi <= lo:
            # Degenerate (constant) marginal on both sides: densities trivially agree.
            tv_per_dim[d] = 0.0
            continue
        edges = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(col_true, bins=edges, density=False)
        q, _ = np.histogram(col_disc, bins=edges, density=False)
        p = p / p.sum()
        q = q / q.sum()
        tv_per_dim[d] = 0.5 * np.sum(np.abs(p - q))

    return float(np.mean(tv_per_dim))


# ---------------------------------------------------------------------------
# 3. Bifurcation-location (period-doubling onset) error, logistic map family
# ---------------------------------------------------------------------------

def bifurcation_location_error(
    true_bifurcation_param: float,
    discovered_system_family: Callable[[float], Callable[[float], float]],
    param_range: tuple[float, float],
    n_param_steps: int = 400,
    x0: float = 0.4,
    transient: int = 2000,
    n_tail: int = 200,
    round_decimals: int = 3,
) -> dict:
    """Locate the first period-1 -> period-2 doubling of a 1D map family as its
    parameter is swept, and compare against the known true bifurcation param.

    discovered_system_family(param) must return a callable map: x -> x_next.

    "Found" means: scanning param_range left-to-right, some grid point i is the
    first where the unique count of the rounded tail-iterate set goes from
    exactly 1 (fixed point) to exactly 2 (period-2 cycle). The reported
    location is that grid parameter value (no sub-grid interpolation).

    "Not found" (location=None, error=None, reason=<str>) covers:
      - every scanned param produces a diverging/non-finite trajectory,
      - the unique-count sequence never contains a clean 1 -> 2 transition
        (e.g. it jumps straight from 1 to >2, stays constant throughout, or
        is never exactly 1 anywhere before the jump),
      - fewer than 2 grid points are usable.

    Returns dict(location, error, reason, param_grid, unique_counts) so callers
    can inspect the raw scan, not just the pass/fail summary.
    """
    if n_param_steps < 2:
        raise ValueError("n_param_steps must be >= 2")

    param_grid = np.linspace(param_range[0], param_range[1], n_param_steps)
    unique_counts: list[Optional[int]] = []

    for p in param_grid:
        map_fn = discovered_system_family(p)
        x = x0
        diverged = False
        for _ in range(transient):
            x = map_fn(x)
            if not np.isfinite(x) or abs(x) > 1e6:
                diverged = True
                break
        if diverged:
            unique_counts.append(None)
            continue
        tail = []
        for _ in range(n_tail):
            x = map_fn(x)
            if not np.isfinite(x) or abs(x) > 1e6:
                diverged = True
                break
            tail.append(round(x, round_decimals))
        if diverged:
            unique_counts.append(None)
            continue
        unique_counts.append(len(set(tail)))

    location = None
    reason = "no clean period-1 -> period-2 transition found in the scanned range"
    for i in range(1, len(unique_counts)):
        prev, cur = unique_counts[i - 1], unique_counts[i]
        if prev == 1 and cur == 2:
            location = float(param_grid[i])
            reason = "found: clean unique-count 1 -> 2 transition"
            break

    if location is None:
        if all(c is None for c in unique_counts):
            reason = "discovered map diverges/non-finite across the entire scanned range"
        error = None
    else:
        if true_bifurcation_param == 0:
            error = abs(location - true_bifurcation_param)
        else:
            error = abs(location - true_bifurcation_param) / abs(true_bifurcation_param)

    return {
        "location": location,
        "error": error,
        "reason": reason,
        "param_grid": param_grid,
        "unique_counts": unique_counts,
    }
