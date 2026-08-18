"""Self-check tests for src/metrics_dynamical.py against known cases.

A system compared against ITSELF must give (near-)zero for every metric;
a genuinely different system must be discriminated by lyapunov_spectrum_error.
"""
import numpy as np
import pytest

from src.metrics_dynamical import (
    DynamicalSystem,
    bifurcation_location_error,
    invariant_measure_tv_distance,
    lyapunov_spectrum_error,
)
from src.simulators import (
    logistic_map,
    logistic_map_jacobian,
    logistic_trajectory,
    lorenz_jacobian,
    lorenz_rhs,
    lorenz_trajectory,
)


def _lorenz_system(sigma=10.0, rho=28.0, beta=8.0 / 3.0) -> DynamicalSystem:
    return DynamicalSystem(
        kind="ode",
        rhs=lambda t, state: lorenz_rhs(t, state, sigma, rho, beta),
        jacobian=lambda state: lorenz_jacobian(state, sigma, rho, beta),
        dim=3,
        params=dict(sigma=sigma, rho=rho, beta=beta),
    )


def _logistic_system(r: float) -> DynamicalSystem:
    return DynamicalSystem(
        kind="map",
        rhs=lambda x: logistic_map(x, r),
        jacobian=lambda x: logistic_map_jacobian(x, r),
        dim=1,
        params=dict(r=r),
    )


# ---------------------------------------------------------------------------
# 1. lyapunov_spectrum_error
# ---------------------------------------------------------------------------

def test_lyapunov_spectrum_error_lorenz_same_system_is_near_zero():
    true_sys = _lorenz_system(rho=28.0)
    disc_sys = _lorenz_system(rho=28.0)
    err = lyapunov_spectrum_error(
        true_sys, disc_sys, x0=np.array([1.0, 1.0, 1.0]),
        n_steps=2000, transient=1000, dt=0.01,
    )
    assert err == pytest.approx(0.0, abs=1e-9)


def test_lyapunov_spectrum_error_lorenz_different_rho_is_large():
    # rho=28 is chaotic (largest LE ~0.9); rho=14 is below the chaos onset
    # (stable fixed point, largest LE < 0), so the gap should be sizable.
    true_sys = _lorenz_system(rho=28.0)
    disc_sys = _lorenz_system(rho=14.0)
    err = lyapunov_spectrum_error(
        true_sys, disc_sys, x0=np.array([1.0, 1.0, 1.0]),
        n_steps=2000, transient=1000, dt=0.01,
    )
    assert err > 0.5  # well above the 0.1 dynamically-distinct threshold


def test_lyapunov_spectrum_error_logistic_same_system_is_near_zero():
    true_sys = _logistic_system(r=4.0)
    disc_sys = _logistic_system(r=4.0)
    err = lyapunov_spectrum_error(true_sys, disc_sys, x0=0.4, n_steps=50000, transient=1000)
    assert err == pytest.approx(0.0, abs=1e-9)


def test_lyapunov_spectrum_error_logistic_different_r_is_large():
    true_sys = _logistic_system(r=4.0)   # chaotic, LE = log(2)
    disc_sys = _logistic_system(r=3.2)   # period-2, LE < 0
    err = lyapunov_spectrum_error(true_sys, disc_sys, x0=0.4, n_steps=50000, transient=1000)
    assert err > 0.5


def test_lyapunov_spectrum_error_rejects_kind_mismatch():
    with pytest.raises(ValueError):
        lyapunov_spectrum_error(_logistic_system(4.0), _lorenz_system(), x0=0.4)


# ---------------------------------------------------------------------------
# 2. invariant_measure_tv_distance
# ---------------------------------------------------------------------------

def test_invariant_measure_tv_distance_logistic_self_comparison_is_near_zero():
    traj = logistic_trajectory(0.4, r=4.0, n_steps=20000, transient=1000)
    tv = invariant_measure_tv_distance(traj, traj, bins=50)
    assert tv == pytest.approx(0.0, abs=1e-9)


def test_invariant_measure_tv_distance_lorenz_self_comparison_is_near_zero():
    _, states = lorenz_trajectory(np.array([1.0, 1.0, 1.0]), t_span=(0, 40), n_points=8000)
    tv = invariant_measure_tv_distance(states, states, bins=40)
    assert tv == pytest.approx(0.0, abs=1e-9)


def test_invariant_measure_tv_distance_discriminates_different_regimes():
    # Chaotic (r=4.0) vs period-2 (r=3.2) logistic occupation densities differ a lot.
    chaotic = logistic_trajectory(0.4, r=4.0, n_steps=20000, transient=1000)
    period2 = logistic_trajectory(0.4, r=3.2, n_steps=20000, transient=1000)
    tv = invariant_measure_tv_distance(chaotic, period2, bins=50)
    assert tv > 0.1  # above the PREREGISTRATION.md section 6 threshold


def test_invariant_measure_tv_distance_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        invariant_measure_tv_distance(np.zeros((10, 3)), np.zeros((10, 2)))


# ---------------------------------------------------------------------------
# 3. bifurcation_location_error
# ---------------------------------------------------------------------------

def test_bifurcation_location_error_logistic_self_comparison_is_near_zero():
    # True period-doubling onset for the logistic map is r=3 exactly.
    def family(r):
        return lambda x: logistic_map(x, r)

    result = bifurcation_location_error(
        true_bifurcation_param=3.0,
        discovered_system_family=family,
        param_range=(2.5, 3.5),
        n_param_steps=400,
    )
    assert result["location"] is not None
    assert result["location"] == pytest.approx(3.0, abs=0.02)
    assert result["error"] == pytest.approx(0.0, abs=0.01)


def test_bifurcation_location_error_reports_not_found_when_no_transition():
    # Constant param range entirely below the first doubling: always period-1.
    def family(r):
        return lambda x: logistic_map(x, r)

    result = bifurcation_location_error(
        true_bifurcation_param=3.0,
        discovered_system_family=family,
        param_range=(1.5, 2.5),
        n_param_steps=100,
    )
    assert result["location"] is None
    assert result["error"] is None
    assert isinstance(result["reason"], str) and len(result["reason"]) > 0


def test_bifurcation_location_error_reports_not_found_when_map_diverges():
    # A pathological "discovered" family that blows up immediately.
    def family(r):
        return lambda x: 1e12 * (x + r)

    result = bifurcation_location_error(
        true_bifurcation_param=3.0,
        discovered_system_family=family,
        param_range=(2.5, 3.5),
        n_param_steps=20,
    )
    assert result["location"] is None
    assert result["error"] is None
    assert "diverge" in result["reason"] or "non-finite" in result["reason"]
