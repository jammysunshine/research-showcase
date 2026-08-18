"""Correctness tests for the trusted simulator (src/simulators.py).

These are the feasibility-test / trusted-comparator checks required by
PROMPT.md before any equation-discovery work: known Lyapunov exponents,
known bifurcation structure, and agreement between two independent
Lyapunov-exponent estimation methods for Lorenz.
"""
import numpy as np
import pytest

from src.simulators import (
    DEFAULT_DUFFING_FORCED_PARAMS,
    DEFAULT_DUFFING_UNFORCED_PARAMS,
    DEFAULT_LORENZ_PARAMS,
    DEFAULT_ROESSLER_PARAMS,
    duffing_forced_lyapunov_spectrum,
    duffing_forced_trajectory,
    duffing_unforced_trajectory,
    harmonic_trajectory,
    logistic_lyapunov_exponent,
    logistic_trajectory,
    lorenz96_jacobian,
    lorenz96_lyapunov_spectrum,
    lorenz96_rhs,
    lorenz96_trajectory,
    lorenz_largest_lyapunov_two_trajectory,
    lorenz_lyapunov_spectrum,
    lorenz_trajectory,
    rossler_lyapunov_spectrum,
    rossler_trajectory,
)


def test_logistic_map_period_doubling_regime():
    # r=3.2 is period-2 (textbook), Lyapunov exponent must be negative.
    traj = logistic_trajectory(0.4, r=3.2, n_steps=200, transient=2000)
    period2_vals = np.round(traj[-4:], 3)
    assert len(np.unique(period2_vals)) <= 2
    lam = logistic_lyapunov_exponent(0.4, r=3.2, n_steps=5000)
    assert lam < 0


def test_logistic_map_chaotic_regime_positive_lyapunov():
    # r=4.0 is fully chaotic (exact map to tent map), analytic exponent = log(2).
    lam = logistic_lyapunov_exponent(0.4, r=4.0, n_steps=200000, transient=1000)
    assert lam == pytest.approx(np.log(2), abs=0.01)


def test_lorenz_trajectory_stays_bounded_on_attractor():
    t, states = lorenz_trajectory(np.array([1.0, 1.0, 1.0]), t_span=(0, 50), n_points=5000)
    # Classic Lorenz attractor for rho=28 stays within a known bounded region.
    assert np.all(np.abs(states) < 60)
    assert t[0] == 0 and t[-1] == 50


def test_lorenz_lyapunov_spectrum_matches_textbook_values():
    spectrum = lorenz_lyapunov_spectrum(
        np.array([1.0, 1.0, 1.0]), dt=0.01, n_steps=3000, transient_steps=1000
    )
    spectrum = np.sort(spectrum)[::-1]
    # Textbook values (sigma=10, rho=28, beta=8/3): approx (0.906, 0, -14.57)
    assert spectrum[0] == pytest.approx(0.906, abs=0.15)
    assert spectrum[1] == pytest.approx(0.0, abs=0.15)
    assert spectrum[2] == pytest.approx(-14.57, abs=1.5)
    # Sum of Lyapunov exponents for a dissipative flow equals -(sigma+1+beta).
    expected_sum = -(DEFAULT_LORENZ_PARAMS["sigma"] + 1 + DEFAULT_LORENZ_PARAMS["beta"])
    assert np.sum(spectrum) == pytest.approx(expected_sum, abs=1.5)


def test_lorenz_largest_lyapunov_cross_check_agrees_with_spectrum_method():
    spectrum = lorenz_lyapunov_spectrum(
        np.array([1.0, 1.0, 1.0]), dt=0.01, n_steps=2000, transient_steps=1000
    )
    lambda1_qr = np.max(spectrum)

    lambda1_direct = lorenz_largest_lyapunov_two_trajectory(
        np.array([1.0, 1.0, 1.0]), dt=0.01, n_steps=2000, transient_steps=1000
    )
    # Two independent methods (Jacobian-based QR vs. direct two-trajectory
    # divergence) should agree within a generous tolerance.
    assert lambda1_direct == pytest.approx(lambda1_qr, abs=0.2)


# ---------------------------------------------------------------------------
# Roessler (MAIN_STUDY_DESIGN.md SS3 held-out family). This is a
# literature-comparison numeric sanity check against known Lyapunov-exponent
# values, not an inspection of discovery-relevant behavior - it does not
# violate PREREGISTRATION.md SS11's blinding (see MAIN_STUDY_DESIGN.md SS4
# step 1 / SS8's explicit note on this point).
# ---------------------------------------------------------------------------

def test_rossler_trajectory_stays_bounded_on_attractor():
    t, states = rossler_trajectory(np.array([1.0, 1.0, 1.0]), t_span=(0, 200), n_points=5000)
    assert np.all(np.abs(states) < 50)
    assert t[0] == 0 and t[-1] == 200


def test_rossler_chaotic_regime_positive_largest_lyapunov():
    # Classic Roessler chaotic parameters (a=b=0.2, c=5.7): literature largest
    # exponent is approximately 0.07 (e.g. Sprott's tables); generous tolerance
    # since this project's estimator/integration settings differ from any
    # single reference implementation.
    spectrum = rossler_lyapunov_spectrum(
        np.array([1.0, 1.0, 1.0]), dt=0.01, n_steps=8000, transient_steps=3000
    )
    assert np.max(spectrum) > 0.02


def test_rossler_reduced_c_control_is_non_chaotic():
    # c=3.0 (a=b=0.2 unchanged) sits in the pre-chaotic period-doubling regime:
    # a simple limit cycle, largest Lyapunov exponent must be <= 0.
    params = dict(a=0.2, b=0.2, c=3.0)
    spectrum = rossler_lyapunov_spectrum(
        np.array([1.0, 1.0, 1.0]), params=params, dt=0.01, n_steps=8000, transient_steps=3000
    )
    assert np.max(spectrum) < 0.01


# ---------------------------------------------------------------------------
# Harmonic oscillator and Duffing (MAIN_STUDY_DESIGN.md Tier A/B regimes).
# ---------------------------------------------------------------------------

def test_harmonic_oscillator_conserves_energy():
    t, states = harmonic_trajectory(np.array([1.0, 0.0]), t_span=(0, 50), n_points=2000)
    energy = 0.5 * (states[:, 0] ** 2 + states[:, 1] ** 2)
    assert (energy.max() - energy.min()) < 1e-6


def test_duffing_unforced_conserves_energy():
    alpha = DEFAULT_DUFFING_UNFORCED_PARAMS["alpha"]
    beta = DEFAULT_DUFFING_UNFORCED_PARAMS["beta"]
    t, states = duffing_unforced_trajectory(np.array([0.5, 0.0]), t_span=(0, 50), n_points=2000)
    energy = (0.5 * states[:, 1] ** 2 + 0.5 * alpha * states[:, 0] ** 2
              + 0.25 * beta * states[:, 0] ** 4)
    assert (energy.max() - energy.min()) < 1e-6


def test_duffing_forced_chaotic_regime_positive_largest_lyapunov():
    spectrum = duffing_forced_lyapunov_spectrum(
        np.array([0.5, 0.0, 0.0]), dt=0.01, n_steps=8000, transient_steps=3000
    )
    assert np.max(spectrum) > 0.02


def test_duffing_forced_trajectory_stays_bounded():
    t, states = duffing_forced_trajectory(np.array([0.5, 0.0, 0.0]), t_span=(0, 200), n_points=5000)
    assert t[0] == 0 and t[-1] == 200
    assert np.all(np.abs(states[:, :2]) < 20)


# ---------------------------------------------------------------------------
# Lorenz-96 (EXTENSION_PLAN.md extension #4, higher-dimensional matched pair).
# ---------------------------------------------------------------------------

def test_lorenz96_jacobian_matches_finite_difference():
    rng = np.random.default_rng(0)
    state = rng.normal(size=6)
    F = 8.0
    J_analytic = lorenz96_jacobian(state, F)
    eps = 1e-6
    J_fd = np.zeros((6, 6))
    for j in range(6):
        dstate = state.copy()
        dstate[j] += eps
        J_fd[:, j] = (lorenz96_rhs(0, dstate, F) - lorenz96_rhs(0, state, F)) / eps
    assert np.max(np.abs(J_analytic - J_fd)) < 1e-6


def test_lorenz96_trajectory_stays_bounded():
    x0 = np.array([8.0, 0.0, 0.0, 0.0, 0.0, 0.0]) + 0.01
    t, states = lorenz96_trajectory(x0, t_span=(0, 50), n_points=2000, params=dict(F=8.0))
    assert np.all(np.abs(states) < 50)
    assert t[0] == 0 and t[-1] == 50


def test_lorenz96_F8_chaotic_regime_positive_largest_lyapunov():
    # F=8, N=6: standard Lorenz-96 forcing known to produce chaos at N=40;
    # verified numerically for this project's N=6 rather than asserted from
    # literature (small-N Lorenz-96 chaos onset is N-dependent).
    x0 = np.array([8.0, 0.0, 0.0, 0.0, 0.0, 0.0]) + 0.01
    spectrum = lorenz96_lyapunov_spectrum(
        x0, params=dict(F=8.0), dt=0.01, n_steps=3000, transient_steps=1000
    )
    assert np.max(spectrum) > 0.3


def test_lorenz96_F1_control_is_non_chaotic():
    # F=1, N=6: numerically verified stable fixed point (entire spectrum
    # negative), the matched non-chaotic control for the F=8 chaotic regime.
    x0 = np.array([8.0, 0.0, 0.0, 0.0, 0.0, 0.0]) + 0.01
    spectrum = lorenz96_lyapunov_spectrum(
        x0, params=dict(F=1.0), dt=0.01, n_steps=3000, transient_steps=1000
    )
    assert np.max(spectrum) < 0.0
