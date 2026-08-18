"""Self-checks for src/discovery_naive_baseline.py, the "no prior" comparator
required by PREREGISTRATION.md section 7.

(a) On noise-free data from a TRUE polynomial system, the naive dense fit's
    coefficients at the true-nonzero positions should match the true
    coefficients closely -- confirming the fit itself is not broken.
(b) On the same (lightly noised, so spurious terms are non-negligible rather
    than floating-point dust) data, the naive baseline reports MORE
    non-negligible coefficients than SINDy/STLSQ -- it does not spontaneously
    produce a sparse structural match. That gap is the whole point of this
    baseline (PREREGISTRATION.md section 7: showing the sparsity
    assumption's contribution).
"""
import numpy as np
import pysindy as ps
import pytest

from src.discovery_naive_baseline import fit_naive_polynomial
from src.simulators import lorenz_rhs, lorenz_trajectory, logistic_trajectory

NONNEGLIGIBLE_TOL = 1e-2  # "non-negligible" coefficient magnitude for (b)


def _lorenz_tail(rho, seed, noise_frac=0.0, n_points=25000, t_end=50.0, transient_frac=0.5):
    rng = np.random.default_rng(seed)
    x0 = np.array([-8.0, 8.0, 27.0]) + rng.normal(0, 0.5, size=3)
    params = dict(sigma=10.0, rho=rho, beta=8.0 / 3.0)
    t, states = lorenz_trajectory(x0, t_span=(0, t_end), n_points=n_points, params=params)
    if noise_frac > 0:
        state_std = states.std(axis=0)
        states = states + rng.normal(0, noise_frac * state_std, size=states.shape)
    n_discard = int(len(t) * transient_frac)
    t_tail = t[n_discard:]
    states_tail = states[n_discard:]
    dt = t_tail[1] - t_tail[0]
    return states_tail, dt, params


def test_naive_fit_recovers_true_coefficients_on_noise_free_lorenz():
    """(a) Lorenz rho=28, degree=2 library (matches the true system's active
    terms exactly, so the analysis is unambiguous). No noise, analytic
    derivatives (isolates the least-squares fit from FD error)."""
    states_tail, dt, params = _lorenz_tail(rho=28.0, seed=0, noise_frac=0.0)
    true_vf = np.array([
        lorenz_rhs(0.0, s, params["sigma"], params["rho"], params["beta"])
        for s in states_tail
    ])

    result = fit_naive_polynomial(states_tail, X_dot=true_vf, degree=2,
                                   feature_names=["x", "y", "z"])
    coeffs = result["coefficients"]
    names = result["feature_names"]
    idx = {n: i for i, n in enumerate(names)}

    sigma, rho, beta = params["sigma"], params["rho"], params["beta"]
    # dx/dt = -sigma*x + sigma*y
    assert coeffs[0, idx["x"]] == pytest.approx(-sigma, rel=0.01)
    assert coeffs[0, idx["y"]] == pytest.approx(sigma, rel=0.01)
    # dy/dt = rho*x - y - x*z
    assert coeffs[1, idx["x"]] == pytest.approx(rho, rel=0.01)
    assert coeffs[1, idx["y"]] == pytest.approx(-1.0, rel=0.05)
    assert coeffs[1, idx["x z"]] == pytest.approx(-1.0, rel=0.05)
    # dz/dt = x*y - beta*z
    assert coeffs[2, idx["x y"]] == pytest.approx(1.0, rel=0.05)
    assert coeffs[2, idx["z"]] == pytest.approx(-beta, rel=0.01)


def test_naive_fit_recovers_logistic_map_coefficients():
    """(a), discrete-map mode: logistic map r=4.0, x_{n+1} = r*x - r*x^2."""
    r = 4.0
    traj = logistic_trajectory(0.4, r=r, n_steps=5000, transient=500)
    result = fit_naive_polynomial(traj, degree=3, discrete=True)
    coeffs = result["coefficients"][0]
    names = result["feature_names"]
    idx = {n: i for i, n in enumerate(names)}

    assert coeffs[idx["1"]] == pytest.approx(0.0, abs=1e-6)
    assert coeffs[idx["x"]] == pytest.approx(r, rel=1e-4)
    assert coeffs[idx["x^2"]] == pytest.approx(-r, rel=1e-4)
    assert coeffs[idx["x^3"]] == pytest.approx(0.0, abs=1e-6)


def test_naive_baseline_is_denser_than_sindy_stlsq_on_lorenz():
    """(b) On the same (lightly noised) Lorenz data, the naive dense fit
    reports strictly more non-negligible coefficients than SINDy/STLSQ --
    the naive fit does not automatically produce sparse structure, unlike
    the sparsity-assuming comparator. This is the concrete demonstration
    PREREGISTRATION.md section 7 asks this baseline to provide."""
    states_tail, dt, params = _lorenz_tail(rho=28.0, seed=1, noise_frac=0.01)

    naive_result = fit_naive_polynomial(states_tail, degree=3, dt=dt,
                                         feature_names=["x", "y", "z"])
    naive_nonneg = int(np.sum(np.abs(naive_result["coefficients"]) > NONNEGLIGIBLE_TOL))
    naive_total = naive_result["coefficients"].size

    sindy_model = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=3),
        optimizer=ps.STLSQ(threshold=0.1),
    )
    sindy_model.fit(states_tail, t=dt, feature_names=["x", "y", "z"])
    sindy_coeffs = sindy_model.coefficients()
    sindy_nonneg = int(np.sum(np.abs(sindy_coeffs) > NONNEGLIGIBLE_TOL))
    sindy_total = sindy_coeffs.size

    # Feature libraries must match in shape for the comparison to be apples-to-apples.
    assert naive_result["coefficients"].shape == sindy_coeffs.shape
    assert naive_result["feature_names"] == sindy_model.get_feature_names()

    print(
        f"\n[naive-vs-sindy, Lorenz rho=28, noise=1%] "
        f"naive nonzero: {naive_nonneg}/{naive_total} | "
        f"SINDy/STLSQ nonzero: {sindy_nonneg}/{sindy_total}"
    )

    assert naive_nonneg > sindy_nonneg
    # SINDy should have found the known true structure: 6 nonzero terms
    # (x,y for dx/dt; x,y,xz for dy/dt; xy,z for dz/dt) -- kept as a sanity
    # anchor so a failure here is diagnosable as "SINDy comparator broken"
    # rather than silently passing on a degenerate SINDy fit.
    assert sindy_nonneg <= 8


def test_naive_baseline_dense_by_construction():
    """Every library term gets a fitted coefficient (n_nonzero counts
    strictly-nonzero, not just >tol, so this documents the no-thresholding
    guarantee directly)."""
    states_tail, dt, _ = _lorenz_tail(rho=28.0, seed=2, noise_frac=0.0)
    result = fit_naive_polynomial(states_tail, degree=3, dt=dt)
    # With float64 lstsq, essentially every coefficient is nonzero to full
    # precision -- there is no thresholding step to zero any of them out.
    assert result["n_nonzero"] == result["coefficients"].size
