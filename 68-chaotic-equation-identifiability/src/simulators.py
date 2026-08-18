"""Truth-known dynamical-system generators with analytic derivatives.

Systems implemented:
  - Logistic map: x_{n+1} = r * x_n * (1 - x_n)
  - Lorenz system: dx/dt = sigma*(y-x), dy/dt = x*(rho-z)-y, dz/dt = x*y - beta*z
  - Coupled van der Pol oscillators (4D, periodic/periodic/chaotic regimes)

Each system exposes:
  - the right-hand side (vector field / map)
  - an analytic Jacobian
  - a trajectory generator (map iteration or high-precision ODE integration)
  - an independent Lyapunov-exponent estimator (Benettin's method via the
    analytic Jacobian, cross-checked against the direct two-trajectory
    divergence method for the largest exponent)

These are the trusted comparators referenced in PROJECT_CHARTER.md and
DECISION_LOG.md: correctness here is a precondition for every later
equation-discovery experiment.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Logistic map
# ---------------------------------------------------------------------------

def logistic_map(x: float, r: float) -> float:
    return r * x * (1.0 - x)


def logistic_map_jacobian(x: float, r: float) -> float:
    return r * (1.0 - 2.0 * x)


def logistic_trajectory(x0: float, r: float, n_steps: int, transient: int = 0) -> np.ndarray:
    """Iterate the logistic map, discarding an optional transient."""
    n_total = n_steps + transient
    xs = np.empty(n_total + 1)
    xs[0] = x0
    for i in range(n_total):
        xs[i + 1] = logistic_map(xs[i], r)
    return xs[transient:]


def generic_map_lyapunov_exponent(
    map_fn: Callable[[float], float],
    jacobian_fn: Callable[[float], float],
    x0: float,
    n_steps: int,
    transient: int = 1000,
) -> float:
    """Benettin-style Lyapunov exponent for an arbitrary 1D map + derivative.

    lambda = lim (1/N) sum_i log |f'(x_i)|. Underlies logistic_lyapunov_exponent
    below; kept generic so other 1D maps (e.g. a discovered model) can reuse it.
    """
    x = x0
    for _ in range(transient):
        x = map_fn(x)
    log_sum = 0.0
    for _ in range(n_steps):
        deriv = abs(jacobian_fn(x))
        # Guard against log(0) at fixed points/superstable cycles.
        log_sum += np.log(max(deriv, 1e-300))
        x = map_fn(x)
    return log_sum / n_steps


def logistic_lyapunov_exponent(x0: float, r: float, n_steps: int, transient: int = 1000) -> float:
    return generic_map_lyapunov_exponent(
        lambda x: logistic_map(x, r),
        lambda x: logistic_map_jacobian(x, r),
        x0, n_steps, transient,
    )


# ---------------------------------------------------------------------------
# Lorenz system
# ---------------------------------------------------------------------------

DEFAULT_LORENZ_PARAMS = dict(sigma=10.0, rho=28.0, beta=8.0 / 3.0)


def lorenz_rhs(t: float, state: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ])


def lorenz_jacobian(state: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    x, y, z = state
    return np.array([
        [-sigma, sigma, 0.0],
        [rho - z, -1.0, -x],
        [y, x, -beta],
    ])


def lorenz_trajectory(
    x0: np.ndarray,
    t_span: tuple[float, float],
    n_points: int,
    params: dict = None,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    method: str = "DOP853",
):
    """High-precision Lorenz integration. Returns (t, states)."""
    params = params or DEFAULT_LORENZ_PARAMS
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        lorenz_rhs,
        t_span,
        x0,
        args=(params["sigma"], params["rho"], params["beta"]),
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Lorenz integration failed: {sol.message}")
    return sol.t, sol.y.T


def generic_ode_lyapunov_spectrum(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    dim: int,
    dt: float = 0.01,
    n_steps: int = 20000,
    transient_steps: int = 2000,
) -> np.ndarray:
    """Full Lyapunov spectrum via the standard tangent-linear + QR (Benettin) method,
    for an arbitrary autonomous ODE (rhs(t, state), jacobian(state)) of any dimension.

    Integrates the state together with an orthonormal frame of tangent vectors,
    using the supplied Jacobian, and periodically re-orthonormalizes (QR) to
    extract the exponents from the accumulated log of the diagonal stretching
    factors. lorenz_lyapunov_spectrum below is a thin wrapper around this.
    """

    def full_rhs(t, y):
        state = y[:dim]
        Q = y[dim:].reshape(dim, dim)
        dstate = rhs(t, state)
        J = jacobian(state)
        dQ = J @ Q
        return np.concatenate([dstate, dQ.flatten()])

    state = np.array(x0, dtype=float)
    Q = np.eye(dim)

    # Transient to reach the attractor.
    for _ in range(transient_steps):
        y0 = np.concatenate([state, Q.flatten()])
        sol = solve_ivp(full_rhs, (0, dt), y0, method="DOP853", rtol=1e-10, atol=1e-11)
        state = sol.y[:dim, -1]
        Q = sol.y[dim:, -1].reshape(dim, dim)
        Q, _ = np.linalg.qr(Q)

    log_sums = np.zeros(dim)
    for _ in range(n_steps):
        y0 = np.concatenate([state, Q.flatten()])
        sol = solve_ivp(full_rhs, (0, dt), y0, method="DOP853", rtol=1e-10, atol=1e-11)
        state = sol.y[:dim, -1]
        Q_raw = sol.y[dim:, -1].reshape(dim, dim)
        Q, R = np.linalg.qr(Q_raw)
        # Fix sign ambiguity of QR so stretching factors are positive-diagonal.
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1.0
        R = R * signs[:, None]
        Q = Q * signs[None, :]
        log_sums += np.log(np.abs(np.diag(R)))

    return log_sums / (n_steps * dt)


def lorenz_lyapunov_spectrum(
    x0: np.ndarray,
    params: dict = None,
    dt: float = 0.01,
    n_steps: int = 20000,
    transient_steps: int = 2000,
) -> np.ndarray:
    params = params or DEFAULT_LORENZ_PARAMS
    sigma, rho, beta = params["sigma"], params["rho"], params["beta"]
    return generic_ode_lyapunov_spectrum(
        rhs=lambda t, state: lorenz_rhs(t, state, sigma, rho, beta),
        jacobian=lambda state: lorenz_jacobian(state, sigma, rho, beta),
        x0=x0, dim=3, dt=dt, n_steps=n_steps, transient_steps=transient_steps,
    )


def lorenz_largest_lyapunov_two_trajectory(
    x0: np.ndarray,
    params: dict = None,
    dt: float = 0.01,
    n_steps: int = 5000,
    transient_steps: int = 2000,
    perturbation: float = 1e-8,
) -> float:
    """Independent cross-check of the largest Lyapunov exponent via direct
    two-trajectory divergence (no Jacobian used) with periodic renormalization.
    """
    params = params or DEFAULT_LORENZ_PARAMS
    sigma, rho, beta = params["sigma"], params["rho"], params["beta"]

    def step(state, dt):
        sol = solve_ivp(
            lorenz_rhs, (0, dt), state, args=(sigma, rho, beta),
            method="DOP853", rtol=1e-10, atol=1e-11,
        )
        return sol.y[:, -1]

    state = np.array(x0, dtype=float)
    for _ in range(transient_steps):
        state = step(state, dt)

    ref = state.copy()
    pert = ref + np.array([perturbation, 0.0, 0.0])

    log_sum = 0.0
    for _ in range(n_steps):
        ref = step(ref, dt)
        pert = step(pert, dt)
        diff = pert - ref
        dist = np.linalg.norm(diff)
        log_sum += np.log(dist / perturbation)
        pert = ref + diff * (perturbation / dist)

    return log_sum / (n_steps * dt)


# ---------------------------------------------------------------------------
# Roessler system (held-out confirmation family, MAIN_STUDY_DESIGN.md SS3).
# Not fit, inspected, or discussed by any discovery method until the
# falsification/independent-replication gate (PREREGISTRATION.md SS11) -
# this simulator + the literature-comparison Lyapunov check below is the
# only code that may touch it before that gate.
# ---------------------------------------------------------------------------

DEFAULT_ROESSLER_PARAMS = dict(a=0.2, b=0.2, c=5.7)


def rossler_rhs(t: float, state: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    x, y, z = state
    return np.array([
        -y - z,
        x + a * y,
        b + z * (x - c),
    ])


def rossler_jacobian(state: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    x, y, z = state
    return np.array([
        [0.0, -1.0, -1.0],
        [1.0, a, 0.0],
        [z, 0.0, x - c],
    ])


def rossler_trajectory(
    x0: np.ndarray,
    t_span: tuple[float, float],
    n_points: int,
    params: dict = None,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    method: str = "DOP853",
):
    """High-precision Roessler integration. Returns (t, states)."""
    params = params or DEFAULT_ROESSLER_PARAMS
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        rossler_rhs,
        t_span,
        x0,
        args=(params["a"], params["b"], params["c"]),
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Roessler integration failed: {sol.message}")
    return sol.t, sol.y.T


def rossler_lyapunov_spectrum(
    x0: np.ndarray,
    params: dict = None,
    dt: float = 0.01,
    n_steps: int = 20000,
    transient_steps: int = 2000,
) -> np.ndarray:
    params = params or DEFAULT_ROESSLER_PARAMS
    a, b, c = params["a"], params["b"], params["c"]
    return generic_ode_lyapunov_spectrum(
        rhs=lambda t, state: rossler_rhs(t, state, a, b, c),
        jacobian=lambda state: rossler_jacobian(state, a, b, c),
        x0=x0, dim=3, dt=dt, n_steps=n_steps, transient_steps=transient_steps,
    )


# ---------------------------------------------------------------------------
# Harmonic oscillator (conservative control regime: energy is a first
# integral, so trajectories are periodic and the regression design matrix is
# rank-deficient off the single conserved-energy shell - MAIN_STUDY_DESIGN.md
# Tier A/B).
# ---------------------------------------------------------------------------

def harmonic_rhs(t: float, state: np.ndarray, omega: float = 1.0) -> np.ndarray:
    x, v = state
    return np.array([v, -(omega ** 2) * x])


def harmonic_jacobian(state: np.ndarray, omega: float = 1.0) -> np.ndarray:
    return np.array([
        [0.0, 1.0],
        [-(omega ** 2), 0.0],
    ])


def harmonic_trajectory(
    x0: np.ndarray,
    t_span: tuple[float, float],
    n_points: int,
    omega: float = 1.0,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    method: str = "DOP853",
):
    """High-precision undamped harmonic-oscillator integration. Returns (t, states)."""
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        harmonic_rhs, t_span, x0, args=(omega,),
        t_eval=t_eval, method=method, rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Harmonic oscillator integration failed: {sol.message}")
    return sol.t, sol.y.T


# ---------------------------------------------------------------------------
# Duffing oscillator: unforced (conservative, 2-state) control regime vs.
# forced-chaotic (3-state autonomous embedding, phi=omega*t carried as a
# state so the system has no explicit t-dependence).
# ---------------------------------------------------------------------------

DEFAULT_DUFFING_UNFORCED_PARAMS = dict(alpha=-1.0, beta=1.0)
DEFAULT_DUFFING_FORCED_PARAMS = dict(delta=0.3, alpha=-1.0, beta=1.0, gamma=0.5, omega=1.2)


def duffing_unforced_rhs(t: float, state: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Conservative (undamped, unforced) double-well Duffing oscillator."""
    x, v = state
    return np.array([v, -alpha * x - beta * x ** 3])


def duffing_unforced_jacobian(state: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    x, v = state
    return np.array([
        [0.0, 1.0],
        [-alpha - 3.0 * beta * x ** 2, 0.0],
    ])


def duffing_unforced_trajectory(
    x0: np.ndarray,
    t_span: tuple[float, float],
    n_points: int,
    params: dict = None,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    method: str = "DOP853",
):
    params = params or DEFAULT_DUFFING_UNFORCED_PARAMS
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        duffing_unforced_rhs, t_span, x0,
        args=(params["alpha"], params["beta"]),
        t_eval=t_eval, method=method, rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Unforced Duffing integration failed: {sol.message}")
    return sol.t, sol.y.T


def duffing_forced_rhs(
    t: float, state: np.ndarray, delta: float, alpha: float, beta: float,
    gamma: float, omega: float,
) -> np.ndarray:
    """Autonomous 3-state embedding of the periodically-forced Duffing
    oscillator: phi tracks omega*t as its own state (dphi/dt = omega) so the
    vector field has no explicit t-dependence, matching this project's other
    discovery-method inputs (state -> state-derivative, no external clock).
    """
    x, v, phi = state
    return np.array([
        v,
        -delta * v - alpha * x - beta * x ** 3 + gamma * np.cos(phi),
        omega,
    ])


def duffing_forced_jacobian(
    state: np.ndarray, delta: float, alpha: float, beta: float, gamma: float, omega: float,
) -> np.ndarray:
    x, v, phi = state
    return np.array([
        [0.0, 1.0, 0.0],
        [-alpha - 3.0 * beta * x ** 2, -delta, -gamma * np.sin(phi)],
        [0.0, 0.0, 0.0],
    ])


def duffing_forced_trajectory(
    x0: np.ndarray,
    t_span: tuple[float, float],
    n_points: int,
    params: dict = None,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    method: str = "DOP853",
):
    """x0 is (x, v, phi); phi0 is conventionally 0.0."""
    params = params or DEFAULT_DUFFING_FORCED_PARAMS
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        duffing_forced_rhs, t_span, x0,
        args=(params["delta"], params["alpha"], params["beta"], params["gamma"], params["omega"]),
        t_eval=t_eval, method=method, rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Forced Duffing integration failed: {sol.message}")
    return sol.t, sol.y.T


def duffing_forced_lyapunov_spectrum(
    x0: np.ndarray,
    params: dict = None,
    dt: float = 0.01,
    n_steps: int = 20000,
    transient_steps: int = 2000,
) -> np.ndarray:
    params = params or DEFAULT_DUFFING_FORCED_PARAMS
    delta, alpha, beta, gamma, omega = (
        params["delta"], params["alpha"], params["beta"], params["gamma"], params["omega"],
    )
    return generic_ode_lyapunov_spectrum(
        rhs=lambda t, state: duffing_forced_rhs(t, state, delta, alpha, beta, gamma, omega),
        jacobian=lambda state: duffing_forced_jacobian(state, delta, alpha, beta, gamma, omega),
        x0=x0, dim=3, dt=dt, n_steps=n_steps, transient_steps=transient_steps,
    )


# ---------------------------------------------------------------------------
# Lorenz-96 system (EXTENSION_PLAN.md extension #4, added 2026-08-17 as a
# higher-dimensional matched pair -- deliberately the same system family
# Gallo, Anselmi, Lazzari (arXiv:2607.18490) used as their own held-out
# zero-refit check, to invite direct comparison). N-dimensional, cyclic:
#   dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F,   indices mod N
# Quadratic (degree-2) right-hand side regardless of N, so this stress-tests
# dimensionality itself rather than confounding it with a higher-degree
# library requirement. N=6 chosen as the smallest N for which Lorenz-96 is
# known to support chaos (N<4 cannot; N=4-5 chaos is parameter-fragile) while
# staying inside this project's local-only compute ceiling (PROJECT_CHARTER.md)
# for a degree-2 SINDy/EDMD monomial dictionary (C(6+2,2)=28 features) and a
# 6-variable symbolic-regression search.
# ---------------------------------------------------------------------------

DEFAULT_LORENZ96_N = 6
DEFAULT_LORENZ96_PARAMS = dict(F=8.0)


def lorenz96_rhs(t: float, state: np.ndarray, F: float) -> np.ndarray:
    n = state.shape[0]
    return (np.roll(state, -1) - np.roll(state, 2)) * np.roll(state, 1) - state + F


def lorenz96_jacobian(state: np.ndarray, F: float) -> np.ndarray:
    n = state.shape[0]
    J = np.zeros((n, n))
    for i in range(n):
        ip1 = (i + 1) % n
        im1 = (i - 1) % n
        im2 = (i - 2) % n
        J[i, im1] += state[ip1] - state[im2]
        J[i, ip1] += state[im1]
        J[i, im2] += -state[im1]
        J[i, i] += -1.0
    return J


def lorenz96_trajectory(
    x0: np.ndarray,
    t_span: tuple[float, float],
    n_points: int,
    params: dict = None,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    method: str = "DOP853",
):
    """High-precision Lorenz-96 integration. Returns (t, states), states shape (n_points, N)."""
    params = params or DEFAULT_LORENZ96_PARAMS
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        lorenz96_rhs, t_span, x0, args=(params["F"],),
        t_eval=t_eval, method=method, rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Lorenz-96 integration failed: {sol.message}")
    return sol.t, sol.y.T


def lorenz96_lyapunov_spectrum(
    x0: np.ndarray,
    params: dict = None,
    dt: float = 0.01,
    n_steps: int = 20000,
    transient_steps: int = 2000,
) -> np.ndarray:
    params = params or DEFAULT_LORENZ96_PARAMS
    F = params["F"]
    n = x0.shape[0]
    return generic_ode_lyapunov_spectrum(
        rhs=lambda t, state: lorenz96_rhs(t, state, F),
        jacobian=lambda state: lorenz96_jacobian(state, F),
        x0=x0, dim=n, dt=dt, n_steps=n_steps, transient_steps=transient_steps,
    )


# ---------------------------------------------------------------------------
# Coupled van der Pol oscillators (4D)
# ---------------------------------------------------------------------------

# State vector: [x1, v1, x2, v2] where vi = dxi/dt
# dx1/dt = v1
# dv1/dt = mu1*(1 - x1^2)*v1 - x1 + k*(x2 - x1)
# dx2/dt = v2
# dv2/dt = mu2*(1 - x2^2)*v2 - omega2^2*x2 + k*(x1 - x2)
#
# Three regimes:
#   Periodic:     mu1=1.0, mu2=1.0, omega2=1.0,  k=0.05  (synchronized limit cycle)
#   Quasi-periodic: mu1=1.0, mu2=1.0, omega2=1.414, k=0.3  (incommensurate beat, 2-torus)
#   Chaotic:      mu1=2.0, mu2=2.5, omega2=1.5,  k=1.5  (torus breakdown)

COUPLED_VDP_REGIMES = {
    "periodic": dict(mu1=1.0, mu2=1.0, omega2=1.0, k=0.01),
    "quasi_periodic": dict(mu1=1.0, mu2=1.0, omega2=1.618, k=0.1),
    "chaotic": dict(mu1=8.0, mu2=8.0, omega2=2.0, k=5.0),
}


def coupled_vdp_rhs(t: float, state: np.ndarray, mu1: float, mu2: float,
                     omega2: float, k: float) -> np.ndarray:
    x1, v1, x2, v2 = state
    dx1 = v1
    dv1 = mu1 * (1.0 - x1**2) * v1 - x1 + k * (x2 - x1)
    dx2 = v2
    dv2 = mu2 * (1.0 - x2**2) * v2 - omega2**2 * x2 + k * (x1 - x2)
    return np.array([dx1, dv1, dx2, dv2])


def coupled_vdp_jacobian(state: np.ndarray, mu1: float, mu2: float,
                          omega2: float, k: float) -> np.ndarray:
    x1, v1, x2, v2 = state
    J = np.array([
        [0.0,                           1.0,  0.0,                           0.0],
        [-2.0*mu1*x1*v1 - 1.0 - k,     mu1*(1.0 - x1**2),  k,             0.0],
        [0.0,                           0.0,  0.0,                           1.0],
        [k,                             0.0,  -omega2**2 - k,  mu2*(1.0 - x2**2)],
    ])
    return J


def coupled_vdp_trajectory(
    x0: np.ndarray,
    t_span: tuple[float, float],
    n_points: int,
    params: dict = None,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    method: str = "DOP853",
):
    """Integrate coupled van der Pol oscillators. Returns (t, states), states shape (n_points, 4)."""
    params = params or COUPLED_VDP_REGIMES["periodic"]
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        coupled_vdp_rhs, t_span, x0,
        args=(params["mu1"], params["mu2"], params["omega2"], params["k"]),
        t_eval=t_eval, method=method, rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Coupled vdP integration failed: {sol.message}")
    return sol.t, sol.y.T


def coupled_vdp_lyapunov_spectrum(
    x0: np.ndarray,
    params: dict = None,
    dt: float = 0.05,
    n_steps: int = 30000,
    transient_steps: int = 5000,
) -> np.ndarray:
    params = params or COUPLED_VDP_REGIMES["periodic"]
    mu1, mu2, omega2, k = params["mu1"], params["mu2"], params["omega2"], params["k"]
    return generic_ode_lyapunov_spectrum(
        rhs=lambda t, state: coupled_vdp_rhs(t, state, mu1, mu2, omega2, k),
        jacobian=lambda state: coupled_vdp_jacobian(state, mu1, mu2, omega2, k),
        x0=x0, dim=4, dt=dt, n_steps=n_steps, transient_steps=transient_steps,
    )


