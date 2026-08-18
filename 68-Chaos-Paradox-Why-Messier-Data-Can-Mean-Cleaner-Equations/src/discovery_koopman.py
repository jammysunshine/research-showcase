"""Koopman / Extended Dynamic Mode Decomposition (EDMD) discovery method.

Third of the three required equation-discovery families in PROMPT.md
requirement 2 (sparse regression/SINDy, symbolic regression, neural/Koopman).

`pykoopman` failed to install cleanly in this environment: pip could not
build its sdist because Python 3.14's stdlib has no `setuptools.build_meta`
available to the isolated build backend (BackendUnavailable). This module is
therefore a self-implemented EDMD fallback, documented as such per the task
instructions.

EDMD approximates the (infinite-dimensional, linear) Koopman operator by:
  1. choosing a finite dictionary of observables Phi(x) = [phi_1(x), ..., phi_k(x)]
     (here: monomials up to a chosen degree in the state; `MonomialDictionary`
     and `fit_edmd`/`fit` accept `degree` as a parameter (default 2) and are
     degree-generic, so degree=3 -- matching PREREGISTRATION.md section 2's
     frozen degree-3 SINDy library -- is fully supported, not just degree=2.
     scripts/run_koopman_smoke_test.py runs and reports both degrees);
  2. evaluating the dictionary on snapshot pairs (x_t, x_{t+1}) drawn from a
     trajectory;
  3. solving the linear least-squares problem
         Phi(X_{t+1}) approx K^T Phi(X_t)
     for the finite-dimensional Koopman matrix K (a k x k matrix acting on
     dictionary space).

Unlike SINDy, EDMD does not directly hand back symbolic coefficients of a
vector field: K is a linear operator on observables, not on the state
directly. What IS meaningful for identifiability comparison:
  - The one-step predictor x_{t+1} approx g(K^T Phi(x_t)) (recovering the raw
    state observables from dictionary space) can be evaluated for one-step
    and short-horizon trajectory error against a held-out trajectory, exactly
    the same off-trajectory generalization check used for SINDy/symbolic
    regression.
  - The eigenvalues of K approximate the Koopman spectrum, which for a
    continuous-time flow relates to exp(lambda_i * dt) for the operator's
    continuous-time generator; comparing this discrete spectrum to the
    known local linearization (e.g. Lorenz Jacobian eigenvalues) at a
    reference point is a spectral consistency check.
  - K restricted to the linear-in-x block of the dictionary approximates a
    locally-linearized vector field, which can be compared to the true
    Jacobian at a reference point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np


# ---------------------------------------------------------------------------
# Dictionary of observables: monomials up to a given degree.
# ---------------------------------------------------------------------------

def _monomial_exponent_tuples(n_vars: int, degree: int):
    """All exponent tuples (e_1, ..., e_n) with 0 <= sum(e_i) <= degree,
    ordered by total degree then lexicographically. Includes the degree-0
    (constant) term first.
    """
    exps = [tuple([0] * n_vars)]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_vars), d):
            e = [0] * n_vars
            for idx in combo:
                e[idx] += 1
            exps.append(tuple(e))
    return exps


def _monomial_name(exponents: tuple, var_names: list[str]) -> str:
    if all(e == 0 for e in exponents):
        return "1"
    parts = []
    for e, name in zip(exponents, var_names):
        if e == 0:
            continue
        parts.append(name if e == 1 else f"{name}^{e}")
    return " ".join(parts)


class MonomialDictionary:
    """Evaluates a dictionary of monomial observables Phi(x) up to `degree`."""

    def __init__(self, n_vars: int, degree: int = 2, var_names: list[str] | None = None):
        self.n_vars = n_vars
        self.degree = degree
        self.var_names = var_names or [f"x{i}" for i in range(n_vars)]
        self.exponents = _monomial_exponent_tuples(n_vars, degree)
        self.n_features = len(self.exponents)
        self.names = [_monomial_name(e, self.var_names) for e in self.exponents]
        # Index of each raw state variable within the dictionary (its
        # degree-1 monomial), needed to recover x from Phi(x).
        self.state_indices = []
        for i in range(n_vars):
            e = tuple(1 if j == i else 0 for j in range(n_vars))
            self.state_indices.append(self.exponents.index(e))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (n_samples, n_vars) -> Phi: (n_samples, n_features)."""
        X = np.atleast_2d(X)
        n_samples = X.shape[0]
        Phi = np.empty((n_samples, self.n_features))
        for j, exps in enumerate(self.exponents):
            col = np.ones(n_samples)
            for var_idx, e in enumerate(exps):
                if e:
                    col = col * X[:, var_idx] ** e
            Phi[:, j] = col
        return Phi


# ---------------------------------------------------------------------------
# EDMD fit
# ---------------------------------------------------------------------------

@dataclass
class EDMDModel:
    """Fitted EDMD (finite-dimensional Koopman) approximation."""

    dictionary: MonomialDictionary
    K: np.ndarray                 # (k, k) Koopman matrix: Phi(x_{t+1}) ~ K^T Phi(x_t)
    dt: float
    residual_rms: float
    n_snapshot_pairs: int
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))

    def predict_dictionary(self, x0: np.ndarray) -> np.ndarray:
        """One-step prediction in dictionary space: Phi(x1) approx K^T Phi(x0)."""
        phi0 = self.dictionary.transform(np.atleast_2d(x0))
        return phi0 @ self.K

    def predict_state(self, x0: np.ndarray) -> np.ndarray:
        """One-step state predictor: apply K in dictionary space, then read the
        raw-state coordinates back out of the propagated dictionary vector.
        """
        phi1 = self.predict_dictionary(x0)
        idx = self.dictionary.state_indices
        return phi1[:, idx][0] if np.atleast_2d(x0).shape[0] == 1 else phi1[:, idx]

    def simulate(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        """Roll out n_steps one-step predictions from x0 (in dictionary space,
        re-lifting the predicted state at each step -- the standard EDMD
        rollout when the dictionary is not exactly Koopman-invariant).
        """
        traj = np.empty((n_steps + 1, len(x0)))
        traj[0] = x0
        x = np.array(x0, dtype=float)
        for i in range(n_steps):
            x = self.predict_state(x)
            traj[i + 1] = x
        return traj

    def linearization(self) -> np.ndarray:
        """Approximate local Jacobian of the one-step map at the origin of
        dictionary space, restricted to the linear-in-state block: the
        (n_vars x n_vars) sub-matrix of K^T mapping degree-1 observables to
        degree-1 observables. This approximates exp(J_true * dt) for small
        dt near a reference point where the monomial dictionary's nonlinear
        cross-terms are locally small -- a rough spectral-consistency probe,
        not an exact linearization.
        """
        idx = self.dictionary.state_indices
        return self.K.T[np.ix_(idx, idx)]


def fit_edmd(
    states: np.ndarray,
    dt: float,
    degree: int = 2,
    var_names: list[str] | None = None,
    ridge_alpha: float = 0.0,
) -> EDMDModel:
    """Fit an EDMD / finite-dimensional Koopman approximation from a single
    trajectory of states sampled at fixed interval dt.

    states: (n_samples, n_vars) array, assumed evenly spaced in time.
    degree: max total degree of the monomial dictionary (default 2; pass
        degree=3 to match PREREGISTRATION.md section 2's frozen degree-3
        SINDy library -- see scripts/run_koopman_smoke_test.py for a
        degree-2 vs degree-3 side-by-side comparison).
    ridge_alpha: Tikhonov/ridge regularization strength (default 0.0 =
        unregularized ordinary least squares).  When > 0, solves
        min ||Phi0 @ K - Phi1||^2 + alpha * ||K||_F^2 via the normal
        equations K = (Phi0^T Phi0 + alpha I)^{-1} Phi0^T Phi1.

    Solves Phi(X_{t+1}) approx Phi(X_t) @ K via ordinary least squares
    (equivalent to the standard EDMD normal-equations solution
    K = pinv(Phi(X_t)) @ Phi(X_{t+1}), computed here via lstsq for
    numerical stability).
    """
    states = np.asarray(states, dtype=float)
    n_vars = states.shape[1]
    dictionary = MonomialDictionary(n_vars, degree=degree, var_names=var_names)

    X0 = states[:-1]
    X1 = states[1:]
    Phi0 = dictionary.transform(X0)
    Phi1 = dictionary.transform(X1)

    if ridge_alpha > 0:
        G = Phi0.T @ Phi0
        n_features = G.shape[0]
        K = np.linalg.solve(G + ridge_alpha * np.eye(n_features), Phi0.T @ Phi1)
    else:
        K, _, _, _ = np.linalg.lstsq(Phi0, Phi1, rcond=None)

    resid = Phi1 - Phi0 @ K
    residual_rms = float(np.sqrt(np.mean(resid ** 2)))

    eigenvalues = np.linalg.eigvals(K)

    return EDMDModel(
        dictionary=dictionary,
        K=K,
        dt=dt,
        residual_rms=residual_rms,
        n_snapshot_pairs=X0.shape[0],
        eigenvalues=eigenvalues,
    )


def fit(states: np.ndarray, dt: float, degree: int = 2, var_names: list[str] | None = None) -> EDMDModel:
    """Alias matching the task's requested `fit(...)` entry point."""
    return fit_edmd(states, dt=dt, degree=degree, var_names=var_names)
