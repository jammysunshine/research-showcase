"""Naive polynomial least-squares baseline ("no prior" comparator), per
PREREGISTRATION.md section 7: "Naive polynomial least-squares fit without
sparsity (no thresholding) as a 'no prior' baseline showing the sparsity
assumption's contribution."

Builds the IDENTICAL polynomial library the SINDy/STLSQ comparator uses
(`ps.PolynomialLibrary(degree=degree)` -- see
experiments/pilot_chaos_vs_periodic.py's `fit_sindy_lorenz`/`fit_sindy_logistic`
and src/discovery_symbolic_regression.py), reusing pysindy's own library
class rather than duplicating library-construction logic. The only
difference from the SINDy comparator is the fit itself: ordinary
(non-sparse) least squares, no STLSQ / no thresholding. Every library term
therefore gets a nonzero coefficient -- this is the point of the baseline
(it shows what the sparsity assumption buys the SINDy comparator, not a
method intended to compete with it on recovery).

Coefficient output format matches `ps.SINDy().coefficients()` /
`ps.SINDy().get_feature_names()` directly, so results are drop-in
comparable to the existing SINDy-fitting code.
"""
from __future__ import annotations

import numpy as np
import pysindy as ps


def fit_naive_polynomial(
    X: np.ndarray,
    X_dot: np.ndarray | None = None,
    degree: int = 3,
    dt: float | None = None,
    discrete: bool = False,
    feature_names: list[str] | None = None,
    zero_tol: float = 1e-8,
) -> dict:
    """Ordinary (dense, non-sparse) least-squares polynomial regression on
    the same PolynomialLibrary(degree=degree) SINDy/STLSQ uses.

    Two modes, matching how the ODE and discrete-map cases are already
    regressed elsewhere in this repo:

    - ODE mode (discrete=False, default): `X` is (n_samples, n_dims)
      states. `X_dot` is the (n_samples, n_dims) derivative target; if
      omitted, it is estimated with pysindy's `FiniteDifference` (`dt`
      required in that case) -- the same convention as
      `discovery_symbolic_regression.estimate_derivatives` and pysindy's
      own default.
    - Discrete-map mode (discrete=True): `X` is a 1D trajectory array of
      length n+1 (e.g. the logistic map's iterate sequence). The library is
      built on x_n and the target is x_{n+1}, matching
      experiments/pilot_chaos_vs_periodic.py's `fit_sindy_logistic` design
      matrix `[1, x, x^2, ..., x^degree]` (PolynomialLibrary on a single
      feature reproduces that same matrix -- verified in tests).

    Returns a dict:
      - "coefficients": (n_targets, n_features) dense array, directly
        comparable to `ps.SINDy().coefficients()`.
      - "feature_names": list[str], same convention/order as
        `ps.SINDy().get_feature_names()`.
      - "n_nonzero": count of coefficients with |c| > zero_tol -- use this
        to compare sparsity against a SINDy/STLSQ fit on the same data.
      - "degree": the library degree used.
    """
    library = ps.PolynomialLibrary(degree=degree)

    if discrete:
        x = np.asarray(X, dtype=float).reshape(-1)
        x_n = x[:-1].reshape(-1, 1)
        target = x[1:].reshape(-1, 1)
        Theta = library.fit_transform(x_n)
        names = library.get_feature_names(feature_names or ["x"])
    else:
        states = np.asarray(X, dtype=float)
        if X_dot is None:
            if dt is None:
                raise ValueError(
                    "dt is required to estimate derivatives when X_dot is not supplied"
                )
            fd = ps.FiniteDifference()
            target = fd(states, t=dt)
        else:
            target = np.asarray(X_dot, dtype=float)
        Theta = library.fit_transform(states)
        n_dims = states.shape[1]
        default_names = [f"x{i}" for i in range(n_dims)]
        names = library.get_feature_names(feature_names or default_names)

    n_targets = target.shape[1]
    n_features = Theta.shape[1]
    coeffs = np.zeros((n_targets, n_features))
    for dim in range(n_targets):
        c, *_ = np.linalg.lstsq(Theta, target[:, dim], rcond=None)
        coeffs[dim] = c

    n_nonzero = int(np.sum(np.abs(coeffs) > zero_tol))

    return dict(
        coefficients=coeffs,
        feature_names=names,
        n_nonzero=n_nonzero,
        degree=degree,
    )
