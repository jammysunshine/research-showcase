"""Symbolic-regression equation discovery (gplearn), the second of the three
required discovery families in PROMPT.md requirement 2 (sparse regression/
SINDy, symbolic regression, neural/Koopman).

Per PREREGISTRATION.md section 2, the symbolic-regression search space is
restricted to +, -, *, / and low-degree polynomial primitives (no trig, exp,
log, etc.), to keep search coverage comparable to the SINDy polynomial
library rather than an open-ended search over gplearn's full default
function set.

Derivative estimation uses pysindy's default `FiniteDifference` operator
(the same one `ps.SINDy().fit()` uses internally when no explicit
differentiation_method is passed, as in scripts/run_sindy_smoke_test.py),
so the two discovery families are compared on identical derivative
estimates rather than confounded by differing numerical-differentiation
schemes.
"""
from __future__ import annotations

import numpy as np
import pysindy as ps
from gplearn.genetic import SymbolicRegressor

# Matched to PREREGISTRATION.md section 2: +, -, *, / only (no trig/exp/log).
# gplearn has no built-in bounded "pdiv"-free divide; it ships a protected
# division ("div") that returns 1 where the denominator is near zero. We use
# that rather than a custom operator, so behavior stays within gplearn's
# documented, reviewable primitive set.
MATCHED_FUNCTION_SET = ("add", "sub", "mul", "div")


def estimate_derivatives(states: np.ndarray, dt: float) -> np.ndarray:
    """Estimate d(states)/dt using pysindy's default FiniteDifference method.

    This is the same derivative-estimation convention used implicitly by
    scripts/run_sindy_smoke_test.py (ps.SINDy().fit(states, t=dt) with no
    differentiation_method override defaults to ps.FiniteDifference()).
    Kept as a standalone, explicit call here so both discovery families are
    demonstrably using the same numerical derivatives.
    """
    fd = ps.FiniteDifference()
    return fd(states, t=dt)


def program_polynomial_degree(program) -> tuple[int | None, bool]:
    """Compute the true polynomial degree of a gplearn discovered program.

    `program` is a gplearn `_Program` object (e.g. `est._program`), whose
    `.program` attribute is a flat, depth-first prefix-notation list mixing
    `_Function` nodes (with `.name`/`.arity`) and terminal leaves (int =
    feature index, float = constant). We walk that tree directly rather than
    round-tripping through `sympy.parse_expr(str(program))`, because
    gplearn's `div` is *protected* division (returns 1.0 near a zero
    denominator rather than raising/inf) — it is not sympy's `/`, and
    naively parsing `div(a, b)` as `a/b` and calling `sympy.together` would
    both mis-model gplearn's actual runtime semantics and still need this
    same subtree-classification logic to decide whether the denominator is
    a constant. Walking `.program` directly is exact and dependency-free.

    Returns `(degree, is_polynomial)`:
      - `is_polynomial` is False iff some `div` node in the tree has a
        denominator subtree that is not a pure constant (i.e. it contains
        at least one feature variable) — such a term (e.g. `x/y`) is a
        rational function, not a polynomial, and its "degree" is undefined.
      - `degree` is the standard polynomial degree (max over `add`/`sub`
        branches, sum over `mul` operands, unchanged by `div` since only
        division by a constant is accepted as polynomial) when
        `is_polynomial` is True, else `None`.

    Feature terminals contribute degree 1 (states are treated as raw,
    degree-1 variables); numeric constants contribute degree 0.
    """

    def walk(i: int) -> tuple[int, bool, int]:
        node = program[i]
        if isinstance(node, bool):
            # Guard: bool is an int subclass; gplearn never emits bool
            # terminals, but avoid silently misclassifying if it did.
            raise TypeError(f"Unexpected boolean terminal in program: {node!r}")
        if isinstance(node, (int, np.integer)):
            # Feature index terminal -> degree-1 variable.
            return 1, True, i + 1
        if isinstance(node, (float, np.floating)):
            # Numeric constant terminal -> degree-0.
            return 0, True, i + 1
        # Otherwise a gplearn _Function node.
        name = node.name
        arity = node.arity
        child_degrees: list[int] = []
        child_polys: list[bool] = []
        j = i + 1
        for _ in range(arity):
            d, p, j = walk(j)
            child_degrees.append(d)
            child_polys.append(p)
        is_poly = all(child_polys)
        if name in ("add", "sub"):
            degree = max(child_degrees)
        elif name == "mul":
            degree = sum(child_degrees)
        elif name == "div":
            num_deg, den_deg = child_degrees
            # Division only preserves polynomial-ness if the denominator
            # subtree is a pure constant (degree 0 *and* itself polynomial,
            # i.e. contains no variable and no non-constant division).
            is_poly = is_poly and (den_deg == 0)
            degree = num_deg
        else:
            raise ValueError(f"Unsupported function in program: {name!r}")
        return degree, is_poly, j

    degree, is_poly, end = walk(0)
    if end != len(program):
        raise ValueError("Program tree did not consume the full node list")
    return (degree if is_poly else None), is_poly


def fit_symbolic_regression(
    states: np.ndarray,
    dt: float,
    feature_names: list[str] | None = None,
    derivatives: np.ndarray | None = None,
    population_size: int = 2000,
    generations: int = 20,
    parsimony_coefficient: float = 0.001,
    max_degree: int = 3,
    random_state: int = 0,
    n_jobs: int = 1,
    stopping_criteria: float = 1e-9,
    max_degree_retries: int = 5,
) -> dict:
    """Fit one gplearn SymbolicRegressor per state dimension, enforcing
    (not just documenting) a polynomial-degree cap.

    states: (n_samples, n_dims) trajectory.
    dt: fixed sampling interval used for finite-difference derivatives
        (ignored if `derivatives` is supplied directly).
    derivatives: optional precomputed (n_samples, n_dims) derivative array;
        if omitted, derivatives are computed with `estimate_derivatives`
        using the same convention as the SINDy comparator.
    max_degree: polynomial degree cap. Enforcement has two parts:
      1. Structural: `init_depth` is tied to `max_degree` (depth `d` bounds
         worst-case `mul`-chain degree at `2**(d-1)` over these binary
         `add/sub/mul/div` primitives), so degree > max_degree becomes
         structurally rarer for smaller `max_degree` rather than being a
         fixed (2,4) range regardless of the requested cap.
      2. Post-hoc verification + refit: gplearn has no native degree
         constraint, so structural tightening alone does not guarantee
         degree <= max_degree. After each fit, `program_polynomial_degree`
         computes the *true* degree of the winning program (see docstring
         there). If it exceeds `max_degree` (or the program is not a
         polynomial at all, e.g. contains `x/y`), the dimension is refit
         with a different `random_state` up to `max_degree_retries` times.
         If no attempt satisfies the cap, the lowest-degree polynomial
         attempt seen is kept (falling back to the lowest-degree attempt
         of any kind if no attempt was ever a polynomial), and
         `degree_ok[dim]` is False so callers can detect and report this
         rather than silently claiming a degree-matched result.

    Returns a dict with keys: "regressors" (list of fitted SymbolicRegressor,
    one per state dim), "feature_names", "programs" (list of str, the raw
    gplearn program repr per dim), "derivatives" (the derivative array used),
    "degrees" (list of int|None, true polynomial degree per dim, None if
    not a polynomial), "is_polynomial" (list of bool per dim), "degree_ok"
    (list of bool per dim: True iff is_polynomial and degree <= max_degree).
    """
    n_samples, n_dims = states.shape
    feature_names = feature_names or [f"x{i}" for i in range(n_dims)]

    if derivatives is None:
        derivatives = estimate_derivatives(states, dt)

    # Tie init_depth to max_degree: worst-case degree of a depth-d binary
    # mul-chain over degree-1 features is 2**(d-1), so pick the smallest d
    # with 2**(d-1) >= max_degree (floor 2, ceiling 4 to stay within the
    # previously validated search-space size).
    min_depth = 2
    max_depth = 2
    while (2 ** (max_depth - 1)) < max_degree and max_depth < 4:
        max_depth += 1
    init_depth = (min_depth, max_depth)

    regressors = []
    programs = []
    degrees: list[int | None] = []
    is_polynomial: list[bool] = []
    degree_ok: list[bool] = []
    for dim in range(n_dims):
        best = None  # (rank_tuple, est, degree, is_poly)
        for attempt in range(max(1, max_degree_retries)):
            est = SymbolicRegressor(
                population_size=population_size,
                generations=generations,
                function_set=MATCHED_FUNCTION_SET,
                init_depth=init_depth,
                parsimony_coefficient=parsimony_coefficient,
                stopping_criteria=stopping_criteria,
                random_state=random_state + attempt,
                n_jobs=n_jobs,
                feature_names=feature_names,
                verbose=0,
            )
            est.fit(states, derivatives[:, dim])
            deg, is_poly = program_polynomial_degree(est._program.program)
            satisfies = is_poly and deg is not None and deg <= max_degree
            # Rank attempts so we can keep the best one even if none
            # satisfies the cap: prefer satisfying, then polynomial with
            # lower degree, then non-polynomial as last resort.
            rank = (
                0 if satisfies else 1,
                0 if is_poly else 1,
                deg if deg is not None else float("inf"),
            )
            if best is None or rank < best[0]:
                best = (rank, est, deg, is_poly)
            if satisfies:
                break

        _, est, deg, is_poly = best
        regressors.append(est)
        programs.append(str(est._program))
        degrees.append(deg)
        is_polynomial.append(is_poly)
        degree_ok.append(bool(is_poly and deg is not None and deg <= max_degree))

    return {
        "regressors": regressors,
        "feature_names": feature_names,
        "programs": programs,
        "derivatives": derivatives,
        "degrees": degrees,
        "is_polynomial": is_polynomial,
        "degree_ok": degree_ok,
    }
