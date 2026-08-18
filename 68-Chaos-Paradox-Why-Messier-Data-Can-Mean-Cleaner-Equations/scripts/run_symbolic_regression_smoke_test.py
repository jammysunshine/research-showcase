"""Symbolic-regression smoke test: gplearn (matched to the SINDy +,-,*,/
low-degree-polynomial primitive set per PREREGISTRATION.md section 2) fit
on a clean, densely sampled Lorenz trajectory, mirroring
scripts/run_sindy_smoke_test.py so the two discovery families are compared
under matched data, derivatives, and library scope.

Per LIMITATIONS.md #2/#3, this runs `fit_symbolic_regression()` at BOTH
max_degree=2 and max_degree=3 on the identical trajectory/derivatives/seed
and reports both side by side (structural match, R^2, degree_ok,
is_polynomial), analogous to `experiments/pilot_chaos_vs_periodic.py`'s
degree-2-vs-degree-3 SINDy comparison. PASS/FAIL gates on the
PREREGISTRATION.md §2-frozen degree-3 run; the degree-2 run is reported for
comparison only, not as an alternative pass condition.

Reports per degree cap:
  - structural-form match (does the discovered expression only involve the
    true active variables for that state dimension?),
  - real polynomial-degree enforcement (is each discovered program an
    actual polynomial, and is its true degree <= max_degree, per
    `program_polynomial_degree`/`fit_symbolic_regression`'s post-hoc
    refit-on-violation logic in src/discovery_symbolic_regression.py), and
  - numeric fit quality (R^2), as gplearn is not guaranteed to produce
    exact, cleanly-thresholded polynomial coefficients the way SINDy's
    STLSQ does.
"""
import numpy as np

from src.discovery_symbolic_regression import estimate_derivatives, fit_symbolic_regression
from src.simulators import DEFAULT_LORENZ_PARAMS, lorenz_trajectory

MAX_DEGREE = 3
COMPARISON_DEGREES = (2, 3)

TRUE_ACTIVE_VARS = {
    "x": {"x", "y"},
    "y": {"x", "y", "z"},
    "z": {"x", "y", "z"},
}


def variables_used(program_str: str, feature_names: list[str]) -> set[str]:
    return {name for name in feature_names if name in program_str}


def run_at_degree(states, dt, feature_names, derivatives, max_degree, random_state):
    """Fit fit_symbolic_regression() at a given max_degree and compute the
    structural-match / R^2 / degree diagnostics for it. Returns a dict
    keyed by state-dim name."""
    result = fit_symbolic_regression(
        states,
        dt,
        feature_names=feature_names,
        derivatives=derivatives,
        population_size=3000,
        generations=25,
        parsimony_coefficient=0.001,
        max_degree=max_degree,
        random_state=random_state,
        n_jobs=1,
    )

    per_dim = {}
    for i, name in enumerate(feature_names):
        prog = result["programs"][i]
        used = variables_used(prog, feature_names)
        true_vars = TRUE_ACTIVE_VARS[name]
        struct_ok = used == true_vars
        reg = result["regressors"][i]
        r2 = reg.score(states, derivatives[:, i])
        per_dim[name] = {
            "program": prog,
            "struct_ok": struct_ok,
            "r2": r2,
            "degree": result["degrees"][i],
            "is_polynomial": result["is_polynomial"][i],
            "degree_ok": result["degree_ok"][i],
        }
    per_dim["struct_all_ok"] = all(per_dim[n]["struct_ok"] for n in feature_names)
    per_dim["degree_all_ok"] = all(per_dim[n]["degree_ok"] for n in feature_names)
    tol_r2 = 0.99
    per_dim["fit_ok"] = all(per_dim[n]["r2"] > tol_r2 for n in feature_names)
    return per_dim


def main():
    t, states = lorenz_trajectory(
        np.array([-8.0, 8.0, 27.0]), t_span=(0, 50), n_points=25000
    )
    dt = t[1] - t[0]
    feature_names = ["x", "y", "z"]

    derivatives = estimate_derivatives(states, dt)

    runs = {}
    for deg in COMPARISON_DEGREES:
        print(f"\n=== Running fit_symbolic_regression(max_degree={deg}), same trajectory/derivatives/seed=0 ===")
        runs[deg] = run_at_degree(states, dt, feature_names, derivatives, deg, random_state=0)
        for name in feature_names:
            print(f"  d{name}/dt = {runs[deg][name]['program']}")

    print("\nDegree-2 vs degree-3 comparison (same Lorenz trajectory, same seed):")
    header = f"{'dim':<5}{'degree':<8}{'struct_match':<14}{'R^2':<12}{'is_polynomial':<16}{'degree_ok':<10}"
    print(header)
    for deg in COMPARISON_DEGREES:
        for name in feature_names:
            d = runs[deg][name]
            print(
                f"{name:<5}{deg:<8}{str(d['struct_ok']):<14}{d['r2']:<12.6f}"
                f"{str(d['is_polynomial']):<16}{str(d['degree_ok']):<10}"
            )

    print("\nAggregate PASS/FAIL gates per degree cap:")
    for deg in COMPARISON_DEGREES:
        r = runs[deg]
        print(
            f"  max_degree={deg}: structural_match={r['struct_all_ok']} "
            f"degree_ok={r['degree_all_ok']} fit_ok={r['fit_ok']}"
        )

    print(
        "\nNote: structural-form match, real degree enforcement, and high "
        "R^2 together are the strongest claim gplearn's raw expression-tree "
        "output supports here; it does NOT by itself imply clean "
        "coefficient recovery within the PREREGISTRATION.md section 6 "
        "tolerance (5% relative / 0.05 absolute), because gplearn programs "
        "are not automatically simplified to a canonical "
        "polynomial-with-coefficients form. Report this method's "
        "structural-recovery result honestly even if it falls short of "
        "SINDy's exact coefficient recovery, and even if degree "
        "enforcement itself trades off against fit quality. Per "
        "PREREGISTRATION.md §2 (frozen degree-3 library), the degree=3 run "
        "is the one that gates PASS/FAIL; the degree=2 run is reported for "
        "comparison only, mirroring experiments/pilot_chaos_vs_periodic.py."
    )

    gate = runs[MAX_DEGREE]
    ok = gate["struct_all_ok"] and gate["degree_all_ok"] and gate["fit_ok"]
    print(
        f"\nSMOKE TEST {'PASSED' if ok else 'FAILED'} at max_degree={MAX_DEGREE} "
        f"(structural_match={gate['struct_all_ok']}, degree_ok={gate['degree_all_ok']}, "
        f"fit_ok={gate['fit_ok']})"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
