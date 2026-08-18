"""Committed, independently-checkable artifact for the shape-only blinding
smoke test described in experiments/main_study_confirmation.py's docstring
and DECISION_LOG.md "Rössler confirmation run".

LIMITATIONS.md #6 flags that the original pre-launch check (run informally
before the frozen confirmation run) left no committed script or log -- the
claim that only dict shape/types, never substantive values, were inspected
rested on a contemporaneous self-report with no independent trail. This
script reproduces the same mechanism as a runnable artifact: it calls
fit_rossler() on ONE small/fast condition and asserts only key names and
Python types on the returned dict, deliberately never printing, comparing,
or reasoning about any numeric value. Running this script (before or after
the fact) cannot itself violate PREREGISTRATION.md SS11, since the actual
frozen confirmation run's results are never touched by it.

This does NOT retroactively prove the original informal check was run this
same way -- it demonstrates the mechanism is legitimate and reproducible,
closing the reproducibility gap going forward, not the historical gap.
"""
from experiments.main_study_confirmation import ROSSLER_REGIMES, fit_rossler

EXPECTED_KEYS_AND_TYPES = {
    "seed": (int,),
    "a": (float, int),
    "b": (float, int),
    "c": (float, int),
    "noise_frac": (float, int),
    "degree": (int,),
    "a_hat": (float,),
    "b_hat": (float,),
    "c_hat": (float,),
    "max_rel_err": (float,),
    "recovered": (bool,),
    "vf_l2_err_confirmation": (float,),
    "dynamically_distinct": (bool,),
    "vf_l2_err_off_attractor_grid": (float,),
    "dynamically_distinct_off_attractor": (bool,),
    "naive_n_nonzero": (int,),
    "naive_n_features": (int,),
    "lyapunov_error": (float, str, type(None)),
    "invariant_measure_tv": (float, type(None)),
}


def _shape_only_check(result: dict) -> list[str]:
    """Return a list of shape/type problems (empty = pass). Never reads or
    reports any numeric value, only key presence and Python type."""
    problems = []
    for key, allowed_types in EXPECTED_KEYS_AND_TYPES.items():
        if key not in result:
            problems.append(f"missing key: {key}")
            continue
        if not isinstance(result[key], allowed_types):
            problems.append(f"key {key}: unexpected type {type(result[key]).__name__}")
    extra = set(result) - set(EXPECTED_KEYS_AND_TYPES)
    if extra:
        problems.append(f"unexpected extra keys: {sorted(extra)}")
    return problems


def main():
    # Deliberately small/fast, and NOT one of the frozen run's actual
    # (regime, noise, degree, seed) conditions from
    # experiments/main_study_results/confirmation_manifest.json -- this
    # smoke test must never touch or report on the frozen run's own data.
    params, label = ROSSLER_REGIMES[0]
    print(f"Running fit_rossler() once, regime={label}, tiny/fast config "
          f"(shape/crash-freedom check only -- no values will be printed).")
    result = fit_rossler(params, seed=999, noise_frac=0.0, degree=2,
                         n_points=2000, t_end=20.0)

    problems = _shape_only_check(result)
    if problems:
        print("SHAPE-ONLY SMOKE TEST FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"SHAPE-ONLY SMOKE TEST PASSED: all {len(EXPECTED_KEYS_AND_TYPES)} "
          f"expected keys present with correct types, no extra keys, "
          f"no numeric value read or reported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
