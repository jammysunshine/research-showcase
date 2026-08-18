"""Trusted-comparator smoke test: SINDy must recover the true Lorenz
coefficients from a clean, densely sampled trajectory before any further
identifiability experiments are trusted. Required by PROMPT.md's
"trusted comparator passes a smoke test" foundation gate.
"""
import numpy as np
import pysindy as ps

from src.simulators import DEFAULT_LORENZ_PARAMS, lorenz_trajectory

TRUE_COEFFS = {
    "x": {"x": -DEFAULT_LORENZ_PARAMS["sigma"], "y": DEFAULT_LORENZ_PARAMS["sigma"]},
    "y": {"x": DEFAULT_LORENZ_PARAMS["rho"], "y": -1.0, "xz": -1.0},
    "z": {"xy": 1.0, "z": -DEFAULT_LORENZ_PARAMS["beta"]},
}


def main():
    t, states = lorenz_trajectory(
        np.array([-8.0, 8.0, 27.0]), t_span=(0, 50), n_points=25000
    )
    dt = t[1] - t[0]

    model = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=2),
        optimizer=ps.STLSQ(threshold=0.1),
    )
    model.fit(states, t=dt, feature_names=["x", "y", "z"])

    print("Discovered equations:")
    model.print()

    coeffs = model.coefficients()
    feature_names = model.get_feature_names()
    print("\nFeature names:", feature_names)

    # Extract discovered sigma, rho, beta from the x and z equations for a
    # numeric pass/fail check, independent of print() formatting.
    x_row = coeffs[0]
    y_row = coeffs[1]
    z_row = coeffs[2]
    idx = {name: i for i, name in enumerate(feature_names)}

    sigma_hat = -x_row[idx["x"]]
    rho_hat = y_row[idx["x"]]
    beta_hat = -z_row[idx["z"]]

    print(f"\nRecovered: sigma={sigma_hat:.3f} rho={rho_hat:.3f} beta={beta_hat:.3f}")
    print(
        f"True:      sigma={DEFAULT_LORENZ_PARAMS['sigma']:.3f} "
        f"rho={DEFAULT_LORENZ_PARAMS['rho']:.3f} beta={DEFAULT_LORENZ_PARAMS['beta']:.3f}"
    )

    tol = 0.05
    ok = (
        abs(sigma_hat - DEFAULT_LORENZ_PARAMS["sigma"]) / DEFAULT_LORENZ_PARAMS["sigma"] < tol
        and abs(rho_hat - DEFAULT_LORENZ_PARAMS["rho"]) / DEFAULT_LORENZ_PARAMS["rho"] < tol
        and abs(beta_hat - DEFAULT_LORENZ_PARAMS["beta"]) / DEFAULT_LORENZ_PARAMS["beta"] < tol
    )
    print(f"\nSMOKE TEST {'PASSED' if ok else 'FAILED'} (tolerance {tol:.0%})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
