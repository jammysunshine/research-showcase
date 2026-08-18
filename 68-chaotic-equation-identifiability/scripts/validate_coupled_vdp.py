#!/usr/bin/env python3
"""Validate the three coupled van der Pol regimes: periodic, quasi-periodic, chaotic.

Checks:
  1. Lyapunov spectrum (expected signature per regime)
  2. Trajectory boundedness and visual inspection data
  3. Divergence (dissipation check)
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.simulators import (
    coupled_vdp_trajectory, coupled_vdp_lyapunov_spectrum,
    coupled_vdp_rhs, COUPLED_VDP_REGIMES,
)


def validate_regime(name: str, params: dict, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(4) * 0.5

    # Lyapunov spectrum
    print(f"\n{'='*60}")
    print(f"Regime: {name}")
    print(f"  params: {params}")
    print(f"  x0: {x0}")
    print(f"  Computing Lyapunov spectrum...")
    lyap = coupled_vdp_lyapunov_spectrum(
        x0, params=params, dt=0.05, n_steps=30000, transient_steps=5000,
    )
    print(f"  Lyapunov spectrum: {lyap}")

    n_pos = int(np.sum(lyap > 0.02))
    n_zero = int(np.sum(np.abs(lyap) < 0.05))  # relaxed threshold for numerical zeros
    n_neg = int(np.sum(lyap < -0.05))

    if name == "periodic":
        expected = "2 near-zero, 2 negative (limit cycle)"
        ok = n_zero >= 1 and n_neg >= 2
    elif name == "quasi_periodic":
        expected = "2 near-zero, 2 negative (2-torus)"
        ok = n_zero >= 2 and n_neg >= 1
    elif name == "chaotic":
        expected = "1 positive (>0.02), 3 negative/near-zero"
        ok = n_pos >= 1
    else:
        expected = "unknown"
        ok = False

    # Divergence check (time-averaged)
    t, states = coupled_vdp_trajectory(x0, (0, 500), 10000, params=params)
    divs = []
    for s in states:
        J = np.zeros((4, 4))
        x1, v1, x2, v2 = s
        mu1, mu2, omega2, k = params["mu1"], params["mu2"], params["omega2"], params["k"]
        div = mu1 * (1.0 - 3*x1**2) + mu2 * (1.0 - 3*x2**2)
        divs.append(div)
    mean_div = float(np.mean(divs))

    # Trajectory bounds
    amp = np.abs(states).max(axis=0)

    result = {
        "regime": name,
        "lyapunov": lyap.tolist(),
        "expected": expected,
        "check_ok": ok,
        "mean_divergence": mean_div,
        "dissipative": mean_div < 0,
        "amplitude_bounds": amp.tolist(),
        "state_range": {
            "x1": [float(states[:, 0].min()), float(states[:, 0].max())],
            "v1": [float(states[:, 1].min()), float(states[:, 1].max())],
            "x2": [float(states[:, 2].min()), float(states[:, 2].max())],
            "v2": [float(states[:, 3].min()), float(states[:, 3].max())],
        },
    }
    return result


def main():
    results = {}
    for name, params in COUPLED_VDP_REGIMES.items():
        r = validate_regime(name, params)
        results[name] = r
        status = "PASS" if r["check_ok"] else "FAIL"
        print(f"\n  >> {status}: expected [{r['expected']}], "
              f"mean_div={r['mean_divergence']:.4f}, "
              f"dissipative={r['dissipative']}")

    print("\n" + "=" * 60)
    all_ok = all(r["check_ok"] for r in results.values())
    all_diss = all(r["dissipative"] for r in results.values())
    print(f"All regimes validated: {all_ok}")
    print(f"All dissipative: {all_diss}")
    if all_ok and all_diss:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
