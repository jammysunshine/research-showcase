"""Lyapunov-normalized Koopman/EDMD one-step error (EXTENSION_PLAN.md #1).

Motivation: Tier B's headline Koopman/EDMD finding is that one_step_rel_rms_err
REVERSES relative to SINDy/SR -- chaotic regimes show WORSE (larger) Koopman
one-step error than their matched non-chaotic control, by roughly 6-11x,
opposite the direction SINDy/SR show. Before reporting this as a genuine
method-specific pathology, it must be checked against the same confound
already diagnosed for Tier C's rollout-error metric: on a chaotic attractor,
ANY one-step predictor's error is inflated relative to a non-chaotic one
simply because nearby-attractor perturbations diverge at the local Lyapunov
rate, independent of whether the fitted model is "wrong". This script
normalizes one_step_rel_rms_err by each regime's own per-dt local expansion
factor exp(lambda_max * dt) and re-checks whether the chaotic/non-chaotic
gap survives.

Reads the frozen tier_b_results.json (no new simulation). Lyapunov exponents
are a property of the true dynamics only (not of noise/seed/fitted model),
so one lambda_max is computed per (family, regime) via the existing Benettin
QR method in src/simulators.py, using each regime's true_rhs/true_jac and a
representative initial condition -- reused, not re-derived.

Writes results to tier_b_koopman_normalization_results.json. Does not modify
any frozen main-study script or result file.
"""
import json

import numpy as np

from src.simulators import (
    DEFAULT_DUFFING_FORCED_PARAMS,
    DEFAULT_DUFFING_UNFORCED_PARAMS,
    duffing_forced_jacobian,
    duffing_forced_rhs,
    duffing_unforced_jacobian,
    duffing_unforced_rhs,
    generic_map_lyapunov_exponent,
    generic_ode_lyapunov_spectrum,
    logistic_map,
    logistic_map_jacobian,
    lorenz_jacobian,
    lorenz_rhs,
)

RESULTS_PATH = "experiments/main_study_results/tier_b_results.json"
OUT_PATH = "experiments/main_study_results/tier_b_koopman_normalization_results.json"

# dt used per family in main_study_tier_b.py's _generate_tier_b_data (MEDIUM dict
# t_end / n_points), needed to convert a continuous-time lambda_max into a
# per-observation-step expansion factor exp(lambda_max * dt) matching the
# Koopman one-step prediction's actual step size. Logistic is a discrete map
# (dt=1 iteration, no continuous-time conversion needed).
DT_BY_FAMILY = {
    "lorenz": 50.0 / 25000,  # t_end / n_points (tail-discard keeps the same spacing)
    "harmonic": 50.0 / 5000,
    "duffing_unforced": 200.0 / 25000,
    "duffing_forced": 200.0 / 25000,
}


def _lyap_logistic(r):
    return generic_map_lyapunov_exponent(
        lambda x: logistic_map(x, r), lambda x: logistic_map_jacobian(x, r),
        x0=0.4, n_steps=20000, transient=2000,
    )


def _lyap_lorenz(rho):
    params = dict(sigma=10.0, rho=rho, beta=8.0 / 3.0)
    spectrum = generic_ode_lyapunov_spectrum(
        rhs=lambda t, s: lorenz_rhs(t, s, params["sigma"], params["rho"], params["beta"]),
        jacobian=lambda s: lorenz_jacobian(s, params["sigma"], params["rho"], params["beta"]),
        x0=np.array([-8.0, 8.0, 27.0]), dim=3, dt=0.01, n_steps=4000, transient_steps=1000,
    )
    return float(np.max(spectrum))


def _lyap_harmonic():
    spectrum = generic_ode_lyapunov_spectrum(
        rhs=lambda t, s: np.array([s[1], -s[0]]),
        jacobian=lambda s: np.array([[0.0, 1.0], [-1.0, 0.0]]),
        x0=np.array([1.0, 0.0]), dim=2, dt=0.01, n_steps=4000, transient_steps=1000,
    )
    return float(np.max(spectrum))


def _lyap_duffing_unforced():
    alpha = DEFAULT_DUFFING_UNFORCED_PARAMS["alpha"]
    beta = DEFAULT_DUFFING_UNFORCED_PARAMS["beta"]
    spectrum = generic_ode_lyapunov_spectrum(
        rhs=lambda t, s: duffing_unforced_rhs(t, s, alpha, beta),
        jacobian=lambda s: duffing_unforced_jacobian(s, alpha, beta),
        x0=np.array([0.5, 0.0]), dim=2, dt=0.01, n_steps=4000, transient_steps=1000,
    )
    return float(np.max(spectrum))


def _lyap_duffing_forced():
    p = DEFAULT_DUFFING_FORCED_PARAMS
    x0 = np.array([0.5, 0.0, 0.0])
    spectrum = generic_ode_lyapunov_spectrum(
        rhs=lambda t, s: duffing_forced_rhs(t, s, p["delta"], p["alpha"], p["beta"], p["gamma"], p["omega"]),
        jacobian=lambda s: duffing_forced_jacobian(s, p["delta"], p["alpha"], p["beta"], p["gamma"], p["omega"]),
        x0=x0, dim=3, dt=0.01, n_steps=4000, transient_steps=1000,
    )
    return float(np.max(spectrum))


def compute_lyapunov_by_regime():
    """One largest Lyapunov exponent per (family, regime) in TIER_B_ITEMS."""
    out = {}
    out[("logistic", "period_2")] = dict(lambda_max=_lyap_logistic(3.2), dt=1.0, discrete=True)
    out[("logistic", "chaotic")] = dict(lambda_max=_lyap_logistic(4.0), dt=1.0, discrete=True)
    out[("lorenz", "stable_fixed_point")] = dict(lambda_max=_lyap_lorenz(14.0), dt=DT_BY_FAMILY["lorenz"], discrete=False)
    out[("lorenz", "classic_chaotic")] = dict(lambda_max=_lyap_lorenz(28.0), dt=DT_BY_FAMILY["lorenz"], discrete=False)
    out[("harmonic", "conservative")] = dict(lambda_max=_lyap_harmonic(), dt=DT_BY_FAMILY["harmonic"], discrete=False)
    out[("duffing_unforced", "conservative")] = dict(lambda_max=_lyap_duffing_unforced(), dt=DT_BY_FAMILY["duffing_unforced"], discrete=False)
    out[("duffing_forced", "forced_chaotic")] = dict(lambda_max=_lyap_duffing_forced(), dt=DT_BY_FAMILY["duffing_forced"], discrete=False)
    return out


def expansion_factor(entry):
    """Per-observation-step local expansion factor exp(lambda_max * dt) (or the
    bare per-iterate factor for discrete maps). Floored at 1.0 -- a negative or
    zero Lyapunov exponent (stable fixed point, conservative orbit) means no
    divergence-driven inflation, so the normalizer should not artificially
    shrink the error below its raw value."""
    lam = entry["lambda_max"]
    if entry["discrete"]:
        raw = np.exp(lam)
    else:
        raw = np.exp(lam * entry["dt"])
    return float(max(raw, 1.0))


def main():
    print("Computing largest Lyapunov exponent per (family, regime)...")
    lyap_by_regime = compute_lyapunov_by_regime()
    for (family, regime), entry in lyap_by_regime.items():
        factor = expansion_factor(entry)
        print(f"  {family}/{regime}: lambda_max={entry['lambda_max']:.4f}  "
              f"expansion_factor={factor:.4f}")

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    rows = []
    for rec in results:
        key = (rec["family"], rec["regime"])
        if key not in lyap_by_regime:
            continue
        raw_err = rec["koopman"]["one_step_rel_rms_err"]
        factor = expansion_factor(lyap_by_regime[key])
        rows.append(dict(
            family=rec["family"], regime=rec["regime"], noise_frac=rec["noise_frac"],
            degree=rec["degree"], seed=rec["seed"],
            koopman_one_step_raw=raw_err,
            lambda_max=lyap_by_regime[key]["lambda_max"],
            expansion_factor=factor,
            koopman_one_step_normalized=raw_err / factor,
        ))

    with open(OUT_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {len(rows)} normalized rows to {OUT_PATH}")

    # Summary: matched chaotic vs non-chaotic pairs, raw vs normalized, across
    # EVERY (noise, degree) cell -- not just one cherry-picked condition.
    pairs = [
        ("logistic", "period_2", "chaotic"),
        ("lorenz", "stable_fixed_point", "classic_chaotic"),
    ]
    summary_rows = []
    print("\n=== Matched-pair ratios across all (noise, degree) cells (median across 5 seeds) ===")
    for family, control, chaotic in pairs:
        print(f"-- {family} ({control} vs {chaotic}) --")
        for noise_frac in [0.0, 0.01, 0.05]:
            for degree in [2, 3]:
                cell = dict(family=family, noise_frac=noise_frac, degree=degree)
                for metric in ("koopman_one_step_raw", "koopman_one_step_normalized"):
                    ctrl_vals = [r[metric] for r in rows
                                 if r["family"] == family and r["regime"] == control
                                 and r["noise_frac"] == noise_frac and r["degree"] == degree]
                    chaos_vals = [r[metric] for r in rows
                                  if r["family"] == family and r["regime"] == chaotic
                                  and r["noise_frac"] == noise_frac and r["degree"] == degree]
                    if not ctrl_vals or not chaos_vals:
                        continue
                    ctrl_med, chaos_med = float(np.median(ctrl_vals)), float(np.median(chaos_vals))
                    ratio = chaos_med / ctrl_med if ctrl_med > 0 else float("inf")
                    cell[f"{metric}_ratio"] = ratio
                    cell[f"{metric}_control_median"] = ctrl_med
                    cell[f"{metric}_chaotic_median"] = chaos_med
                print(f"  noise={noise_frac:.0%} degree={degree}: "
                      f"raw_ratio={cell.get('koopman_one_step_raw_ratio', float('nan')):.3g}  "
                      f"normalized_ratio={cell.get('koopman_one_step_normalized_ratio', float('nan')):.3g}")
                summary_rows.append(cell)

    with open("experiments/main_study_results/tier_b_koopman_normalization_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)


if __name__ == "__main__":
    main()
