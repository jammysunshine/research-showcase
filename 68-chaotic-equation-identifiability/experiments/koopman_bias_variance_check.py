"""Numerical sanity check for KOOPMAN_BIAS_VARIANCE_THEORY.md (NEXT_STEPS.md
item #8).

The theory doc argues EDMD's chaos-vs-control one-step-error reversal is a
DICTIONARY-TRUNCATION BIAS effect (grows with phase-space coverage/domain
diameter, because a fixed finite monomial dictionary cannot exactly represent
Phi(F(x)) for a nonlinear F, and the mu-weighted best-linear-fit residual
mechanically grows as mu is spread over a larger region) rather than a
regression-VARIANCE effect (which would predict one_step_err falling as
lambda_min(G) rises, the opposite of what is observed).

This script does not run new dynamics -- it reuses koopman_gram_matrix_
analysis.py's exact (family, regime_args, seed, noise_frac) tuples and
regenerates the same training states via `_generate_tier_b_data` (bit-
identical to what produced koopman_gram_matrix_results.json), then adds one
new quantity per row: a state-space SCALE/coverage proxy independent of the
monomial dictionary (RMS state norm, and mean per-dimension std), to check
two things honestly:

1. Does one_step_err rise with domain diameter/coverage (consistent with the
   bias story), not just with lambda_min(G) itself (already known)?
2. CONFOUND CHECK: monomial Gram matrices are not scale-invariant (lambda_min
   for a degree-d dictionary scales roughly like a constant times the
   dominant state magnitude to the 2d power), so lambda_min(G) and raw state
   MAGNITUDE are likely highly correlated by construction. This script
   reports that correlation honestly rather than assuming lambda_min's
   correlation with error reflects "coverage/diversity" and not merely
   "regime B happens to have larger numbers than regime A."

Spearman rank correlation is used throughout (robust to the exact lambda_min
= 0.0 ties in the period_2 rows, and to the arbitrary units of each proxy).
"""
import json

import numpy as np
from scipy.stats import spearmanr

from experiments.koopman_gram_matrix_analysis import MATCHED_PAIRS, _regime_args
from experiments.main_study_tier_b import (
    LIBRARY_DEGREES,
    NOISE_LEVELS_B,
    SEEDS,
    _generate_tier_b_data,
)

IN_PATH = "experiments/main_study_results/koopman_gram_matrix_results.json"
OUT_PATH = "experiments/main_study_results/koopman_bias_variance_check_results.json"


def _scale_proxies(states):
    rms_norm = float(np.sqrt(np.mean(np.sum(states ** 2, axis=1))))
    mean_std = float(np.mean(np.std(states, axis=0)))
    diameter = float(np.max(states, axis=0).__sub__(np.min(states, axis=0)).max())
    return rms_norm, mean_std, diameter


def main():
    with open(IN_PATH) as f:
        gram_rows = json.load(f)
    gram_by_key = {r["key"]: r for r in gram_rows}

    rows = []
    for family, (label_ctrl, label_chaos) in MATCHED_PAIRS.items():
        for label in (label_ctrl, label_chaos):
            regime_args = _regime_args(family, label)
            for noise_frac in NOISE_LEVELS_B:
                for degree in LIBRARY_DEGREES:
                    for seed in SEEDS:
                        key = f"{family}|{label}|{noise_frac}|{degree}|{seed}"
                        g = gram_by_key.get(key)
                        if g is None:
                            print(f"MISSING: {key}")
                            continue
                        data = _generate_tier_b_data(family, regime_args, seed, noise_frac)
                        rms_norm, mean_std, diameter = _scale_proxies(data["states"])
                        rows.append(dict(
                            key=key, family=family, regime=label,
                            is_chaotic=(label == label_chaos),
                            noise_frac=noise_frac, degree=degree, seed=seed,
                            lambda_min=g["lambda_min"], cond=g["cond"],
                            one_step_rel_rms_err=g["one_step_rel_rms_err"],
                            rms_norm=rms_norm, mean_std=mean_std, diameter=diameter,
                        ))

    out = {"rows": rows, "correlations": {}}
    for family in MATCHED_PAIRS:
        fam_rows = [r for r in rows if r["family"] == family]
        err = np.array([r["one_step_rel_rms_err"] for r in fam_rows])
        lam = np.array([r["lambda_min"] for r in fam_rows])
        cond = np.array([r["cond"] for r in fam_rows])
        diam = np.array([r["diameter"] for r in fam_rows])
        rms = np.array([r["rms_norm"] for r in fam_rows])

        def srho(a, b):
            rho, p = spearmanr(a, b)
            return {"rho": float(rho), "p": float(p)}

        out["correlations"][family] = dict(
            n=len(fam_rows),
            err_vs_lambda_min=srho(err, lam),
            err_vs_cond=srho(err, cond),
            err_vs_diameter=srho(err, diam),
            err_vs_rms_norm=srho(err, rms),
            lambda_min_vs_diameter=srho(lam, diam),
            lambda_min_vs_rms_norm=srho(lam, rms),
        )
        print(f"{family} (n={len(fam_rows)}):")
        for k, v in out["correlations"][family].items():
            if isinstance(v, dict):
                print(f"  {k}: rho={v['rho']:.3f} p={v['p']:.3g}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(rows)} rows + correlations to {OUT_PATH}")


if __name__ == "__main__":
    main()
