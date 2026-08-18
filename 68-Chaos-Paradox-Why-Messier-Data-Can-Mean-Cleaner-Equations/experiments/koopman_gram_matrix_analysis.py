"""Mechanistic theory for Tier B's Koopman/EDMD reversal (EXTENSION_PLAN.md
extension #1, deepened 2026-08-17 in response to a "make this thesis-worthy"
scope-up request).

Gallo, Anselmi, Lazzari (arXiv:2607.18490) predict SINDy/PySR's identifiability
ceiling with a single derived quantity: lambda_min(M), the smallest eigenvalue
of the invariant-measure moment matrix built from the SINDy feature library
evaluated along the trajectory -- a persistent-excitation/state-coverage
argument. This project's own independent review (LIMITATIONS.md #7) already
found the same persistent-excitation reading for SINDy/SR: chaos is not
special, broad phase-space coverage is.

Tier B's Koopman/EDMD arm reverses that pattern -- chaotic regimes have
WORSE (not better) one-step prediction error than their matched non-chaotic
control. This script tests whether the SAME KIND of quantity Gallo et al.
use to predict SINDy/PySR's ceiling -- lambda_min of the feature-dictionary
Gram/moment matrix, here built from EDMD's own monomial dictionary Phi(x)
rather than SINDy's library -- also predicts (and thereby explains) the
Koopman-specific REVERSAL. The hypothesis: for EDMD's least-squares
regression, broader phase-space coverage (larger lambda_min, normally a
*good* persistent-excitation signal for SINDy/PySR) also stretches the
dictionary's dynamic range far more unevenly across monomial degrees on a
chaotic attractor than a confined non-chaotic orbit, which shows up as WORSE
conditioning of the design matrix used to fit K -- i.e. persistent excitation
that is good for coefficient identifiability can simultaneously be bad for
EDMD's regression conditioning if it is unevenly distributed across the
dictionary. lambda_min(G) alone is Gallo et al.'s exact quantity; this script
also reports the design matrix's condition number cond(G) = lambda_max/lambda_min
as the more EDMD-relevant diagnostic, since EDMD's error is a regression-
conditioning story, not a persistent-excitation-floor story.

No new trajectory data is generated beyond what Tier B already used --
`_generate_tier_b_data` (main_study_tier_b.py) is called with the exact same
(family, regime_args, seed, noise_frac) tuples Tier B used, so states are
bit-for-bit identical to what produced tier_b_results.json's koopman.
one_step_rel_rms_err values (same RNG seeding scheme, no changes to that
function). Only the two true matched pairs (logistic period_2/chaotic,
Lorenz stable_fixed_point/classic_chaotic) are covered -- harmonic/Duffing
have no chaotic counterpart and are out of scope for a chaotic-vs-non-chaotic
comparison, matching Tier B's own matched-pair design.
"""
import json

import numpy as np

from experiments.main_study_tier_b import (
    LIBRARY_DEGREES,
    NOISE_LEVELS_B,
    SEEDS,
    TIER_B_ITEMS,
    _generate_tier_b_data,
    _job_key,
)
from src.discovery_koopman import MonomialDictionary

MATCHED_PAIRS = {
    "logistic": ("period_2", "chaotic"),
    "lorenz": ("stable_fixed_point", "classic_chaotic"),
}

RESULTS_PATH = "experiments/main_study_results/tier_b_results.json"
OUT_PATH = "experiments/main_study_results/koopman_gram_matrix_results.json"


def _regime_args(family, label):
    for fam, args, lbl in TIER_B_ITEMS:
        if fam == family and lbl == label:
            return args
    raise KeyError((family, label))


def _gram_stats(Phi0):
    n = Phi0.shape[0]
    G = (Phi0.T @ Phi0) / n
    eigs = np.linalg.eigvalsh(G)
    eigs = np.clip(eigs, 0.0, None)  # numerical noise near 0 for near-singular G
    lam_min = float(eigs[0])
    lam_max = float(eigs[-1])
    lam_min_nonzero = float(eigs[eigs > 1e-300][0]) if np.any(eigs > 1e-300) else 0.0
    cond = float(lam_max / lam_min_nonzero) if lam_min_nonzero > 0 else float("inf")
    return lam_min, lam_max, cond


def main():
    with open(RESULTS_PATH) as f:
        tier_b = json.load(f)
    tier_b_by_key = {r["key"]: r for r in tier_b}

    rows = []
    for family, (label_ctrl, label_chaos) in MATCHED_PAIRS.items():
        for label in (label_ctrl, label_chaos):
            regime_args = _regime_args(family, label)
            for noise_frac in NOISE_LEVELS_B:
                for degree in LIBRARY_DEGREES:
                    for seed in SEEDS:
                        key = _job_key(family, label, noise_frac, degree, seed)
                        tb_row = tier_b_by_key.get(key)
                        if tb_row is None:
                            print(f"MISSING from tier_b_results.json: {key}")
                            continue
                        one_step_err = tb_row["koopman"]["one_step_rel_rms_err"]

                        data = _generate_tier_b_data(family, regime_args, seed, noise_frac)
                        dictionary = MonomialDictionary(
                            n_vars=data["states"].shape[1], degree=degree,
                            var_names=data["feature_names"])
                        Phi0 = dictionary.transform(data["states"][:-1])
                        lam_min, lam_max, cond = _gram_stats(Phi0)

                        rows.append(dict(
                            key=key, family=family, regime=label,
                            is_chaotic=(label == label_chaos),
                            noise_frac=noise_frac, degree=degree, seed=seed,
                            lambda_min=lam_min, lambda_max=lam_max, cond=cond,
                            one_step_rel_rms_err=one_step_err,
                        ))
        print(f"{family}: done")

    with open(OUT_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
