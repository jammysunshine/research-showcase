#!/usr/bin/env python3
"""Compute effect sizes with bootstrap CIs for the main study results."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

RESULTS_DIR = Path("experiments/main_study_results")


def cliff_delta(x, y):
    """Cliff's delta effect size. >0.147=small, >0.33=medium, >0.474=large."""
    x, y = np.asarray(x), np.asarray(y)
    dominance = sum((1 if xi > yj else -1 if xi < yj else 0)
                    for xi in x for yj in y)
    return dominance / (len(x) * len(y))


def bootstrap_ratio_ci(x, y, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    ratios = []
    for _ in range(n_boot):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        my = np.mean(by)
        if my > 1e-15:
            ratios.append(np.mean(bx) / my)
    if not ratios:
        return (0, 0)
    return float(np.percentile(ratios, 100 * alpha / 2)), float(np.percentile(ratios, 100 * (1 - alpha / 2)))


def bootstrap_diff_ci(x, y, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    diffs = [np.mean(rng.choice(x, len(x), True)) - np.mean(rng.choice(y, len(y), True))
             for _ in range(n_boot)]
    return float(np.percentile(diffs, 100 * alpha / 2)), float(np.percentile(diffs, 100 * (1 - alpha / 2)))


def effect_label(d):
    d = abs(d)
    if d < 0.147: return "negligible"
    if d < 0.33: return "small"
    if d < 0.474: return "medium"
    return "large"


def analyze_tier_b():
    with open(RESULTS_DIR / "tier_b_results.json") as f:
        data = json.load(f)

    print("=" * 80)
    print("EFFECT SIZES: Tier B Headline Comparisons")
    print("=" * 80)

    # Koopman reversal
    pairs = [
        ("logistic", "period_2", "logistic", "chaotic", "Logistic map (Koopman)"),
        ("lorenz", "stable_fixed_point", "lorenz", "classic_chaotic", "Lorenz system (Koopman)"),
    ]

    for ctrl_fam, ctrl_reg, cha_fam, cha_reg, label in pairs:
        ctrl = [r["koopman"]["one_step_rel_rms_err"] for r in data
                if r["family"] == ctrl_fam and r["regime"] == ctrl_reg
                and r.get("koopman", {}).get("one_step_rel_rms_err") is not None]
        cha = [r["koopman"]["one_step_rel_rms_err"] for r in data
               if r["family"] == cha_fam and r["regime"] == cha_reg
               and r.get("koopman", {}).get("one_step_rel_rms_err") is not None]

        if len(ctrl) < 2 or len(cha) < 2:
            print(f"{label}: insufficient data (n_ctrl={len(ctrl)}, n_cha={len(cha)})")
            continue

        ctrl, cha = np.array(ctrl), np.array(cha)
        d = cliff_delta(cha, ctrl)
        ratio = np.mean(cha) / np.mean(ctrl) if np.mean(ctrl) > 1e-15 else float("inf")
        ratio_lo, ratio_hi = bootstrap_ratio_ci(cha, ctrl)
        diff_lo, diff_hi = bootstrap_diff_ci(cha, ctrl)

        print(f"\n{label}")
        print(f"  Control mean: {np.mean(ctrl):.6f} (n={len(ctrl)})")
        print(f"  Chaotic mean: {np.mean(cha):.6f} (n={len(cha)})")
        print(f"  Ratio (chaos/control): {ratio:.2f}x [{ratio_lo:.2f}x, {ratio_hi:.2f}x] 95% CI")
        print(f"  Difference: {np.mean(cha)-np.mean(ctrl):.6f} [{diff_lo:.6f}, {diff_hi:.6f}]")
        print(f"  Cliff's delta: {d:.3f} ({effect_label(d)})")

    # SINDy recovery contingency
    print("\n" + "=" * 80)
    print("SINDY COEFFICIENT RECOVERY")
    print("=" * 80)

    systems = [("logistic", "period_2", "logistic", "chaotic"),
               ("lorenz", "stable_fixed_point", "lorenz", "classic_chaotic")]

    for ctrl_fam, ctrl_reg, cha_fam, cha_reg in systems:
        ctrl_rec = sum(1 for r in data if r["family"] == ctrl_fam and r["regime"] == ctrl_reg
                       and r.get("sindy", {}).get("recovered"))
        ctrl_tot = sum(1 for r in data if r["family"] == ctrl_fam and r["regime"] == ctrl_reg)
        cha_rec = sum(1 for r in data if r["family"] == cha_fam and r["regime"] == cha_reg
                      and r.get("sindy", {}).get("recovered"))
        cha_tot = sum(1 for r in data if r["family"] == cha_fam and r["regime"] == cha_reg)

        print(f"\n{cha_fam}/{cha_reg} vs {ctrl_fam}/{ctrl_reg} (SINDy):")
        print(f"  Control: {ctrl_rec}/{ctrl_tot} recovered")
        print(f"  Chaotic: {cha_rec}/{cha_tot} recovered")

        if ctrl_tot > 0 and cha_tot > 0:
            odds, p = fisher_exact([[cha_rec, cha_tot - cha_rec],
                                     [ctrl_rec, ctrl_tot - ctrl_rec]],
                                    alternative="greater")
            print(f"  Fisher's exact p (one-sided): {p:.2e}")
            print(f"  Odds ratio: {odds:.1f}" if odds < 1e10 else f"  Odds ratio: Inf")


def analyze_tier_c():
    with open(RESULTS_DIR / "tier_c_results.json") as f:
        data = json.load(f)

    print("\n" + "=" * 80)
    print("TIER C: DELAY EMBEDDING — NO GAP")
    print("=" * 80)

    ctrl = [r["sindy_delay_embedded"]["one_step_rel_rms_err"] for r in data
            if r["regime"] in ("period_2", "stable_fixed_point")
            and r.get("sindy_delay_embedded", {}).get("one_step_rel_rms_err") is not None]
    cha = [r["sindy_delay_embedded"]["one_step_rel_rms_err"] for r in data
           if r["regime"] in ("chaotic", "classic_chaotic")
           and r.get("sindy_delay_embedded", {}).get("one_step_rel_rms_err") is not None]

    if ctrl and cha:
        d = cliff_delta(cha, ctrl)
        print(f"Control one-step error: mean={np.mean(ctrl):.6f} (n={len(ctrl)})")
        print(f"Chaotic one-step error: mean={np.mean(cha):.6f} (n={len(cha)})")
        print(f"Cliff's delta: {d:.3f} ({effect_label(d)})")
        print(f"Both below threshold (0.10): control={max(ctrl)<.1}, chaotic={max(cha)<.1}")


def analyze_coverage():
    cov_file = RESULTS_DIR / "coverage_experiment_results.json"
    if not cov_file.exists():
        print("\ncoverage_experiment_results.json not yet available")
        return

    with open(cov_file) as f:
        data = json.load(f)

    print("\n" + "=" * 80)
    print("COVERAGE EXPERIMENT: Coupled van der Pol (3 regimes)")
    print("=" * 80)

    regimes = ["periodic", "quasi_periodic", "chaotic"]
    for noise in [0.0, 0.01, 0.05]:
        print(f"\n--- Noise={noise:.1%} ---")
        for metric in ["sindy_one_step_err", "koopman_one_step_err"]:
            errs = {}
            for regime in regimes:
                vals = [r[metric] for r in data if r["regime"] == regime
                        and abs(r["noise_pct"] - noise) < 1e-6
                        and r.get(metric) is not None]
                errs[regime] = np.array(vals) if vals else np.array([])
                if len(vals) > 0:
                    print(f"  {metric:25s} {regime:15s}: mean={np.mean(vals):.6f} n={len(vals)}")

            # Pairwise Cliff's delta
            for r1, r2 in [("periodic", "quasi_periodic"), ("quasi_periodic", "chaotic"),
                            ("periodic", "chaotic")]:
                if len(errs.get(r1, [])) >= 2 and len(errs.get(r2, [])) >= 2:
                    d = cliff_delta(errs[r2], errs[r1])
                    print(f"    {r2} vs {r1}: Cliff's d={d:.3f} ({effect_label(d)})")


if __name__ == "__main__":
    analyze_tier_b()
    analyze_tier_c()
    analyze_coverage()
