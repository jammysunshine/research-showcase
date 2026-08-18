#!/usr/bin/env python3
"""Generate publication-quality figures for the chaotic equation identifiability project.

Reads committed JSON result files and produces PDF/PNG figures.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("experiments/main_study_results")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def figure1_main_results():
    """2x2 panel: Tier A heatmap, Koopman reversal bars, Tier C boxplot, Gram scatter."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Panel A: Tier A SINDy heatmap (degree=2) ---
    ax = axes[0, 0]
    with open(RESULTS_DIR / "tier_a_results.json") as f:
        tier_a = json.load(f)

    regime_order = []
    seen = set()
    for fam, degs in tier_a.items():
        for deg_key in ["degree2", "degree3"]:
            if deg_key not in degs:
                continue
            for reg, noise_dict in degs[deg_key].items():
                key = (fam, reg)
                if key not in seen:
                    regime_order.append(key)
                    seen.add(key)
    regime_order.sort()

    noises = [0.0, 0.001, 0.01, 0.05]
    grid = np.full((len(regime_order), len(noises)), np.nan)
    for fam, degs in tier_a.items():
        if "degree2" not in degs:
            continue
        for reg, noise_dict in degs["degree2"].items():
            i = regime_order.index((fam, reg))
            for noise_str, info in noise_dict.items():
                nf = info.get("noise_frac", float(noise_str))
                j = noises.index(nf) if nf in noises else None
                if j is None:
                    continue
                n_recovered = info.get("n_recovered", 0)
                n_total = info.get("n_total", 1)
                grid[i, j] = n_recovered / n_total

    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(noises)))
    ax.set_xticklabels([f"{n:.1%}" for n in noises], fontsize=7)
    ax.set_yticks(range(len(regime_order)))
    labels = []
    for f, r in regime_order:
        short_f = f.replace("_forced", "+F").replace("_unforced", "")
        short_r = r.replace("_fixed_point", "_fp").replace("classic_", "").replace("pre_", "pre-")
        labels.append(f"{short_f}: {short_r}")
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Noise level")
    ax.set_title("A. SINDy recovery (degree=2)", fontsize=10, fontweight="bold")
    for i in range(len(regime_order)):
        for j in range(len(noises)):
            v = grid[i, j]
            if not np.isnan(v):
                symbol = "\u2713" if v >= 0.9 else ("\u25cb" if v > 0 else "\u2717")
                ax.text(j, i, symbol, ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recovery fraction")

    # --- Panel B: Koopman reversal (paired bars with per-seed dots) ---
    ax = axes[0, 1]
    with open(RESULTS_DIR / "tier_b_results.json") as f:
        tier_b = json.load(f)

    pairs = [
        ("logistic", "period_2", "logistic", "chaotic", "Logistic"),
        ("lorenz", "stable_fixed_point", "lorenz", "classic_chaotic", "Lorenz"),
    ]

    x_pos = np.arange(len(pairs))
    width = 0.35
    for k, (ctrl_f, ctrl_r, cha_f, cha_r, lbl) in enumerate(pairs):
        ctrl_by_n = {}
        cha_by_n = {}
        for r in tier_b:
            if r["family"] == ctrl_f and r["regime"] == ctrl_r and r.get("koopman"):
                n = r["noise_frac"]
                ctrl_by_n.setdefault(n, []).append(r["koopman"]["one_step_rel_rms_err"])
            if r["family"] == cha_f and r["regime"] == cha_r and r.get("koopman"):
                n = r["noise_frac"]
                cha_by_n.setdefault(n, []).append(r["koopman"]["one_step_rel_rms_err"])

        n_key = 0.05
        c_mean = np.mean(ctrl_by_n.get(n_key, [0]))
        ch_mean = np.mean(cha_by_n.get(n_key, [0]))
        ax.bar(k - width / 2, c_mean, width, color="#9E9E9E",
               label="Control" if k == 0 else "")
        ax.bar(k + width / 2, ch_mean, width, color="#F44336",
               label="Chaotic" if k == 0 else "")
        for v in ctrl_by_n.get(n_key, []):
            ax.plot(k - width / 2, v, "k.", markersize=4, alpha=0.5)
        for v in cha_by_n.get(n_key, []):
            ax.plot(k + width / 2, v, "k.", markersize=4, alpha=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([p[4] for p in pairs])
    ax.set_ylabel("One-step relative RMS error")
    ax.set_title("B. Koopman reversal (noise=5%)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # --- Panel C: Tier C delay embedding boxplot ---
    ax = axes[1, 0]
    with open(RESULTS_DIR / "tier_c_results.json") as f:
        tier_c = json.load(f)

    ctrl_errs = [r["sindy_delay_embedded"]["one_step_rel_rms_err"] for r in tier_c
                 if r["regime"] in ("period_2", "stable_fixed_point")
                 and r.get("sindy_delay_embedded", {}).get("one_step_rel_rms_err") is not None]
    cha_errs = [r["sindy_delay_embedded"]["one_step_rel_rms_err"] for r in tier_c
                if r["regime"] in ("chaotic", "classic_chaotic")
                and r.get("sindy_delay_embedded", {}).get("one_step_rel_rms_err") is not None]

    bp = ax.boxplot([ctrl_errs, cha_errs], tick_labels=["Control", "Chaotic"],
                    patch_artist=True, widths=0.5, showfliers=True)
    bp["boxes"][0].set_facecolor("#9E9E9E")
    bp["boxes"][1].set_facecolor("#F44336")
    ax.axhline(y=0.10, color="red", linestyle="--", alpha=0.5, label="Threshold")
    ax.set_ylabel("One-step error")
    ax.set_title("C. Delay embedding: no gap", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)

    # --- Panel D: Gram-matrix lambda_min scatter ---
    ax = axes[1, 1]
    with open(RESULTS_DIR / "koopman_gram_matrix_results.json") as f:
        gram = json.load(f)

    ctrl_l, ctrl_e = [], []
    cha_l, cha_e = [], []
    for r in gram:
        if r.get("lambda_min") and r.get("one_step_rel_rms_err"):
            if r["is_chaotic"]:
                cha_l.append(r["lambda_min"])
                cha_e.append(r["one_step_rel_rms_err"])
            else:
                ctrl_l.append(r["lambda_min"])
                ctrl_e.append(r["one_step_rel_rms_err"])

    ax.scatter(ctrl_l, ctrl_e, c="#9E9E9E", marker="o", label="Control",
               alpha=0.6, s=30)
    ax.scatter(cha_l, cha_e, c="#F44336", marker="^", label="Chaotic",
               alpha=0.6, s=30)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda_{\min}(G)$ (persistent excitation)")
    ax.set_ylabel("Koopman one-step error")
    ax.set_title(r"D. Sign reversal: $\lambda_{\min}$ vs EDMD",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)

    fig.suptitle("When Does Phase-Space Coverage Aid Equation Identifiability?",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        out = FIGURES_DIR / f"fig1_main_results.{fmt}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def figure2_koopman_per_seed():
    """Per-seed dot plot of Koopman error by noise level."""
    with open(RESULTS_DIR / "tier_b_results.json") as f:
        tier_b = json.load(f)

    pairs = [
        ("logistic", "period_2", "logistic", "chaotic", "Logistic map"),
        ("lorenz", "stable_fixed_point", "lorenz", "classic_chaotic", "Lorenz system"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, (cf, cr, chf, chr_, title) in zip(axes, pairs):
        noises = sorted(set(r["noise_frac"] for r in tier_b if r["noise_frac"] > 0))
        x = np.arange(len(noises))
        width = 0.35

        for k, noise in enumerate(noises):
            ctrl = [r["koopman"]["one_step_rel_rms_err"] for r in tier_b
                    if r["family"] == cf and r["regime"] == cr
                    and abs(r["noise_frac"] - noise) < 1e-6
                    and r.get("koopman", {}).get("one_step_rel_rms_err") is not None]
            cha = [r["koopman"]["one_step_rel_rms_err"] for r in tier_b
                   if r["family"] == chf and r["regime"] == chr_
                   and abs(r["noise_frac"] - noise) < 1e-6
                   and r.get("koopman", {}).get("one_step_rel_rms_err") is not None]

            ax.bar(k - width / 2, np.mean(ctrl) if ctrl else 0, width,
                   color="#9E9E9E", alpha=0.8)
            ax.bar(k + width / 2, np.mean(cha) if cha else 0, width,
                   color="#F44336", alpha=0.8)
            for v in ctrl:
                ax.plot(k - width / 2, v, "k.", markersize=5, alpha=0.6)
            for v in cha:
                ax.plot(k + width / 2, v, "k.", markersize=5, alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{n:.1%}" for n in noises])
        ax.set_xlabel("Noise level")
        ax.set_title(title)
        if ax == axes[0]:
            ax.set_ylabel("Koopman one-step error")
            ax.legend(["Control", "Chaotic"], fontsize=7, loc="upper left")
        ax.set_yscale("log")

    fig.suptitle("Koopman/EDMD Reversal: Per-Seed Error by Noise",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        out = FIGURES_DIR / f"fig2_koopman_per_seed.{fmt}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def figure3_coverage_experiment():
    """Coverage experiment: 3-regime comparison."""
    cov_file = RESULTS_DIR / "coverage_experiment_results.json"
    if not cov_file.exists():
        print("coverage_experiment_results.json not yet available, skipping figure 3")
        return

    with open(cov_file) as f:
        data = json.load(f)

    regimes = ["periodic", "quasi_periodic", "chaotic"]
    labels = ["Periodic\n(limit cycle)", "Quasi-periodic\n(2-torus)", "Chaotic\n(strange)"]
    noises = [0.0, 0.01, 0.05]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    ax = axes[0]
    for k, noise in enumerate(noises):
        means = []
        for regime in regimes:
            vals = [r["sindy_one_step_err"] for r in data if r["regime"] == regime
                    and abs(r["noise_pct"] - noise) < 1e-6]
            means.append(np.mean(vals) if vals else 0)
        offset = (k - 1) * 0.25
        ax.bar(np.arange(3) + offset, means, 0.22,
               label=f"Noise={noise:.1%}", alpha=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("SINDy one-step error")
    ax.set_title("SINDy (degree=2)", fontsize=10)
    ax.legend(fontsize=7)
    ax.axhline(y=0.10, color="red", linestyle="--", alpha=0.3)

    ax = axes[1]
    for k, noise in enumerate(noises):
        means = []
        for regime in regimes:
            vals = [r["koopman_one_step_err"] for r in data if r["regime"] == regime
                    and abs(r["noise_pct"] - noise) < 1e-6]
            means.append(np.mean(vals) if vals else 0)
        offset = (k - 1) * 0.25
        ax.bar(np.arange(3) + offset, means, 0.22,
               label=f"Noise={noise:.1%}", alpha=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Koopman one-step error")
    ax.set_title("Koopman/EDMD (degree=2)", fontsize=10)
    ax.legend(fontsize=7)
    ax.set_yscale("log")

    ax = axes[2]
    colors = ["#4CAF50", "#2196F3", "#FF9800"]
    ridge_keys = ["koopman_ridge_0e+00", "koopman_ridge_1e-08", "koopman_ridge_1e-06",
                  "koopman_ridge_1e-04", "koopman_ridge_1e-02"]
    alpha_labels = ["OLS", "1e-8", "1e-6", "1e-4", "1e-2"]
    for k, (regime, color) in enumerate(zip(regimes, colors)):
        errs = []
        for rk in ridge_keys:
            vals = [r.get(rk) for r in data if r["regime"] == regime
                    and abs(r["noise_pct"] - 0.05) < 1e-6
                    and r.get(rk) is not None]
            errs.append(np.mean(vals) if vals else np.nan)
        ax.plot(range(len(ridge_keys)), errs, "o-", color=color,
                label=labels[k].replace("\n", " "), markersize=5)
    ax.set_xticks(range(len(ridge_keys)))
    ax.set_xticklabels(alpha_labels, fontsize=8)
    ax.set_xlabel(r"Ridge regularization $\alpha$")
    ax.set_ylabel("Koopman one-step error")
    ax.set_title("Ridge sensitivity (noise=5%)", fontsize=10)
    ax.legend(fontsize=7)
    ax.set_yscale("log")

    fig.suptitle("Coverage Experiment: Periodic vs Quasi-periodic vs Chaotic",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        out = FIGURES_DIR / f"fig3_coverage_experiment.{fmt}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating publication figures...")
    figure1_main_results()
    figure2_koopman_per_seed()
    figure3_coverage_experiment()
    print("Done.")
