"""NN (deep) Koopman arm on Tier B's logistic and Lorenz matched pairs.

Companion to experiments/main_study_tier_b.py's EDMD arm
(src/discovery_koopman.fit_edmd). Tests whether the EDMD "chaos reversal"
(DECISION_LOG.md 2026-08-17: EDMD one-step error is WORSE in chaotic
regimes than matched non-chaotic controls) is specific to EDMD's fixed
monomial dictionary, or also shows up when the "dictionary" is a learned
nonlinear encoder (autoencoder + linear latent dynamics, see
src/discovery_koopman_nn.py).

Data generation for the logistic and lorenz families is copied verbatim
(same RNG offsets, same trajectory lengths, same noise-injection logic)
from experiments/main_study_tier_b.py's `_generate_tier_b_data`, so the NN
arm is fit and evaluated on IDENTICAL trajectories to the existing EDMD
numbers in experiments/main_study_results/tier_b_results.json -- an
apples-to-apples comparison, not a re-derived one.

The one-step-error metric replicates _run_koopman's exactly:
  - fit on the (noisy) training trajectory `states`
  - evaluate one-step prediction error on a DIFFERENT, held-out
    "confirmation" trajectory `states_conf` (different initial condition,
    same regime/noise), via
        one_step_rel_rms_err = ||pred(X0_conf) - X1_conf|| / ||X1_conf||
    (relative L2 norm over the whole array; matches src.discovery_koopman's
    _vf_err / EDMD's `one_step_rel_rms_err` field name and definition).

Scope (kept compute-bounded per PROJECT_CHARTER.md, local CPU only):
  - families: logistic {period_2 r=3.2, chaotic r=4.0},
              lorenz    {stable_fixed_point rho=14, classic_chaotic rho=28}
    (the two required TIER_B_ITEMS families)
  - noise levels: {0.0, 0.01, 0.05} (all three Tier B levels)
  - seeds: [0, 1, 2] (3 of Tier B's 5 seeds -- task minimum is 3)
  - degree: not applicable (NN arm has no "library degree"); latent_dim is
    fixed per family (see LATENT_DIM below), chosen somewhat larger than
    the corresponding EDMD degree-2 monomial dictionary size, so neither
    arm is capacity-starved relative to the other.
This gives 2 families x 2 regimes x 3 noise x 3 seeds = 36 fits.
"""
import json
import time

import numpy as np

from src.discovery_koopman_nn import fit_deep_koopman
from src.simulators import logistic_map, logistic_trajectory, lorenz_rhs, lorenz_trajectory

NOISE_LEVELS = [0.0, 0.01, 0.05]
SEEDS = [0, 1, 2]

MEDIUM = dict(logistic_n_steps=5500, lorenz_n_points=25000, lorenz_t_end=50.0)
N_OFF_ATTRACTOR_GRID_POINTS = 500  # unused here (no grid VF-error concept for Koopman), kept for parity

TIER_B_ITEMS = [
    ("logistic", 3.2, "period_2"),
    ("logistic", 4.0, "chaotic"),
    ("lorenz", 14.0, "stable_fixed_point"),
    ("lorenz", 28.0, "classic_chaotic"),
]

# EDMD degree-2 dictionary sizes: logistic (1 var) -> 3, lorenz (3 vars) -> 10.
# Latent dims chosen "somewhat larger" per task instructions.
LATENT_DIM = dict(logistic=6, lorenz=16)
HIDDEN = dict(logistic=16, lorenz=32)
N_EPOCHS = dict(logistic=400, lorenz=400)

RESULTS_PATH = "experiments/main_study_results/nn_koopman_results.json"


def _confirmation_offset(seed):
    return seed + 10_000


def _generate_data(family, regime_args, seed, noise_frac):
    """Verbatim copy of the logistic/lorenz branches of
    experiments/main_study_tier_b.py's _generate_tier_b_data, so the NN arm
    sees identical trajectories to the EDMD arm for the same (family,
    regime_args, seed, noise_frac).
    """
    rng = np.random.default_rng(seed)
    conf_rng = np.random.default_rng(_confirmation_offset(seed))

    if family == "logistic":
        r = regime_args
        x0 = 0.1 + 0.3 * rng.random()
        n_steps = MEDIUM["logistic_n_steps"]
        traj = logistic_trajectory(x0, r=r, n_steps=n_steps, transient=500)
        if noise_frac > 0:
            traj = traj + rng.normal(0, noise_frac * traj.std(), size=traj.shape)
        x0_conf = 0.1 + 0.3 * conf_rng.random()
        traj_conf = logistic_trajectory(x0_conf, r=r, n_steps=n_steps, transient=500)
        if noise_frac > 0:
            traj_conf = traj_conf + conf_rng.normal(0, noise_frac * traj_conf.std(), size=traj_conf.shape)
        return dict(dt=None, states=traj.reshape(-1, 1), states_conf=traj_conf.reshape(-1, 1))

    if family == "lorenz":
        rho = regime_args
        params = dict(sigma=10.0, rho=rho, beta=8.0 / 3.0)
        x0 = np.array([-8.0, 8.0, 27.0]) + rng.normal(0, 0.5, size=3)
        t, states = lorenz_trajectory(x0, t_span=(0, MEDIUM["lorenz_t_end"]),
                                       n_points=MEDIUM["lorenz_n_points"], params=params)
        if noise_frac > 0:
            states = states + rng.normal(0, noise_frac * states.std(axis=0), size=states.shape)
        n_discard = int(len(t) * 0.5)
        t_tail, states_tail = t[n_discard:], states[n_discard:]
        dt = t_tail[1] - t_tail[0]

        x0_conf = np.array([-8.0, 8.0, 27.0]) + conf_rng.normal(0, 0.5, size=3)
        t_conf, states_conf = lorenz_trajectory(x0_conf, t_span=(0, MEDIUM["lorenz_t_end"]),
                                                  n_points=MEDIUM["lorenz_n_points"], params=params)
        if noise_frac > 0:
            states_conf = states_conf + conf_rng.normal(0, noise_frac * states_conf.std(axis=0), size=states_conf.shape)
        states_conf_tail = states_conf[n_discard:]

        return dict(dt=dt, states=states_tail, states_conf=states_conf_tail)

    raise ValueError(f"unknown family {family!r}")


def _vf_err(pred, true):
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-300))


def _run_one(family, regime_args, label, noise_frac, seed):
    data = _generate_data(family, regime_args, seed, noise_frac)
    dt = data["dt"] if data["dt"] is not None else 1.0

    t0 = time.time()
    model = fit_deep_koopman(
        data["states"], dt=dt,
        latent_dim=LATENT_DIM[family], hidden=HIDDEN[family],
        n_hidden_layers=1, n_epochs=N_EPOCHS[family], lr=1e-3,
        seed=seed, verbose=False,
    )
    wall = time.time() - t0

    states_conf = data["states_conf"]
    X0, X1 = states_conf[:-1], states_conf[1:]
    pred1 = model.predict_state(X0)
    one_step_err = _vf_err(pred1, X1)

    return dict(
        family=family, regime=label, noise_frac=noise_frac, seed=seed,
        latent_dim=LATENT_DIM[family], hidden=HIDDEN[family], n_epochs=N_EPOCHS[family],
        one_step_rel_rms_err=one_step_err,
        final_train_loss=model.final_train_loss,
        train_loss_first=model.train_loss_history[0],
        train_loss_diverged=bool(not np.isfinite(model.final_train_loss)
                                  or model.final_train_loss > 10 * model.train_loss_history[0]),
        wall_clock_s=wall,
    )


def main():
    results = []
    t_start = time.time()
    for family, regime_args, label in TIER_B_ITEMS:
        for noise_frac in NOISE_LEVELS:
            for seed in SEEDS:
                rec = _run_one(family, regime_args, label, noise_frac, seed)
                results.append(rec)
                print(f"{family:10s} {label:20s} noise={noise_frac:.1%} seed={seed}: "
                      f"one_step_err={rec['one_step_rel_rms_err']:.4g} "
                      f"(wall={rec['wall_clock_s']:.1f}s, final_loss={rec['final_train_loss']:.4g}"
                      f"{' DIVERGED' if rec['train_loss_diverged'] else ''})")
    total_wall = time.time() - t_start
    print(f"\nTotal wall clock: {total_wall:.1f}s for {len(results)} fits")

    with open(RESULTS_PATH, "w") as f:
        json.dump(dict(results=results, total_wall_clock_s=total_wall,
                        n_fits=len(results), seeds=SEEDS, noise_levels=NOISE_LEVELS,
                        latent_dim=LATENT_DIM, hidden=HIDDEN, n_epochs=N_EPOCHS), f, indent=2)
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
