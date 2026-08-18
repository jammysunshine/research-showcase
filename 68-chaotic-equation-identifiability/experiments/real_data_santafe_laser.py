"""Real-world data transfer check (NEXT_STEPS.md Tier-3 item #9,
PROJECT_CHARTER.md's 2026-08-18 amendment permitting ONE bounded real-world
dataset for this specific check).

Dataset: Santa Fe Time Series Competition Dataset A -- a scalar intensity
time series from a real NH3 far-infrared laser experiment, physically
documented as exhibiting "Lorenz-like chaos" (Hübner, Weiss, Abraham, Tang
1994 -- see SOURCES.json). `data/santafe_laser/A.dat` (1000 points, the
original competition training series) and `A.cont` (9093 further points
from the same physical run, downloaded alongside A.dat -- see
`scripts/download_santafe_laser.py`). There is NO known governing ODE for
this system (a physical laser, not a synthetic fixture), so coefficient
recovery is not evaluable -- see DECISION_LOG.md for why this is treated as
a purely structural/qualitative transfer check, not a hypothesis test.

Pipeline (adapted from `main_study_tier_c.py`'s single-coordinate delay-
embedded SINDy approach, since real data here is exactly the single-scalar-
series case Tier C's design already targets):
  1. Normalize both series (z-score against A.dat's own mean/std) --
     necessary because STLSQ_THRESHOLD=0.1 (this project's pinned default,
     `main_study_tier_b.STLSQ_THRESHOLD`) is calibrated against synthetic
     trajectories whose typical magnitude is O(1-30); the raw laser
     intensity units (tens to hundreds) would make that threshold
     meaningless without rescaling. This is a real preprocessing decision,
     disclosed rather than silently applied.
  2. Delay-embed at dim=3 -- NOT an arbitrary choice: the physical NH3-FIR
     laser is modeled by the Lorenz-Haken equations, a genuine 3-variable
     (field, polarization, population inversion) system isomorphic to the
     Lorenz equations (this is literally why Hübner et al. call it
     "Lorenz-like" -- see SOURCES.json summary), so dim=3 is the
     domain-motivated embedding dimension, matching this project's own
     EMBEDDING_DIM["lorenz"]=3 convention in `main_study_tier_c.py`.
  3. tau chosen via the same `choose_delay_by_autocorrelation` heuristic
     Tier C already uses, computed on the normalized A.dat series.
  4. Fit a continuous-time SINDy model (pysindy `PolynomialLibrary(degree)`,
     `STLSQ(threshold=0.1)`, default finite-difference derivative
     estimation) with dt=1.0 (an arbitrary unit-spaced time axis, since the
     laser's true physical sampling interval is not documented in the
     archived files) -- IDENTICAL code path to Tier C's ODE-family branch
     (`main_study_tier_c._run_tier_c_sindy`'s `else` clause). A discrete-
     time one-step-map fit (`pysindy`'s `discrete_time=True` mode, matching
     the logistic map's treatment elsewhere in this project) was the
     original plan and is arguably better-conditioned for noisy real data,
     but was DROPPED after discovering the installed `pysindy==2.1.0`'s
     `SINDy.__init__` no longer accepts a `discrete_time` argument at all
     (removed/refactored in this version) -- pinning down or vendoring an
     older pysindy solely for this one Tier-3 extension was judged not
     worth the reproducibility risk to the rest of the project's frozen
     pysindy-dependent results, so the continuous-time/dt=1 path (already
     proven to work via Tier C) was used instead. Disclosed as a real
     implementation deviation from the plan, not a silent substitution.
  5. Evaluate one-step-from-ground-truth prediction error (Euler-forward:
     `Y[k] + dt * model.predict(Y[k])` compared against the TRUE `Y[k+1]`
     at every step, matching Tier C's/Tier B's Koopman one-step convention
     exactly -- avoids compounding chaotic divergence into what should be
     a measure of local model fit) and a 50-step free rollout (does it
     stay bounded or blow up/collapse) on the held-out A.cont confirmation
     series, embedded with the SAME tau chosen from A.dat.
  6. Compare resulting one-step error magnitude against this project's own
     synthetic Tier C numbers for the matched Lorenz pair (classic_chaotic
     vs. stable_fixed_point, both already delay-embedded single-coordinate
     SINDy fits at the same STLSQ threshold and comparable noise levels).

No `dynamically_distinct` pass/fail gate is computed here (that gate
requires a KNOWN true right-hand side to compare against, which does not
exist for real data) -- this script reports raw one-step/rollout error
magnitudes only, for qualitative/structural comparison, per NEXT_STEPS.md
item #9's own instructions ("report structural/qualitative findings
instead").
"""
import json
import os

import numpy as np
import pysindy as ps

from src.delay_embedding import choose_delay_by_autocorrelation, delay_embed

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "santafe_laser")
RESULTS_PATH = "experiments/main_study_results/real_data_santafe_laser_results.json"

EMBEDDING_DIM = 3  # domain-motivated: Lorenz-Haken laser model, see module docstring
STLSQ_THRESHOLD = 0.1  # matches main_study_tier_b.STLSQ_THRESHOLD
DEGREES = [2, 3]
ROLLOUT_HORIZON = 50
N_CONF_POINTS = 2000  # truncate A.cont to keep confirmation length comparable to Tier B/C trajectories


def _vf_err(pred, true):
    scale = max(float(np.abs(true).max()), 1e-12)
    return float(np.sqrt(np.mean((pred - true) ** 2)) / scale)


def _load_series():
    train_raw = np.loadtxt(os.path.join(DATA_DIR, "A.dat"))
    cont_raw = np.loadtxt(os.path.join(DATA_DIR, "A.cont"))
    mu, sigma = train_raw.mean(), train_raw.std()
    train = (train_raw - mu) / sigma
    cont = (cont_raw - mu) / sigma  # normalized against A.dat's own stats, not its own
    return train, cont, dict(mu=float(mu), sigma=float(sigma),
                              n_train_raw=len(train_raw), n_cont_raw=len(cont_raw))


def _fit_and_eval(degree, x_train, x_conf):
    tau = choose_delay_by_autocorrelation(x_train)
    Y_train = delay_embed(x_train, EMBEDDING_DIM, tau)
    Y_conf_full = delay_embed(x_conf, EMBEDDING_DIM, tau)
    Y_conf = Y_conf_full[:N_CONF_POINTS] if len(Y_conf_full) > N_CONF_POINTS else Y_conf_full

    dt = 1.0  # arbitrary unit-spaced time axis; see module docstring
    feature_names = [f"y{i}" for i in range(EMBEDDING_DIM)]
    model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=degree),
                      optimizer=ps.STLSQ(threshold=STLSQ_THRESHOLD))
    model.fit(Y_train, t=dt, feature_names=feature_names)

    # One-step-from-ground-truth (Euler-forward from the TRUE Y[k]),
    # matching main_study_tier_c._run_tier_c_sindy's ODE-family convention.
    pred_next = Y_conf[:-1] + dt * model.predict(Y_conf[:-1])
    one_step_err = _vf_err(pred_next, Y_conf[1:])

    # 50-step free rollout from the confirmation series' own first point.
    # Manual bounded Euler stepping, NOT model.simulate() (which calls
    # scipy solve_ivp internally) -- a diverging/stiff polynomial fit on
    # noisy real data can make solve_ivp take adaptive steps that shrink
    # toward zero without ever reaching the horizon, hanging indefinitely
    # (observed in practice during development: a first attempt using
    # model.simulate() had to be killed after exceeding a 120s budget with
    # no output at all). Manual Euler with an explicit blow-up check is
    # slower to converge in principle but cannot hang, and it directly
    # answers this item's own question ("does it stay bounded"), which is
    # exactly what an adaptive integrator silently struggling to resolve a
    # blow-up would obscure.
    dt_step = 0.1  # sub-steps of dt=1.0 for stability, matches STLSQ_THRESHOLD-scale fits
    horizon = min(ROLLOUT_HORIZON, len(Y_conf) - 1)
    rollout = np.zeros((horizon + 1, EMBEDDING_DIM))
    rollout[0] = Y_conf[0]
    blew_up = False
    for i in range(horizon):
        y = rollout[i].copy()
        for _ in range(int(round(dt / dt_step))):
            deriv = model.predict(y.reshape(1, -1))[0]
            y = y + dt_step * deriv
            if not np.all(np.isfinite(y)) or np.abs(y).max() > 1e6:
                blew_up = True
                break
        if blew_up:
            rollout[i + 1:] = np.nan
            break
        rollout[i + 1] = y
    if blew_up:
        rollout_err = float("inf")
    else:
        rollout_err = _vf_err(rollout, Y_conf[:horizon + 1])

    coeffs = model.coefficients()
    n_nonzero = int(np.sum(np.abs(coeffs) > 1e-10))
    n_total = coeffs.size

    return dict(
        degree=degree, tau=int(tau), embedding_dim=EMBEDDING_DIM,
        n_train_embedded=int(Y_train.shape[0]), n_conf_embedded=int(Y_conf.shape[0]),
        one_step_rel_rms_err=one_step_err,
        rollout_horizon=horizon, rollout_blew_up=bool(blew_up),
        rollout_rel_rms_err=rollout_err,
        n_nonzero_coeffs=n_nonzero, n_total_coeffs=n_total,
        sparsity_frac=float(n_nonzero / n_total),
    )


def main():
    os.makedirs("experiments/main_study_results", exist_ok=True)
    x_train, x_conf, meta = _load_series()
    print(f"Loaded: {meta}")

    records = []
    for degree in DEGREES:
        rec = _fit_and_eval(degree, x_train, x_conf)
        records.append(rec)
        print(f"degree={degree}: tau={rec['tau']} one_step_err={rec['one_step_rel_rms_err']:.4g} "
              f"rollout_err={rec['rollout_rel_rms_err']:.4g} blew_up={rec['rollout_blew_up']} "
              f"sparsity={rec['n_nonzero_coeffs']}/{rec['n_total_coeffs']}", flush=True)

    metadata = dict(
        dataset="Santa Fe Time Series Competition Dataset A (NH3 far-infrared laser, chaotic)",
        source_files=["data/santafe_laser/A.dat", "data/santafe_laser/A.cont"],
        preprocessing=meta,
        embedding_dim=EMBEDDING_DIM, stlsq_threshold=STLSQ_THRESHOLD,
        degrees=DEGREES, rollout_horizon=ROLLOUT_HORIZON, n_conf_points_used=N_CONF_POINTS,
        comparison_note=(
            "No dynamically_distinct pass/fail gate computed -- no known true RHS for "
            "real data. Compare one_step_rel_rms_err/rollout magnitudes qualitatively "
            "against experiments/main_study_results/tier_c_results.json and "
            "tier_c_multi_coordinate_results.json's Lorenz classic_chaotic vs. "
            "stable_fixed_point one-step-error numbers at the same STLSQ_THRESHOLD=0.1, "
            "degree in {2,3} (see DECISION_LOG.md for the actual comparison)."
        ),
    )
    with open(RESULTS_PATH, "w") as f:
        json.dump(dict(metadata=metadata, records=records), f, indent=2, default=float)
    print(f"\nWrote {len(records)} records to {RESULTS_PATH}.")


if __name__ == "__main__":
    main()
