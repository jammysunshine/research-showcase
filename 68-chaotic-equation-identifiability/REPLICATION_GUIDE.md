# Replication Guide

How to reproduce the key results in this repository.

## Prerequisites

- Python 3.14+ (tested on macOS arm64)
- ~10 GB disk for result files
- ~4 GB RAM peak (Tier B with 6 workers)
- Local CPU only; no cloud or GPU required

## Setup

```bash
# Clone and set up environment
cd 68-chaotic-equation-identifiability
python3 -m venv .venv
.venv/bin/pip install numpy scipy sympy matplotlib pysindy pytest gplearn scikit-learn torch
```

For the PySR cross-check (extension 2), a separate environment is needed:
```bash
python3 -m venv .venv_pysr311
.venv_pysr311/bin/pip install pysr numpy scipy
# Julia backend (SymbolicRegression.jl) must be installed separately
```

For the pykoopman cross-check:
```bash
python3 -m venv .venv_pykoopman311
.venv_pykoopman311/bin/pip install pykoopman numpy scipy
```

## Running Tests (verify environment)

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v
# Expected: 21/21 passed (or more as tests are added)
```

## Reproducing Individual Tiers

### Pilot (exploratory, not frozen)
```bash
PYTHONPATH=. .venv/bin/python experiments/pilot_chaos_vs_periodic.py
# Output: experiments/pilot_chaos_vs_periodic_results.json
```

### Tier A (SINDy grid, 132 conditions)
```bash
PYTHONPATH=. .venv/bin/python experiments/main_study.py
# Output: experiments/main_study_results/tier_a_results.json, tier_a_manifest.json
# Wall-clock: ~1000s on a single core
```

### Tier B (3 methods × 7 pairs, 210 conditions)
```bash
PYTHONPATH=. .venv/bin/python experiments/main_study_tier_b.py
# Output: experiments/main_study_results/tier_b_results.json, tier_b_manifest.json
# Wall-clock: ~6500s across 6 workers (N_WORKERS=6 in script)
# Checkpointed: delete tier_b_checkpoint.jsonl to rerun from scratch
```

### Tier C (delay-embedded partial observation, 210 conditions)
```bash
PYTHONPATH=. .venv/bin/python experiments/main_study_tier_c.py
# Output: experiments/main_study_results/tier_c_results.json
PYTHONPATH=. .venv/bin/python experiments/main_study_tier_c_noise5.py
# Output: experiments/main_study_results/tier_c_noise5_results.json
# Wall-clock: ~6s total (SINDy on scalar series is cheap)
```

### Rössler confirmation (80 fits, run once)
```bash
PYTHONPATH=. .venv/bin/python experiments/main_study_confirmation.py
# Output: experiments/main_study_results/confirmation_results.json, confirmation_manifest.json
# Wall-clock: ~103s
```

### Lorenz ρ=45 held-out regime (40 fits, run once)
```bash
PYTHONPATH=. .venv/bin/python experiments/main_study_confirmation_regime.py
# Output: experiments/main_study_results/confirmation_regime_results.json
# Wall-clock: ~65s
```

### Extensions

```bash
# Extension 1: Koopman Gram-matrix mechanistic analysis
PYTHONPATH=. .venv/bin/python experiments/koopman_gram_matrix_analysis.py

# Extension 2: PySR cross-check (requires .venv_pysr311)
PYTHONPATH=. .venv_pysr311/bin/python experiments/pysr_crosscheck_duffing.py

# Extension 3: Neural-network Koopman
PYTHONPATH=. .venv/bin/python experiments/nn_koopman_tier_b.py

# Extension 4: Lorenz-96 (N=6)
PYTHONPATH=. .venv/bin/python experiments/lorenz96_pipeline.py

# Koopman cross-check against pykoopman
PYTHONPATH=. .venv/bin/python experiments/koopman_crosscheck_generate.py
PYTHONPATH=. .venv_pykoopman311/bin/python experiments/koopman_crosscheck_pykoopman.py
```

### Smoke Tests (quick validation, ~seconds each)
```bash
PYTHONPATH=. .venv/bin/python scripts/run_sindy_smoke_test.py
PYTHONPATH=. .venv/bin/python scripts/run_symbolic_regression_smoke_test.py
PYTHONPATH=. .venv/bin/python scripts/run_koopman_smoke_test.py
PYTHONPATH=. .venv/bin/python scripts/run_rossler_blinding_smoke_test.py
```

### Counterexamples
```bash
PYTHONPATH=. .venv/bin/python src/counterexamples_first_integral.py
PYTHONPATH=. .venv/bin/python src/counterexamples_conjugacy.py      # REFUTED — preserved as failed attempt
PYTHONPATH=. .venv/bin/python src/counterexamples_conjugacy_v2.py    # CONFIRMED (generic finite-data effect)
```

## Key Result Files

| File | Contents |
|---|---|
| `experiments/main_study_results/tier_a_results.json` | Tier A: 132 SINDy conditions |
| `experiments/main_study_results/tier_b_results.json` | Tier B: 210 cross-method conditions |
| `experiments/main_study_results/tier_b_koopman_normalization_results.json` | Lyapunov-normalization analysis |
| `experiments/main_study_results/tier_c_results.json` | Tier C: partial observation |
| `experiments/main_study_results/tier_c_noise5_results.json` | Tier C noise=5% extension |
| `experiments/main_study_results/confirmation_results.json` | Rössler held-out |
| `experiments/main_study_results/confirmation_regime_results.json` | Lorenz ρ=45 held-out |
| `experiments/main_study_results/koopman_gram_matrix_results.json` | Gram-matrix mechanistic theory |
| `experiments/main_study_results/nn_koopman_results.json` | Neural-network Koopman |
| `experiments/main_study_results/lorenz96_results.json` | Lorenz-96 (N=6) |
| `experiments/main_study_results/pysr_crosscheck_duffing_results.json` | PySR cross-check |

## Known Nondeterminism

- **gplearn with n_jobs=-1:** parallel worker scheduling causes non-reproducible y-dimension R^2 across runs (e.g., 0.9697 vs 0.957). Fixed in the smoke test by pinning n_jobs=1; frozen-tier runs also use n_jobs=1.
- **PySR/Julia:** JIT compilation timing varies; results are deterministic given identical random seeds.
- **NN Koopman:** gradient-descent training has minor seed-dependent variance; 3 seeds used, not 5.

## Troubleshooting

- **gplearn install fails on Python 3.14:** may need `pip install --no-build-isolation gplearn` or a Python 3.11/3.12 environment.
- **Tier B hangs:** the 60s-per-job timeout (added post-stall) should prevent infinite loops; delete `tier_b_checkpoint.jsonl` and rerun to resume.
- **Memory pressure on 16GB Mac:** Tier B's 6-worker parallelism peaks around 3–4 GB; reduce N_WORKERS in the script if needed.
