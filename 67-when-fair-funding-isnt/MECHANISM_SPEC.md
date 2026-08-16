# Public-project mechanism specification

## Frozen domain

There are `n=3` agents. Each private value is an integer in `{0,1,2}` for a binary public project. The project has known cost `c`; the main frontier evaluates `c=1,...,6`. A deterministic direct mechanism maps reports to `q∈{0,1}` and payments `p_i`, where positive payment is paid by the agent. Utility is `u_i(v,r)=v_i q(r)-p_i(r)`.

The exhaustive class is every anonymous monotone Boolean allocation rule. An anonymous state is a sorted report vector. Monotonicity is coordinatewise: increasing any report cannot change `q=1` to `q=0`. The three-agent, three-level domain has 10 anonymous states and exactly 16 monotone rules; all 16 are enumerated, not sampled.

Payments are the normalized discrete critical values: if agent `i` is selected, `p_i` is the smallest report at which selection occurs with other reports fixed; otherwise `p_i=0`. This is the standard no-subsidy normalization for a single-parameter binary allocation rule. It gives DSIC and ex-post IR by construction, and the verifier checks them independently anyway.

## Acceptance predicate

A mechanism is accepted when, on every profile, it is feasible, DSIC against every report in `{0,1,2}`, ex-post IR, anonymous, and weakly budget balanced (`Σ_i p_i ≥ c` when `q=1`; no payments when `q=0`). A rule must also select the project at the all-2 profile so that the frontier excludes the trivial never-build mechanism.

## Objectives and comparators

The primary objective is worst-case additive social-welfare regret relative to the efficient rule `q*=1{Σv_i≥c}`; secondary outcomes are expected welfare under the uniform finite distribution, project rate, and maximum revenue. The efficient allocation with critical payments is a DSIC/IR comparator and a deliberate budget-balance counterexample. A seeded sum-threshold proposal loop is a discovery probe, never a proof of coverage. A separate checker replays serialized tables without importing the primary verifier.
