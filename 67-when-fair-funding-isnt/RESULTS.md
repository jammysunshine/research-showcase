# Results

The preregistered three-agent class contains 16 anonymous monotone rules. At
costs `1..6`, accepted counts are `4,4,4,1,1,1`; best worst-case regrets are
`3,2,1,1,0,0`. The four cost-3 rows all pass the independent checker.

The exact finite cross-check enumerates 16, 32, and 64 rules for `n=3,4,5`.
Across costs `1..2n`, accepted counts are:

| agents | accepted counts by cost |
|---:|---|
| 3 | 4,4,4,1,1,1 |
| 4 | 5,5,5,5,1,1,1,1 |
| 5 | 6,6,6,6,6,1,1,1,1,1 |

All 74 serialized accepted rows pass the standalone checker. The all-agent
theorem below explains these counts in the declared model. The efficient
critical-payment rule still fails budget balance at `(0,2,2)` (payments
`(0,1,1)` against cost 3). The held-out `{0,1,2,3}` stress audit records 207
failures for the efficient threshold family, preserving the generalization
boundary.

The harder six-agent extension enumerates 128 rules over 28 sorted states at
each cost `1..12`. Accepted counts are `7,7,7,7,7,7,1,1,1,1,1,1`; all 48
serialized accepted rows pass independent replay. The complete run took
56.394 seconds and recorded 29,474,816 bytes peak resident memory on Darwin.
This is a finite computational cross-check of the all-agent theorem; its
runtime and memory remain useful reproducibility measurements.

## Finite value-lattice theorem

For every `n>=1`, integer value cap `m>=1`, and cost `c`, within the declared
deterministic anonymous coordinatewise-monotone class with normalized critical
payments and required build at the all-`m` profile, put `k=ceil(c/n)`. There
is no accepted rule if `k>m`; otherwise accepted rules are exactly the
nonempty upward-closed subsets of the sorted restricted lattice
`{k,...,m}^n`. `m=2` gives the prior suffix count: `n+1` rules through cost
`n`, one through cost `2n`, and none above it. The proof is in
`PUBLIC_PROJECT_THEOREM.md`.

The theorem was frozen before a separate full-domain confirmation at
`n=3,m=4,c=1..12`. Predicted and exhaustive accepted rule sets were identical
at every cost; counts are `65,65,65,15,15,15,4,4,4,1,1,1`. All 255 serialized
accepted rules pass the independent checker with zero failures. The run took
24.00 seconds and 45,842,432 bytes peak resident memory. The immutable input
and generated record are `configs/phase_viii_value_lattice_theorem.json` and
`artifacts/phase_viii_value_lattice_theorem.json`.

The earlier post-hoc `m=3` run (20 sorted states, 66 anonymous monotone rules)
is retained as an exploratory precursor. Its counts over costs `1..9` were
`15,15,15,4,4,4,1,1,1`; all 60 serialized accepted rows passed independent
replay. The newer preregistered `m=4` test—not that precursor—is the theorem
confirmation.

Artifacts: `artifacts/public_project_study.json`,
`artifacts/public_project_certificate.json`,
`artifacts/public_project_scaling.csv`,
`artifacts/public_project_frontier.csv`, and `reports/public_project_frontier.svg`.
The value-lattice extension is in `artifacts/public_project_value_extension.json`.
The six-agent extension is in `artifacts/public_project_n6_extension.json`.

## Executable-formula certification extension

The separately preregistered IJCAI-2019 Equation (2) positive control was
evaluated over the continuous ordered three-agent cube using 23 exact
piecewise-affine arrangement vertices. Primary and independent evaluators
agree on charge ratios `[2, 7/3]`, hence efficiency `2/3`. It differs from the
AAAI-2024 printed three-agent formula at `(0,1/3,1/3)`, where the total charges
are `2` and `19/9`. This establishes the audit method can distinguish two
published rules with the same headline efficiency; it does not identify a new
mechanism. The artifact is
`artifacts/guo_2019_three_agent_optimal_audit.json`.

The PRIMA-2016 Equation (3) redistribution rule is now also certified on the
continuous ordered three-agent cube. To preserve its source convention, the
typed certificate uses `sum externalities - total redistribution`; subtracting
twice first-best cost gives exactly VCG revenue minus redistribution. All 23
certificate vertices agree with the separately written source evaluator. The
minimum slack is `64/81`; the exact minimum retained-efficiency expression is
`-47/162` at `(0,0,1)`. This is a continuous strengthening of the existing
grid audit, not a refutation of the source's asymptotic competitive claim.

## Frozen rational-ReLU cross-check

Six preregistered deterministic rational one-hidden-layer ReLU fixtures cover
three development and three confirmation cases with three to five agents and
widths two or three. Direct source evaluation and compiler-lowered certificates
match exactly. Their `(candidate bases, feasible vertices)` pairs are `(165,6)`,
`(3060,14)`, `(6188,49)`, `(364,23)`, `(3060,8)`, and `(792,19)`.
An independent exact-real Z3 encoding returns `unsat` for all 18 strict queries
that attempt to improve recorded slack or ratio bounds; every reported rational
witness re-evaluates exactly. Artifacts: `artifacts/relu_benchmark_results.json`
and `artifacts/relu_benchmark_z3_certificate.json`. This is a bounded source
semantic check, not a discovered mechanism or general neural verification.

## Exact uniform-repair synthesis

For a source used once per omitted agent, increasing its output bias by
`delta` increases total charge, and hence budget slack, by exactly `n*delta`
at every report profile. The frozen seven-source study therefore synthesizes
the smallest uniform nonnegative repair as `max(0,-s/n)` from baseline slack
`s`. The printed four-agent decimal control gives `1/20000`, reproducing the
published repair. Of the six frozen fixtures, five require positive offsets
(`178/49`, `144/49`, `190/49`, `25/7`, and `23/7`) and one has zero slack and
zero offset. All repaired slacks are exactly zero; each positive half-offset
still fails at the original slack witness. Direct-source and compiled
certificates agree exactly, and seven Z3 exact-real no-deficit challenges are
`unsat`. The large synthetic offsets are an important negative result: this
does not demonstrate an economically good repair, only a sound exact repair
primitive within a deliberately narrow scalar family. Artifacts:
`artifacts/uniform_repair_study.json` and
`artifacts/uniform_repair_z3_certificate.json`.

## Exact budget--IR trade-off

For an efficient Groves rule, truthful utility is `S-h`. A uniform output-bias
repair by `delta` therefore reduces every utility by exactly `delta`. The
Phase-VII frozen seven-source corpus has no repaired source satisfying ex-post
IR. The displayed four-agent control changes from `-833/2500` to `-1333/4000`;
the four initially nonnegative synthetic margins are each smaller than their
required offset. Direct source replay matches the compiler minima, 28 Z3
strict-lower-bound queries are `unsat`, and seven Z3 IR queries agree. See
`artifacts/repair_ir_tradeoff_study.json` and
`artifacts/repair_ir_tradeoff_z3_certificate.json`.
