# Verification report

Primary verification is in `src/mechanism_discovery/public_project.py`; it
checks every profile, unilateral report, payment, and anonymity permutation.
`public_project_independent.py` reconstructs the table and critical payments
without importing the primary verifier.

The preregistered run enumerates 16 rules; the exploratory extension enumerates
32 and 64 rules for four and five agents. The standalone replay accepts all 74
serialized rows (`cross_n_failure_count=0`, cross-agent digest
`a04706cd4d754debd5847529e3b3ebe22a14de45efa9b94db8edfd91823a9cc8`); the
original four cost-3 rows retain digest
`4ad7d6cfc60828e3b042a72cac79d0e4ce3737d1b3579aa9fa5bc612ca0361e1`.
The efficient comparator has a budget witness at `(0,2,2)` with payments
`(0,1,1)` and cost 3. Held-out checks cover all 64 profiles for each threshold
1–6 and report 207 failures.

For the frozen rational-ReLU benchmark, source-direct and compiler-lowered
certificate extrema match on all six cases. The independent Z3 verifier checks
three strict counterexample predicates for each source (budget slack below its
certificate, ratio below its minimum, and ratio above its maximum); all 18 are
`unsat`. It also evaluates each serialized rational extremum witness directly.
The resulting artifact is `artifacts/relu_benchmark_z3_certificate.json`.

Phase VI uses the certified baseline minimum slack to derive the smallest
uniform per-term output-bias repair. The direct source route agrees exactly
with the compiler certificate on all seven repaired sources; for every
positive repair, half the offset remains deficit-producing at the baseline
witness. `scripts/verify_uniform_repair_z3.py` encodes the repaired sources
over exact reals and returns `unsat` for all seven strict-negative-slack
queries. The artifact is `artifacts/uniform_repair_z3_certificate.json`.

Phase VII independently replays minimum truthful utility directly from source
coefficients and issues 28 exact-real strict-lower-bound Z3 queries, one per
source and omitted report. All are `unsat`; seven separate negative-utility
queries match the compiled conclusion that every repaired source fails IR.

Phase X post-hoc coalition robustness adds bounded-group deviation replay on the
finite public-project class used in Phase VIII:

- `scripts/run_public_project_coalition_frontier.py` writes
  `artifacts/public_project_coalition_frontier.json` and
  `scripts/verify_public_project_coalition_frontier.py` writes
  `artifacts/public_project_coalition_frontier_certificate.json` with 6
  rejected cap-2 rows, 0 rejected survivors, matching frozen names
  (`anonymous_monotone_mask_512`, `anonymous_monotone_mask_960`), and
  independent digest
  `495ec29b412247f4278f6bc2493c0eabe6dc76e421c21a24e835c8986aea239d`.
- `scripts/run_public_project_coalition_scaling.py` and
  `scripts/verify_public_project_coalition_scaling.py` cover `n=3..5` with
  `max_coalition_size=3`, no independent failures, and independent digest
  `18c3aec976f666e62af712a057000588586fc9fefc606affc1ed546d57a1009e`.
- `scripts/run_public_project_coalition_scaling_extended.py` and
  `scripts/verify_public_project_coalition_scaling_extended.py` cover `n=3..6`
  with `max_coalition_size=3`, 34 selected checks, and independent digest
  `5fb957dc1ef2fa327800d6072139f3d89fe0aa2df06a6f5beca23cd4dd2aedf1`.

- `scripts/run_public_project_coalition_value3_frontier.py` and
  `scripts/verify_public_project_coalition_value3_frontier.py` extend the
  same coalition filter to `n=3`, `max_value=3`, `cost=1..9` with cap up to
  3, checking selected costs `1,3,9`, and independent digest
  `cfe838d49e8814069f0c6e08da310cedba1df6f6d492a234c775ae5de34e41c1`.

Phase X independent baseline audit: `scripts/run_public_project_coalition_baseline_audit.py`
and `scripts/verify_public_project_coalition_baseline_audit.py` test the
canonical efficient/pivotal mechanism (welfare-maximizing sum-threshold
decision, critical-value payment) against the same bounded-coalition bar used
above, across all four prior domains (75 `(domain, n, cost)` rows). This is
decoupled from that mechanism's separately documented weak-budget-balance
deficit: fragility is judged strictly on the `dsic`/`coalitional_dsic`
verifier fields, never the bundled `accepted` flag. Result: the baseline is
single-agent DSIC everywhere tested but fails at coalition size 2 in 66 of 75
rows. Among the 11 selected spot-check rows only one is coalition-robust
(`value3_frontier`, `n=3`, `cost=9`, no failing cap found up to 3); the other
ten fail at coalition size 2, including every selected row in `frontier`,
`scaling`, and `scaling_extended`. The independent replay checks all 75 rows
with 0 mismatches and digest
`825ded27f5889dd1a48b9ff4ae6459a4fc566dfb5919bf64bfdf345217e86f96`.

Phase X false-name manipulation supplement: `scripts/run_public_project_false_name_audit.py`
and `scripts/verify_public_project_false_name_audit.py` test a different attack
against the same canonical efficient/pivotal mechanism: a single real agent
fabricating extra fake report identities, instead of a coalition of distinct
real agents. Because the sum-threshold rule with critical-value payments is
defined identically for any agent count, the check compares the mechanism at
`n_real` real agents (baseline, truthful) against the same rule at
`n_real+fake_budget` agents (attacker controls one real slot plus
`fake_budget` fake slots, other `n_real-1` real agents held truthful), across
`n_real=3,4,5` and `fake_budget in {0,1,2}` (72 rows). `fake_budget=0` is a
positive control and shows 0 manipulable rows everywhere, confirming the
harness reduces to ordinary single-agent DSIC when there is no attack. At
`fake_budget=1` or `2`, 48 of 72 rows are manipulable, and 6 of 9 selected
spot checks are manipulable (every selected row with `fake_budget>=1`). A
concrete witness at `n_real=3`, `cost=3`: truthful profile `(0,1,1)` does not
build the project (attacker utility 0); if the true-value-1 agent reports
`2` in their own slot and fabricates one fake identity also reporting `2`,
the extended 4-agent profile `(0,1,2,2)` builds, and both the real and fake
slot's critical-value payment computes to 0 (each slot's threshold, holding
the other inflated slot fixed, is already met at report 0), so the attacker
nets a utility gain of 1 for free. The independent verifier recomputes the
sum-threshold/critical-value rule from its closed-form definition without
importing `public_project.py`, and reproduces every row's manipulable count
with 0 mismatches and digest
`9e2e5ca254dfde9b2c3987f6cf35973e836fc15e429213a5356dcb3bd1ea69d3`.

Phase X general coalition lemma (analytical, not search-based):
`scripts/verify_public_project_coalition_lemma.py` proves a closed-form
sufficient condition for coalition-manipulability of the sum-threshold/
critical-value mechanism: for every integer `n>=2`, `max_value=m>=1`, and
`cost c<=(n-1)*m`, the grand-coalition deviation "every agent reports `m`"
builds the project and forces payment `0` for every agent, because the other
`n-1` agents' reports alone already sum to `(n-1)*m>=c`. This is a proof by
direct construction, holding for every `(n,m,c)` satisfying the bound, not
only the five domains searched above. The script cross-checks the condition
against all 75 rows of `artifacts/public_project_coalition_baseline_audit.json`:
0 false positives (every row the condition predicts fragile is fragile in the
search data) and 0 mismatches between the predicted zero-payment construction
and direct simulation. It also settles the baseline audit's single robust
exception (`value3_frontier`, `n=3`, `cost=9=n*m`): at `cost=n*m`, the
`(n-1)*m` bound is never met, so the construction cannot apply; a separate
argument shows any proper-coalition deviation that reaches `cost=n*m` forces
outsiders to have true value exactly `m` (else the coalition cannot reach the
threshold with reports capped at `m`) and forces every coalition member's
report to `m`, at which point each member's critical-value payment equals `m`
exactly — their own full report — eliminating any possible gain. This is
checked by exhaustive enumeration over all truthful profiles and all proper
coalition subsets for `n=3,4` and `m=2,3` (`0` counterexamples). Digest
`fc030d4f60bc63f815224c070a4626b579f6c0a6efe8c91e079739d27a760db5`.

Phase X complete coalition characterization (analytical, not search-based):
`scripts/verify_public_project_coalition_characterization.py` closes the gap
left by the Section-4.11-equivalent sufficient condition above. It derives
two closed-form facts by convexity of the critical-value payment: (1) the
minimum total payment a coalition of size `k` can achieve against outsider
true-value sum `S_O` is exactly
`k*max(0,(cost-S_O)-(k-1)*max_value)`, attained by every member reporting
`max_value`; (2) the truthful payment that is hardest to beat, for a fixed
coalition value-sum, comes from the extremal "bang-bang" distribution (as
many members as possible at `max_value`, one remainder, the rest `0`).
Combining both gives an existence check for coalition-cap-`k`
manipulability via a bounded sweep over integer value sums, not a search
over reports or profiles. Cross-checked against all 75 rows of
`artifacts/public_project_coalition_baseline_audit.json`, it reproduces the
searched `min_failing_coalition_size` exactly on every row
(`rows_checked=75 exact_matches=75 mismatches=0`) -- an exact match, not
merely a no-false-positive bound like the Section-4.11 sufficient condition,
closing the 14-row gap that condition left unexplained. Because it never
enumerates reports, it also evaluates at agent counts unreachable by brute
force (e.g. `n=20`, `max_value=8` in microseconds, versus `9^20` infeasible
profiles). Digest
`9dd70ad48733e6be95cd8ff4b0f37e5638e7347ccb7d044ab0a2adf80ebe7be0`.

Phase X extended cross-check and gain sizes:
`scripts/verify_public_project_coalition_characterization_extended.py` runs
the primary verifier's own brute-force coalition search (`max_coalition_size=3`)
on 70 `(n,max_value,cost)` cells the original 75-row baseline audit never
covered -- `max_value in {3,4,5}` at every previously tested agent count
(`n=3,4,5`) -- and compares against the closed-form formula's prediction:
`0` mismatches, extending the total independently brute-force-verified row
count from 75 to 145. Of the 70 new cells, 65 are fragile; the script also
computes, from the closed-form formula, the exact size of the free value gain
the optimal coalition captures at each fragile cell (up to `5` value units at
`n=3`, `max_value=5`, cost 5-9). A separate, explicitly formula-only sweep
(not independently re-verified by brute force, since enumerating reports at
these agent counts is combinatorially infeasible) evaluates 252
`(n,max_value,cost)` triples up to `n=40`, `max_value=15` on a coarse cost
grid and finds `99.2%` already manipulable by a coalition of size 2, with
gains up to `15` value units. Digest
`c10e6217b03415820f95c65fbfbab7dc159796c8099441935961ec5e796e5f48`.

All ten scripts are fully replayable from frozen config files in
`configs/public_project_coalition_*.json` and
`configs/public_project_false_name_audit.json`; the lemma and characterization
scripts need no config of their own since they are direct proofs
cross-checked against the existing baseline-audit artifact (the extended
characterization script instead calls the primary verifier directly on new
cells rather than reading a config file).
