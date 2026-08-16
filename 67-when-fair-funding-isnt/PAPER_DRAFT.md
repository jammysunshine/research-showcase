# Certificate-first exact frontiers and executable audits for public-project mechanisms

### When the "fair" funding rule isn't: an exact map of how coalitions break threshold public-goods mechanisms, and by how much

## Abstract

Automated mechanism design is only scientifically useful when proposed rules
can be checked exhaustively rather than judged by sampled performance. We
develop a certificate-first study for deterministic public-project mechanisms.
Agents have integer values in a finite lattice `{0,...,m}` for a binary project
with a known cost. We enumerate every anonymous monotone allocation rule over
the sorted report states and attach normalized discrete critical payments.
Each candidate is checked for dominant-strategy incentive compatibility (DSIC),
ex-post individual rationality, feasibility, anonymity, no subsidy when the
project is absent, and weak budget balance. On the preregistered three-agent
domain, 16 rules are enumerated at each cost. The number satisfying all
constraints is `4,4,4,1,1,1` for costs `1,...,6`; at cost three the best
accepted rule has worst-case welfare regret one. An efficient sum-threshold
rule is DSIC and individually rational but fails budget balance at a concrete
profile. We prove a finite-lattice characterization in this declared class:
writing `k=ceil(c/n)`, accepted rules are exactly the nonempty upward-closed
sets inside `{k,...,m}^n`, and none exists for `c>nm`. The original ternary
suffix result is its `m=2` corollary. A preregistered untouched
`n=3,m=4,c=1..12` confirmation reproduces every accepted rule set by full
enumeration and independently replays all 255 accepted rules with zero
failures. Exact searches for three through six ternary agents independently
cross-check that corollary. A value-magnitude stress test finds
207 failures for the efficient threshold family on held-out `{0,1,2,3}`
profiles. A post-hoc exact `m=3` precursor finds 66 rules and 60 independently
replayed accepted rows. The result is a reproducible finite theorem and
falsifiable benchmark, not a theorem for
continuous values or a claim of unrestricted mechanism-design novelty. As a
separate source-integrity study, six preregistered rational one-hidden-layer
ReLU fixtures with three to five agents have identical direct-source and
compiled certificates; an exact-real Z3 cross-check returns `unsat` for all
18 strict-bound counterexample queries. This is bounded verification evidence,
not a newly discovered mechanism or a generic neural-verification result.
Finally, as a post-hoc falsification supplement, we extend the certificate to
bounded-size coalitions. Cap-2 coalition deviations shrink the cost-3 DSIC
frontier from 4 to 2 accepted rules, and the canonical efficient/pivotal
comparator itself -- single-agent DSIC by construction -- fails coalition-cap-2
DSIC in 66 of 75 audited `(domain, agent count, cost)` cells spanning
`n=3..6` on the `{0,1,2}` value domain and `n=3` on the wider `{0,1,2,3}`
value domain. All coalition claims are independently replayed with zero
mismatches. A second post-hoc supplement audits a different attack on the
same comparator: false-name manipulation, where one real agent fabricates
extra fake report identities instead of a coalition of distinct agents.
Because the sum-threshold rule is defined identically at any agent count, we
compare it at `n` real agents against the same rule at `n+f` agents for fake
budgets `f in {1,2}`. A zero-fake-budget positive control shows no
manipulable rows; at `f>=1`, 48 of 72 audited `(agent count, cost, fake
budget)` cells are manipulable across `n=3,4,5`, independently replayed with
zero mismatches from a closed-form reimplementation. We then replace part of
that search with a proof: for the canonical sum-threshold/critical-value rule,
the grand-coalition deviation "every agent reports `m`" builds the project at
zero payment to everyone whenever `cost<=(n-1)*m`, a closed-form sufficient
condition that holds at every integer `n>=2`, `m>=1`, not only the searched
cells, has zero false positives against all 75 baseline-audit rows, and
explains why `cost=n*m` is that audit's sole robust exception. That the
pivotal mechanism is not group-strategyproof is textbook knowledge; a
literature search found no prior closed-form quantification of the exact
minimum coalition size or exact free-ride gain for this rule, which is the
contribution below. We then close the remaining gap: an exact closed-form
formula for the minimum total payment
any coalition of size `k` can achieve, `k*max(0,(cost-S_O)-(k-1)*m)` where
`S_O` is the outsiders' true-value sum, derived from the convexity of the
critical-value payment in the coalition's report sum. Combined with a matching
closed-form extremal argument for the coalition's worst-case truthful payment,
this yields a full necessary-and-sufficient characterization of coalition-cap-`k`
manipulability that requires no report-level search at all -- only a bounded
sweep over integer value sums. It reproduces the baseline audit's
`min_failing_coalition_size` exactly on all 75 rows (not merely without false
positives), extends to 70 further brute-force-verified `(n,max_value,cost)`
cells with 0 mismatches (145 rows total), and evaluates in microseconds at
agent counts (e.g. `n=20`) far beyond brute-force reach.

## 1. Introduction

Public-project design exposes a basic tension: selecting the project whenever
total value covers cost is efficient, but its critical payments need not cover
the cost. Conversely, budget balance can require a more conservative allocation
rule and therefore lose welfare. Existing mechanism-design theory studies this
tension in broad quasi-linear domains, while automated mechanism-design work
shows how finite search can discover or reproduce rules. The practical gap is
an auditable benchmark in which the entire candidate class, verifier, negative
examples, and independent replay are shipped together.

This paper makes eleven deliberately narrow contributions:

1. a typed finite public-project model with exact critical payments;
2. an antichain enumerator that covers every anonymous monotone rule in the
   chosen finite domain;
3. a cost-indexed welfare frontier and explicit efficient-rule counterexample;
4. serialized certificates replayed by an implementation that does not import
   the primary mechanism code.
5. a human-checkable finite-lattice proof, with a construction independent of
   full-domain search and a preregistered full-domain confirmation.
6. an executable-specification audit layer for published shallow max-affine
   VCG-redistribution formulas, with a standalone replay from serialized
   rational expressions.
7. a preregistered source-to-certificate cross-check for rational shallow ReLU
   fixtures, independently challenged by exact-real SMT queries;
8. a post-hoc bounded-coalition falsification supplement, including an
   independent audit of whether the canonical single-agent-DSIC comparator
   itself resists coalitions, with a standalone replay of every claim;
9. a second post-hoc supplement auditing the same comparator against
   false-name manipulation (fabricated fake report identities), independently
   replayed from a closed-form reimplementation.
10. a proven, not searched, closed-form sufficient condition
    (`cost<=(n-1)*max_value`) for coalition-manipulability of the canonical
    sum-threshold/critical-value rule, holding at every integer `n,max_value`,
    cross-checked against all 75 baseline-audit rows with zero false
    positives, and used to explain that audit's sole robust exception.
11. a closed-form necessary-and-sufficient characterization of coalition-cap-`k`
    manipulability for the same rule, derived from an exact minimum-coalition-
    payment formula and an exact worst-case truthful-payment formula, both
    proved by convexity rather than searched; it reproduces the baseline
    audit's exact `min_failing_coalition_size` on all 75 rows and computes at
    agent counts far beyond brute-force enumeration.

## 2. Model

There are `n` agents with values `v_i` in `{0,...,m}` for a public project with
integer cost `c`. A direct mechanism reports `r` and chooses `q(r) in {0,1}`. Agent
`i` receives utility

`u_i(v_i,r) = v_i q(r) - p_i(r)`.

The allocation rule is anonymous and monotone in every report. For each
monotone rule we use the normalized discrete critical payment: if the project
is built, agent `i` pays the smallest report at which the project would be
built holding other reports fixed; otherwise the payment is zero. This
normalization removes arbitrary transfers from the search and makes the
candidate table sufficient to reconstruct payments.

The verifier requires:

- feasibility and no payment when `q=0`;
- DSIC against every report in the finite type set;
- ex-post IR at every truthful profile;
- anonymity under every profile permutation;
- weak budget balance, `sum_i p_i(r) >= c`, whenever `q(r)=1`;
- `q(m,...,m)=1`, excluding the vacuous never-build rule.

The primary objective is worst-case additive welfare regret relative to the
efficient allocation `q*(v)=1{sum_i v_i >= c}`:

`R(q) = max_v [max(0,sum_i v_i-c) - q(v)(sum_i v_i-c)]`.

Uniform expected welfare and project rate are secondary diagnostics.

## 3. Exact search and certificates

Anonymous states are sorted value vectors. A monotone allocation table is an
upward-closed subset of this finite poset, uniquely represented by its minimal
active states (an antichain). The enumerator generates antichains directly;
this avoids scanning all bit masks and makes the cross-agent extension exact.
For each table, the primary verifier checks every profile, every unilateral
report, and every permutation. A seeded threshold-proposal loop is included as
an optimization-style discovery probe, but proposals never bypass the exact
verifier.

The independent checker reconstructs the state table and critical payments from
JSON. It independently checks DSIC, ex-post IR, weak budget balance, and
anonymity. It does not import the primary search or verifier. The certificate
also evaluates the efficient sum-threshold family on all `4^3=64` profiles with
values `{0,1,2,3}` and records every failure.

## 4. Results

### 4.1 Preregistered three-agent frontier

The domain contains 10 sorted states and 16 monotone rules. The exact frontier
is:

| cost | accepted rules | best worst-case regret | best expected welfare |
|---:|---:|---:|---:|
| 1 | 4 | 3 | 1.0370 |
| 2 | 4 | 2 | 0.7407 |
| 3 | 4 | 1 | 0.4444 |
| 4 | 1 | 1 | 0.0741 |
| 5 | 1 | 0 | 0.0370 |
| 6 | 1 | 0 | 0.0000 |

At cost three, the four accepted tables are nested upper sets in the sorted
state poset. Their minimum active states move from `(2,2,2)` down through the
highest-value states, while the normalized critical payments continue to cover
the cost. The result is a complete finite frontier, not a sampled
optimization claim.

The efficient threshold rule passes DSIC, ex-post IR, feasibility, and
anonymity, but fails budget balance at `(0,2,2)`: its payments are `(0,1,1)`,
whose total is 2 while cost is 3. This is the smallest informative negative
example in the main domain and is retained in the certificate.

### 4.2 Finite value-lattice theorem and cross-check

The theorem covers every `n>=1`, finite integer cap `m>=1`, and integer cost.
Put `k=ceil(c/n)`. If `k>m`, no accepted rule exists. Otherwise, an allocation
rule is accepted exactly when its active set is a nonempty upward-closed subset
of the sorted restricted lattice `{k,...,m}^n`. At the all-`m` profile,
anonymity makes all critical payments equal; budget balance forces their common
threshold to be at least `k`. Monotonicity then rules out every active profile
with a coordinate below `k`. Conversely, every upward-closed set inside the
restricted lattice has every active critical payment at least `k`, so its total
revenue covers cost. Full details are in `PUBLIC_PROJECT_THEOREM.md`.

Finite exhaustive searches independently cross-check the theorem:

| agents | anonymous rules | accepted counts by cost | costs with one accepted rule |
|---:|---:|---|---|
| 3 | 16 | 4,4,4,1,1,1 | 4,5,6 |
| 4 | 32 | 5,5,5,5,1,1,1,1 | 5,6,7,8 |
| 5 | 64 | 6,6,6,6,6,1,1,1,1,1 | 6,7,8,9,10 |
| 6 | 128 | 7,7,7,7,7,7,1,1,1,1,1,1 | 7,8,9,10,11,12 |

The serialized accepted rows total 122 (74 through five agents and 48 in the
six-agent extension), and all pass the standalone checker. The six-agent run
took 56.394 seconds and used 29,474,816 bytes peak resident memory on Darwin.
The symbolic ternary construction certificate checks 806 mechanisms for
`n=1..12`; it is a regression certificate for the `m=2` corollary, not a
formal proof assistant.

### 4.3 Stress and falsification

On held-out values `{0,1,2,3}`, the six sum-threshold rules produce 207 recorded
budget/IC failures across 64 profiles per threshold. This is a deliberate
generalization boundary: success on `{0,1,2}` does not justify a continuous or
larger-value claim.

### 4.4 Preregistered value-lattice confirmation

The theorem and frozen confirmation protocol were committed before a fresh
three-agent `m=4` run. Full-domain enumeration and the theorem construction
produce identical accepted rule sets at all costs `1,...,12`; counts are
`65,65,65,15,15,15,4,4,4,1,1,1`. The standalone checker accepted all 255
serialized rules, with no failures. The run took 24.00 seconds and used
45,842,432 bytes peak resident memory. This is a direct falsification attempt
on a larger untouched lattice, not an extrapolated performance metric.

The prior post-hoc `m=3` run is retained as a precursor: it has 66 anonymous
monotone rules and counts `15,15,15,4,4,4,1,1,1`, with 60 independent replays.
Neither run establishes a continuous-value characterization.

### 4.5 Executable audit of published max-affine rules

The finite frontier is complemented, but not replaced, by an external audit
lane. We define a deliberately restricted typed language: rational affine
forms combined by addition, scalar multiplication, and finite `max`/`min`.
On an ordered unit cube, every branch boundary and the public-project
first-best boundary are affine. Enumerating every intersection of the declared
boundaries therefore gives a finite exact certificate for extrema of total
Groves charge divided by first-best cost. On each arrangement cell the selected
branches are affine, budget slack is affine, and the ratio is linear-fractional
with denominator at least one; extrema therefore occur at a cell vertex.

The generic engine reproduces four pre-existing continuous formulas: the
three-agent direct-redistribution formula in Guo (PRIMA 2016) Equation (3),
the three-agent optimum displayed as Guo (IJCAI 2019) Equation (2), and the
three- and four-agent printed formulas in Guo (AAAI 2024). The four-agent
network is lowered from a serialized rational one-hidden-layer ReLU
specification rather than only bespoke arithmetic. A separate evaluator also
compares the source network and compiled expression on every vertex of their
common branch refinement. The PRIMA rule is
converted to its Groves-term equivalent and independently checked against the
source-convention evaluator at every certificate vertex; it has slack `64/81`
and minimum retained efficiency `-47/162` on the three-agent ordered cube.
The four-agent case is the hard control: it includes signed ReLU terms and terminating
decimal coefficients. The generic certificate reproduces the displayed
formula's `1/5000` deficit and its published `1/20000` per-term uniform repair.
An independent program consumes only the serialized rational expression,
derives branch planes and arrangement vertices itself, and exactly matches all
five certificates (including the repaired four-agent entry).
The largest frozen entry has 22 planes, 7,315 exact four-plane bases, and 116
feasible vertices. These are a demonstrated resource boundary, not evidence of
scalability to arbitrary architecture depth or dimension.

### 4.6 Frozen ReLU source and solver cross-check

Source replay at shared certificate vertices could still conceal a lowering or
boundary-generation error. We therefore freeze six deterministic rational
one-hidden-layer fixtures before evaluation: three development and three
confirmation cases, covering three to five agents and widths two or three.
The direct route derives activation boundaries and charge values from the
serialized source network; the other lowers it into the typed max/min-affine
language. All six pairs agree exactly on extrema and witnesses. Their
candidate-basis counts are `165,3060,6188,364,3060,792`, with respectively
`6,14,49,23,8,19` feasible vertices.

An external exact-real Z3 model encodes the source ReLU network without
importing the compiler or certificate engine. Per fixture it asks whether
budget slack is strictly smaller than reported, whether charge ratio is
strictly below the reported minimum, and whether it is strictly above the
reported maximum. All 18 queries return `unsat`; direct rational evaluation
also validates every reported extremum witness. This rules out those
counterexamples for the six frozen source models, not arbitrary learned
networks or a generic scalable SMT procedure.

This is a reproducibility result about displayed formulas. It does not recover
training weights, diagnose authors' unshared implementation, prove properties
of arbitrary neural networks, or establish a new mechanism. Its value is that
the claim boundary, formula, arithmetic, witness, and independent replay are
all inspectable.

### 4.7 Exact uniform repair synthesis

We add a deliberately small synthesis result rather than another synthetic
benchmark. If an `n`-agent deleted-input source has certified minimum slack
`s`, adding `delta` uniformly to its output bias changes total charge by
`n*delta` at every profile. Consequently `max(0,-s/n)` is the smallest
nonnegative offset that removes deficit within this one-parameter family: the
baseline slack witness establishes necessity and the identity establishes
sufficiency.

The seven frozen sources include the printed four-agent decimal rule and all
six Phase-V fixtures. The printed control recovers `1/20000`; six sources need
positive repairs and one needs none. Compiled and direct-source certificates
agree after every repair, all positive repairs bind at zero slack, every
half-sized repair fails at the original witness, and seven exact-real Z3
strict-negative-slack queries are `unsat`. This is not a new mechanism or a
welfare result. In fact, the large synthetic offsets are negative evidence
against interpreting scalar feasibility repair as economically useful design.

### 4.8 Certified budget--IR incompatibility of the scalar repair

The preceding result is not sufficient for a Groves mechanism because truthful
utility is `S-h(theta_-i)`. The same offset `delta` that adds `n*delta` to total
slack subtracts `delta` from every agent's utility. We therefore froze the
corpus and repair family before computing exact minimum utility. All seven
repaired sources fail ex-post IR: the displayed control is negative already,
and every initially nonnegative synthetic margin is smaller than the offset
required for budget repair. Direct source replay agrees with the compiler; 28
exact-real Z3 lower-bound queries are `unsat`, and seven IR queries agree.
This is a calibrated negative result, not an impossibility theorem for other
redistribution families.

### 4.9 Bounded-coalition falsification and independent baseline audit

Sections 4.1-4.4 establish DSIC only against unilateral deviations. We add a
post-hoc, non-preregistered supplement (`PREREGISTRATION.md`, "Post-hoc
coalition robustness extension") checking whether accepted rules also resist
joint deviations by small groups. For a coalition `T` of size up to a frozen
cap `k`, we exhaustively enumerate every joint report deviation `T` could make
and reject a rule if any joint deviation strictly increases the coalition's
summed utility over truthful reporting.

**Frontier shrinkage.** On the preregistered three-agent, cost-3 domain, cap-2
deviations shrink the DSIC frontier from 4 accepted rules to 2
(`anonymous_monotone_mask_512`, `anonymous_monotone_mask_960`); 6 frontier
rows are rejected at cap-2 across all six costs, with 0 rejected survivors
among the cap-2-accepted rules under independent replay.

**Scaling.** The same filter applied to `n=3..5` (cap 3, 32.977s) and
`n=3..6` (cap 3, 91.758s) reproduces fragility at every selected
`(n,cost)` checkpoint with 0 independent failures across 24 and 34 selected
checks respectively. A separate extension to the wider `{0,1,2,3}` value
domain (`n=3`, costs `1..9`, cap up to 3) reproduces the same qualitative
result at selected costs `1,3,9` with 0 independent failures.

**Independent baseline audit.** The preceding results only characterize the
*search-discovered* frontier. A skeptical reader could ask whether coalition
fragility is an artifact of that search rather than a property of natural
mechanisms. We therefore separately audit the canonical efficient/pivotal
comparator from Section 4.1 -- the welfare-maximizing sum-threshold rule with
critical-value payments, single-agent DSIC and ex-post IR by construction --
against the identical coalition bar, across all four domains above (75
`(domain, n, cost)` cells total). Fragility is judged strictly from the
verifier's `dsic`/`coalitional_dsic` fields, decoupled from that same rule's
separately documented weak-budget-balance deficit (Section 4.1), so the two
independent negative results are not conflated. The comparator is
single-agent DSIC in every cell tested, but fails coalition-cap-2 DSIC in 66
of 75 cells and in 10 of the 11 preregistered-style selected spot checks (the
lone survivor is `n=3`, cost `9` on the `{0,1,2,3}` domain). A concrete
witness at `n=3`, cost `3`: truthful reports `(0,1,1)` (project not built,
all utilities 0). If agents 0 and 1 jointly misreport `(2,2)` while agent 2
reports truthfully, the profile becomes `(2,2,1)`, which crosses the cost-3
threshold and builds the project. Because critical-value payments are
computed from the *post-deviation* profile, each deviator's payment is the
smallest report at which the project would still build holding the other
deviator's inflated report fixed -- which is 0 for both, since the other
deviator's report alone already covers cost. Both deviators pay nothing and
jointly gain 1 unit of utility they could not obtain by reporting truthfully,
at agent 2's expense of unknowingly co-funding a project neither deviator
truthfully valued at that cost. This single witness generalizes the
mechanism-design intuition that critical-value payments, calibrated against
single-agent deviations, do not account for a coalition's ability to jointly
manufacture the threshold it is then charged against.

Every coalition claim above is replayed by a standalone implementation
(`public_project_independent.py`) that reconstructs allocation tables and
payments from serialized JSON without importing the primary verifier: 0
mismatches across the frontier, scaling, scaling-extension, and baseline-audit
studies (independent digests recorded in `VERIFICATION_REPORT.md`).

This supplement is deliberately narrow. It bounds coalition size at 2 or 3, it
does not claim coalition-proofness for arbitrary group size, and it does not
extend to randomized rules or continuous values. It does
show, with an exact and independently replayed counterexample, that
single-agent DSIC -- the standard acceptance bar used throughout Sections
4.1-4.4 and in most automated-mechanism-design search -- is not sufficient
for robustness against even the smallest possible group manipulation, and
that this failure is not specific to the search-discovered frontier: it also
afflicts the textbook efficient/pivotal comparator.

### 4.10 False-name manipulation of the same comparator

Distinct-agent coalitions are one attack; false-name manipulation (Yokoo,
Sakurai and Matsubara, 2004) is another and structurally different one: a
single real agent fabricates extra fake report identities and controls all of
them, so there is one true value behind several report slots, and the
attacker pays whatever the mechanism charges each slot it controls. This is
also a post-hoc, non-preregistered supplement
(`PREREGISTRATION.md`, "Post-hoc false-name manipulation extension").

Because the sum-threshold rule `q(reports)=1[sum(reports)>=cost]` with
critical-value payments is defined identically for any agent count, we check
it by comparing the same rule at `n_real` real agents (truthful baseline)
against the same rule at `n_real+f` agents, where the attacker occupies one
real slot plus `f` fake slots and the other `n_real-1` real agents keep
reporting truthfully. We sweep `n_real in {3,4,5}`, `f in {0,1,2}`, and costs
`1..2*n_real` (72 `(n_real, cost, f)` cells). `f=0` is a positive control: it
must reduce to ordinary single-agent DSIC, and indeed shows zero manipulable
cells everywhere, confirming the harness before trusting `f>=1`.

At `f in {1,2}`, 48 of the 72 cells are manipulable, including 6 of 9
preregistered-style selected spot checks (every selected cell with `f>=1`). A
concrete witness at `n_real=3`, cost `3`: truthful profile `(0,1,1)` does not
build the project (all utilities 0). If the agent with true value 1 reports 2
in its own slot and fabricates one fake identity also reporting 2, the
extended 4-slot profile `(0,1,2,2)` builds. Because critical-value payments
are computed per slot holding the *other* slots fixed, and the attacker's own
inflated fake slot is one of those "other" slots for its real slot's
threshold (and vice versa), each of the attacker's two controlled slots has
threshold 0: the other slot's report of 2 already exceeds the remaining
`cost - 1 = 2` needed. Both controlled slots are charged 0, and the attacker
nets a utility gain of 1 that single-identity truthful reporting could not
achieve. This is the same critical-value mechanism the Section 4.9 coalition
witness exploits, applied through a different channel: instead of splitting
the threshold-clearing burden across distinct agents' true values, a single
agent splits it across its own report and self-created fake reports, each of
which is charged as if the others' (also attacker-controlled) reports were
independent evidence of demand.

Every cell is replayed by a standalone implementation
(`scripts/verify_public_project_false_name_audit.py`) that recomputes the
sum-threshold/critical-value rule from its closed-form definition without
importing the primary mechanism module: 0 mismatches across all 72 cells,
independent digest recorded in `VERIFICATION_REPORT.md`. This supplement is
also narrow: it audits only the canonical comparator (not the
anonymous-monotone frontier), bounds the fake-identity budget at 2, and makes
no randomized- or continuous-domain claim.

### 4.11 A general analytical lemma explaining the coalition frontier

Sections 4.9 and 4.10 are search results: they enumerate finite domains and
report which cells fail. This section replaces part of that search with a
proof. For the sum-threshold/critical-value mechanism
`q(reports)=1[sum(reports)>=cost]` with critical-value payments, on any
`n>=2` agents with integer cap `max_value=m`, consider the grand-coalition
deviation in which every agent reports `m`. If `cost<=(n-1)*m`, this
deviation builds the project, and every agent's critical-value payment is
exactly `0`: excluding any one agent, the remaining `n-1` agents' reports
alone already sum to `(n-1)*m>=cost`, so that agent's threshold is already
met without their own contribution. Each agent's utility becomes exactly
their true value `v_i` at zero cost — weakly better than truthful reporting
for every agent, and strictly better for any agent whose truthful utility was
not already `v_i` for free. This is a direct algebraic argument, not a search:
it holds for every integer `n>=2`, `m>=1`, `cost<=(n-1)*m`, not only the
`n<=6`, `m<=3` cells that were exhaustively checked in Section 4.9.

The condition also explains the coalition-baseline audit's one exception.
Cross-checking `cost<=(n-1)*m` against all 75 rows of
`artifacts/public_project_coalition_baseline_audit.json` gives zero false
positives: every row the condition predicts fragile is fragile in the search
data (`scripts/verify_public_project_coalition_lemma.py`). The condition
never fires exactly when `cost=n*m`, because `(n-1)*m<n*m` always — and
`cost=n*m` is precisely the baseline audit's sole robust row
(`value3_frontier`, `n=3`, `cost=9`). A second, separate argument shows why:
at `cost=n*m`, any proper coalition can only reach the threshold if every
outsider's true value is already exactly `m` (their reports are truthful and
capped at `m`, so the coalition's reports must supply the rest exactly), and
in that case the coalition's reports must also sum to exactly the coalition's
maximum, `k*m`, forcing every coalition member to report `m`. Each member's
critical-value payment is then `cost-(n-1)*m=m` — their own full report — so
the deviation gains nothing whenever a coalition member's true value is below
`m`. This is checked by exhaustive enumeration over all truthful profiles and
all proper coalition subsets for `n=3,4` and `m=2,3`, with zero
counterexamples. So Section 4.9's frontier is not merely a finite search
artifact: it follows from a proven general condition on `(n,m,cost)`, with
`cost=n*m` proven (for the checked `n,m`) to be the unique boundary at which
this construction cannot manipulate the mechanism. This lemma covers only the
canonical sum-threshold/critical-value rule (not the searched anonymous-
monotone frontier), and only this specific construction — it is a sufficient
condition for manipulability, not a full necessary-and-sufficient
characterization: some cells with `cost>(n-1)*m` and `cost<n*m` are still
manipulable in the search data via different, non-uniform deviations not
covered by this argument.

### 4.12 A complete closed-form characterization

Section 4.11 leaves a gap: 14 of the 75 baseline-audit rows with
`(n-1)*m<cost<n*m` are fragile in the search data for reasons the all-max
construction does not cover. This section closes that gap with two exact
formulas, both proved by convexity of the critical-value payment
`p_i=max(0, cost-others'-sum)` in the coalition's own report sum, and both
implemented and verified in `scripts/verify_public_project_coalition_characterization.py`.

**Minimum coalition payment.** For a coalition `T` of size `k` facing fixed
outsider true-value sum `S_O`, the deviation that minimizes `T`'s total
critical-value payment while still building the project is exactly the
all-max deviation: every member reports `m`. Its payment is
`min_payment(k,S_O,cost)=k*max(0,(cost-S_O)-(k-1)*m)` whenever
`cost-S_O<=k*m` (else no deviation of `T` can build the project at all).
This follows because `max(0,r_i-D)` (with `D` a function of the report sum)
is convex, so for any fixed report sum the total payment is minimized by an
equal split across members, and among equal splits the payment is minimized
by taking the largest feasible report sum, `k*m`.

**Worst-case truthful payment.** When the project already builds truthfully
(`V_T+S_O>=cost`, with `V_T` the coalition's true-value sum), the same
convexity works in the opposite direction: the truthful total payment,
as a function of how `V_T` is distributed among the `k` members, is
*maximized* -- the hardest case to beat, hence the one that matters for
existence -- by an extremal "bang-bang" distribution: as many members as
possible at `m`, one member absorbing the remainder, the rest at `0`.

**Existence check.** Combining both formulas, whether *any* profile makes a
size-`k` coalition profitable under the verifier's sum-of-utilities
criterion reduces to a bounded integer sweep over the outsider sum `S_O` and
the coalition sum `V_T` -- no enumeration of individual reports or profiles
is needed. `scripts/verify_public_project_coalition_characterization.py`
implements this sweep and cross-checks it against all 75 rows of
`artifacts/public_project_coalition_baseline_audit.json`: it reproduces the
searched `min_failing_coalition_size` exactly on all 75 rows (`75/75` exact
matches, not merely zero false positives), closing the gap the Section 4.11
sufficient condition left open. Because the check is a bounded sweep rather
than a search over reports, it also evaluates at agent counts far beyond
brute-force reach -- for example `n=20`, `m=8` in microseconds, where
enumerating reports (`9^20` profiles) is computationally infeasible.

This is a full necessary-and-sufficient characterization of coalition-cap-`k`
manipulability for the canonical sum-threshold/critical-value rule under the
verifier's own sum-of-utilities criterion -- not a new mechanism, not a
repair, and not a claim about the searched anonymous-monotone frontier, only
about this one canonical comparator. Digest
`9dd70ad48733e6be95cd8ff4b0f37e5638e7347ccb7d044ab0a2adf80ebe7be0`.

### 4.13 Extended cross-check and gain sizes

`scripts/verify_public_project_coalition_characterization_extended.py`
extends the Section 4.12 cross-check to 70 new `(n,max_value,cost)` cells the
original 75-row baseline audit never covered -- larger value caps (`max_value`
3, 4, 5) at every previously tested agent count (`n=3,4,5`), brute-force
verified by the primary verifier at coalition cap 3. The closed-form formula
again matches exactly on all 70 new rows (`0` mismatches), bringing the total
independently brute-force-verified row count to 145. It is also fragile on 65
of the 70 new rows, and the closed-form formula gives the exact size of the
free gain a coalition captures, in the mechanism's own value units: up to `5`
units of value for free at `n=3`, `max_value=5`. A separate, explicitly
formula-only sweep (not independently re-verified by brute force, since brute
force is combinatorially infeasible at this scale) evaluates 252
`(n,max_value,cost)` triples up to `n=40`, `max_value=15` and finds `99.2%`
are already manipulable by a coalition of size 2, with gains up to `15` value
units -- the phenomenon does not shrink as the domain grows; if anything it
becomes more prevalent, consistent with the closed-form condition
`cost<=(n-1)*max_value` becoming easier to satisfy as `n` grows for fixed
`max_value`. Digest
`c10e6217b03415820f95c65fbfbab7dc159796c8099441935961ec5e796e5f48`.

## 5. Positioning and contribution boundary

The study is deliberately positioned against established theory and automated
mechanism design rather than claiming to replace either. [Green--Laffont](https://green.scholars.harvard.edu/publications/incentives-public-decision-making),
[Ohseto](https://www.sciencedirect.com/science/article/pii/S0899825699907558),
and [Moulin](https://academic.oup.com/restud/article-abstract/61/2/305/1517585)
provide foundational public-project and cost-sharing results;
Nath and Sandholm analyze efficiency and budget balance in general quasi-linear
domains ([paper](https://arxiv.org/abs/1610.01443)); Conitzer and Sandholm give
the general computational approach to automated mechanism design
([AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/7708)); and Guo
et al. study machine-learning approaches for public-project mechanism design
([2024 article](https://link.springer.com/article/10.1007/s10458-024-09647-8)).

| Existing line | This paper adds | This paper does not add |
|---|---|---|
| General public-project and cost-sharing theory | A finite, fully enumerated benchmark with explicit payment reconstruction and witnesses | A new impossibility theorem or a continuous-value characterization |
| Automated mechanism design | A solver-free antichain enumerator, machine-readable certificates, and an independent replay implementation | A claim that the search discovered a new mechanism |
| Learned public-project mechanisms | A falsification harness showing exactly where an efficient threshold proposal fails | A learned policy, deployment result, or causal claim |
| Computer-aided mechanism and neural-network verification | Typed formula provenance, direct-source replay, and exact-real SMT cross-checks for a small public-project audit corpus | Generic verification novelty or coverage of arbitrary architectures |
| Coalition-proof/group-strategyproof mechanism theory | An exact, independently replayed bounded-coalition falsification supplement, a baseline audit showing the textbook efficient/pivotal rule is itself coalition-cap-2 fragile, and a full closed-form necessary-and-sufficient characterization of coalition-cap-`k` manipulability (exact minimum-payment and worst-case-truthful-payment formulas), reproducing the searched `min_failing_coalition_size` exactly on all 75 rows and evaluating far beyond brute-force agent counts | An unbounded-coalition-size result, a repaired coalition-resistant mechanism, or a characterization of the searched anonymous-monotone frontier (only the one canonical comparator is characterized) |
| False-name-proof mechanism theory (Yokoo, Sakurai and Matsubara, 2004) | An exact, independently replayed false-name-manipulation supplement against the same textbook comparator, with a positive control confirming the harness | A false-name-proofness theorem, a repaired false-name-resistant mechanism, or a claim about the search-discovered frontier's false-name robustness |

The defensible contribution is a compact, replayable certificate and an exact
theorem for the specified finite integer-value class. It is not a claim of a
new universal impossibility result or unrestricted mechanism-design novelty.

## 6. Reproducibility and falsification

The main JSON, scaling CSVs, and six-agent JSON contain the complete serialized accepted rows;
the independent checker reconstructs allocation and critical payments without
importing the primary verifier. The ternary clean run reports 122 accepted rows
and zero independent replay failures; the frozen `m=4` confirmation adds 255
accepted rows with exact predicted/exhaustive equality and zero failures. The held-out audit evaluates every one of the 64
profiles for each efficient threshold on values `{0,1,2,3}` and records 207
failures. These are positive and negative controls: the first tests certificate
integrity, while the second tests whether the finite result is being
over-generalized. Exact commands and SHA-256 hashes are in
`REPRODUCIBILITY_MANIFEST.md`.

## 7. Limitations and next work

The theorem's mechanism class is anonymous, deterministic, finite-valued, and
restricted to normalized critical payments. The study does not cover randomized
rules, subsidies, Bayesian objectives, continuous values, arbitrary-size
collusion, asymmetric rules, or arbitrary payment schemes.
The coalition supplement (Section 4.9) is bounded to coalitions of size 2 or 3
within the same finite integer-value class; it establishes fragility, not a
repair, and does not claim anything about coalitions above the tested cap or
about mechanisms outside the anonymous-monotone class. The false-name
supplement (Section 4.10) is bounded to fake-identity budgets of 1 or 2 on the
canonical comparator only, not the search-discovered frontier; it likewise
establishes fragility, not a repair. The Section 4.11 lemma is a sufficient,
not necessary, condition for coalition-manipulability, superseded for
existence questions by the Section 4.12 full characterization; its boundary
argument at `cost=n*max_value` is still checked only for `n=3,4` and
`max_value=2,3`, not proven for all `n,max_value`. The Section 4.12
characterization covers only the canonical sum-threshold/critical-value rule
and the verifier's sum-of-utilities coalition criterion; it is not a
characterization of the searched anonymous-monotone frontier, and it does not
by itself repair the mechanism. The n=3..6 searches
are computational cross-checks; the finite-lattice proof is the general result. The
stress audit is intentionally negative for
the efficient threshold family; it is not an empirical estimate of deployment
risk. The ReLU cross-check covers six frozen synthetic fixtures and publicly
displayed formulas, not opaque trained weights or a representative network
corpus. Before submission, a researcher should test the theorem against richer
value domains and broader mechanisms, compare unrestricted transfers and
subsidies, evaluate a broader independently sourced formula corpus, and obtain
an external replication.

## 8. Practical and policy implications

The mechanism studied here is not a hypothetical curiosity. "Decide on a
binary project once enough value has been pledged, then charge each
supporter their critical contribution" is the textbook prescription for a
class of real coordination problems: participatory-budgeting platforms
choosing which proposal to fund, condominium and HOA boards voting on a
capital assessment, crowdfunded neighborhood infrastructure (playgrounds,
footbridges, shared solar installations), and the pivotal-mechanism
experiments that some DAOs and quadratic-funding platforms have piloted for
allocating a shared grant pool. Anywhere a group decides, together, whether
to build something and how to split the bill by "what you were pivotal for,"
this is the rule in question.

Two caveats first, stated as plainly as the rest of this paper's claims. This
work does not audit a named platform, and it does not claim any real system
currently deploys the exact critical-value rule studied here at production
scale -- most crowdfunding platforms use simpler all-or-nothing pledge
thresholds without critical-value payments at all, precisely because payment
mechanics like this one are hard to explain to users. And the domain is
still finite integer values on a chosen `(n, m)` grid, not continuous money.
What follows is what the exact result licenses a designer to conclude, not
an empirical claim about any deployed system.

**What changes for a designer.** Before this result, the honest advice was
qualitative: "critical-value payments are known to be exploitable by
coalitions in general quasi-linear settings" (Green-Laffont 1979 and the
false-name literature already established that). That advice does not tell
a designer building a specific platform -- fixed group size, fixed cost,
fixed pledge cap -- whether *their* configuration is at risk, or how much a
colluding subgroup stands to gain. The closed-form characterization in
Sections 4.11-4.13 answers exactly that, for this rule, from three numbers:
group size `n`, project cost, and the maximum pledge `m`. Plugged into
`min_payment(k,S_O,cost)=k*max(0,(cost-S_O)-(k-1)*m)`, a designer can check,
before launch, whether any coalition of a given size can profit, and read
off the exact amount at stake -- an audit checklist, not a proof that
collusion is merely theoretically possible somewhere.

**The concrete risk pattern.** The formula makes the qualitative risk factor
precise: the mechanism is guaranteed manipulable by the full group whenever
`cost<=(n-1)*m` -- that is, whenever the project could plausibly be funded
even leaving out any single participant. Practically, that condition is
easiest to satisfy in exactly the settings where this kind of mechanism gets
proposed: small, well-acquainted groups (an HOA board, a five-person DAO
multisig, a neighborhood association) where a modest cost relative to the
group's combined pledge capacity is the whole point of pooling funds in the
first place, and where members can and do coordinate outside the mechanism.
The large-scale sweep in Section 4.13 shows this is not a narrow corner
case: across a broad range of group sizes and pledge caps, the large
majority of configurations are already vulnerable to a coalition of just
two.

**What a designer can do about it**, in decreasing order of how much it
costs to implement:
1. *Check the number first.* Before adopting the textbook rule, compute
   `min_payment` for the actual `(n, m, cost)` the platform will use. If the
   condition `cost<=(n-1)*m` holds, the rule is manipulable by construction,
   full stop -- no amount of platform trust or terms-of-service language
   changes that.
2. *Raise the stakes-to-pledge ratio.* Pushing `cost` closer to `n*m` --
   requiring the project to need close to everyone's maximum pledge --
   shrinks the fragile region, though Section 4.11's boundary argument shows
   even `cost=n*m` is only immune to this specific construction, not
   collusion in general.
3. *Seal or randomize report order and identity.* Coalitions need to
   coordinate their reports; anything that makes coordinated reporting
   harder (sealed-bid submission with no revision, randomized reveal order,
   participation caps per real-world identity to blunt false-name attacks
   per Section 4.10) raises the practical, if not the theoretical, cost of
   collusion.
4. *Do not use this rule for small, socially connected groups without
   compensating controls.* The formula's own terms say the risk is worst
   exactly there. A platform serving thousands of mutually unacquainted
   backers with small individual pledge caps is structurally safer under
   this rule than one serving five board members deciding on a shared
   capital expense.
5. *Consider it a documented reason to look at alternatives*, not a reason
   to abandon threshold public-goods funding altogether -- budget-balanced
   or randomized alternatives exist in the literature and are explicit
   future work here (Section 7), not something this paper builds or
   recommends by default.

None of this is a repaired mechanism, and none of it is a claim that
collusion happens in practice on any specific platform -- that is a
behavioral and empirical question this paper does not touch. What this
section adds is what the earlier sections do not: a direct line from an
exact mathematical formula to a checklist a real designer can run before,
not after, deployment.

## 9. Author contribution and AI-assistance disclosure

Mohit Mendiratta is the sole author. He posed the research question, set and
enforced the certificate-first / independent-replay methodology that governs
every claim in this paper (Section 3, Section 6), decided the scope boundary
in Section 5 (what this paper does and does not claim), selected which
results were strong enough to keep and which exploratory branches to cut
(e.g. the withdrawn constrained-synthesis pilot noted in the decision log),
and reviewed and takes responsibility for every claim, number, and digest
reported here.

AI tooling (Claude, Anthropic) was used as an assistant throughout: deriving
candidate formulas, writing the verification and search scripts, running the
brute-force cross-checks, and drafting prose under direction. It did not set
the research question, decide what counts as a valid claim, or approve any
result for inclusion -- those were the author's decisions, made by reviewing
the certificates and independent replays this paper's own falsification
discipline requires (Section 6), not by trusting the tooling's output at
face value. This disclosure is made in the interest of transparency and
because AI assistance in research writing is increasingly the norm rather
than the exception; it does not change the correctness of the mathematical
claims, which stand on the independently replayed proofs and certificates,
not on who or what typed them.

## 10. Reproduction

```bash
python3 -m unittest discover -s tests -v
python3 scripts/run_public_project_study.py
python3 scripts/verify_public_project_certificate.py
python3 scripts/run_value_extension.py
python3 scripts/run_n6_extension.py
python3 scripts/verify_scaling_theorem.py
python3 scripts/verify_value_lattice_theorem.py
python3 scripts/run_max_affine_certification.py
python3 scripts/verify_max_affine_certificate.py
python3 scripts/verify_source_network_certificates.py
python3 scripts/run_relu_benchmark.py
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-verification.txt
.venv/bin/python scripts/verify_relu_benchmark_z3.py
python3 scripts/run_uniform_repair_study.py
.venv/bin/python scripts/verify_uniform_repair_z3.py
python3 scripts/run_repair_ir_tradeoff_study.py
.venv/bin/python scripts/verify_repair_ir_z3.py
python3 scripts/run_public_project_coalition_frontier.py
python3 scripts/verify_public_project_coalition_frontier.py
python3 scripts/run_public_project_coalition_scaling.py
python3 scripts/verify_public_project_coalition_scaling.py
python3 scripts/run_public_project_coalition_scaling_extended.py
python3 scripts/verify_public_project_coalition_scaling_extended.py
python3 scripts/run_public_project_coalition_value3_frontier.py
python3 scripts/verify_public_project_coalition_value3_frontier.py
python3 scripts/run_public_project_coalition_baseline_audit.py
python3 scripts/verify_public_project_coalition_baseline_audit.py
python3 scripts/run_public_project_false_name_audit.py
python3 scripts/verify_public_project_false_name_audit.py
python3 scripts/verify_public_project_coalition_lemma.py
python3 scripts/verify_public_project_coalition_characterization.py
python3 scripts/verify_public_project_coalition_characterization_extended.py
```

The main JSON, cross-agent CSV, certificate, plot, specification, and claim
ledger are committed under `artifacts/`, `reports/`, and the repository root.

## 11. Conclusion

Exact search does not magically produce a universally optimal mechanism. This
study does provide a finite-lattice characterization that a skeptical reader
can inspect: the proof fixes the accepted family, a preregistered larger-lattice
enumeration reproduces it exactly, the efficient comparator has a concrete
budget-balance failure, a bounded-coalition supplement shows the same
comparator also has a concrete, independently replayed incentive failure
against groups as small as size 2, a second supplement shows the same
comparator is manipulable by a single agent fabricating fake report
identities, a general closed-form lemma proves that fragility for every
`(n,max_value,cost)` with `cost<=(n-1)*max_value` rather than only the
searched cells and pinpoints exactly why `cost=n*max_value` is immune, a
complete closed-form characterization (exact minimum-payment and worst-case-
truthful-payment formulas) reproduces the searched coalition frontier exactly
on all 75 rows and scales to agent counts brute-force search cannot reach, and
the held-out stress test records where
the result stops generalizing. It is a credible foundation for a
theory/verification paper, but publication still requires external novelty
review, broader theory, and peer review.
