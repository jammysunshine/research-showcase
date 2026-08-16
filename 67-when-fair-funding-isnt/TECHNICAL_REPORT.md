# Technical report

The public-project study reduces a single-parameter DSIC/EPIR search to
allocation rules plus critical payments. Anonymous states are sorted report
vectors. The enumerator generates minimal-active-state antichains of the
sorted-state poset, constructs the corresponding up-sets, and therefore covers
every monotone table without scanning arbitrary masks. It yields 16, 32, and
64 candidates for three, four, and five agents.

For each rule and cost, the verifier computes critical payments, tests every
deviation, checks anonymity and budget coverage, and computes welfare regret
against the efficient allocation. The independent checker consumes only
serialized tables and replays all 74 accepted cross-agent rows. A seeded
threshold proposal loop is instrumentation only; it never establishes
completeness. The held-out value-magnitude audit probes the strongest natural
extrapolation and records its failures.

## Exact max-affine audit method

Phase IV adds a separate, deliberately restricted certificate method for a
publicly displayed continuous formula. Its declared language is a finite sum
of rational affine forms and finite `max`/`min` groups of rational affine
forms, evaluated on the ordered cube `0 <= x_1 <= ... <= x_n <= 1`. The
certificate includes the full rational expression, all feasible arrangement
vertices, extrema witnesses, and the number of planes and candidate bases
examined.

**Proposition (declared language only).** For a represented charge formula,
enumerating every intersection of `n` planes from the ordered-cube facets,
the formula's branch-equality planes, and the first-best boundary exactly
certifies the extrema of charge/first-best, the derived efficiency value, and
budget slack on this domain.

Within each cell of that arrangement, all selected branches are affine. Budget
slack is consequently affine and charge/first-best is linear-fractional with a
strictly positive denominator (the first-best value is at least one). Each
such objective has an extremum at a cell vertex; degenerate vertices are still
included because all `n`-plane subsets are considered. This is not a result
for arbitrary programs, arbitrary networks, hidden source weights, or domains
outside the ordered unit cube.

## Frozen source and exact-real solver cross-check

Six deterministic rational one-hidden-layer ReLU fixtures were frozen before
evaluation: three development and three confirmation cases with three to five
agents and widths two or three. The compiler lowers each deleted-input charge
into the declared max/min-affine language. An independent direct evaluator
instead obtains activation boundaries and charge values from the serialized
source network. Exact extrema and witnesses agree for all six fixtures. The
basis/feasible-vertex pairs are `(165,6)`, `(3060,14)`, `(6188,49)`,
`(364,23)`, `(3060,8)`, and `(792,19)`.

`scripts/verify_relu_benchmark_z3.py` separately encodes each source network
over Z3 exact reals. It asks three strict counterexample questions per fixture:
lower budget slack, lower charge ratio, and higher charge ratio than the
recorded certificate. All 18 queries are `unsat`, and the script independently
re-evaluates all rational extremum witnesses. This is a bounded semantic
cross-check, not a claim about arbitrary architectures or solver scalability.

## Uniform-repair synthesis

Let a serialized network `h` be evaluated on every deleted-input vector for
`n` agents, and let `s` be the exact minimum budget slack of its total charge.
Adding `delta` to the output bias of `h` adds `n*delta` to total charge at every
profile. Thus `delta*=max(0,-s/n)` is sufficient for non-deficit; when `s<0`,
the original minimum-slack witness proves every smaller nonnegative scalar
offset fails. This is an exact theorem for the stated one-parameter family,
not an optimization over mechanisms.

The frozen seven-source study applies the construction to the displayed
four-agent decimal control and all six Phase-V fixtures. Compiler and direct
source certificates agree after each repair, and Z3 exact-real searches find
no negative slack for any repaired source. The large offsets in the synthetic
cases are retained as a practical limitation: correctness of a scalar repair
does not imply individual rationality, welfare quality, or economic relevance.

## Budget--IR compatibility

The scalar repair has an exact adverse utility effect: since truthful utility
is `S-h(theta_-i)`, an offset `delta` subtracts `delta` from every agent's
utility. The independent Phase-VII source evaluator and Z3 audit certify that
none of the seven repaired sources is ex-post IR. This invalidates any claim
that the repair is, by itself, an economically viable mechanism.
