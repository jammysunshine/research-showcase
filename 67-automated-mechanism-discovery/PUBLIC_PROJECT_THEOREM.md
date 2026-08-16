# Exact finite-lattice characterization

This theorem concerns the declared finite model only. It does not cover
continuous values, arbitrary transfers, randomized mechanisms, asymmetry, or
coalitional incentives.

**Model.** There are `n >= 1` agents with reports in `{0,...,m}`, a binary
project with integer cost `c >= 1`, deterministic anonymous coordinatewise
monotone allocation, and normalized discrete critical payments. An accepted
rule must build at `(m,...,m)`, satisfy DSIC, ex-post IR, feasibility, no
payment when the project is absent, and weak budget balance whenever it builds.

**Theorem.** Put `k = ceil(c/n)`. If `k>m`, there is no accepted rule. If
`k<=m`, the accepted allocation rules are exactly the nonempty upward-closed
subsets of the sorted state poset contained in `{k,...,m}^n`. Equivalently, a
rule is accepted exactly when it is monotone, non-vacuous, and it never builds
at a profile with any coordinate below `k`. Its count is therefore the number
of nonempty upper sets in the sorted `{k,...,m}^n` lattice.

For `m=2`, the positive sorted states form a chain. The general theorem then
reduces to the earlier closed form: `n+1` rules for `c<=n`, one for
`n<c<=2n`, and none above `2n`.

## Proof

Let `h=(m,...,m)` and define

```text
t = min {x in {0,...,m}: q(x,m,...,m)=1}.
```

The set is nonempty because `q(h)=1`. By anonymity, each critical payment at
`h` is `t`, so weak budget balance gives `n t >= c`; hence `t>=k`.

Suppose an active state has a coordinate `r<k`. After permuting that
coordinate first, the state is coordinatewise below `(r,m,...,m)`. Monotonicity
would make the latter state active, giving `t<=r<k`, a contradiction. Thus an
accepted rule cannot build outside `{k,...,m}^n`.

Conversely, take any nonempty upward-closed set inside that restricted lattice.
At every active profile, holding any other reports fixed, the agent's critical
report is at least `k`; otherwise the rule would build outside the restricted
lattice. Total critical-payment revenue is therefore at least `nk>=c`.
Monotone critical payments are DSIC and ex-post IR; feasibility, no payment
when absent, and anonymity hold by construction. The rule is accepted.

The two directions characterize exactly the stated rule set.

## Evidence boundary

`src/mechanism_discovery/public_project_theorem.py` constructs the predicted
rules without using the full-domain frontier. The original ternary regression
certificate covers `n=1..12`. `scripts/verify_value_lattice_theorem.py` is a
preregistered confirmation on the untouched `n=3, m=4` grid: it exhaustively
enumerates the full domain, compares exact rule sets, and independently replays
every accepted mechanism. These scripts are regression certificates for the
human-checkable finite-model proof, not a proof-assistant development.
