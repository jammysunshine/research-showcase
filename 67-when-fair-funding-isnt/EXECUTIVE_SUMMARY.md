# Executive summary

Repository: https://github.com/jammysunshine/research-showcase/tree/main/67-when-fair-funding-isnt

Experiment 67 is a certificate-first study of deterministic public-project
mechanisms. Its core result is an exact finite value-lattice characterization:
for any agent count `n`, integer value cap `m`, and integer cost `c`, accepted
rules are precisely the nonempty upward-closed allocation sets inside
`{ceil(c/n),...,m}^n`. The original ternary count is the `m=2` corollary.
An untouched preregistered `n=3,m=4,c=1..12` run exactly matched the theorem's
full-domain rule sets and independently replayed all 255 accepted rules with
zero failures.

The original ternary accepted-count sequences are `4,4,4,1,1,1` (`n=3`),
`5,5,5,5,1,1,1,1` (`n=4`), and `6,6,6,6,6,1,1,1,1,1` (`n=5`). All 74
serialized accepted rows pass an independent checker. The efficient threshold
rule fails cost coverage at `(0,2,2)`, and a held-out value-3 audit records 207
failures. The defensible contribution is a finite discrete theorem plus
reproducible frontier and falsification benchmarks. It is not a continuous-type
result, deployment evidence, a generic AI discovery claim, or a guarantee of
publication or a prize.

The later exact repair audit makes the limitation sharper: a scalar offset can
remove certified budget deficits, but it lowers every truthful Groves utility
by that offset. On the unchanged seven-source corpus, none of the repaired
rules is ex-post IR; compiler, direct-source, and exact-real Z3 checks agree.
