# Soundness boundary for exact max-affine certificates

This note proves the claim implemented by
`certify_ordered_public_project_charge`; it is not a proof of properties of an
unrepresented neural network or of a source's private implementation.

## Declared language and domain

Let `Expr` be a finite sum of rational affine forms and terms of the form
`max(a_1,...,a_k)` or `min(a_1,...,a_k)`, where every `a_j` is affine. Let
`D_n = {x in Q^n : 0 <= x_1 <= ... <= x_n <= 1}` and
`F(x)=max(1,sum_i x_i)`. The certificate reports extrema on the real polytope
`D_n` of `C(x)/F(x)` and `C(x)-(n-1)F(x)`, with exact rational arithmetic.

## Theorem 1 — arrangement certificate

For every declared `Expr C`, the procedure enumerates witnesses attaining the
minimum and maximum of `C/F` and the minimum budget slack on `D_n`.

Proof. Add the ordered-cube facets, the plane `sum_i x_i=1`, and every
pairwise branch-equality plane. These planes partition `D_n` into finitely many
polytopes. In each relative interior, every branch choice is fixed, so `C` and
`F` are affine. Budget slack is affine and has a minimum at a polytope vertex.
Restricting the ratio to a line segment gives `(at+b)/(ct+d)`, whose derivative
has constant sign (or is zero) because `ct+d=F>=1`; repeated endpoint moves
give a vertex with the same or a more extreme ratio. Every cell vertex is an
intersection of `n` listed planes, so all are enumerated and evaluated exactly.
∎

## Lemma 2 — rational ReLU lowering

For a declared one-hidden-layer rational network
`b + sum_j w_j x_j + sum_h alpha_h max(0, beta_h + sum_j gamma_hj x_j)`,
`compile_one_hidden_layer` denotes the same function at every real input.

Proof. The output and preactivations are exact affine combinations. For
`alpha_h >= 0`, its term is `max(0, alpha_h preactivation)`; for `alpha_h < 0`,
scaling changes max to min and gives `min(0, alpha_h preactivation)`. Summing
these pointwise identities proves the result. ∎

## Lemma 3 — independent source binding

For a declared finite-agent symmetric deleted-input network, the standalone
checker compares the source evaluator and serialized `Expr` at every vertex of
the common refinement of their branch planes and `D_n`. If it accepts, both
representations agree throughout `D_n`.

Proof. Both functions are affine on each common-refinement cell. Their
difference is affine and zero at every vertex, hence zero on that cell's convex
hull. ∎

## Explicit computational envelope

With `P` distinct arrangement planes in dimension `n`, at most `binom(P,n)`
bases are solved; degenerate bases may be rejected. For an `h`-unit
one-hidden-layer network on `d` inputs, lowering creates one affine term and
at most `h` binary max/min terms per network evaluation. The symmetric
`n`-agent deletion construction has `n` such evaluations. This is a bound
for the declared language, not a polynomial-time claim in network depth or
dimension. The certificate records actual plane, basis, and vertex counts.

## Falsification boundary

The proof excludes floating-point coefficients, products of variables,
non-piecewise-affine activations, undisclosed preprocessing, deeper networks
unless lowered into this language, and source formulas interpreted differently
from the serialized rational specification.
