"""Exact finite-grid synthesis for anonymous VCG redistribution rules.

The public project costs one.  Each agent type is in a finite subset of
``[0, 1]`` and the efficient allocation builds exactly when the reported sum
reaches one.  Following the Groves representation used in the public-project
redistribution literature, a symmetric rule is represented by ``h(theta_-i)``.
At a profile theta, truthful utility of agent i is ``S(theta) - h(theta_-i)``,
where ``S(theta) = max(sum(theta), 1)``.  Hence individual rationality is
``h(theta_-i) <= S(theta)`` and no deficit is
``sum_i h(theta_-i) >= (n - 1) S(theta)``.

This module deliberately uses only exact ``Fraction`` arithmetic.  It is a
finite-grid oracle, not a claim about the continuous public-project problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from typing import Iterable, Sequence


Vector = tuple[Fraction, ...]


@dataclass(frozen=True)
class Inequality:
    """A single constraint ``coefficients · x >= bound``."""

    coefficients: Vector
    bound: Fraction
    label: str


@dataclass(frozen=True)
class ExactSolution:
    values: Vector
    objective: Fraction
    active_labels: tuple[str, ...]
    examined_bases: int
    feasible_vertices: int


@dataclass(frozen=True)
class CegisResult:
    """Trace from counterexample-guided constraint generation."""

    solution: ExactSolution
    rounds: int
    added_labels: tuple[str, ...]


def sorted_tuples(values: Sequence[Fraction], width: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(combinations_with_replacement(values, width))


def combinations_with_replacement(values: Sequence[Fraction], width: int) -> Iterable[tuple[Fraction, ...]]:
    if width == 0:
        yield ()
        return
    for indices in combinations(range(len(values) + width - 1), width):
        shifted = tuple(index - position for position, index in enumerate(indices))
        yield tuple(values[index] for index in shifted)


def profile_space(values: Sequence[Fraction], agents: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(product(values, repeat=agents))


def first_best(profile: Sequence[Fraction]) -> Fraction:
    return max(sum(profile, Fraction(0)), Fraction(1))


def _others(profile: Sequence[Fraction], agent: int) -> tuple[Fraction, ...]:
    return tuple(sorted(profile[:agent] + profile[agent + 1:]))


def synthesis_constraints(values: Sequence[Fraction], agents: int) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Inequality, ...]]:
    """Return anonymous h inputs and all frozen grid constraints.

    We include nonnegative Groves terms as a pre-specified normalization.
    A Groves term is not, by itself, a realized payment: the outcome-dependent
    term also enters transfers.  Thus this restriction must not be described
    as ruling out subsidies; no-deficit is the economic constraint checked
    separately below.
    """
    if agents < 2:
        raise ValueError("at least two agents are required")
    inputs = sorted_tuples(values, agents - 1)
    index = {entry: position for position, entry in enumerate(inputs)}
    constraints: list[Inequality] = []
    zero = tuple(Fraction(0) for _ in inputs)
    for position, entry in enumerate(inputs):
        coefficients = list(zero)
        coefficients[position] = Fraction(1)
        constraints.append(Inequality(tuple(coefficients), Fraction(0), f"nonnegative:{entry}"))

    # For a fixed h input, IR only needs its tightest S(theta) upper bound.
    ir_bounds: dict[tuple[Fraction, ...], Fraction] = {entry: Fraction(10**9) for entry in inputs}
    deficit: dict[tuple[Fraction, ...], Fraction] = {}
    for profile in profile_space(values, agents):
        s_value = first_best(profile)
        other_inputs = tuple(_others(profile, agent) for agent in range(agents))
        for entry in other_inputs:
            ir_bounds[entry] = min(ir_bounds[entry], s_value)
        canonical_profile = tuple(sorted(profile))
        row = [Fraction(0) for _ in inputs]
        for entry in other_inputs:
            row[index[entry]] += 1
        deficit[canonical_profile] = (tuple(row), Fraction(agents - 1) * s_value)

    for entry, upper in ir_bounds.items():
        coefficients = [Fraction(0) for _ in inputs]
        coefficients[index[entry]] = Fraction(-1)
        constraints.append(Inequality(tuple(coefficients), -upper, f"ir:{entry}"))
    for profile, (coefficients, bound) in deficit.items():
        constraints.append(Inequality(coefficients, bound, f"no_deficit:{profile}"))
    return inputs, tuple(constraints)


def expected_charge_coefficients(values: Sequence[Fraction], probabilities: Sequence[Fraction], agents: int,
                                 inputs: Sequence[tuple[Fraction, ...]]) -> Vector:
    if len(values) != len(probabilities) or sum(probabilities, Fraction(0)) != 1:
        raise ValueError("probabilities must match values and sum exactly to one")
    index = {entry: position for position, entry in enumerate(inputs)}
    coefficients = [Fraction(0) for _ in inputs]
    for profile in profile_space(values, agents):
        probability = Fraction(1)
        for value in profile:
            probability *= probabilities[values.index(value)]
        for agent in range(agents):
            coefficients[index[_others(profile, agent)]] += probability
    return tuple(coefficients)


def _dot(left: Sequence[Fraction], right: Sequence[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def _solve_square(rows: Sequence[Sequence[Fraction]], bounds: Sequence[Fraction]) -> Vector | None:
    """Exact Gauss-Jordan elimination; ``None`` means singular."""
    size = len(bounds)
    augmented = [list(row) + [bound] for row, bound in zip(rows, bounds)]
    for column in range(size):
        pivot = next((row for row in range(column, size) if augmented[row][column]), None)
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [item / scale for item in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            scale = augmented[row][column]
            if scale:
                augmented[row] = [item - scale * base for item, base in zip(augmented[row], augmented[column])]
    return tuple(row[-1] for row in augmented)


def solve_by_vertex_enumeration(constraints: Sequence[Inequality], objective: Vector) -> ExactSolution:
    """Minimize a linear objective by complete exact vertex enumeration.

    The caller must provide a bounded, pointed finite-dimensional polyhedron.
    Every candidate basis is checked against every inequality, so the returned
    solution is a certificate for this explicit finite LP.
    """
    dimension = len(objective)
    if dimension == 0:
        raise ValueError("empty LP")
    best: ExactSolution | None = None
    examined = 0
    feasible = 0
    for basis in combinations(constraints, dimension):
        examined += 1
        candidate = _solve_square([item.coefficients for item in basis], [item.bound for item in basis])
        if candidate is None or any(_dot(item.coefficients, candidate) < item.bound for item in constraints):
            continue
        feasible += 1
        solution = ExactSolution(candidate, _dot(objective, candidate),
                                 tuple(item.label for item in basis), examined, feasible)
        if best is None or solution.objective < best.objective or (
            solution.objective == best.objective and solution.values < best.values
        ):
            best = solution
    if best is None:
        raise ValueError("LP has no feasible vertex")
    return ExactSolution(best.values, best.objective, best.active_labels, examined, feasible)


def verify_solution(constraints: Sequence[Inequality], objective: Vector, solution: ExactSolution) -> list[str]:
    """Return human-readable violations; empty means primal feasibility/value consistency."""
    failures = []
    if len(solution.values) != len(objective):
        return ["dimension mismatch"]
    if _dot(objective, solution.values) != solution.objective:
        failures.append("objective mismatch")
    for constraint in constraints:
        if _dot(constraint.coefficients, solution.values) < constraint.bound:
            failures.append(constraint.label)
    return failures


def solve_counterexample_guided(constraints: Sequence[Inequality], objective: Vector) -> CegisResult:
    """Solve the same LP by adding one maximally violated profile constraint per round.

    Nonnegativity and IR constraints bound the initial restricted LP.  The
    remaining no-deficit constraints are deliberately withheld until the
    verifier produces a concrete violating profile.  This is a second,
    algorithmically distinct discovery path; its final candidate is always
    checked against the full constraint set.
    """
    active = [item for item in constraints if not item.label.startswith("no_deficit:")]
    withheld = [item for item in constraints if item.label.startswith("no_deficit:")]
    added: list[str] = []
    while True:
        solution = solve_by_vertex_enumeration(active, objective)
        violations = [item for item in withheld if _dot(item.coefficients, solution.values) < item.bound]
        if not violations:
            return CegisResult(solution, len(added), tuple(added))
        # Deterministic strongest violation; label resolves exact ties.
        worst = min(violations, key=lambda item: (_dot(item.coefficients, solution.values) - item.bound, item.label))
        active.append(worst)
        withheld.remove(worst)
        added.append(worst.label)
