"""Independent exact replay for the finite VCG-redistribution LP.

This intentionally does not import the synthesis module.  It reconstructs the
anonymous type pairs, constraints, and the candidate's objective using only
primitive fractions, providing a meaningful check on emitted certificates.
"""

from fractions import Fraction
from itertools import combinations, product
from typing import Sequence


def _pairs(values: Sequence[Fraction]) -> tuple[tuple[Fraction, Fraction], ...]:
    return tuple((values[left], values[right]) for left in range(len(values)) for right in range(left, len(values)))


def _first_best(profile: tuple[Fraction, Fraction, Fraction]) -> Fraction:
    return max(sum(profile, Fraction(0)), Fraction(1))


def _dot(left, right):
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def _solve(rows, bounds):
    size = len(bounds)
    table = [list(row) + [bound] for row, bound in zip(rows, bounds)]
    for column in range(size):
        pivot = next((row for row in range(column, size) if table[row][column]), None)
        if pivot is None:
            return None
        table[column], table[pivot] = table[pivot], table[column]
        divisor = table[column][column]
        table[column] = [value / divisor for value in table[column]]
        for row in range(size):
            if row != column and table[row][column]:
                multiplier = table[row][column]
                table[row] = [value - multiplier * pivot_value for value, pivot_value in zip(table[row], table[column])]
    return tuple(row[-1] for row in table)


def _constraints(values: Sequence[Fraction]):
    pairs = _pairs(values)
    index = {pair: position for position, pair in enumerate(pairs)}
    size = len(pairs)
    result = []
    for position in range(size):
        row = [Fraction(0)] * size
        row[position] = 1
        result.append((tuple(row), Fraction(0), f"nonnegative:{pairs[position]}"))
    upper = {pair: Fraction(10**9) for pair in pairs}
    deficit = {}
    for profile in product(values, repeat=3):
        s_value = _first_best(profile)
        omitted = []
        for agent in range(3):
            pair = tuple(sorted(profile[:agent] + profile[agent + 1:]))
            omitted.append(pair)
            upper[pair] = min(upper[pair], s_value)
        row = [Fraction(0)] * size
        for pair in omitted:
            row[index[pair]] += 1
        deficit[tuple(sorted(profile))] = (tuple(row), 2 * s_value)
    for pair, bound in upper.items():
        row = [Fraction(0)] * size
        row[index[pair]] = -1
        result.append((tuple(row), -bound, f"ir:{pair}"))
    for profile, (row, bound) in deficit.items():
        result.append((row, bound, f"no_deficit:{profile}"))
    return pairs, tuple(result)


def replay(values: Sequence[Fraction], probabilities: Sequence[Fraction], candidate: Sequence[Fraction]) -> dict:
    """Check feasibility and independently recompute the global vertex optimum."""
    if len(values) != 3 or len(probabilities) != 3 or sum(probabilities, Fraction(0)) != 1:
        raise ValueError("the frozen independent replay supports the three-point, three-agent grid")
    pairs, constraints = _constraints(values)
    if len(candidate) != len(pairs):
        return {"accepted": False, "failures": ["candidate dimension mismatch"]}
    weights = [Fraction(0)] * len(pairs)
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    for profile in product(values, repeat=3):
        probability = probabilities[values.index(profile[0])] * probabilities[values.index(profile[1])] * probabilities[values.index(profile[2])]
        for agent in range(3):
            pair = tuple(sorted(profile[:agent] + profile[agent + 1:]))
            weights[pair_index[pair]] += probability
    violations = [label for row, bound, label in constraints if _dot(row, candidate) < bound]
    optimum = None
    bases = 0
    feasible = 0
    for basis in combinations(constraints, len(pairs)):
        bases += 1
        point = _solve([row for row, _, _ in basis], [bound for _, bound, _ in basis])
        if point is None or any(_dot(row, point) < bound for row, bound, _ in constraints):
            continue
        feasible += 1
        value = _dot(weights, point)
        if optimum is None or value < optimum:
            optimum = value
    candidate_value = _dot(weights, candidate)
    return {
        "accepted": not violations and candidate_value == optimum,
        "feasibility_failures": violations,
        "candidate_objective": candidate_value,
        "optimal_objective": optimum,
        "examined_bases": bases,
        "feasible_vertices": feasible,
    }
