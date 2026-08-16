"""Standalone replay of the IJCAI-2019 rational-grid baseline audit.

It intentionally does not import the primary baseline implementation.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product


def _f(a: Fraction, b: Fraction, z: Fraction) -> Fraction:
    if z >= 1:
        return (a + b) / 2 + z / 3
    top = max(a + b, 1 - z)
    result = z / 3 + top / 3
    if a > 0:
        result += (top - max(b, 1 - z)) / 6
    if b > 0:
        result += (top - max(a, 1 - z)) / 6
    if a > 0 and b > 0:
        result += (1 - z) / 3
    return result


def _term(others: tuple[Fraction, ...]) -> Fraction:
    n = len(others) + 1
    total = Fraction(0)
    for first in range(n - 1):
        for second in range(n - 1):
            if first != second:
                total += _f(others[first], others[second],
                            sum((value for index, value in enumerate(others)
                                 if index != first and index != second), Fraction(0)))
    return Fraction(3, n * (n - 2)) * total


def replay(agents: int, denominator: int = 4) -> dict[str, object]:
    values = tuple(Fraction(index, denominator) for index in range(denominator + 1))
    rows = []
    for profile in product(values, repeat=agents):
        charge = sum((_term(profile[:index] + profile[index + 1:])
                      for index in range(agents)), Fraction(0))
        first_best = max(sum(profile, Fraction(0)), Fraction(1))
        rows.append((charge - (agents - 1) * first_best, charge / first_best, profile))
    minimum_slack, _, min_witness = min(rows)
    _, maximum_ratio, max_witness = max(rows, key=lambda row: row[1])
    return {
        "agents": agents,
        "denominator": denominator,
        "profiles_examined": len(rows),
        "minimum_budget_slack": minimum_slack,
        "minimum_witness": min_witness,
        "maximum_charge_ratio": maximum_ratio,
        "maximum_witness": max_witness,
        "worst_case_efficiency": Fraction(agents) - maximum_ratio,
    }
