"""Exact rational-grid audit of Guo's IJCAI-2019 symbolic baseline.

This is a *baseline reproduction*, not a discovery claim.  The implementation
transcribes Equation (6) and the symmetrisation immediately preceding it in
Guo, ``An Asymptotically Optimal VCG Redistribution Mechanism for the Public
Project Problem`` (IJCAI 2019, pp. 317--18).  Every operation is rational.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product


def _max(*values: Fraction) -> Fraction:
    return max(values)


def local_f(a: Fraction, b: Fraction, z: Fraction) -> Fraction:
    """Equation (6), including its `z >= 1` branch, exactly as printed."""
    if z >= 1:
        return (a + b) / 2 + z / 3
    threshold = _max(a + b, 1 - z)
    value = z / 3 + threshold / 3
    if a > 0:
        value += (threshold - _max(b, 1 - z)) / 6
    if b > 0:
        value += (threshold - _max(a, 1 - z)) / 6
    if a > 0 and b > 0:
        value += (1 - z) / 3
    return value


def groves_term(profile_without_agent: tuple[Fraction, ...]) -> Fraction:
    """The anonymous average over ordered pairs in the paper's construction."""
    n = len(profile_without_agent) + 1
    if n < 3:
        raise ValueError("the published dimension-reduction construction requires n >= 3")
    total = Fraction(0)
    for first, a in enumerate(profile_without_agent):
        for second, b in enumerate(profile_without_agent):
            if first == second:
                continue
            z = sum((value for index, value in enumerate(profile_without_agent)
                     if index not in (first, second)), Fraction(0))
            total += local_f(a, b, z)
    return Fraction(3, n * (n - 2)) * total


def total_charge(profile: tuple[Fraction, ...]) -> Fraction:
    return sum((groves_term(profile[:index] + profile[index + 1:])
                for index in range(len(profile))), Fraction(0))


def first_best(profile: tuple[Fraction, ...]) -> Fraction:
    return max(sum(profile, Fraction(0)), Fraction(1))


@dataclass(frozen=True)
class GridAudit:
    agents: int
    denominator: int
    profiles_examined: int
    minimum_budget_slack: Fraction
    minimum_witness: tuple[Fraction, ...]
    maximum_charge_ratio: Fraction
    maximum_witness: tuple[Fraction, ...]
    worst_case_efficiency: Fraction


def audit_grid(agents: int, denominator: int = 4) -> GridAudit:
    """Exhaustively audit the named rational grid; no sampling is involved."""
    if agents < 3 or denominator < 1:
        raise ValueError("agents must be >= 3 and denominator must be positive")
    values = tuple(Fraction(index, denominator) for index in range(denominator + 1))
    rows = []
    for profile in product(values, repeat=agents):
        charge = total_charge(profile)
        s_value = first_best(profile)
        rows.append((charge - (agents - 1) * s_value, charge / s_value, profile))
    minimum_slack, _, minimum_witness = min(rows)
    _, maximum_ratio, maximum_witness = max(rows, key=lambda row: row[1])
    return GridAudit(
        agents=agents,
        denominator=denominator,
        profiles_examined=len(rows),
        minimum_budget_slack=minimum_slack,
        minimum_witness=minimum_witness,
        maximum_charge_ratio=maximum_ratio,
        maximum_witness=maximum_witness,
        worst_case_efficiency=Fraction(agents) - maximum_ratio,
    )
