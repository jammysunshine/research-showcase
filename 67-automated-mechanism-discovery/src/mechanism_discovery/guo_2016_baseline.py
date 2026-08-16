"""Exact grid audit of Guo's PRIMA-2016 VCG redistribution rule.

This transcribes Equation (3) and Theorem 1 of Guo, ``Competitive VCG
Redistribution Mechanism for Public Project Problem`` (PRIMA 2016).  Unlike
the ``h``-form rules used in the later papers, Equation (3) directly specifies
redistribution.  We therefore check no-deficit as total VCG revenue minus total
redistribution, precisely in the paper's convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product


def first_best(profile: tuple[Fraction, ...]) -> Fraction:
    return max(sum(profile, Fraction(0)), Fraction(1))


def pivot_groves_term(without_agent: tuple[Fraction, ...]) -> Fraction:
    """Normalized pivotal Groves term for one omitted report."""
    agents = len(without_agent) + 1
    return max(sum(without_agent, Fraction(0)), Fraction(agents - 1, agents))


def vcg_revenue(profile: tuple[Fraction, ...]) -> Fraction:
    """Total VCG payment in the normalized public-project model."""
    n = len(profile)
    return sum(
        (pivot_groves_term(profile[:i] + profile[i + 1:] ) for i in range(n)),
        Fraction(0),
    ) - (n - 1) * first_best(profile)


def raw_redistribution(without_agent: tuple[Fraction, ...]) -> Fraction:
    """Equation (3), with all arithmetic exact."""
    n = len(without_agent) + 1
    if n < 3:
        raise ValueError("Equation (3) requires at least three agents")
    subtotal = sum(without_agent, Fraction(0))
    pair_total = Fraction(0)
    for index in range(n - 1):
        pair_total += max(subtotal - without_agent[index], Fraction(n - 1, n))
    return (pair_total - (n - 2) * max(subtotal, Fraction(1))) / (n - 1)


def upper_bound(agents: int) -> Fraction:
    """The U(n) correction bound from Proposition 1."""
    n = Fraction(agents)
    return Fraction(1, agents - 1) + Fraction(agents - 1, 4 * agents) + Fraction(4 * (agents + 1) ** 3, 27 * agents * (agents - 1) ** 2)


def corrected_redistribution(without_agent: tuple[Fraction, ...]) -> Fraction:
    n = len(without_agent) + 1
    return raw_redistribution(without_agent) - upper_bound(n) / n


def total_corrected_redistribution(profile: tuple[Fraction, ...]) -> Fraction:
    return sum(
        (corrected_redistribution(profile[:i] + profile[i + 1:]) for i in range(len(profile))),
        Fraction(0),
    )


def efficiency_ratio(profile: tuple[Fraction, ...]) -> Fraction:
    """Expression (2) after the theorem's correction."""
    n = len(profile)
    retained = n * first_best(profile) - sum(
        (pivot_groves_term(profile[:i] + profile[i + 1:] ) for i in range(n)),
        Fraction(0),
    ) + total_corrected_redistribution(profile)
    return retained / first_best(profile)


@dataclass(frozen=True)
class GridAudit:
    agents: int
    denominator: int
    profiles_examined: int
    minimum_budget_slack: Fraction
    minimum_budget_witness: tuple[Fraction, ...]
    minimum_efficiency: Fraction
    minimum_efficiency_witness: tuple[Fraction, ...]


def audit_grid(agents: int, denominator: int = 4) -> GridAudit:
    values = tuple(Fraction(i, denominator) for i in range(denominator + 1))
    rows = []
    for profile in product(values, repeat=agents):
        rows.append((
            vcg_revenue(profile) - total_corrected_redistribution(profile),
            efficiency_ratio(profile),
            profile,
        ))
    budget, _, budget_witness = min(rows)
    _, efficiency, efficiency_witness = min(rows, key=lambda row: row[1])
    return GridAudit(agents, denominator, len(rows), budget, budget_witness, efficiency, efficiency_witness)
