"""Independent exact replay for the PRIMA-2016 Equation (3) audit."""

from fractions import Fraction
from itertools import product


def replay(agents: int, denominator: int = 4) -> dict:
    n = Fraction(agents)
    correction = (Fraction(1, agents - 1) + Fraction(agents - 1, 4 * agents)
                  + Fraction(4 * (agents + 1) ** 3, 27 * agents * (agents - 1) ** 2)) / n
    values = tuple(Fraction(i, denominator) for i in range(denominator + 1))
    rows = []
    for profile in product(values, repeat=agents):
        social = max(sum(profile, Fraction(0)), Fraction(1))
        external = sum((max(sum(profile[:i] + profile[i + 1:], Fraction(0)), Fraction(agents - 1, agents)) for i in range(agents)), Fraction(0))
        vcg = external - (agents - 1) * social
        redistribution = Fraction(0)
        for i in range(agents):
            others = profile[:i] + profile[i + 1:]
            subtotal = sum(others, Fraction(0))
            redistribution += (sum((max(subtotal - value, Fraction(agents - 1, agents)) for value in others), Fraction(0)) - (agents - 2) * max(subtotal, Fraction(1))) / (agents - 1) - correction
        rows.append((vcg - redistribution, (agents * social - external + redistribution) / social, profile))
    budget, _, budget_witness = min(rows)
    _, efficiency, efficiency_witness = min(rows, key=lambda row: row[1])
    return {
        "agents": agents, "denominator": denominator, "profiles_examined": len(rows),
        "minimum_budget_slack": budget, "minimum_budget_witness": budget_witness,
        "minimum_efficiency": efficiency, "minimum_efficiency_witness": efficiency_witness,
    }
