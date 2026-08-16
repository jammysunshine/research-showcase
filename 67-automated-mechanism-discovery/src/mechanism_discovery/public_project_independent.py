"""Independent checker for public-project certificates.

This module intentionally does not import ``public_project``.  It reconstructs
the payment rule and all finite constraints from serialized allocation tables.
"""

from __future__ import annotations

from itertools import combinations, permutations, product


def check(table: dict, *, max_coalition_size: int = 1) -> dict:
    if max_coalition_size < 1:
        raise ValueError("max_coalition_size must be at least 1")
    n = int(table["n_agents"])
    max_value = int(table["max_value"])
    cost = int(table["cost"])
    allocation = {tuple(row[0]): int(row[1]) for row in table["allocation_by_state"]}

    def q(reports):
        return allocation[tuple(sorted(reports))]

    def payment(reports, agent):
        if not q(reports):
            return 0
        probe = list(reports)
        for value in range(max_value + 1):
            probe[agent] = value
            if q(tuple(probe)):
                return value
        raise AssertionError("allocation is active but has no finite threshold")

    def utility(true_values, reports, agent):
        return true_values[agent] * q(reports) - payment(reports, agent)

    witnesses = []
    profiles = list(product(range(max_value + 1), repeat=n))
    for reports in profiles:
        pays = tuple(payment(reports, i) for i in range(n))
        if q(reports) == 0 and any(pays):
            witnesses.append({"property": "no_payment_without_project", "profile": reports})
        if q(reports) and sum(pays) < cost:
            witnesses.append({"property": "weak_budget_balance", "profile": reports, "payments": pays})
        for agent in range(n):
            truthful = utility(reports, reports, agent)
            if truthful < 0:
                witnesses.append({"property": "ex_post_ir", "profile": reports, "agent": agent})
            for alternative in range(max_value + 1):
                if alternative == reports[agent]:
                    continue
                deviating = list(reports)
                deviating[agent] = alternative
                if utility(reports, tuple(deviating), agent) > truthful:
                    witnesses.append({"property": "dsic", "profile": reports, "agent": agent, "deviation": alternative})
        for perm in permutations(range(n)):
            if q(reports) != q(tuple(reports[i] for i in perm)):
                witnesses.append({"property": "anonymity", "profile": reports, "permutation": perm})
    if max_coalition_size >= 2:
        coalition_sizes = range(2, min(max_coalition_size, n) + 1)
        for reports in profiles:
            truthful_utils = [utility(reports, reports, agent) for agent in range(n)]
            for coalition_size in coalition_sizes:
                for coalition in combinations(range(n), coalition_size):
                    for deviations in product(range(max_value + 1), repeat=coalition_size):
                        if all(reports[agent] == deviated for agent, deviated in zip(coalition, deviations)):
                            continue
                        deviating = list(reports)
                        for agent, deviated in zip(coalition, deviations):
                            deviating[agent] = deviated
                        deviating_reports = tuple(deviating)
                        baseline_total = sum(truthful_utils[agent] for agent in coalition)
                        deviating_total = sum(
                            utility(reports, deviating_reports, agent)
                            for agent in coalition
                        )
                        if deviating_total > baseline_total:
                            witnesses.append(
                                {
                                    "property": "coalitional_dsic",
                                    "profile": reports,
                                    "coalition": list(coalition),
                                    "deviation": list(deviations),
                                    "gain": deviating_total - baseline_total,
                                    "coalition_size": coalition_size,
                                }
                            )
                            break
                    else:
                        continue
                    break
    properties = {w["property"] for w in witnesses}
    return {
        "accepted": not properties,
        "witnesses": witnesses,
        "dsic": "dsic" not in properties,
        "ex_post_ir": "ex_post_ir" not in properties,
        "weak_budget_balance": "weak_budget_balance" not in properties,
        "anonymity": "anonymity" not in properties,
        "coalitional_dsic": "coalitional_dsic" not in properties,
        "max_coalition_size": max_coalition_size,
    }


def independent_frontier(
    table_rows: list[dict],
    *,
    max_coalition_size: int = 1,
) -> list[str]:
    """Return frontier mechanism names under an independent replay with optional coalition cap."""
    return [
        row["mechanism"]["name"]
        for row in table_rows
        if check(row["mechanism"], max_coalition_size=max_coalition_size)["accepted"]
    ]
