"""Standalone finite checker for Experiment 67.

This module deliberately imports neither the primary verifier nor its utility
function.  It uses a row-table encoding and reconstructs the finite search
space itself, so agreement is a meaningful implementation cross-check.
"""

from itertools import product


ROWS = ((0, 0), (0, 1), (1, 0), (1, 1))
ROW_INDEX = {profile: index for index, profile in enumerate(ROWS)}
PAYMENT_GRID = (-1, 0, 1)


def _payoff(true_preference: int, row: tuple[int, int, int], agent: int) -> int:
    choice, payment_0, payment_1 = row
    return int(choice == true_preference) - (payment_0, payment_1)[agent]


def table_from_mechanism(mechanism) -> tuple[tuple[int, int, int], ...]:
    """Adapter kept at the edge: verification itself is model-independent."""
    return tuple(
        (outcome.choice, outcome.payments[0], outcome.payments[1])
        for outcome in mechanism.outcomes
    )


def check_table(table: tuple[tuple[int, int, int], ...]) -> dict:
    """Check every finite constraint and emit primitive, serializable witnesses."""
    failures: list[dict] = []
    if len(table) != len(ROWS):
        return {"accepted": False, "failures": [{"property": "totality", "detail": str(len(table))}]}
    for profile, row in zip(ROWS, table):
        if len(row) != 3 or row[0] not in (0, 1):
            failures.append({"property": "feasibility", "profile": list(profile), "detail": repr(row)})
            continue
        if row[1] + row[2] != 0:
            failures.append({"property": "budget_balance", "profile": list(profile), "detail": repr(row[1:])})
        for agent in (0, 1):
            truth = profile[agent]
            honest = _payoff(truth, row, agent)
            if honest < 0:
                failures.append({"property": "individual_rationality", "profile": list(profile), "agent": agent,
                                 "detail": str(honest)})
            lie_profile = list(profile)
            lie_profile[agent] = 1 - truth
            lied = _payoff(truth, table[ROW_INDEX[tuple(lie_profile)]], agent)
            if lied > honest:
                failures.append({"property": "dsic", "profile": list(profile), "agent": agent,
                                 "deviation": 1 - truth, "detail": f"{honest}->{lied}"})
    # Frozen fairness predicate: anonymity plus truthful disparity at most one.
    for profile in ROWS:
        swapped = (profile[1], profile[0])
        row, swapped_row = table[ROW_INDEX[profile]], table[ROW_INDEX[swapped]]
        if row[0] != swapped_row[0] or row[1:] != (swapped_row[2], swapped_row[1]):
            failures.append({"property": "anonymity", "profile": list(profile),
                             "detail": f"vs {list(swapped)}"})
    for profile, row in zip(ROWS, table):
        disparity = abs(_payoff(profile[0], row, 0) - _payoff(profile[1], row, 1))
        if disparity > 1:
            failures.append({"property": "fairness", "profile": list(profile),
                             "detail": f"disparity={disparity}"})
    # Frozen coalition predicate: no joint report makes both fixed identities
    # strictly better off at their true profile.
    for true_profile in ROWS:
        honest = table[ROW_INDEX[true_profile]]
        for report in ROWS:
            if report == true_profile:
                continue
            proposed = table[ROW_INDEX[report]]
            if all(_payoff(true_profile[agent], proposed, agent) >
                   _payoff(true_profile[agent], honest, agent) for agent in (0, 1)):
                failures.append({"property": "coalition_strategyproof", "profile": list(true_profile),
                                 "detail": f"joint_report={list(report)}"})
    return {"accepted": not failures, "failures": failures}


def independent_frontier() -> list[tuple[tuple[int, int, int], ...]]:
    """Enumerate the 6^4 budget-balanced tables without primary-search imports."""
    rows = tuple((choice, payment_0, -payment_0)
                 for choice in (0, 1) for payment_0 in PAYMENT_GRID)
    return [table for table in product(rows, repeat=len(ROWS)) if check_table(table)["accepted"]]


def check(mechanism) -> dict:
    """Backward-compatible adapter for callers using the canonical model object."""
    return check_table(table_from_mechanism(mechanism))
