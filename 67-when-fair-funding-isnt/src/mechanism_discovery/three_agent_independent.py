"""Independent checker for the three-agent extension.

No primary extension module is imported.  Rows are primitive
``(choice,payment_0,payment_1,payment_2)`` tuples and the candidate generator
reconstructs the anonymous budget-balance equations independently.
"""

from itertools import combinations, product

ROWS = tuple(product((0, 1), repeat=3))
ROW_INDEX = {profile: index for index, profile in enumerate(ROWS)}
GRID = (-2, -1, 0, 1, 2)
PERMUTATIONS = ((1, 0, 2), (2, 1, 0), (0, 2, 1))


def _utility(preference, row, agent):
    return int(preference == row[0]) - row[agent + 1]


def check_table(table):
    failures = []
    if len(table) != 8:
        return {"accepted": False, "failures": [{"property": "totality"}]}
    for profile, row in zip(ROWS, table):
        if len(row) != 4 or row[0] not in (0, 1):
            failures.append({"property": "feasibility", "profile": list(profile), "detail": repr(row)})
            continue
        if sum(row[1:]) != 0:
            failures.append({"property": "budget_balance", "profile": list(profile), "detail": repr(row[1:])})
        for agent in range(3):
            honest = _utility(profile[agent], row, agent)
            if honest < 0:
                failures.append({"property": "individual_rationality", "profile": list(profile), "agent": agent})
            lie = list(profile)
            lie[agent] = 1 - lie[agent]
            deviating = _utility(profile[agent], table[ROW_INDEX[tuple(lie)]], agent)
            if deviating > honest:
                failures.append({"property": "dsic", "profile": list(profile), "agent": agent,
                                 "detail": f"{honest}->{deviating}"})

    for profile in ROWS:
        source = table[ROW_INDEX[profile]]
        for permutation in PERMUTATIONS:
            permuted_profile = tuple(profile[i] for i in permutation)
            target = table[ROW_INDEX[permuted_profile]]
            expected_payments = tuple(source[i + 1] for i in permutation)
            if target[0] != source[0] or target[1:] != expected_payments:
                failures.append({"property": "anonymity", "profile": list(profile),
                                 "detail": f"permutation={permutation}"})

    for profile, row in zip(ROWS, table):
        utilities = [_utility(profile[i], row, i) for i in range(3)]
        if max(utilities) - min(utilities) > 1:
            failures.append({"property": "fairness", "profile": list(profile),
                             "detail": f"disparity={max(utilities)-min(utilities)}"})

    agents = (0, 1, 2)
    for truthful in ROWS:
        honest = table[ROW_INDEX[truthful]]
        honest_utilities = [_utility(truthful[i], honest, i) for i in agents]
        for size in (1, 2, 3):
            for coalition in combinations(agents, size):
                for reports in product((0, 1), repeat=size):
                    report = list(truthful)
                    for agent, bit in zip(coalition, reports):
                        report[agent] = bit
                    report = tuple(report)
                    if report == truthful:
                        continue
                    proposed = table[ROW_INDEX[report]]
                    proposed_utilities = [_utility(truthful[i], proposed, i) for i in agents]
                    if all(proposed_utilities[i] > honest_utilities[i] for i in coalition):
                        failures.append({"property": "coalition_strategyproof", "profile": list(truthful),
                                         "coalition": list(coalition), "detail": f"joint_report={list(report)}"})
    for profile in ROWS:
        complement = tuple(1 - bit for bit in profile)
        if table[ROW_INDEX[complement]][0] != 1 - table[ROW_INDEX[profile]][0]:
            failures.append({"property": "neutrality", "profile": list(profile)})
    # Neutrality is an audited diagnostic, not part of the frozen acceptance
    # predicate (matching the primary verifier and the preregistration).
    accepted = not any(failure["property"] != "neutrality" for failure in failures)
    return {"accepted": accepted, "failures": failures}


def _anonymous_tables():
    count_one = tuple((p1, p0) for p1 in GRID for p0 in GRID if p1 + 2 * p0 == 0)
    count_two = tuple((p1, p0) for p1 in GRID for p0 in GRID if 2 * p1 + p0 == 0)
    for choices in product((0, 1), repeat=4):
        for row_one in count_one:
            for row_two in count_two:
                rows = []
                for profile in ROWS:
                    count = sum(profile)
                    if count == 1:
                        p1, p0 = row_one
                        payments = tuple(p1 if bit else p0 for bit in profile)
                    elif count == 2:
                        p1, p0 = row_two
                        payments = tuple(p1 if bit else p0 for bit in profile)
                    else:
                        payments = (0, 0, 0)
                    rows.append((choices[count], *payments))
                yield tuple(rows)


def candidate_tables():
    return tuple(_anonymous_tables())


def independent_frontier():
    return [table for table in _anonymous_tables() if check_table(table)["accepted"]]
