"""Check the grand-coalition zero-payment lemma against the baseline-audit artifact.

Lemma: for the sum-threshold/critical-value mechanism (n agents, integer
max_value m, integer cost c), if c <= (n-1)*m, the grand-coalition deviation
"every agent reports m" builds the project and charges every agent payment 0
(since the other n-1 agents' reports alone already sum to (n-1)*m >= c). Every
agent's utility becomes exactly v_i, weakly beating their truthful utility and
strictly beating it whenever v_i>0 and the project did not already build
truthfully with payment 0. This is a closed-form sufficient condition for
coalition-manipulability: no search is needed to establish it for a given
(n, m, c). This script cross-checks the condition against every row of the
existing baseline-audit artifact (produced by search) and directly verifies
the zero-payment construction by simulation for a spot set of cases, then
separately checks the payment-forcing argument for the boundary case c=n*m.
"""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_AUDIT = REPO_ROOT / "artifacts" / "public_project_coalition_baseline_audit.json"
OUTPUT = REPO_ROOT / "artifacts" / "public_project_coalition_lemma_certificate.json"


def _allocation(reports: tuple[int, ...], cost: int) -> int:
    return int(sum(reports) >= cost)


def _payment(reports: tuple[int, ...], agent: int, cost: int) -> int:
    if not _allocation(reports, cost):
        return 0
    others_sum = sum(reports) - reports[agent]
    return max(0, cost - others_sum)


def _grand_coalition_all_max_is_zero_payment(n: int, m: int, cost: int) -> bool:
    reports = tuple([m] * n)
    if not _allocation(reports, cost):
        return False
    return all(_payment(reports, i, cost) == 0 for i in range(n))


def _rows_from_domain(domain: dict[str, Any]) -> list[dict[str, Any]]:
    if "cost_rows" in domain:
        return domain["cost_rows"]
    return [row for block in domain["by_n"] for row in block["cost_rows"]]


def _check_against_baseline_audit() -> dict[str, Any]:
    data = json.loads(BASELINE_AUDIT.read_text())
    checked = 0
    false_positives: list[dict[str, Any]] = []
    predicted_fragile_count = 0
    construction_mismatches: list[dict[str, Any]] = []
    for domain in data["domains"]:
        for row in _rows_from_domain(domain):
            n = row["n_agents"]
            m = row["max_value"]
            c = row["cost"]
            actually_fragile = row["min_failing_coalition_size"] is not None
            bound = (n - 1) * m
            predicted_fragile = c <= bound
            checked += 1
            if predicted_fragile:
                predicted_fragile_count += 1
                if not actually_fragile:
                    false_positives.append(
                        {"domain": domain["domain"], "n_agents": n, "max_value": m, "cost": c}
                    )
                if not _grand_coalition_all_max_is_zero_payment(n, m, c):
                    construction_mismatches.append(
                        {"domain": domain["domain"], "n_agents": n, "max_value": m, "cost": c}
                    )
    return {
        "rows_checked": checked,
        "predicted_fragile_count": predicted_fragile_count,
        "false_positive_count": len(false_positives),
        "false_positives": false_positives,
        "construction_mismatch_count": len(construction_mismatches),
        "construction_mismatches": construction_mismatches,
    }


def _boundary_payment_forcing(n: int, m: int) -> dict[str, Any]:
    """At cost=n*m, show every profile that lets any proper coalition build via
    all-reports-at-most-m forces coalition members to pay their full report
    (no free ride), by direct enumeration over all truthful profiles and all
    proper coalition subsets of size 1..n-1."""
    cost = n * m
    counterexamples: list[dict[str, Any]] = []
    checked = 0
    for v in product(range(m + 1), repeat=n):
        for k in range(1, n):
            for coalition in _combinations(range(n), k):
                outsiders = [j for j in range(n) if j not in coalition]
                sum_o = sum(v[j] for j in outsiders)
                if sum_o != (n - k) * m:
                    continue  # outsiders must be maxed for coalition reports alone to reach cost
                truthful_util = sum(
                    v[i] * _allocation(v, cost) - _payment(v, i, cost) for i in coalition
                )
                reports = tuple(m if j in coalition else v[j] for j in range(n))
                checked += 1
                if not _allocation(reports, cost):
                    continue
                deviated_util = sum(
                    v[i] * _allocation(reports, cost) - _payment(reports, i, cost)
                    for i in coalition
                )
                if deviated_util > truthful_util:
                    counterexamples.append(
                        {
                            "n": n,
                            "m": m,
                            "cost": cost,
                            "coalition": list(coalition),
                            "truthful_profile": list(v),
                            "truthful_utility": truthful_util,
                            "deviated_utility": deviated_util,
                        }
                    )
    return {"n": n, "m": m, "cost": cost, "checked": checked, "counterexamples": counterexamples}


def _combinations(iterable, r):
    from itertools import combinations

    return combinations(iterable, r)


def main() -> None:
    baseline_check = _check_against_baseline_audit()
    boundary_checks = [_boundary_payment_forcing(n, m) for n in (3, 4) for m in (2, 3)]
    boundary_counterexample_count = sum(len(b["counterexamples"]) for b in boundary_checks)

    payload = {
        "lemma": (
            "For the sum-threshold/critical-value public-project mechanism, "
            "if cost <= (n-1)*max_value, the grand-coalition all-max deviation "
            "builds the project at zero payment to every agent, weakly "
            "Pareto-dominating truthful reporting for the whole coalition."
        ),
        "baseline_audit_cross_check": baseline_check,
        "boundary_payment_forcing_checks": boundary_checks,
        "boundary_counterexample_count": boundary_counterexample_count,
    }
    digest_source = json.dumps(
        {
            "false_positive_count": baseline_check["false_positive_count"],
            "construction_mismatch_count": baseline_check["construction_mismatch_count"],
            "boundary_counterexample_count": boundary_counterexample_count,
        },
        sort_keys=True,
    ).encode("utf-8")
    payload["digest"] = hashlib.sha256(digest_source).hexdigest()

    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"rows_checked={baseline_check['rows_checked']} "
        f"false_positives={baseline_check['false_positive_count']} "
        f"construction_mismatches={baseline_check['construction_mismatch_count']} "
        f"boundary_counterexamples={boundary_counterexample_count}"
    )


if __name__ == "__main__":
    main()
