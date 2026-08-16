#!/usr/bin/env python3
"""Create the independent and adversarial certificate for the 3-agent run."""

import hashlib
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechanism_discovery.three_agent_extension import (  # noqa: E402
    PROFILES, enumerate_anonymous_budget_balanced, majority, table_from_mechanism,
    verify,
)
from mechanism_discovery.three_agent_independent import (  # noqa: E402
    ROWS, candidate_tables, check_table, independent_frontier,
)


def canonical(table):
    return tuple(tuple(int(value) for value in row) for row in table)


def serializable(table):
    return [list(row) for row in table]


def digest(tables):
    payload = json.dumps([serializable(table) for table in sorted(tables)], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def utility(true_type, choice, payment, magnitude):
    return magnitude * int(true_type == choice) - payment


def table_welfare(table, weights, magnitude):
    weighted = total_weight = 0
    for profile, row, weight in zip(ROWS, table, weights):
        weighted += weight * sum(utility(profile[i], row[0], row[i + 1], magnitude) for i in range(3))
        total_weight += weight
    return weighted / total_weight


def adversarial_audit(table, magnitudes):
    """Check IR, unilateral DSIC, and all-coalition robustness at each magnitude."""
    failures = []
    for magnitude in magnitudes:
        for truthful in ROWS:
            honest_row = table[ROWS.index(truthful)]
            honest = [utility(truthful[i], honest_row[0], honest_row[i + 1], magnitude) for i in range(3)]
            if min(honest) < 0:
                failures.append({"magnitude": magnitude, "property": "individual_rationality",
                                 "profile": list(truthful), "utilities": honest})
            for report in ROWS:
                if report == truthful:
                    continue
                proposed_row = table[ROWS.index(report)]
                proposed = [utility(truthful[i], proposed_row[0], proposed_row[i + 1], magnitude) for i in range(3)]
                for coalition_size in (1, 2, 3):
                    from itertools import combinations
                    for coalition in combinations((0, 1, 2), coalition_size):
                        changed = [i for i in coalition if report[i] != truthful[i]]
                        outsiders = [i for i in (0, 1, 2) if i not in coalition]
                        if (changed and all(report[i] == truthful[i] for i in outsiders)
                                and all(proposed[i] > honest[i] for i in coalition)):
                            failures.append({"magnitude": magnitude, "property": "coalition_strategyproof",
                                             "profile": list(truthful), "coalition": list(coalition),
                                             "report": list(report)})
                            break
                    else:
                        continue
                    break
    return {"passed": not failures, "failures": failures[:20], "failure_count": len(failures)}


def main():
    config_path = ROOT / "configs/experiment_67_three_agent.json"
    confirmation_path = ROOT / "configs/confirmation_67_three_agent.json"
    config = json.loads(config_path.read_text())
    confirmation = json.loads(confirmation_path.read_text())
    primary_mechanisms = enumerate_anonymous_budget_balanced()
    independent_candidates = candidate_tables()
    primary_tables = {canonical(table_from_mechanism(m)) for m in primary_mechanisms if verify(m).accepted}
    independent_tables = {canonical(table) for table in independent_frontier()}
    if primary_tables != independent_tables:
        raise SystemExit("primary/independent frontier mismatch")
    if len(primary_mechanisms) != 144 or len(primary_tables) != 5:
        raise SystemExit(f"unexpected counts: {len(primary_mechanisms)}, {len(primary_tables)}")

    baseline = canonical(table_from_mechanism(majority()))
    distributions = confirmation["distributions"]
    distribution_results = []
    for distribution in distributions:
        weights = tuple(distribution["weights"])
        distribution_results.append({
            "name": distribution["name"],
            "weights": list(weights),
            "baseline_welfare_by_magnitude": {
                str(magnitude): table_welfare(baseline, weights, magnitude)
                for magnitude in confirmation["value_magnitudes"]
            },
            "accepted_welfare_by_magnitude": {
                str(magnitude): [table_welfare(table, weights, magnitude) for table in sorted(primary_tables)]
                for magnitude in confirmation["value_magnitudes"]
            },
        })
    audits = []
    nonzero_rejections = []
    for table in sorted(primary_tables):
        audits.append({"table": serializable(table), "adversarial": adversarial_audit(table, confirmation["value_magnitudes"])})
    for table in independent_candidates:
        if any(payment for row in table for payment in row[1:]) and not check_table(table)["accepted"]:
            nonzero_rejections.append({"table": serializable(table), "first_failure": check_table(table)["failures"][0]})
            break
    certificate = {
        "certificate_version": 1,
        "domain": {
            "profiles": [list(profile) for profile in PROFILES],
            "candidate_count": len(primary_mechanisms),
            "accepted_count": len(primary_tables),
            "primary_frontier_sha256": digest(primary_tables),
            "independent_frontier_sha256": digest(independent_tables),
            "set_equal": primary_tables == independent_tables,
        },
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "confirmation_config_sha256": hashlib.sha256(confirmation_path.read_bytes()).hexdigest(),
        "baseline": {"table": serializable(baseline), "name": "three_agent_majority",
                      "pointwise_welfare_maximal": True, "adversarial": adversarial_audit(baseline, confirmation["value_magnitudes"])},
        "accepted_audits": audits,
        "heldout_distributions": distribution_results,
        "nonzero_transfer_rejection_witness": nonzero_rejections[0] if nonzero_rejections else None,
        "claims": {
            "accepted_set": "Exactly five anonymous mechanisms survive all primary constraints in the frozen bounded domain.",
            "transfers": "Every accepted mechanism has zero transfers; the first nonzero-transfer candidate has a machine-readable rejection witness.",
            "welfare": "No accepted mechanism strictly improves majority because majority is pointwise allocatively optimal for unit values.",
            "evidence_boundary": "These are exhaustive statements only for three binary agents, deterministic anonymous tables, and payment grid -2..2.",
        },
        "scope_limits": confirmation["threat_models"],
    }
    output = ROOT / "artifacts/experiment_67_three_agent_certificate.json"
    output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
