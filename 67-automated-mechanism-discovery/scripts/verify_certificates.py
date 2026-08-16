#!/usr/bin/env python3
"""Produce the independent finite-verification and adversarial certificates."""

import hashlib
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechanism_discovery.adversarial_audit import (  # noqa: E402
    audit_baseline, distributional_welfare, parse_distributions,
    pointwise_no_welfare_improvement,
)
from mechanism_discovery.independent_verifier import (  # noqa: E402
    ROWS, check_table, independent_frontier, table_from_mechanism,
)
from mechanism_discovery.model import Mechanism, Outcome, anonymous_or  # noqa: E402
from mechanism_discovery.search import exhaustive_search  # noqa: E402
from mechanism_discovery.verifier import metrics, verify  # noqa: E402


def canonical(table):
    return tuple(tuple(int(value) for value in row) for row in table)


def serializable(table):
    return [list(row) for row in table]


def digest(tables):
    payload = json.dumps([serializable(table) for table in sorted(tables)], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def as_mechanism(table, name="certificate_table"):
    return Mechanism(tuple(Outcome(row[0], (row[1], row[2])) for row in table), name)


def minimal_zero_transfer_dsic_counterexample(baseline_table):
    candidates = []
    for choices in product((0, 1), repeat=len(ROWS)):
        table = tuple((choice, 0, 0) for choice in choices)
        result = check_table(table)
        if any(failure["property"] == "dsic" for failure in result["failures"]):
            distance = sum(row != reference for row, reference in zip(table, baseline_table))
            candidates.append((distance, table, result["failures"]))
    distance, table, failures = min(candidates, key=lambda row: (row[0], row[1]))
    return {"distance_from_baseline_rows": distance, "table": serializable(table),
            "first_failure": failures[0], "all_failures": failures}


def main():
    confirmation_config = json.loads((ROOT / "configs" / "confirmation_67.json").read_text())
    distributions = parse_distributions(confirmation_config)
    config_hash = hashlib.sha256((ROOT / "configs" / "confirmation_67.json").read_bytes()).hexdigest()

    baseline = anonymous_or()
    baseline_table = table_from_mechanism(baseline)
    primary_rows = exhaustive_search()
    primary_tables = {canonical(table_from_mechanism(row["mechanism"])) for row in primary_rows}
    independent_tables = set(independent_frontier())
    disagreements = sorted(primary_tables ^ independent_tables)
    if disagreements:
        raise SystemExit(f"primary/independent frontier mismatch: {serializable(disagreements[0])}")
    if len(primary_tables) != 4 or len(independent_tables) != 4:
        raise SystemExit("unexpected accepted-frontier cardinality")

    checker_rows = []
    strict_uniform_improvers = []
    neutral_accepted = []
    for table in sorted(primary_tables):
        mechanism = as_mechanism(table)
        primary = verify(mechanism).as_dict()
        independent = check_table(table)
        if primary["accepted"] != independent["accepted"] or not independent["accepted"]:
            raise SystemExit("accepted table did not survive checker comparison")
        metric = metrics(mechanism)
        if primary["neutrality"]:
            neutral_accepted.append(serializable(table))
        if metric["expected_welfare"] > metrics(baseline)["expected_welfare"]:
            strict_uniform_improvers.append(serializable(table))
        checker_rows.append({"table": serializable(table), "primary_accepted": primary["accepted"],
                             "independent_accepted": independent["accepted"], "metrics": metric,
                             "pointwise_vs_baseline": pointwise_no_welfare_improvement(table, baseline_table),
                             "heldout_welfare": distributional_welfare(table, distributions)})

    certificate = {
        "certificate_version": 1,
        "finite_domain": {"profiles": [list(profile) for profile in ROWS], "candidate_count": 1296,
                          "accepted_count": len(primary_tables)},
        "confirmation_config": {"path": "configs/confirmation_67.json", "sha256": config_hash},
        "independent_frontier_comparison": {
            "primary_frontier_sha256": digest(primary_tables),
            "independent_frontier_sha256": digest(independent_tables),
            "set_equal": primary_tables == independent_tables,
        },
        "baseline": {
            "name": baseline.name, "table": serializable(baseline_table),
            "primary": verify(baseline).as_dict(), "independent": check_table(baseline_table),
            "heldout_welfare": distributional_welfare(baseline_table, distributions),
            "adversarial_audit": audit_baseline(baseline),
        },
        "accepted_tables": checker_rows,
        "strict_uniform_welfare_improvers_over_baseline": strict_uniform_improvers,
        "neutral_accepted_tables": neutral_accepted,
        "finite_neutrality_impossibility": {
            "accepted_and_neutral_count": len(neutral_accepted),
            "statement": "No accepted table in the frozen search space is neutral.",
        },
        "minimal_zero_transfer_dsic_counterexample": minimal_zero_transfer_dsic_counterexample(baseline_table),
        "scope_limits": confirmation_config["threat_models"],
    }
    output = ROOT / "artifacts" / "experiment_67_independent_certificate.json"
    output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
