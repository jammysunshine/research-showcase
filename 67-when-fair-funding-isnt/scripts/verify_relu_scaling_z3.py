#!/usr/bin/env python3
"""Independently challenge Phase-IX certificates with Z3 exact reals."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import z3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_relu_benchmark_z3 import charge, ordered_cube, q, strict_query, verify_witness


def main() -> None:
    source = ROOT / "artifacts" / "phase_ix_relu_scaling_results.json"
    payload = json.loads(source.read_text())
    checks = []
    for entry in payload["entries"]:
        agents = int(entry["agents"])
        reports = [z3.Real(f"phase_ix_{entry['seed']}_{index}") for index in range(agents)]
        total = sum(reports, z3.IntVal(0))
        first_best = z3.If(total >= 1, total, 1)
        total_charge = charge(entry["source_network"], reports)
        certificate = entry["certificate"]
        verify_witness(entry["source_network"], agents, certificate["minimum_slack_witness"], certificate["minimum_budget_slack"], "slack")
        verify_witness(entry["source_network"], agents, certificate["minimum_witness"], certificate["minimum_charge_ratio"], "ratio")
        verify_witness(entry["source_network"], agents, certificate["maximum_witness"], certificate["maximum_charge_ratio"], "ratio")
        common = ordered_cube(reports)
        queries = {
            "strictly_lower_budget_slack": total_charge - (agents - 1) * first_best < q(certificate["minimum_budget_slack"]),
            "strictly_lower_charge_ratio": total_charge < q(certificate["minimum_charge_ratio"]) * first_best,
            "strictly_higher_charge_ratio": total_charge > q(certificate["maximum_charge_ratio"]) * first_best,
        }
        for name, condition in queries.items():
            result = strict_query(common, condition)
            if result != "unsat":
                raise SystemExit(f"Z3 found unresolved certificate challenge ({name}, seed {entry['seed']}): {result}")
            checks.append({"seed": entry["seed"], "query": name, "result": result})
    result = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "z3_version": z3.get_version_string(), "checks": checks}
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = ROOT / "artifacts" / "phase_ix_relu_scaling_z3_certificate.json"
    output.write_text(serialized)
    print(output)
    print(hashlib.sha256(serialized.encode()).hexdigest())


if __name__ == "__main__":
    main()
