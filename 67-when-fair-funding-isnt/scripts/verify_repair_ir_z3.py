#!/usr/bin/env python3
"""Independently challenge Phase-VII repaired utility minima with Z3."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import z3

ROOT = Path(__file__).resolve().parents[1]


def q(value):
    value = Fraction(value)
    return z3.Q(value.numerator, value.denominator)


def network_value(network, inputs):
    value = q(network["output_bias"]) + sum((q(weight) * item for weight, item in zip(network["output_weights"], inputs)), z3.IntVal(0))
    for unit in network["hidden"]:
        preactivation = q(unit["bias"]) + sum((q(weight) * item for weight, item in zip(unit["weights"], inputs)), z3.IntVal(0))
        value += q(unit["output_weight"]) * z3.If(preactivation >= 0, preactivation, 0)
    return value


def main() -> None:
    source = ROOT / "artifacts" / "repair_ir_tradeoff_study.json"
    payload = json.loads(source.read_text())
    checks = []
    for entry_index, entry in enumerate(payload["entries"]):
        agents = int(entry["agents"])
        minimum = q(entry["repaired_utility_certificate"]["minimum_utility"])
        expected_ir = bool(entry["repair_preserves_ex_post_ir"])
        for deleted in range(agents):
            reports = [z3.Real(f"ir_{entry_index}_{deleted}_{coordinate}") for coordinate in range(agents)]
            first_best = z3.If(sum(reports, z3.IntVal(0)) >= 1, sum(reports, z3.IntVal(0)), 1)
            utility = first_best - network_value(entry["repaired_source"], reports[:deleted] + reports[deleted + 1:])
            solver = z3.Solver()
            solver.add(reports[0] >= 0, reports[-1] <= 1)
            solver.add(*(reports[index] <= reports[index + 1] for index in range(agents - 1)))
            solver.add(utility < minimum)
            result = str(solver.check())
            if result != "unsat":
                raise SystemExit(f"Z3 found lower utility for {entry['name']} agent {deleted}: {result}")
            checks.append({"name": entry["name"], "agent": deleted, "query": "below_reported_minimum_utility", "result": result})
        reports = [z3.Real(f"ir_zero_{entry_index}_{coordinate}") for coordinate in range(agents)]
        first_best = z3.If(sum(reports, z3.IntVal(0)) >= 1, sum(reports, z3.IntVal(0)), 1)
        utility = first_best - network_value(entry["repaired_source"], reports[1:])
        solver = z3.Solver()
        solver.add(reports[0] >= 0, reports[-1] <= 1)
        solver.add(*(reports[index] <= reports[index + 1] for index in range(agents - 1)))
        solver.add(utility < 0)
        result = str(solver.check())
        if (result == "unsat") != expected_ir:
            raise SystemExit(f"Z3 IR predicate disagrees for {entry['name']}: {result}")
        checks.append({"name": entry["name"], "agent": 0, "query": "negative_repaired_utility", "result": result})
    output_payload = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "z3_version": z3.get_version_string(), "checks": checks}
    serialized = json.dumps(output_payload, indent=2, sort_keys=True) + "\n"
    output = ROOT / "artifacts" / "repair_ir_tradeoff_z3_certificate.json"
    output.write_text(serialized)
    print(output)
    print(hashlib.sha256(serialized.encode()).hexdigest())


if __name__ == "__main__":
    main()
