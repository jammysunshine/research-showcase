#!/usr/bin/env python3
"""Independently challenge Phase-VI repaired sources with exact-real Z3."""

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
    result = q(network["output_bias"]) + sum(
        (q(weight) * item for weight, item in zip(network["output_weights"], inputs)), z3.IntVal(0)
    )
    for unit in network["hidden"]:
        preactivation = q(unit["bias"]) + sum(
            (q(weight) * item for weight, item in zip(unit["weights"], inputs)), z3.IntVal(0)
        )
        result += q(unit["output_weight"]) * z3.If(preactivation >= 0, preactivation, 0)
    return result


def charge(network, reports):
    return sum((network_value(network, reports[:index] + reports[index + 1:])
                for index in range(len(reports))), z3.IntVal(0))


def main() -> None:
    source = ROOT / "artifacts" / "uniform_repair_study.json"
    payload = json.loads(source.read_text())
    checks = []
    for index, entry in enumerate(payload["entries"]):
        agents = int(entry["agents"])
        reports = [z3.Real(f"repair_{index}_{coordinate}") for coordinate in range(agents)]
        total = sum(reports, z3.IntVal(0))
        first_best = z3.If(total >= 1, total, 1)
        solver = z3.Solver()
        solver.add(reports[0] >= 0, reports[-1] <= 1)
        solver.add(*(reports[coordinate] <= reports[coordinate + 1] for coordinate in range(agents - 1)))
        solver.add(charge(entry["repaired_source"], reports) - (agents - 1) * first_best < 0)
        result = str(solver.check())
        if result != "unsat":
            raise SystemExit(f"Z3 found repaired deficit for {entry['name']}: {result}")
        checks.append({"name": entry["name"], "query": "strictly_negative_repaired_slack", "result": result})
    output_payload = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "z3_version": z3.get_version_string(),
        "checks": checks,
    }
    serialized = json.dumps(output_payload, indent=2, sort_keys=True) + "\n"
    output = ROOT / "artifacts" / "uniform_repair_z3_certificate.json"
    output.write_text(serialized)
    print(output)
    print(hashlib.sha256(serialized.encode()).hexdigest())


if __name__ == "__main__":
    main()
