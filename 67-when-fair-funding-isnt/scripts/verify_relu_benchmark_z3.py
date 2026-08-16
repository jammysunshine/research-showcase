#!/usr/bin/env python3
"""Use Z3 exact-real feasibility queries to challenge frozen ReLU certificates."""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import z3

ROOT = Path(__file__).resolve().parents[1]


def q(value: str | int | Fraction) -> z3.ArithRef:
    value = Fraction(value)
    return z3.Q(value.numerator, value.denominator)


def network_value(network: dict, inputs: list[z3.ArithRef]) -> z3.ArithRef:
    value = q(network["output_bias"]) + sum(
        (q(weight) * item for weight, item in zip(network["output_weights"], inputs)), z3.IntVal(0)
    )
    for unit in network["hidden"]:
        preactivation = q(unit["bias"]) + sum(
            (q(weight) * item for weight, item in zip(unit["weights"], inputs)), z3.IntVal(0)
        )
        value += q(unit["output_weight"]) * z3.If(preactivation >= 0, preactivation, 0)
    return value


def charge(network: dict, reports: list[z3.ArithRef]) -> z3.ArithRef:
    return sum((network_value(network, reports[:index] + reports[index + 1:])
                for index in range(len(reports))), z3.IntVal(0))


def fraction_charge(network: dict, reports: tuple[Fraction, ...]) -> Fraction:
    total = Fraction(0)
    for deleted in range(len(reports)):
        inputs = reports[:deleted] + reports[deleted + 1:]
        value = Fraction(network["output_bias"]) + sum(
            (Fraction(weight) * item for weight, item in zip(network["output_weights"], inputs)), Fraction(0)
        )
        for unit in network["hidden"]:
            preactivation = Fraction(unit["bias"]) + sum(
                (Fraction(weight) * item for weight, item in zip(unit["weights"], inputs)), Fraction(0)
            )
            value += Fraction(unit["output_weight"]) * max(Fraction(0), preactivation)
        total += value
    return total


def verify_witness(network: dict, agents: int, witness: list[str], expected: str, kind: str) -> None:
    reports = tuple(Fraction(value) for value in witness)
    total = sum(reports, Fraction(0))
    first_best = max(total, Fraction(1))
    source_charge = fraction_charge(network, reports)
    observed = source_charge - Fraction(agents - 1) * first_best if kind == "slack" else source_charge / first_best
    if observed != Fraction(expected):
        raise SystemExit(f"certificate witness mismatch for {kind}: {observed} != {expected}")


def ordered_cube(reports: list[z3.ArithRef]) -> list[z3.BoolRef]:
    return [reports[0] >= 0, reports[-1] <= 1] + [
        reports[index] <= reports[index + 1] for index in range(len(reports) - 1)
    ]


def strict_query(constraints: list[z3.BoolRef], condition: z3.BoolRef) -> str:
    solver = z3.Solver()
    solver.add(*constraints, condition)
    return str(solver.check())


def main() -> None:
    source = ROOT / "artifacts" / "relu_benchmark_results.json"
    payload = json.loads(source.read_text())
    checks = []
    for entry in payload["entries"]:
        agents = int(entry["agents"])
        reports = [z3.Real(f"x_{entry['seed']}_{index}") for index in range(agents)]
        total = sum(reports, z3.IntVal(0))
        first_best = z3.If(total >= 1, total, 1)
        total_charge = charge(entry["source_network"], reports)
        certificate = entry["certificate"]
        verify_witness(entry["source_network"], agents, certificate["minimum_slack_witness"],
                       certificate["minimum_budget_slack"], "slack")
        verify_witness(entry["source_network"], agents, certificate["minimum_witness"],
                       certificate["minimum_charge_ratio"], "ratio")
        verify_witness(entry["source_network"], agents, certificate["maximum_witness"],
                       certificate["maximum_charge_ratio"], "ratio")
        common = ordered_cube(reports)
        specifications = {
            "strictly_lower_budget_slack": total_charge - (agents - 1) * first_best < q(certificate["minimum_budget_slack"]),
            "strictly_lower_charge_ratio": total_charge < q(certificate["minimum_charge_ratio"]) * first_best,
            "strictly_higher_charge_ratio": total_charge > q(certificate["maximum_charge_ratio"]) * first_best,
        }
        for name, condition in specifications.items():
            result = strict_query(common, condition)
            if result != "unsat":
                raise SystemExit(f"Z3 found unresolved certificate challenge ({name}, seed {entry['seed']}): {result}")
            checks.append({"seed": entry["seed"], "query": name, "result": result})
    result = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "z3_version": z3.get_version_string(),
        "checks": checks,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = ROOT / "artifacts" / "relu_benchmark_z3_certificate.json"
    output.write_text(serialized)
    print(output)
    print(hashlib.sha256(serialized.encode()).hexdigest())


if __name__ == "__main__":
    main()
