#!/usr/bin/env python3
"""Run the frozen Phase-IX exact rational-ReLU scaling study."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mechanism_discovery.max_affine_independent import replay_deleted_input_network
from src.mechanism_discovery.piecewise_affine import certify_ordered_public_project_charge
from src.mechanism_discovery.relu_benchmark import deleted_input_charge, deterministic_network


def encode(value):
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, tuple):
        return [encode(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: encode(item) for key, item in value.__dict__.items()}
    return value


def mutated_output_bias(source: dict) -> dict:
    """Return a deterministic nonzero source mutation for the negative control."""
    mutated = {**source}
    mutated["output_bias"] = str(Fraction(source["output_bias"]) + Fraction(1, 7))
    return mutated


def main() -> None:
    config_path = ROOT / "configs" / "phase_ix_relu_scaling.json"
    config = json.loads(config_path.read_text())
    entries = []
    started = time.perf_counter()
    for case in config["cases"]:
        agents, width, seed = int(case["agents"]), int(case["width"]), int(case["seed"])
        source = deterministic_network(seed, agents - 1, width, int(config["coefficient_denominator"]))
        case_started = time.perf_counter()
        compiled = encode(certify_ordered_public_project_charge(deleted_input_charge(source, agents), agents))
        direct = replay_deleted_input_network(source, agents)
        if compiled != direct:
            raise SystemExit(f"source/compiler disagreement for seed {seed}")
        mutation_changes = replay_deleted_input_network(mutated_output_bias(source), agents) != direct
        if not mutation_changes:
            raise SystemExit(f"mutation control failed for seed {seed}")
        entries.append({
            **case,
            "source_network": source,
            "certificate": compiled,
            "mutation_changes_certificate": mutation_changes,
            "elapsed_seconds": round(time.perf_counter() - case_started, 6),
        })
    payload = {
        "benchmark": config["benchmark"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "total_elapsed_seconds": round(time.perf_counter() - started, 6),
        "entries": entries,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output = ROOT / "artifacts" / "phase_ix_relu_scaling_results.json"
    output.write_text(serialized)
    print(output)
    print(hashlib.sha256(serialized.encode()).hexdigest())


if __name__ == "__main__":
    main()
