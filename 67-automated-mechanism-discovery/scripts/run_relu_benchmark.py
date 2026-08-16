#!/usr/bin/env python3
"""Run the frozen Phase-V source/compiler exact-certificate benchmark."""

from __future__ import annotations

import hashlib
import json
import sys
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


def main() -> None:
    config_path = ROOT / "configs" / "relu_benchmark.json"
    config = json.loads(config_path.read_text())
    entries = []
    for case in config["cases"]:
        agents, width, seed = int(case["agents"]), int(case["width"]), int(case["seed"])
        source = deterministic_network(seed, agents - 1, width, int(config["coefficient_denominator"]))
        compiled = encode(certify_ordered_public_project_charge(deleted_input_charge(source, agents), agents))
        direct = replay_deleted_input_network(source, agents)
        if compiled != direct:
            raise SystemExit(f"source/compiler disagreement for seed {seed}")
        entries.append({**case, "source_network": source, "certificate": compiled})
    payload = {
        "benchmark": config["benchmark"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "entries": entries,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output = ROOT / "artifacts" / "relu_benchmark_results.json"
    output.write_text(serialized)
    print(output)
    print(hashlib.sha256(serialized.encode()).hexdigest())


if __name__ == "__main__":
    main()
