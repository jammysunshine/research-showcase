#!/usr/bin/env python3
"""Execute the frozen Phase-VII exact budget--IR trade-off study."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mechanism_discovery.max_affine_corpus import guo_2024_four_agent_network_spec
from src.mechanism_discovery.max_affine_independent import replay_deleted_input_network_utility_margin
from src.mechanism_discovery.piecewise_affine import certify_minimum_groves_utility, certify_ordered_public_project_charge
from src.mechanism_discovery.relu_benchmark import deleted_input_charge, deleted_input_terms, deterministic_network
from src.mechanism_discovery.uniform_repair import add_output_bias_offset, synthesize_minimal_uniform_repair


def encode(value):
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [encode(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: encode(item) for key, item in asdict(value).items()}
    return value


def utility_certificate(source, agents):
    return certify_minimum_groves_utility(deleted_input_terms(source, agents), agents)


def entry(name, source, agents, metadata):
    baseline_budget = certify_ordered_public_project_charge(deleted_input_charge(source, agents), agents)
    repair = synthesize_minimal_uniform_repair(baseline_budget, agents)
    repaired_source = add_output_bias_offset(source, repair.per_term_offset)
    baseline_utility = utility_certificate(source, agents)
    repaired_utility = utility_certificate(repaired_source, agents)
    direct = replay_deleted_input_network_utility_margin(repaired_source, agents)
    encoded_repaired = encode(repaired_utility)
    if direct["minimum_utility"] != encoded_repaired["minimum_utility"]:
        raise SystemExit(f"direct/compiled utility minimum disagreement for {name}")
    predicted = baseline_utility.minimum_utility - repair.per_term_offset
    if repaired_utility.minimum_utility != predicted:
        raise SystemExit(f"utility shift identity failed for {name}")
    preserves_ir = baseline_utility.minimum_utility >= 0 and repair.per_term_offset <= baseline_utility.minimum_utility
    if preserves_ir != (repaired_utility.minimum_utility >= 0):
        raise SystemExit(f"IR feasibility predicate failed for {name}")
    return {
        "name": name,
        **metadata,
        "agents": agents,
        "baseline_source": source,
        "repaired_source": repaired_source,
        "baseline_budget_slack": encode(baseline_budget.minimum_budget_slack),
        "per_term_offset": encode(repair.per_term_offset),
        "baseline_utility_certificate": encode(baseline_utility),
        "repaired_utility_certificate": encoded_repaired,
        "independent_repaired_utility": direct,
        "predicted_repaired_minimum_utility": encode(predicted),
        "repair_preserves_ex_post_ir": preserves_ir,
    }


def main() -> None:
    config_path = ROOT / "configs" / "relu_benchmark.json"
    config = json.loads(config_path.read_text())
    entries = [entry("guo_aaai_2024_printed_4_agent", guo_2024_four_agent_network_spec(), 4,
                     {"kind": "published_printed_decimal_control"})]
    for case in config["cases"]:
        agents, width, seed = int(case["agents"]), int(case["width"]), int(case["seed"])
        source = deterministic_network(seed, agents - 1, width, int(config["coefficient_denominator"]))
        entries.append(entry(f"fixture_{seed}", source, agents, {"kind": "frozen_fixture", **case}))
    payload = {
        "study": "phase_vii_exact_uniform_repair_ir_tradeoff_v1",
        "benchmark_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "entries": entries,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output = ROOT / "artifacts" / "repair_ir_tradeoff_study.json"
    output.write_text(serialized)
    print(output)
    print(hashlib.sha256(serialized.encode()).hexdigest())


if __name__ == "__main__":
    main()
