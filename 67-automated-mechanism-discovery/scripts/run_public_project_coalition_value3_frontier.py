#!/usr/bin/env python3
"""Coalition-robust frontier run on the ternary-lattice (`max_value=3`) public-project class."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechanism_discovery.public_project import (  # noqa: E402
    PublicProjectSpec,
    verify_public_project,
    frontier,
    enumerate_anonymous_monotone,
)


def _serialise_mechanism(mechanism) -> dict[str, Any]:
    return {
        "name": mechanism.name,
        "n_agents": mechanism.spec.n_agents,
        "max_value": mechanism.spec.max_value,
        "cost": mechanism.spec.cost,
        "allocation_by_state": [[list(state), allocation] for state, allocation in mechanism.allocation_by_state],
    }


def _serialise_survivor(mechanism, metrics: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
    return {
        "mechanism": _serialise_mechanism(mechanism),
        "metrics": metrics,
        "verification": verification,
    }


def _survivors_by_cap(spec, max_coalition_size: int) -> dict[str, Any]:
    base_frontier = list(frontier(spec))
    by_cap: dict[int, list[dict[str, Any]]] = {}
    metrics_by_name: dict[str, dict[str, Any]] = {}
    base_mechanisms: list = []
    for row in base_frontier:
        mechanism = row["mechanism"]
        base_mechanisms.append(mechanism)
        metrics_by_name[mechanism.name] = row["metrics"]
        by_cap.setdefault(1, []).append(_serialise_survivor(mechanism, row["metrics"], row["verification"]))

    min_failing = {name: None for name in metrics_by_name}
    survivors = base_mechanisms

    for coalition_size in range(2, max_coalition_size + 1):
        next_survivors = []
        for mechanism in survivors:
            verification = mechanism_verification(mechanism, coalition_size)
            if verification["accepted"]:
                next_survivors.append((mechanism, verification))
            elif min_failing[mechanism.name] is None:
                min_failing[mechanism.name] = coalition_size
        by_cap[coalition_size] = [
            _serialise_survivor(mechanism, metrics_by_name[mechanism.name], verification)
            for mechanism, verification in next_survivors
        ]
        survivors = [mechanism for mechanism, _ in next_survivors]

    for mechanism in survivors:
        min_failing[mechanism.name] = None

    base_names = {row["mechanism"]["name"] for row in by_cap[1]}
    fragile_by_cap = {}
    for coalition_size in range(2, max_coalition_size + 1):
        cap_names = {row["mechanism"]["name"] for row in by_cap[coalition_size]}
        fragile_by_cap[str(coalition_size)] = sorted(base_names - cap_names)

    fragility_distribution: dict[str, int] = {}
    for name in sorted(min_failing):
        key = str(min_failing[name]) if min_failing[name] is not None else "none"
        fragility_distribution[key] = fragility_distribution.get(key, 0) + 1

    return {
        "survivors_by_cap": {str(size): by_cap[size] for size in sorted(by_cap)},
        "counts_by_cap": {str(size): len(by_cap[size]) for size in sorted(by_cap)},
        "fragile_against_unrestricted": fragile_by_cap,
        "fragility_distribution": fragility_distribution,
        "min_failing_coalition_size": [
            {"name": name, "min_failing_coalition_size": min_failing[name]}
            for name in sorted(min_failing)
        ],
        "survivor_signature_by_cap": {str(size): [row["mechanism"]["name"] for row in by_cap[size]] for size in sorted(by_cap)},
    }


def mechanism_verification(mechanism, coalition_size: int) -> dict[str, Any]:
    return verify_public_project(mechanism, max_coalition_size=coalition_size)


def main() -> None:
    started = time.perf_counter()
    config = json.loads((ROOT / "configs" / "public_project_coalition_value3_frontier.json").read_text())
    n_agents = int(config["n_agents"])
    max_value = int(config["max_value"])
    costs = list(config["costs"])
    max_coalition_size = int(config["max_coalition_size"])

    by_cost: list[dict[str, Any]] = []
    for cost in costs:
        spec = PublicProjectSpec(n_agents=n_agents, max_value=max_value, cost=int(cost))
        candidate_count = len(list(enumerate_anonymous_monotone(spec)))
        by_cap_payload = _survivors_by_cap(spec, max_coalition_size)
        by_cost.append({
            "cost": int(cost),
            "candidate_count": candidate_count,
            **by_cap_payload,
        })

    selected_costs = {int(item["cost"]) for item in config.get("selected", [])}
    selected_cost_rows = [row for row in by_cost if row["cost"] in selected_costs]

    artifact = {
        "study": "public_project_coalition_value3_frontier",
        "question": (
            "How much of the finite m=3 anonymous-monotone public-project frontier "
            "survives coalition deviations of bounded size 2 and 3?"
        ),
        "configuration": config,
        "max_coalition_size": max_coalition_size,
        "by_cost": by_cost,
        "selected_cost_rows": selected_cost_rows,
        "selected_costs": sorted(selected_costs),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    artifact_path = ROOT / "artifacts" / "public_project_coalition_value3_frontier.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(artifact_path), "elapsed_seconds": artifact["elapsed_seconds"]}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
