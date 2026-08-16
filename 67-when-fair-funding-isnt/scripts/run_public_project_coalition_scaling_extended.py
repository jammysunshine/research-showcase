#!/usr/bin/env python3
"""Extended coalition-robust frontier scaling with n=6 support.

This run keeps the same mechanism class as earlier coalition studies and adds
an exact n=6 sweep on the ternary public-project lattice to test whether the
coalition-robust frontier structure is stable as n grows.
"""

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
    enumerate_anonymous_monotone,
    frontier,
    verify_public_project,
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


def _survivors_by_cap(spec: PublicProjectSpec, max_coalition_size: int) -> dict[str, Any]:
    """
    Evaluate exact DSIC frontier (cap=1), then filter only the survivors for each
    larger coalition-size cap.
    """
    base_frontier = list(frontier(spec, check_anonymity=False))
    by_cap: dict[int, list[dict[str, Any]]] = {}

    metrics_by_name: dict[str, dict[str, Any]] = {}
    base_mechanisms: list = []
    for row in base_frontier:
        mechanism = row["mechanism"]
        base_mechanisms.append(mechanism)
        metrics_by_name[mechanism.name] = row["metrics"]
        by_cap.setdefault(1, []).append(
            _serialise_survivor(mechanism, row["metrics"], row["verification"]),
        )

    min_failing: dict[str, int | None] = {name: None for name in metrics_by_name}
    survivors = base_mechanisms

    for cap in range(2, max_coalition_size + 1):
        next_survivors: list = []
        for mechanism in survivors:
            verification = verify_public_project(
                mechanism,
                check_anonymity=False,
                max_coalition_size=cap,
            )
            if verification["accepted"]:
                next_survivors.append((mechanism, verification))
            elif min_failing[mechanism.name] is None:
                min_failing[mechanism.name] = cap
        by_cap[cap] = [
            _serialise_survivor(mechanism, metrics_by_name[mechanism.name], verification)
            for mechanism, verification in next_survivors
        ]
        survivors = [mechanism for mechanism, _ in next_survivors]

    for mechanism in survivors:
        min_failing[mechanism.name] = None

    base_names = {row["mechanism"]["name"] for row in by_cap[1]}
    fragile_by_cap = {}
    for cap in range(2, max_coalition_size + 1):
        cap_names = {row["mechanism"]["name"] for row in by_cap[cap]}
        fragile_by_cap[str(cap)] = sorted(base_names - cap_names)

    fragility_distribution: dict[str, int] = {}
    for name in sorted(min_failing):
        key = str(min_failing[name]) if min_failing[name] is not None else "none"
        fragility_distribution[key] = fragility_distribution.get(key, 0) + 1

    return {
        "survivors_by_cap": {str(cap): by_cap[cap] for cap in sorted(by_cap)},
        "counts_by_cap": {str(cap): len(by_cap[cap]) for cap in sorted(by_cap)},
        "fragile_against_unrestricted": fragile_by_cap,
        "fragility_distribution": fragility_distribution,
        "min_failing_coalition_size": [
            {"name": name, "min_failing_coalition_size": min_failing[name]}
            for name in sorted(min_failing)
        ],
        "survivor_signature_by_cap": {str(cap): [row["mechanism"]["name"] for row in by_cap[cap]] for cap in sorted(by_cap)},
    }


def _costs_for_n(config: dict[str, Any], n_agents: int) -> list[int]:
    costs = config["costs_by_n_agents"].get(str(n_agents))
    if not costs:
        raise KeyError(f"missing costs_by_n_agents entry for n={n_agents}")
    return [int(cost) for cost in costs]


def _selected_cost_rows(config: dict[str, Any], n_agents: int, cost_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = {(int(item["n_agents"]), int(item["cost"])) for item in config.get("selected", [])}
    return [row for row in cost_rows if (n_agents, row["cost"]) in selected]


def main() -> None:
    started = time.perf_counter()
    config = json.loads((ROOT / "configs" / "public_project_coalition_scaling_extended.json").read_text())
    n_agents = [int(n) for n in config["n_agents"]]
    max_value = int(config["max_value"])
    max_coalition_size = int(config["max_coalition_size"])

    by_n: list[dict[str, Any]] = []
    for n in n_agents:
        cost_rows: list[dict[str, Any]] = []
        for cost in _costs_for_n(config, n):
            spec = PublicProjectSpec(n_agents=n, max_value=max_value, cost=cost)
            candidate_count = len(list(enumerate_anonymous_monotone(spec)))
            by_cap_payload = _survivors_by_cap(spec, max_coalition_size)
            cost_rows.append({
                "cost": cost,
                "candidate_count": candidate_count,
                **by_cap_payload,
            })

        by_n.append({
            "n_agents": n,
            "max_value": max_value,
            "cost_rows": cost_rows,
            "selected_cost_rows": _selected_cost_rows(config, n, cost_rows),
        })

    artifact = {
        "study": "public_project_coalition_scaling_extended",
        "question": (
            "Do DSIC frontier survivors remain coalition-robust under cap-3 deviations "
            "as n increases from 3 to 6 on the ternary lattice?"
        ),
        "configuration": config,
        "max_coalition_size": max_coalition_size,
        "by_n": by_n,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "artifact_count": sum(len(block["cost_rows"]) for block in by_n),
    }

    path = ROOT / "artifacts" / "public_project_coalition_scaling_extended.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(path), "elapsed_seconds": artifact["elapsed_seconds"]}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
