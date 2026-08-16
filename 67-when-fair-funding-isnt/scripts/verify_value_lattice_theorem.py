#!/usr/bin/env python3
"""Frozen confirmation of the arbitrary integer-value lattice theorem."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechanism_discovery.public_project import PublicProjectSpec, frontier  # noqa: E402
from mechanism_discovery.public_project_independent import check  # noqa: E402
from mechanism_discovery.public_project_theorem import value_lattice_mechanisms  # noqa: E402


def serialise(mechanism):
    return {
        "name": mechanism.name,
        "n_agents": mechanism.spec.n_agents,
        "max_value": mechanism.spec.max_value,
        "cost": mechanism.spec.cost,
        "allocation_by_state": [[list(state), allocation] for state, allocation in mechanism.allocation_by_state],
    }


def normalized_rows(mechanisms):
    return {
        tuple((state, allocation) for state, allocation in mechanism.allocation_by_state)
        for mechanism in mechanisms
    }


def main() -> None:
    config = json.loads((ROOT / "configs/phase_viii_value_lattice_theorem.json").read_text())
    primary = config["primary"]
    n_agents, max_value = primary["n_agents"], primary["max_value"]
    rows = []
    accepted_rules = []
    all_serialized = []
    for cost in primary["costs"]:
        spec = PublicProjectSpec(n_agents=n_agents, max_value=max_value, cost=cost)
        predicted = value_lattice_mechanisms(spec)
        exhaustive = tuple(item["mechanism"] for item in frontier(spec))
        if normalized_rows(predicted) != normalized_rows(exhaustive):
            raise AssertionError(("rule_set_mismatch", n_agents, max_value, cost))
        serialized = [serialise(mechanism) for mechanism in exhaustive]
        checks = [check(item) for item in serialized]
        if any(not report["accepted"] for report in checks):
            raise AssertionError(("independent_failure", cost, checks))
        rows.append({
            "cost": cost,
            "predicted_count": len(predicted),
            "exhaustive_count": len(exhaustive),
            "independent_failure_count": sum(not report["accepted"] for report in checks),
        })
        accepted_rules.append({"cost": cost, "rules": serialized})
        all_serialized.extend(json.dumps(item, sort_keys=True, separators=(",", ":")) for item in serialized)
    payload = {
        "study": config["study"],
        "config_sha256": hashlib.sha256((ROOT / "configs/phase_viii_value_lattice_theorem.json").read_bytes()).hexdigest(),
        "scope": primary,
        "rows": rows,
        "accepted_rules": accepted_rules,
        "accepted_rule_digest": hashlib.sha256("\n".join(sorted(all_serialized)).encode()).hexdigest(),
        "independent_failure_count": sum(row["independent_failure_count"] for row in rows),
        "result": "exact predicted/exhaustive rule-set equality on the frozen confirmation grid",
    }
    path = ROOT / "artifacts/phase_viii_value_lattice_theorem.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
