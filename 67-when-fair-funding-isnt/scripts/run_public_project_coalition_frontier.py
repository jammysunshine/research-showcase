#!/usr/bin/env python3
"""Run bounded coalition-robust frontier checks for the public-project study."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechanism_discovery.public_project import (  # noqa: E402
    PublicProjectSpec,
    enumerate_anonymous_monotone,
    frontier,
)


def _serialise_mechanism(mechanism) -> dict[str, Any]:
    return {
        "name": mechanism.name,
        "n_agents": mechanism.spec.n_agents,
        "max_value": mechanism.spec.max_value,
        "cost": mechanism.spec.cost,
        "allocation_by_state": [[list(state), allocation] for state, allocation in mechanism.allocation_by_state],
    }


def _serialise_frontier_rows(rows: list[dict]) -> list[dict[str, Any]]:
    return [
        {
            "mechanism": _serialise_mechanism(row["mechanism"]),
            "metrics": row["metrics"],
            "verification": row["verification"],
        }
        for row in rows
    ]


def main() -> None:
    config = json.loads((ROOT / "configs" / "public_project_coalition_frontier.json").read_text())
    max_coalition_size = int(config["max_coalition_size"])
    n_agents = int(config["n_agents"])
    max_value = int(config["max_value"])
    costs = list(config["costs"])

    by_cost = {}
    for cost in costs:
        spec = PublicProjectSpec(n_agents=n_agents, max_value=max_value, cost=int(cost))
        no_coalition = frontier(spec)
        coalition = frontier(spec, max_coalition_size=max_coalition_size)
        by_cost[str(cost)] = {
            "candidate_count": len(list(enumerate_anonymous_monotone(spec))),
            "accepted_count_unrestricted": len(no_coalition),
            "accepted_count_coalitional": len(coalition),
            "survivors_unrestricted": _serialise_frontier_rows(no_coalition),
            "survivors_coalitional": _serialise_frontier_rows(coalition),
        }

    frozen_cost = 3
    frozen_spec = PublicProjectSpec(n_agents=n_agents, max_value=max_value, cost=frozen_cost)
    coalition_frontier = frontier(frozen_spec, max_coalition_size=max_coalition_size)
    # Keep a focused witness set for the headline domain in this extension.
    selected_cost_3 = by_cost[str(frozen_cost)]["survivors_coalitional"]

    payload = {
        "study": "public_project_coalition_frontier",
        "question": "Do frontier rules remain DSIC under bounded coalition deviations (up to size 2) in the frozen ternary public-project domain?",
        "configuration": config,
        "frozen_cost": frozen_cost,
        "candidate_count_at_frozen_cost": len(list(enumerate_anonymous_monotone(frozen_spec))),
        "selected_frontier": selected_cost_3,
        "cost_frontier": by_cost,
        "coalitional_frontier": _serialise_frontier_rows(coalition_frontier),
    }

    artifact = ROOT / "artifacts" / "public_project_coalition_frontier.json"
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "artifact": str(artifact),
                "max_coalition_size": max_coalition_size,
                "accepted_count_cost3_unrestricted": by_cost[str(frozen_cost)]["accepted_count_unrestricted"],
                "accepted_count_cost3_coalitional": by_cost[str(frozen_cost)]["accepted_count_coalitional"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
