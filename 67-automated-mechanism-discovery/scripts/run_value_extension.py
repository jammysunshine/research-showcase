#!/usr/bin/env python3
"""Exact value-lattice sensitivity extension for the public-project study."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechanism_discovery.public_project import (  # noqa: E402
    PublicProjectSpec,
    enumerate_anonymous_monotone,
    frontier,
    public_project_metrics,
)
from mechanism_discovery.public_project_independent import check  # noqa: E402


def serialise(mechanism):
    return {
        "name": mechanism.name,
        "n_agents": mechanism.spec.n_agents,
        "max_value": mechanism.spec.max_value,
        "cost": mechanism.spec.cost,
        "allocation_by_state": [[list(state), allocation] for state, allocation in mechanism.allocation_by_state],
    }


def main() -> None:
    rows = []
    candidate_count = None
    for cost in range(1, 10):
        spec = PublicProjectSpec(n_agents=3, max_value=3, cost=cost)
        candidates = frontier(spec)
        candidate_count = len(list(enumerate_anonymous_monotone(spec)))
        for result in candidates:
            mechanism = serialise(result["mechanism"])
            rows.append({"mechanism": mechanism, "metrics": public_project_metrics(result["mechanism"])})

    checks = [check(row["mechanism"]) for row in rows]
    names = sorted(f"{row['mechanism']['cost']}:{row['mechanism']['name']}" for row in rows)
    payload = {
        "study": "public_project_value_lattice_extension",
        "scope": {"n_agents": 3, "max_value": 3, "costs": list(range(1, 10))},
        "candidate_count": candidate_count,
        "accepted_count_by_cost": [sum(row["mechanism"]["cost"] == cost for row in rows) for cost in range(1, 10)],
        "accepted_rows": rows,
        "independent_failure_count": sum(not report["accepted"] for report in checks),
        "independent_digest": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "interpretation": "Exploratory exact value-lattice sensitivity result; it does not change the preregistered three-agent max_value=2 headline.",
    }
    path = ROOT / "artifacts" / "public_project_value_extension.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("scope", "candidate_count", "accepted_count_by_cost", "independent_failure_count", "independent_digest")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
