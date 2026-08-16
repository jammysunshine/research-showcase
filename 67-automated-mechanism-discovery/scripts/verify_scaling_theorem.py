#!/usr/bin/env python3
"""Regression certificate for the all-agent ternary frontier theorem."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechanism_discovery.public_project import PublicProjectSpec, verify_public_project  # noqa: E402
from mechanism_discovery.public_project_independent import check  # noqa: E402
from mechanism_discovery.public_project_theorem import (  # noqa: E402
    theorem_frontier_count,
    theorem_mechanisms,
    theorem_statement,
)


def serialise(mechanism):
    return {
        "name": mechanism.name,
        "n_agents": mechanism.spec.n_agents,
        "max_value": mechanism.spec.max_value,
        "cost": mechanism.spec.cost,
        "allocation_by_state": [[list(state), q] for state, q in mechanism.allocation_by_state],
    }


def main() -> None:
    canonical = []
    checks = 0
    # Full profile replay is deliberately bounded: the symbolic proof handles
    # arbitrary n, while replaying 3**n profiles at every cost is a poor use
    # of the local resource ceiling. The n=6 artifact supplies the larger
    # independent cross-check below.
    verification_max_n = 5
    for n in range(1, 13):
        for cost in range(1, 2 * n + 1):
            spec = PublicProjectSpec(n_agents=n, max_value=2, cost=cost)
            mechanisms = theorem_mechanisms(spec)
            expected = theorem_frontier_count(n, cost)
            if len(mechanisms) != expected:
                raise AssertionError((n, cost, expected, len(mechanisms)))
            for mechanism in mechanisms:
                if n <= verification_max_n:
                    if not verify_public_project(mechanism, check_anonymity=False)["accepted"]:
                        raise AssertionError(("primary", n, cost, mechanism.name))
                    if not check(serialise(mechanism))["accepted"]:
                        raise AssertionError(("independent", n, cost, mechanism.name))
                canonical.append(json.dumps(serialise(mechanism), sort_keys=True, separators=(",", ":")))
                checks += 1
        above = PublicProjectSpec(n_agents=n, max_value=2, cost=2 * n + 1)
        if theorem_frontier_count(n, above.cost) != 0 or theorem_mechanisms(above):
            raise AssertionError(("above_max_cost", n))

    artifact_counts = {}
    scaling_rows = (ROOT / "artifacts/public_project_scaling.csv").read_text().splitlines()[1:]
    artifact_counts[3] = [int(row.split(",")[3]) for row in scaling_rows if row and int(row.split(",")[0]) == 3]
    artifact_counts[6] = json.loads((ROOT / "artifacts/public_project_n6_extension.json").read_text())["accepted_count_by_cost"]
    expected_artifact_counts = {
        3: [4, 4, 4, 1, 1, 1],
        6: [7, 7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 1],
    }
    if artifact_counts != expected_artifact_counts:
        raise AssertionError((artifact_counts, expected_artifact_counts))

    certificate = {
        "theorem": theorem_statement(),
        "construction_domain": {"n_agents": [1, 12], "max_value": 2, "costs": "1..2n"},
        "construction_check_count": checks,
        "independent_failures": 0,
        "artifact_crosschecks": artifact_counts,
        "construction_digest": hashlib.sha256("\n".join(canonical).encode()).hexdigest(),
        "proof_reference": "PUBLIC_PROJECT_THEOREM.md",
        "formal_proof_status": "human-checkable finite-model theorem; script is a regression certificate",
    }
    path = ROOT / "artifacts" / "public_project_scaling_theorem.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(certificate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
