#!/usr/bin/env python3
"""Exact six-agent scaling extension with independent certificate replay."""

from __future__ import annotations

import hashlib
import json
import resource
import platform
import sys
import time
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
    started = time.perf_counter()
    rows = []
    candidate_counts = []
    accepted_counts = []
    costs = list(range(1, 13))
    for cost in costs:
        spec = PublicProjectSpec(n_agents=6, max_value=2, cost=cost)
        candidate_counts.append(len(list(enumerate_anonymous_monotone(spec))))
        # Sorted-state encoding makes anonymity structural; the standalone
        # checker below still verifies every permutation in the certificate.
        accepted = frontier(spec, check_anonymity=False)
        accepted_counts.append(len(accepted))
        rows.extend(
            {"mechanism": serialise(result["mechanism"]), "metrics": result["metrics"]}
            for result in accepted
        )
        print(json.dumps({"cost": cost, "candidates": candidate_counts[-1], "accepted": accepted_counts[-1]}), flush=True)

    checks = [check(row["mechanism"]) for row in rows]
    canonical_rows = [json.dumps(row["mechanism"], sort_keys=True, separators=(",", ":")) for row in rows]
    digest = hashlib.sha256("\n".join(sorted(canonical_rows)).encode()).hexdigest()
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes; Linux reports KiB.
    max_rss_bytes = raw_rss if platform.system() == "Darwin" else raw_rss * 1024
    payload = {
        "study": "public_project_n6_exact_extension",
        "scope": {"n_agents": 6, "max_value": 2, "costs": costs},
        "candidate_counts_by_cost": candidate_counts,
        "accepted_count_by_cost": accepted_counts,
        "accepted_row_count": len(rows),
        "accepted_rows": rows,
        "independent_failure_count": sum(not report["accepted"] for report in checks),
        "independent_digest": digest,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "max_rss_bytes": max_rss_bytes,
        "resource_platform": platform.system(),
        "interpretation": (
            "Exploratory exact six-agent extension. Anonymity is structural in the "
            "sorted-state representation and is independently replayed over all "
            "profile permutations; this does not establish an asymptotic theorem."
        ),
    }
    path = ROOT / "artifacts" / "public_project_n6_extension.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "scope", "candidate_counts_by_cost", "accepted_count_by_cost",
        "accepted_row_count", "independent_failure_count", "independent_digest",
        "elapsed_seconds", "max_rss_bytes", "resource_platform",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
