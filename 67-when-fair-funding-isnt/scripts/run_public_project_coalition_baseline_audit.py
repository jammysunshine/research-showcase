#!/usr/bin/env python3
"""Independent baseline audit: does the canonical efficient/pivotal public-project
mechanism itself survive the same bounded-coalition deviation bar used to prune
the anonymous-monotone frontier search in Phase X?

`efficient_mechanism` implements the welfare-maximizing sum-threshold decision
with critical-value payments -- the natural single-parameter comparator for
this binary-decision domain. It is DSIC and ex-post IR by construction; this
study asks whether that guarantee extends to bounded coalitions, independent
of and without reusing the frontier-search machinery's acceptance path.

Note: this mechanism already has a separately documented weak-budget-balance
deficit at some profiles (`VERIFICATION_REPORT.md`'s efficient-comparator
witness). That is unrelated to incentive compatibility, so this audit reports
fragility strictly from the `dsic`/`coalitional_dsic` verifier fields, never
from the bundled `accepted` flag, to avoid re-surfacing the known budget
finding as a spurious new coalition result.
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
    efficient_mechanism,
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


def _incentive_ok(report: dict[str, Any]) -> bool:
    """DSIC/coalitional-DSIC status only; ignores unrelated properties (this
    mechanism's budget-balance deficit is a separate, already-documented fact
    and must not be conflated with coalition-incentive fragility)."""
    return bool(report["dsic"]) and bool(report["coalitional_dsic"])


def _min_failing_coalition_size(spec: PublicProjectSpec, max_coalition_size: int) -> int | None:
    mechanism = efficient_mechanism(spec)
    for cap in range(1, max_coalition_size + 1):
        if not _incentive_ok(verify_public_project(mechanism, max_coalition_size=cap)):
            return cap
    return None


def _audit_cost_row(n_agents: int, max_value: int, cost: int, max_coalition_size: int) -> dict[str, Any]:
    spec = PublicProjectSpec(n_agents=n_agents, max_value=max_value, cost=cost)
    mechanism = efficient_mechanism(spec)
    unrestricted = verify_public_project(mechanism, max_coalition_size=1)
    coalitional = verify_public_project(mechanism, max_coalition_size=max_coalition_size)
    return {
        "n_agents": n_agents,
        "max_value": max_value,
        "cost": cost,
        "mechanism": _serialise_mechanism(mechanism),
        "unrestricted_verification": unrestricted,
        "coalitional_verification": coalitional,
        "min_failing_coalition_size": _min_failing_coalition_size(spec, max_coalition_size),
    }


def _audit_frontier_like_domain(domain: dict[str, Any]) -> dict[str, Any]:
    n_agents = int(domain["n_agents"])
    max_value = int(domain["max_value"])
    max_coalition_size = int(domain["max_coalition_size"])
    cost_rows = [
        _audit_cost_row(n_agents, max_value, int(cost), max_coalition_size)
        for cost in domain["costs"]
    ]
    selected = {int(c) for c in domain["selected_costs"]}
    return {
        "domain": domain["domain"],
        "max_coalition_size": max_coalition_size,
        "cost_rows": cost_rows,
        "selected_cost_rows": [row for row in cost_rows if row["cost"] in selected],
    }


def _audit_scaling_domain(domain: dict[str, Any]) -> dict[str, Any]:
    max_value = int(domain["max_value"])
    max_coalition_size = int(domain["max_coalition_size"])
    n_rows = []
    for n in domain["n_agents"]:
        n = int(n)
        costs = [int(c) for c in domain["costs_by_n_agents"][str(n)]]
        cost_rows = [_audit_cost_row(n, max_value, cost, max_coalition_size) for cost in costs]
        n_rows.append({"n_agents": n, "cost_rows": cost_rows})
    selected_pairs = {(int(item["n_agents"]), int(item["cost"])) for item in domain["selected"]}
    selected_rows = [
        row
        for block in n_rows
        for row in block["cost_rows"]
        if (row["n_agents"], row["cost"]) in selected_pairs
    ]
    return {
        "domain": domain["domain"],
        "max_coalition_size": max_coalition_size,
        "by_n": n_rows,
        "selected_cost_rows": selected_rows,
    }


def main() -> None:
    started = time.perf_counter()
    config = json.loads((ROOT / "configs" / "public_project_coalition_baseline_audit.json").read_text())

    domain_results = []
    for domain in config["domains"]:
        if "costs_by_n_agents" in domain:
            domain_results.append(_audit_scaling_domain(domain))
        else:
            domain_results.append(_audit_frontier_like_domain(domain))

    fragile_selected = []
    robust_selected = []
    for result in domain_results:
        for row in result["selected_cost_rows"]:
            entry = {
                "domain": result["domain"],
                "n_agents": row["n_agents"],
                "cost": row["cost"],
                "min_failing_coalition_size": row["min_failing_coalition_size"],
            }
            if row["min_failing_coalition_size"] is None:
                robust_selected.append(entry)
            else:
                fragile_selected.append(entry)

    payload = {
        "study": "public_project_coalition_baseline_audit",
        "question": (
            "Does the canonical efficient/pivotal public-project mechanism (welfare-maximizing "
            "sum-threshold decision, critical-value payment) itself survive bounded coalition "
            "deviations across every domain already studied for the anonymous-monotone frontier?"
        ),
        "configuration": config,
        "domains": domain_results,
        "selected_summary": {
            "robust": robust_selected,
            "fragile": fragile_selected,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    artifact = ROOT / "artifacts" / "public_project_coalition_baseline_audit.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "artifact": str(artifact),
                "elapsed_seconds": payload["elapsed_seconds"],
                "selected_robust_count": len(robust_selected),
                "selected_fragile_count": len(fragile_selected),
                "fragile_selected": fragile_selected,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
