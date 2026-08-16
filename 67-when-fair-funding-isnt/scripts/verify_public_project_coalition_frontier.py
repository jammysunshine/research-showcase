#!/usr/bin/env python3
"""Independent verification for the coalition-robust public-project frontier."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechanism_discovery.public_project_independent import check  # noqa: E402


def _serialised_key(mechanism: dict[str, Any]) -> str:
    return json.dumps(mechanism, sort_keys=True)


def _check(cache: dict[tuple[str, int], dict], mechanism: dict[str, Any], max_coalition_size: int) -> dict:
    key = (_serialised_key(mechanism), int(max_coalition_size))
    if key not in cache:
        cache[key] = check(mechanism, max_coalition_size=max_coalition_size)
    return cache[key]


def _min_failing_size(cache: dict[tuple[str, int], dict], mechanism: dict[str, Any]) -> int | None:
    n_agents = int(mechanism["n_agents"])
    for size in range(1, n_agents + 1):
        if not _check(cache, mechanism, size)["accepted"]:
            return size
    return None


def _first_witness(report: dict, property_name: str) -> dict[str, Any]:
    for witness in report.get("witnesses", []):
        if witness.get("property") == property_name:
            return dict(witness)
    return {}


def main() -> None:
    artifact_path = ROOT / "artifacts" / "public_project_coalition_frontier.json"
    if not artifact_path.exists():
        raise SystemExit(f"missing artifact: {artifact_path}")

    artifact = json.loads(artifact_path.read_text())
    max_coalition_size = int(artifact["configuration"]["max_coalition_size"])
    frozen_cost = int(artifact["frozen_cost"])

    cache: dict[tuple[str, int], dict] = {}
    failures: list[dict] = []
    cost_rows: list[dict] = []

    for cost in artifact["configuration"]["costs"]:
        cost_key = str(int(cost))
        row = artifact["cost_frontier"][cost_key]
        unrestricted_rows = row["survivors_unrestricted"]
        coalition_rows = row["survivors_coalitional"]

        unrestricted_k1_passes = []
        unrestricted_kmax_passes = []
        for frontier_row in unrestricted_rows:
            mechanism = frontier_row["mechanism"]
            rep_k1 = _check(cache, mechanism, 1)
            rep_kmax = _check(cache, mechanism, max_coalition_size)
            unrestricted_k1_passes.append(rep_k1["accepted"])
            unrestricted_kmax_passes.append(rep_kmax["accepted"])
            if not rep_kmax["accepted"]:
                failures.append({
                    "scope": "cost",
                    "cost": int(cost),
                    "name": mechanism["name"],
                    "max_coalition_size": max_coalition_size,
                    "verification": rep_kmax,
                    "witness": _first_witness(rep_kmax, "coalitional_dsic"),
                })

        coalition_failures = []
        coalition_names = []
        for frontier_row in coalition_rows:
            mechanism = frontier_row["mechanism"]
            rep_kmax = _check(cache, mechanism, max_coalition_size)
            coalition_names.append(mechanism["name"])
            if not rep_kmax["accepted"]:
                coalition_failures.append(mechanism["name"])
                failures.append({
                    "scope": "cost_coalitional",
                    "cost": int(cost),
                    "name": mechanism["name"],
                    "max_coalition_size": max_coalition_size,
                    "verification": rep_kmax,
                })

        fragile_names = [
            entry["mechanism"]["name"]
            for entry in unrestricted_rows
            if not _check(cache, entry["mechanism"], max_coalition_size)["accepted"]
        ]

        cost_rows.append({
            "cost": int(cost),
            "candidate_count": row["candidate_count"],
            "expected_unrestricted": len(unrestricted_rows),
            "expected_coalitional": len(coalition_rows),
            "observed_unrestricted_k1": sum(unrestricted_k1_passes),
            "observed_unrestricted_kmax": sum(unrestricted_kmax_passes),
            "observed_coalitional_kmax": len(coalition_rows) - len(coalition_failures),
            "survivor_names_after_coalition_check": coalition_names,
            "fragile_against_kmax": sorted(fragile_names),
        })

    selected = artifact["selected_frontier"]
    selected_checks = []
    for row in selected:
        mechanism = row["mechanism"]
        selected_checks.append({
            "name": mechanism["name"],
            "verification": _check(cache, mechanism, max_coalition_size),
            "min_failing_coalition_size": _min_failing_size(cache, mechanism),
        })
    selected_failures = [r for r in selected_checks if not r["verification"]["accepted"]]

    all_serialized_rows = []
    for cost in artifact["configuration"]["costs"]:
        for coalition_row in artifact["cost_frontier"][str(int(cost))]["survivors_coalitional"]:
            all_serialized_rows.append(
                json.dumps(
                    {
                        "name": coalition_row["mechanism"]["name"],
                        "verification": _check(cache, coalition_row["mechanism"], max_coalition_size),
                    },
                    sort_keys=True,
                )
            )
    independent_digest = hashlib.sha256("\n".join(sorted(all_serialized_rows)).encode()).hexdigest()

    selected_names = {row["mechanism"]["name"] for row in selected}
    frozen_row = artifact["cost_frontier"][str(frozen_cost)]
    frozen_selected_match = set(r["mechanism"]["name"] for r in frozen_row["survivors_coalitional"])

    certificate = {
        "study": "public_project_coalition_frontier",
        "artifact": str(artifact_path),
        "max_coalition_size": max_coalition_size,
        "selected_cost": frozen_cost,
        "frozen_cost_match": {
            "selected_names": sorted(selected_names),
            "frozen_cost_coalitional_survivors": sorted(frozen_selected_match),
            "match": selected_names == frozen_selected_match,
        },
        "cost_rows": cost_rows,
        "selected_frontier_count": len(selected),
        "selected_frontier_failures": len(selected_failures),
        "selected_frontier_reports": selected_checks,
        "independent_failure_count": len(failures),
        "independent_digest": independent_digest,
        "statement": (
            "Coalitional-robust frontier claims are replayed with the independent checker. "
            "Every serialized unrestricted frontier row is rechecked at DSIC cap 1 and cap "
            f"{max_coalition_size}; coalition survivor rows are rechecked at cap "
            f"{max_coalition_size}; and the frozen selected frontier equals the "
            "cost-3 coalition survivor set."
        ),
    }

    out = ROOT / "artifacts" / "public_project_coalition_frontier_certificate.json"
    out.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "artifact": str(artifact_path),
        "certificate": str(out),
        "independent_failures": certificate["independent_failure_count"],
        "selected_frontier_failures": certificate["selected_frontier_failures"],
        "frozen_cost_match": certificate["frozen_cost_match"]["match"],
        "independent_digest": certificate["independent_digest"],
        "frozen_selected_names": certificate["frozen_cost_match"]["frozen_cost_coalitional_survivors"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
