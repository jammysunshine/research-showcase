#!/usr/bin/env python3
"""Independent verification for bounded-coalition public-project scaling artifact."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechanism_discovery.public_project_independent import check as independent_check  # noqa: E402


def _serialised_key(mechanism: dict[str, Any]) -> str:
    return json.dumps(mechanism, sort_keys=True)


def _check(
    cache: dict[tuple[str, int], dict[str, Any]],
    mechanism: dict[str, Any],
    cap: int,
) -> dict[str, Any]:
    key = (_serialised_key(mechanism), int(cap))
    if key not in cache:
        cache[key] = independent_check(mechanism, max_coalition_size=cap)
    return cache[key]


def _rows_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["mechanism"]["name"]: row for row in rows}


def _first_witness(report: dict[str, Any], property_name: str) -> dict[str, Any]:
    for witness in report.get("witnesses", []):
        if witness.get("property") == property_name:
            return dict(witness)
    return {}


def main() -> None:
    artifact_path = ROOT / "artifacts" / "public_project_coalition_scaling.json"
    if not artifact_path.exists():
        raise SystemExit(f"missing artifact: {artifact_path}")

    artifact = json.loads(artifact_path.read_text())
    config = json.loads((ROOT / "configs" / "public_project_coalition_scaling.json").read_text())

    max_coalition_size = int(config["max_coalition_size"])
    config_costs_by_n = {int(n): [int(c) for c in costs] for n, costs in config["costs_by_n_agents"].items()}
    selected_expected = {(int(item["n_agents"]), int(item["cost"])) for item in config.get("selected", [])}

    cache: dict[tuple[str, int], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    selected_checks: list[dict[str, Any]] = []

    for block in artifact.get("by_n", []):
        n_agents = int(block["n_agents"])
        configured_costs = config_costs_by_n.get(n_agents)
        if configured_costs is None:
            failures.append({"scope": "n_agents_unconfigured", "n_agents": n_agents})
            continue

        configured_costs = sorted(set(configured_costs))
        row_by_cost = {int(row["cost"]): row for row in block.get("cost_rows", [])}
        artifact_costs = sorted(row_by_cost)

        if artifact_costs != configured_costs:
            failures.append(
                {
                    "scope": "cost_grid_mismatch",
                    "n_agents": n_agents,
                    "observed_costs": artifact_costs,
                    "expected_costs": configured_costs,
                },
            )

        cap_keys = [str(i) for i in range(1, max_coalition_size + 1)]
        cost_summaries: list[dict[str, Any]] = []

        for cost in configured_costs:
            row = row_by_cost.get(cost)
            if row is None:
                continue

            survivors_by_cap = row.get("survivors_by_cap", {})
            if sorted(survivors_by_cap.keys()) != cap_keys:
                failures.append(
                    {
                        "scope": "cap_keys_mismatch",
                        "n_agents": n_agents,
                        "cost": cost,
                        "observed": sorted(survivors_by_cap.keys()),
                        "expected": cap_keys,
                    },
                )

            counts_by_cap = row.get("counts_by_cap", {})
            cap_signatures = row.get("survivor_signature_by_cap", {})
            fragile_expected = row.get("fragile_against_unrestricted", {})
            min_failing_expected = {entry["name"]: entry["min_failing_coalition_size"] for entry in row.get("min_failing_coalition_size", [])}

            cap_replayed_counts: dict[str, int] = {}
            cap_failures = 0
            cap_digests: dict[str, str] = {}
            by_cap_names: dict[str, list[str]] = {}

            # Revalidate each serialized row at every cap.
            for cap in range(1, max_coalition_size + 1):
                cap_key = str(cap)
                serialized_rows = survivors_by_cap.get(cap_key, [])
                by_cap_names[cap_key] = []
                cap_row_records: list[dict[str, Any]] = []

                for frontier_row in serialized_rows:
                    mechanism = frontier_row["mechanism"]
                    mechanism_name = mechanism["name"]
                    by_cap_names[cap_key].append(mechanism_name)
                    verification = _check(cache, mechanism, cap)
                    frontier_verification = frontier_row.get("verification", {})

                    if verification.get("max_coalition_size") != cap:
                        failures.append(
                            {
                                "scope": "max_coalition_size_field_mismatch",
                                "n_agents": n_agents,
                                "cost": cost,
                                "cap": cap,
                                "name": mechanism_name,
                                "expected": cap,
                                "observed": verification.get("max_coalition_size"),
                            },
                        )

                    if not verification.get("accepted", False):
                        cap_failures += 1
                        failures.append(
                            {
                                "scope": "serialized_row_rejected",
                                "n_agents": n_agents,
                                "cost": cost,
                                "cap": cap,
                                "name": mechanism_name,
                                "witness": _first_witness(verification, "coalitional_dsic"),
                                "verification": verification,
                                "serialized_verification": frontier_verification,
                            },
                        )
                    elif frontier_verification.get("accepted") is not True:
                        failures.append(
                            {
                                "scope": "primary_serialized_inconsistent",
                                "n_agents": n_agents,
                                "cost": cost,
                                "cap": cap,
                                "name": mechanism_name,
                                "primary_verification": frontier_verification,
                                "independent_verification": verification,
                            },
                        )

                    if verification.get("max_coalition_size") != int(frontier_verification.get("max_coalition_size", -1)):
                        failures.append(
                            {
                                "scope": "serialized_max_coalition_size_mismatch",
                                "n_agents": n_agents,
                                "cost": cost,
                                "cap": cap,
                                "name": mechanism_name,
                                "serialized": frontier_verification.get("max_coalition_size"),
                                "independent": verification.get("max_coalition_size"),
                            },
                        )

                    cap_row_records.append(
                        {
                            "name": mechanism_name,
                            "mechanism": mechanism,
                            "verification": verification,
                        },
                    )

                by_cap_names[cap_key] = sorted(by_cap_names[cap_key])
                cap_replayed_counts[cap_key] = len(serialized_rows)
                if cap_row_records:
                    cap_digests[cap_key] = hashlib.sha256(
                        "\n".join(
                            sorted(
                                json.dumps(
                                    {
                                        "name": record["name"],
                                        "verification": record["verification"],
                                        "cap": cap,
                                    },
                                    sort_keys=True,
                                )
                                for record in cap_row_records
                            ),
                        ).encode(),
                    ).hexdigest()
                else:
                    cap_digests[cap_key] = hashlib.sha256(b"").hexdigest()

                expected_count = int(counts_by_cap.get(cap_key, -1))
                if expected_count != len(serialized_rows):
                    failures.append(
                        {
                            "scope": "counts_by_cap_mismatch",
                            "n_agents": n_agents,
                            "cost": cost,
                            "cap": cap,
                            "expected": expected_count,
                            "observed": len(serialized_rows),
                        },
                    )

                expected_signature = sorted(cap_signatures.get(cap_key, []))
                if expected_signature != by_cap_names[cap_key]:
                    failures.append(
                        {
                            "scope": "signature_mismatch",
                            "n_agents": n_agents,
                            "cost": cost,
                            "cap": cap,
                            "expected": expected_signature,
                            "observed": by_cap_names[cap_key],
                        },
                    )

            # Fragility summaries are from cap-1 baseline survivors minus current cap survivors.
            cap1_names = set(by_cap_names.get("1", []))
            for cap in range(2, max_coalition_size + 1):
                cap_key = str(cap)
                observed_fragile = sorted(cap1_names - set(by_cap_names.get(cap_key, [])))
                expected_fragile = sorted(fragile_expected.get(cap_key, []))
                if observed_fragile != expected_fragile:
                    failures.append(
                        {
                            "scope": "fragile_against_unrestricted_mismatch",
                            "n_agents": n_agents,
                            "cost": cost,
                            "cap": cap,
                            "expected": expected_fragile,
                            "observed": observed_fragile,
                        },
                    )

            # Recompute minimum failing coalition size from accepted/unaccepted progression.
            recomputed_min_fail: list[dict[str, Any]] = []
            for name, base_row in _rows_by_name(survivors_by_cap.get("1", [])).items():
                mechanism = base_row["mechanism"]
                mechanism_observed_min_fail = None
                for cap in range(1, int(mechanism["n_agents"]) + 1):
                    if not _check(cache, mechanism, cap).get("accepted", False):
                        mechanism_observed_min_fail = cap
                        break
                recomputed_min_fail.append(
                    {"name": name, "min_failing_coalition_size": mechanism_observed_min_fail},
                )
                if min_failing_expected.get(name) != mechanism_observed_min_fail:
                    failures.append(
                        {
                            "scope": "min_failing_mismatch",
                            "n_agents": n_agents,
                            "cost": cost,
                            "name": name,
                            "expected": min_failing_expected.get(name),
                            "recomputed": mechanism_observed_min_fail,
                        },
                    )

            # Selected row consistency checks for this selected cost.
            selected_rows_for_cost = [
                row
                for row in block.get("selected_cost_rows", [])
                if int(row.get("cost")) == cost
            ]
            for selected_row in selected_rows_for_cost:
                if selected_row.get("cost") != cost:
                    failures.append(
                        {
                            "scope": "selected_cost_row_misbound",
                            "n_agents": n_agents,
                            "selected_cost": selected_row.get("cost"),
                            "expected_cost": cost,
                        },
                    )
                    continue
                if sorted(selected_row.get("survivors_by_cap", {}).keys()) != cap_keys:
                    failures.append(
                        {
                            "scope": "selected_cap_keys_mismatch",
                            "n_agents": n_agents,
                            "cost": cost,
                            "expected": cap_keys,
                            "observed": sorted(selected_row.get("survivors_by_cap", {}).keys()),
                        },
                    )
                for cap_key, serialized_rows in selected_row.get("survivors_by_cap", {}).items():
                    cap = int(cap_key)
                    for serialized_row in serialized_rows:
                        mechanism = serialized_row["mechanism"]
                        verification = _check(cache, mechanism, cap)
                        selected_checks.append(
                            {
                                "n_agents": n_agents,
                                "cost": cost,
                                "cap": cap,
                                "name": mechanism["name"],
                                "verification": verification,
                                "min_failing_coalition_size": next(
                                    (
                                        entry["min_failing_coalition_size"]
                                        for entry in row.get("min_failing_coalition_size", [])
                                        if entry.get("name") == mechanism["name"]
                                    ),
                                    None,
                                ),
                            },
                        )

            fragility_distribution = row.get("fragility_distribution", {})
            recomputed_distribution: dict[str, int] = {}
            for entry in recomputed_min_fail:
                key = str(entry["min_failing_coalition_size"]) if entry["min_failing_coalition_size"] is not None else "none"
                recomputed_distribution[key] = recomputed_distribution.get(key, 0) + 1
            if fragility_distribution and fragility_distribution != recomputed_distribution:
                failures.append(
                    {
                        "scope": "fragility_distribution_mismatch",
                        "n_agents": n_agents,
                        "cost": cost,
                        "expected": fragility_distribution,
                        "recomputed": recomputed_distribution,
                    },
                )

            cost_summaries.append(
                {
                    "cost": cost,
                    "candidate_count": row.get("candidate_count"),
                    "counts_by_cap_observed": cap_replayed_counts,
                    "counts_by_cap": counts_by_cap,
                    "cap_failures": cap_failures,
                    "cap_digests": cap_digests,
                },
            )

        # Check that selected-cost row declarations match the configured selected map.
        selected_for_n = [(n, c) for (n, c) in selected_expected if n == n_agents]
        selected_observed = [
            (n_agents, int(row.get("cost")))
            for row in block.get("selected_cost_rows", [])
        ]
        if sorted(selected_for_n) != sorted(selected_observed):
            failures.append(
                {
                    "scope": "selected_cost_mismatch",
                    "n_agents": n_agents,
                    "expected": sorted(selected_for_n),
                    "observed": sorted(selected_observed),
                },
            )

        diagnostics.append(
            {
                "n_agents": n_agents,
                "max_cost": max(configured_costs) if configured_costs else None,
                "cost_rows": len(configured_costs),
                "selected_cost_rows": [int(row["cost"]) for row in block.get("selected_cost_rows", [])],
                "cost_summaries": cost_summaries,
            },
        )

    # Independent digest over max-cap survivors across all rows in fixed order.
    maxcap_rows: list[str] = []
    for block in artifact.get("by_n", []):
        n_agents = int(block["n_agents"])
        for row in block.get("cost_rows", []):
            cap_key = str(max_coalition_size)
            for mechanism_row in row.get("survivors_by_cap", {}).get(cap_key, []):
                maxcap_rows.append(
                    json.dumps(
                        {
                            "n_agents": n_agents,
                            "cost": int(row["cost"]),
                            "mechanism": mechanism_row["mechanism"],
                            "max_coalition_size": max_coalition_size,
                        },
                        sort_keys=True,
                    ),
                )
    independent_digest = hashlib.sha256("\n".join(sorted(maxcap_rows)).encode()).hexdigest()

    selected_failures = [item for item in selected_checks if not item["verification"].get("accepted", False)]
    certificate = {
        "study": "public_project_coalition_scaling",
        "artifact": str(artifact_path),
        "max_coalition_size": max_coalition_size,
        "independent_failures": len(failures),
        "selected_checks_count": len(selected_checks),
        "selected_failures": len(selected_failures),
        "selected_checks": sorted(selected_checks, key=lambda item: (item["n_agents"], item["cost"], item["cap"], item["name"])),
        "independent_digest": independent_digest,
        "diagnostics": diagnostics,
        "statement": (
            "Every serialized mechanism in the scaling artifact is rechecked at every cap "
            "up to the configured coalition cap. Selected rows are rechecked independently "
            "at each cap and all fragile/min-failing summaries are recomputed."
        ),
    }

    out = ROOT / "artifacts" / "public_project_coalition_scaling_certificate.json"
    out.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "certificate": str(out),
                "independent_failures": certificate["independent_failures"],
                "selected_failures": certificate["selected_failures"],
                "selected_checks_count": certificate["selected_checks_count"],
                "independent_digest": certificate["independent_digest"],
            },
            indent=2,
            sort_keys=True,
        )
    )

    if failures:
        print(json.dumps({"failures": failures[:20]}, indent=2, sort_keys=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
