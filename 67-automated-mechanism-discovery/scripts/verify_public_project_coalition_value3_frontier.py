#!/usr/bin/env python3
"""Independent verification for the value-3 frontier coalition-fragility extension."""

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


def _check(cache: dict[tuple[str, int], dict[str, Any]], mechanism: dict[str, Any], cap: int) -> dict[str, Any]:
    key = (_serialised_key(mechanism), int(cap))
    if key not in cache:
        cache[key] = independent_check(mechanism, max_coalition_size=cap)
    return cache[key]


def _first_witness(report: dict[str, Any], property_name: str) -> dict[str, Any]:
    for witness in report.get("witnesses", []):
        if witness.get("property") == property_name:
            return dict(witness)
    return {}


def _rows_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["mechanism"]["name"]: row for row in rows}


def _recomputed_min_failing(cache: dict[tuple[str, int], dict[str, Any]], mechanism: dict[str, Any]) -> int | None:
    n_agents = int(mechanism["n_agents"])
    for cap in range(1, n_agents + 1):
        if not _check(cache, mechanism, cap).get("accepted", False):
            return cap
    return None


def main() -> None:
    artifact_path = ROOT / "artifacts" / "public_project_coalition_value3_frontier.json"
    if not artifact_path.exists():
        raise SystemExit(f"missing artifact: {artifact_path}")

    config_path = ROOT / "configs" / "public_project_coalition_value3_frontier.json"
    if not config_path.exists():
        raise SystemExit(f"missing config: {config_path}")

    artifact = json.loads(artifact_path.read_text())
    config = json.loads(config_path.read_text())

    max_coalition_size = int(artifact["max_coalition_size"])
    if str(max_coalition_size) != str(config["max_coalition_size"]):
        raise SystemExit(
            f"config/run coalition size mismatch: config={config['max_coalition_size']}, "
            f"artifact={artifact['max_coalition_size']}",
        )

    expected_costs = [int(cost) for cost in config["costs"]]
    expected_costs_set = sorted(expected_costs)
    selected_costs = sorted({int(item["cost"]) for item in config.get("selected", [])})

    by_cost = artifact.get("by_cost", [])
    artifact_costs = sorted({int(row.get("cost")) for row in by_cost if isinstance(row, dict) and "cost" in row})
    if artifact_costs != expected_costs_set:
        raise SystemExit(f"cost grid mismatch: expected {expected_costs_set}, observed {artifact_costs}")

    cache: dict[tuple[str, int], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []

    selected_cost_rows = {int(row["cost"]): row for row in artifact.get("selected_cost_rows", []) if isinstance(row, dict) and "cost" in row}
    observed_selected_costs = sorted(selected_cost_rows)
    if observed_selected_costs != selected_costs:
        failures.append(
            {
                "scope": "selected_cost_mismatch",
                "expected": selected_costs,
                "observed": observed_selected_costs,
            },
        )

    cap_keys = [str(i) for i in range(1, max_coalition_size + 1)]

    for row in by_cost:
        if not isinstance(row, dict) or "cost" not in row:
            failures.append({"scope": "invalid_cost_row", "observed": row})
            continue

        cost = int(row["cost"])
        if cost not in expected_costs_set:
            failures.append({"scope": "unexpected_cost_row", "cost": cost})
            continue

        survivors_by_cap = row.get("survivors_by_cap", {})
        if sorted(survivors_by_cap.keys()) != cap_keys:
            failures.append(
                {
                    "scope": "cap_keys_mismatch",
                    "cost": cost,
                    "observed": sorted(survivors_by_cap.keys()),
                    "expected": cap_keys,
                },
            )

        counts_by_cap = row.get("counts_by_cap", {})
        fragile_expected = row.get("fragile_against_unrestricted", {})
        min_expected = {
            entry["name"]: entry["min_failing_coalition_size"] for entry in row.get("min_failing_coalition_size", [])
        }
        signature_expected = row.get("survivor_signature_by_cap", {})

        by_cap_names: dict[str, list[str]] = {}
        cap_digests: dict[str, str] = {}
        cap_failures = 0

        for cap_key in cap_keys:
            cap = int(cap_key)
            serialized_rows = survivors_by_cap.get(cap_key, [])
            names: list[str] = []
            replay_rows: list[dict[str, Any]] = []

            for frontier_row in serialized_rows:
                mechanism = frontier_row["mechanism"]
                mechanism_name = mechanism["name"]
                names.append(mechanism_name)

                verification = _check(cache, mechanism, cap)
                primary_verification = frontier_row.get("verification", {})

                if verification.get("max_coalition_size") != cap:
                    failures.append(
                        {
                            "scope": "max_coalition_size_field_mismatch",
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
                            "cost": cost,
                            "cap": cap,
                            "name": mechanism_name,
                            "witness": _first_witness(verification, "coalitional_dsic"),
                            "verification": verification,
                            "serialized_verification": primary_verification,
                        },
                    )
                elif primary_verification.get("accepted") is not True:
                    failures.append(
                        {
                            "scope": "primary_serialized_inconsistent",
                            "cost": cost,
                            "cap": cap,
                            "name": mechanism_name,
                            "primary_verification": primary_verification,
                            "independent_verification": verification,
                        },
                    )

                replay_rows.append(
                    {
                        "name": mechanism_name,
                        "mechanism": mechanism,
                        "verification": verification,
                    },
                )

            by_cap_names[cap_key] = sorted(names)
            expected_count = int(counts_by_cap.get(cap_key, -1))
            if expected_count != len(serialized_rows):
                failures.append(
                    {
                        "scope": "count_mismatch",
                        "cost": cost,
                        "cap": cap,
                        "expected": expected_count,
                        "observed": len(serialized_rows),
                    },
                )

            expected_signature = sorted(signature_expected.get(cap_key, []))
            if expected_signature != by_cap_names[cap_key]:
                failures.append(
                    {
                        "scope": "signature_mismatch",
                        "cost": cost,
                        "cap": cap,
                        "expected": expected_signature,
                        "observed": by_cap_names[cap_key],
                    },
                )

            if replay_rows:
                cap_digests[cap_key] = hashlib.sha256(
                    "\n".join(
                        sorted(
                            json.dumps(
                                {
                                    "name": item["name"],
                                    "verification": item["verification"],
                                    "cap": cap,
                                },
                                sort_keys=True,
                            )
                            for item in replay_rows
                        ),
                    ).encode(),
                ).hexdigest()
            else:
                cap_digests[cap_key] = hashlib.sha256(b"").hexdigest()

        cap1_names = set(by_cap_names.get("1", []))
        for cap in range(2, max_coalition_size + 1):
            cap_key = str(cap)
            expected_fragile = sorted(fragile_expected.get(cap_key, []))
            observed_fragile = sorted(cap1_names - set(by_cap_names.get(cap_key, [])))
            if expected_fragile != observed_fragile:
                failures.append(
                    {
                        "scope": "fragile_against_unrestricted_mismatch",
                        "cost": cost,
                        "cap": cap,
                        "expected": expected_fragile,
                        "observed": observed_fragile,
                    },
                )

        recomputed_min_fail: list[dict[str, Any]] = []
        fragility_distribution: dict[str, int] = {}
        for name, frontier_row in _rows_by_name(survivors_by_cap.get("1", [])).items():
            mechanism = frontier_row["mechanism"]
            min_failing = _recomputed_min_failing(cache, mechanism)
            recomputed_min_fail.append({"name": name, "min_failing_coalition_size": min_failing})
            key = str(min_failing) if min_failing is not None else "none"
            fragility_distribution[key] = fragility_distribution.get(key, 0) + 1
            if min_expected.get(name) != min_failing:
                failures.append(
                    {
                        "scope": "min_failing_mismatch",
                        "cost": cost,
                        "name": name,
                        "expected": min_expected.get(name),
                        "recomputed": min_failing,
                    },
                )

        if row.get("fragility_distribution", {}) != fragility_distribution:
            failures.append(
                {
                    "scope": "fragility_distribution_mismatch",
                    "cost": cost,
                    "expected": row.get("fragility_distribution", {}),
                    "recomputed": fragility_distribution,
                },
            )

        verification_rows.append(
            {
                "cost": cost,
                "candidate_count": row.get("candidate_count"),
                "cap_failures": cap_failures,
                "counts_by_cap": counts_by_cap,
                "cap_digests": cap_digests,
                "recomputed_min_fail": recomputed_min_fail,
            },
        )

        if cost in selected_cost_rows:
            selected_row = selected_cost_rows[cost]
            selected_survivors_by_cap = selected_row.get("survivors_by_cap", {})
            if sorted(selected_survivors_by_cap.keys()) != cap_keys:
                failures.append(
                    {
                        "scope": "selected_cap_keys_mismatch",
                        "cost": cost,
                        "observed": sorted(selected_survivors_by_cap.keys()),
                        "expected": cap_keys,
                    },
                )

            for cap_key in cap_keys:
                cap = int(cap_key)
                for selected_frontier_row in selected_survivors_by_cap.get(cap_key, []):
                    mechanism = selected_frontier_row["mechanism"]
                    verification = _check(cache, mechanism, cap)
                    if not verification.get("accepted"):
                        failures.append(
                            {
                                "scope": "selected_row_rejected",
                                "cost": cost,
                                "cap": cap,
                                "name": mechanism["name"],
                                "verification": verification,
                            },
                        )

    row_by_cost = {int(row["cost"]): row for row in by_cost if isinstance(row, dict) and "cost" in row}
    selected_frontier: list[str] = []
    selected_frontier_reference: list[str] = []
    for cost in selected_costs:
        if cost not in selected_cost_rows or cost not in row_by_cost:
            continue
        selected_frontier.extend(
            [frontier_row["mechanism"]["name"] for frontier_row in selected_cost_rows[cost]["survivors_by_cap"]["1"]],
        )
        selected_frontier_reference.extend(
            [frontier_row["mechanism"]["name"] for frontier_row in row_by_cost[cost]["survivors_by_cap"]["1"]],
        )

    maxcap_payload: list[str] = []
    for row in by_cost:
        cost_key = row["cost"]
        for serialized_row in row.get("survivors_by_cap", {}).get(str(max_coalition_size), []):
            maxcap_payload.append(
                json.dumps(
                    {
                        "cost": cost_key,
                        "max_coalition_size": max_coalition_size,
                        "mechanism": serialized_row["mechanism"],
                    },
                    sort_keys=True,
                ),
            )
    independent_digest = hashlib.sha256("\n".join(sorted(maxcap_payload)).encode()).hexdigest()

    certificate = {
        "study": "public_project_coalition_value3_frontier",
        "artifact": str(artifact_path),
        "max_coalition_size": max_coalition_size,
        "cost_rows": verification_rows,
        "cost_count": len(verification_rows),
        "selected_costs": sorted(selected_costs),
        "selected_frontier_count": len(selected_frontier),
        "selected_frontier": selected_frontier,
        "selected_frontier_reference": sorted(selected_frontier_reference),
        "selected_frontier_match": sorted(selected_frontier) == sorted(selected_frontier_reference),
        "selected_failures": len([f for f in failures if str(f.get("scope", "")).startswith("selected")]),
        "independent_failure_count": len(failures),
        "independent_digest": independent_digest,
        "statement": (
            "Value-3 coalition-fragility extension verification: every serialized mechanism "
            f"is independently replayed at cap 1..{max_coalition_size}, selected costs are "
            "revalidated, and fragility summaries are recomputed from base survivors."
        ),
    }

    if certificate["selected_frontier_match"] is False:
        failures.append({"scope": "selected_frontier_match", "match": False})

    if failures:
        print(json.dumps({"independent_failures": failures[:20]}, indent=2, sort_keys=True))
        raise SystemExit(1)

    certificate_path = ROOT / "artifacts" / "public_project_coalition_value3_frontier_certificate.json"
    certificate_path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "certificate": str(certificate_path),
                "cost_rows": len(certificate["cost_rows"]),
                "independent_failure_count": certificate["independent_failure_count"],
                "independent_digest": independent_digest,
            },
            indent=2,
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
