#!/usr/bin/env python3
"""Independent verification for the public-project coalition baseline audit."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechanism_discovery.public_project_independent import check  # noqa: E402


def _incentive_ok(report: dict[str, Any]) -> bool:
    """DSIC/coalitional-DSIC status only; ignores this mechanism's separately
    documented, unrelated weak-budget-balance deficit."""
    return bool(report["dsic"]) and bool(report["coalitional_dsic"])


def _min_failing_coalition_size(mechanism: dict[str, Any], max_coalition_size: int) -> int | None:
    for cap in range(1, max_coalition_size + 1):
        if not _incentive_ok(check(mechanism, max_coalition_size=cap)):
            return cap
    return None


def _replay_cost_row(row: dict[str, Any], max_coalition_size: int) -> dict[str, Any]:
    mechanism = row["mechanism"]
    unrestricted = check(mechanism, max_coalition_size=1)
    coalitional = check(mechanism, max_coalition_size=max_coalition_size)
    min_failing = _min_failing_coalition_size(mechanism, max_coalition_size)
    mismatches = []
    if _incentive_ok(unrestricted) != _incentive_ok(row["unrestricted_verification"]):
        mismatches.append("unrestricted_incentive_ok")
    if _incentive_ok(coalitional) != _incentive_ok(row["coalitional_verification"]):
        mismatches.append("coalitional_incentive_ok")
    if min_failing != row["min_failing_coalition_size"]:
        mismatches.append("min_failing_coalition_size")
    return {
        "n_agents": row["n_agents"],
        "cost": row["cost"],
        "name": mechanism["name"],
        "unrestricted_incentive_ok": _incentive_ok(unrestricted),
        "coalitional_incentive_ok": _incentive_ok(coalitional),
        "min_failing_coalition_size": min_failing,
        "mismatches": mismatches,
    }


def main() -> None:
    artifact_path = ROOT / "artifacts" / "public_project_coalition_baseline_audit.json"
    if not artifact_path.exists():
        raise SystemExit(f"missing artifact: {artifact_path}")

    artifact = json.loads(artifact_path.read_text())

    replayed_rows: list[dict[str, Any]] = []
    for result in artifact["domains"]:
        max_coalition_size = int(result["max_coalition_size"])
        if "by_n" in result:
            cost_rows = [row for block in result["by_n"] for row in block["cost_rows"]]
        else:
            cost_rows = result["cost_rows"]
        for row in cost_rows:
            replayed = _replay_cost_row(row, max_coalition_size)
            replayed["domain"] = result["domain"]
            replayed_rows.append(replayed)

    failures = [row for row in replayed_rows if row["mismatches"]]
    fragile_rows = [row for row in replayed_rows if row["min_failing_coalition_size"] is not None]

    independent_digest = hashlib.sha256(
        "\n".join(
            json.dumps(
                {
                    "domain": row["domain"],
                    "n_agents": row["n_agents"],
                    "cost": row["cost"],
                    "name": row["name"],
                    "unrestricted_incentive_ok": row["unrestricted_incentive_ok"],
                    "coalitional_incentive_ok": row["coalitional_incentive_ok"],
                    "min_failing_coalition_size": row["min_failing_coalition_size"],
                },
                sort_keys=True,
            )
            for row in sorted(replayed_rows, key=lambda r: (r["domain"], r["n_agents"], r["cost"]))
        ).encode()
    ).hexdigest()

    certificate = {
        "study": "public_project_coalition_baseline_audit",
        "artifact": str(artifact_path),
        "row_count": len(replayed_rows),
        "independent_failure_count": len(failures),
        "independent_failures": failures,
        "fragile_row_count": len(fragile_rows),
        "fragile_rows": [
            {
                "domain": row["domain"],
                "n_agents": row["n_agents"],
                "cost": row["cost"],
                "min_failing_coalition_size": row["min_failing_coalition_size"],
            }
            for row in fragile_rows
        ],
        "independent_digest": independent_digest,
        "statement": (
            "Every audited (domain, n, cost) row's canonical efficient/pivotal mechanism is "
            "independently rebuilt and rechecked at coalition cap 1 and at each domain's frozen "
            "max_coalition_size using a standalone checker that does not import public_project.py."
        ),
    }

    out = ROOT / "artifacts" / "public_project_coalition_baseline_audit_certificate.json"
    out.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "certificate": str(out),
                "row_count": certificate["row_count"],
                "independent_failures": certificate["independent_failure_count"],
                "fragile_row_count": certificate["fragile_row_count"],
                "independent_digest": certificate["independent_digest"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
