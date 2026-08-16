#!/usr/bin/env python3
"""Independent verification for the public-project false-name audit.

This does not import `public_project.py`. It recomputes the sum-threshold
allocation rule and critical-value payment directly from the closed-form
definition (build iff sum(reports) >= cost; payment_i = max(0, cost -
sum(reports excluding i)) when the project builds, else 0) -- the same
formula `public_project_independent.py` already uses for the coalition
audits, applied here to a cross-agent-count comparison instead of a
serialized allocation table.
"""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _allocation(reports: tuple[int, ...], cost: int) -> int:
    return int(sum(reports) >= cost)


def _payment(reports: tuple[int, ...], agent: int, cost: int) -> int:
    if not _allocation(reports, cost):
        return 0
    others_sum = sum(reports) - reports[agent]
    return max(0, cost - others_sum)


def _false_name_gains(n_real: int, max_value: int, cost: int, fake_budget: int) -> dict[str, Any]:
    controlled = list(range(n_real - 1, n_real + fake_budget))
    manipulable = []
    checked = 0
    for v in product(range(max_value + 1), repeat=n_real):
        base_alloc = _allocation(v, cost)
        for attacker in range(n_real):
            checked += 1
            baseline_utility = v[attacker] * base_alloc - _payment(v, attacker, cost)
            others = [v[j] for j in range(n_real) if j != attacker]
            best_gain = None
            for own_report in range(max_value + 1):
                for fake_reports in product(range(max_value + 1), repeat=fake_budget):
                    extended = tuple(others) + (own_report,) + fake_reports
                    alloc = _allocation(extended, cost)
                    total_payment = sum(_payment(extended, idx, cost) for idx in controlled)
                    utility = v[attacker] * alloc - total_payment
                    gain = utility - baseline_utility
                    if best_gain is None or gain > best_gain:
                        best_gain = gain
            if best_gain is not None and best_gain > 0:
                manipulable.append({"truthful_profile": list(v), "attacker": attacker, "gain": best_gain})
    return {"checked": checked, "manipulable_count": len(manipulable), "manipulable": manipulable}


def main() -> None:
    artifact_path = ROOT / "artifacts" / "public_project_false_name_audit.json"
    if not artifact_path.exists():
        raise SystemExit(f"missing artifact: {artifact_path}")

    artifact = json.loads(artifact_path.read_text())

    replayed_rows: list[dict[str, Any]] = []
    for domain in artifact["domains"]:
        n_real = int(domain["n_real"])
        max_value = int(domain["max_value"])
        for row in domain["cost_rows"]:
            cost = int(row["cost"])
            fake_budget = int(row["fake_budget"])
            replay = _false_name_gains(n_real, max_value, cost, fake_budget)
            mismatches = []
            if replay["checked"] != row["checked"]:
                mismatches.append("checked")
            if replay["manipulable_count"] != row["manipulable_count"]:
                mismatches.append("manipulable_count")
            replayed_rows.append(
                {
                    "n_real": n_real,
                    "cost": cost,
                    "fake_budget": fake_budget,
                    "checked": replay["checked"],
                    "manipulable_count": replay["manipulable_count"],
                    "mismatches": mismatches,
                }
            )

    failures = [row for row in replayed_rows if row["mismatches"]]
    manipulable_rows = [row for row in replayed_rows if row["manipulable_count"] > 0]
    control_failures = [
        row for row in replayed_rows if row["fake_budget"] == 0 and row["manipulable_count"] > 0
    ]

    independent_digest = hashlib.sha256(
        "\n".join(
            json.dumps(
                {
                    "n_real": row["n_real"],
                    "cost": row["cost"],
                    "fake_budget": row["fake_budget"],
                    "checked": row["checked"],
                    "manipulable_count": row["manipulable_count"],
                },
                sort_keys=True,
            )
            for row in sorted(replayed_rows, key=lambda r: (r["n_real"], r["cost"], r["fake_budget"]))
        ).encode()
    ).hexdigest()

    certificate = {
        "study": "public_project_false_name_audit",
        "artifact": str(artifact_path),
        "row_count": len(replayed_rows),
        "independent_failure_count": len(failures),
        "independent_failures": failures,
        "manipulable_row_count": len(manipulable_rows),
        "control_failure_count": len(control_failures),
        "independent_digest": independent_digest,
        "statement": (
            "Every audited (n_real, cost, fake_budget) row is independently rebuilt from the "
            "closed-form sum-threshold/critical-value formula, without importing public_project.py, "
            "and rechecked for manipulable-count agreement. fake_budget=0 is a positive control and "
            "must show zero manipulable rows."
        ),
    }

    out = ROOT / "artifacts" / "public_project_false_name_audit_certificate.json"
    out.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "certificate": str(out),
                "row_count": certificate["row_count"],
                "independent_failures": certificate["independent_failure_count"],
                "manipulable_row_count": certificate["manipulable_row_count"],
                "control_failure_count": certificate["control_failure_count"],
                "independent_digest": certificate["independent_digest"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
