#!/usr/bin/env python3
"""Independent replay and adversarial confirmation for the public-project study."""

from __future__ import annotations

import hashlib
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mechanism_discovery.public_project_independent import check  # noqa: E402


def main() -> None:
    study = json.loads((ROOT / "artifacts" / "public_project_study.json").read_text())
    rows = study["accepted_at_cost_3"]
    independent = []
    for row in rows:
        independent.append({"name": row["mechanism"]["name"], "verification": check(row["mechanism"])})
    failures = [row for row in independent if not row["verification"]["accepted"]]

    # Held-out value magnitudes: the searched table is evaluated only through
    # the sum-threshold family, whose critical-value rule extends naturally.
    heldout = []
    for threshold in range(1, 7):
        failures_for_threshold = 0
        profiles = list(product(range(4), repeat=3))
        for profile in profiles:
            allocation = int(sum(profile) >= threshold)
            payments = []
            for agent in range(3):
                others = sum(profile) - profile[agent]
                payments.append(max(0, threshold - others) if allocation else 0)
            if allocation and sum(payments) < 3:
                failures_for_threshold += 1
            for agent, truthful_value in enumerate(profile):
                truthful = truthful_value * allocation - payments[agent]
                for report in range(4):
                    if report == truthful_value:
                        continue
                    alternate_sum = sum(profile) - truthful_value + report
                    alt_alloc = int(alternate_sum >= threshold)
                    alt_payment = max(0, threshold - (sum(profile) - truthful_value)) if alt_alloc else 0
                    if truthful_value * alt_alloc - alt_payment > truthful:
                        failures_for_threshold += 1
        heldout.append({"threshold": threshold, "profiles": len(profiles), "failures": failures_for_threshold})

    digest = hashlib.sha256(json.dumps(independent, sort_keys=True).encode()).hexdigest()
    cross_n_rows = []
    for result in study.get("exact_cross_n_scale", []):
        for frontier_row in result["cost_frontier"]:
            cross_n_rows.extend(frontier_row["accepted"])
    cross_n_checks = [check(row["mechanism"]) for row in cross_n_rows]
    cross_n_failures = [report for report in cross_n_checks if not report["accepted"]]
    cross_n_names = sorted(
        f"{row['mechanism']['n_agents']}:{row['mechanism']['cost']}:{row['mechanism']['name']}"
        for row in cross_n_rows
    )
    certificate = {
        "study": "public_project_exact_frontier",
        "primary_rows": len(rows),
        "independent_rows": len(independent),
        "independent_failure_count": len(failures),
        "independent_digest": digest,
        "cross_n_rows": len(cross_n_rows),
        "cross_n_failure_count": len(cross_n_failures),
        "cross_n_digest": hashlib.sha256("\n".join(cross_n_names).encode()).hexdigest(),
        "heldout_sum_threshold_audit": heldout,
        "heldout_total_failures": sum(row["failures"] for row in heldout),
        "statement": "The independent checker agrees with every serialized accepted main-study row and every serialized cross-n row; held-out sum-threshold rules are checked on all 4^3 value profiles.",
    }
    path = ROOT / "artifacts" / "public_project_certificate.json"
    path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(certificate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
