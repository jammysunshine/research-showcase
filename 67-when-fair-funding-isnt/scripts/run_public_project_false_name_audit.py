#!/usr/bin/env python3
"""Post-hoc false-name-manipulation audit of the canonical efficient/pivotal
public-project mechanism.

Phase X's coalition-robustness studies checked joint deviations by a group of
distinct real agents, each with their own true value. False-name manipulation
is a different, well-known attack (Yokoo, Sakurai & Matsubara 2004): a single
real agent fabricates extra fake report identities and controls all of them,
so there is exactly one true value behind several report slots, and the
attacker pays whatever the mechanism charges every slot they control.

`sum_threshold_mechanism`/`efficient_mechanism` build q(reports)=1[sum>=cost]
with critical-value payments, and this rule is defined identically for any
agent count -- so a false-name attack with a fake budget `f` is checked by
comparing the SAME rule (same cost) evaluated at `n_real` real reports against
that rule evaluated at `n_real+f` reports, where the attacker occupies one
real slot plus `f` fake slots and the other `n_real-1` real agents keep
reporting truthfully (they are unaware of the attack). `f=0` is a positive
control: it must reduce to ordinary single-agent DSIC (zero manipulable
rows), confirming the harness itself before trusting `f>=1`.

This is a bounded, non-preregistered supplement (see `PREREGISTRATION.md`'s
"Post-hoc coalition robustness extension" note on scope): fake budgets 1-2,
n_real up to 5, the same finite integer-value domain as Phase X.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechanism_discovery.public_project import (  # noqa: E402
    PublicProjectSpec,
    efficient_mechanism,
)


def _false_name_gains(n_real: int, max_value: int, cost: int, fake_budget: int) -> dict[str, Any]:
    base = efficient_mechanism(PublicProjectSpec(n_agents=n_real, max_value=max_value, cost=cost))
    attack = efficient_mechanism(
        PublicProjectSpec(n_agents=n_real + fake_budget, max_value=max_value, cost=cost)
    )
    controlled = list(range(n_real - 1, n_real - 1 + 1 + fake_budget))  # own slot + fake slots

    manipulable = []
    checked = 0
    for v in product(range(max_value + 1), repeat=n_real):
        base_alloc = base.allocation(v)
        for attacker in range(n_real):
            checked += 1
            baseline_utility = v[attacker] * base_alloc - base.payment(v, attacker)
            others = [v[j] for j in range(n_real) if j != attacker]
            best_gain = None
            best_witness = None
            for own_report in range(max_value + 1):
                for fake_reports in product(range(max_value + 1), repeat=fake_budget):
                    extended = tuple(others) + (own_report,) + fake_reports
                    alloc = attack.allocation(extended)
                    total_payment = sum(attack.payment(extended, idx) for idx in controlled)
                    utility = v[attacker] * alloc - total_payment
                    gain = utility - baseline_utility
                    if best_gain is None or gain > best_gain:
                        best_gain = gain
                        best_witness = {
                            "own_report": own_report,
                            "fake_reports": list(fake_reports),
                            "allocation": alloc,
                            "total_payment": total_payment,
                            "attacker_utility": utility,
                        }
            if best_gain is not None and best_gain > 0:
                manipulable.append(
                    {
                        "truthful_profile": list(v),
                        "attacker": attacker,
                        "baseline_utility": baseline_utility,
                        "gain": best_gain,
                        "witness": best_witness,
                    }
                )
    return {
        "n_real": n_real,
        "max_value": max_value,
        "cost": cost,
        "fake_budget": fake_budget,
        "checked": checked,
        "manipulable_count": len(manipulable),
        "manipulable": manipulable,
    }


def _audit_domain(domain: dict[str, Any]) -> dict[str, Any]:
    n_real = int(domain["n_real"])
    max_value = int(domain["max_value"])
    cost_rows = [
        _false_name_gains(n_real, max_value, int(cost), int(f))
        for cost in domain["costs"]
        for f in domain["fake_budgets"]
    ]
    selected_keys = {(int(item["cost"]), int(item["fake_budget"])) for item in domain["selected"]}
    selected_rows = [row for row in cost_rows if (row["cost"], row["fake_budget"]) in selected_keys]
    return {
        "n_real": n_real,
        "max_value": max_value,
        "cost_rows": cost_rows,
        "selected_rows": selected_rows,
    }


def main() -> None:
    started = time.perf_counter()
    config = json.loads((ROOT / "configs" / "public_project_false_name_audit.json").read_text())

    domain_results = [_audit_domain(domain) for domain in config["domains"]]

    selected_summary = []
    for result in domain_results:
        for row in result["selected_rows"]:
            selected_summary.append(
                {
                    "n_real": row["n_real"],
                    "cost": row["cost"],
                    "fake_budget": row["fake_budget"],
                    "manipulable_count": row["manipulable_count"],
                }
            )

    control_failures = [
        entry for entry in selected_summary if entry["fake_budget"] == 0 and entry["manipulable_count"] > 0
    ]

    payload = {
        "study": "public_project_false_name_audit",
        "question": (
            "Can a single real agent gain by fabricating extra fake report identities "
            "against the canonical efficient/pivotal public-project mechanism, compared "
            "to truthful single-identity reporting?"
        ),
        "configuration": config,
        "domains": domain_results,
        "selected_summary": selected_summary,
        "control_failures": control_failures,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    artifact = ROOT / "artifacts" / "public_project_false_name_audit.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "artifact": str(artifact),
                "elapsed_seconds": payload["elapsed_seconds"],
                "selected_summary": selected_summary,
                "control_failures": control_failures,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
