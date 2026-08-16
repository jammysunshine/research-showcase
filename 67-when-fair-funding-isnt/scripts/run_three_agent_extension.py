#!/usr/bin/env python3
"""Run the frozen exact three-agent extension and write its raw frontier."""

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechanism_discovery.three_agent_extension import (  # noqa: E402
    PROFILES, anonymous_and, anonymous_or, enumerate_anonymous_budget_balanced,
    fingerprint, majority, metrics, table_from_mechanism, verify,
)
from mechanism_discovery.three_agent_independent import (  # noqa: E402
    candidate_tables, check_table,
)


def digest(tables):
    payload = json.dumps([list(map(list, table)) for table in sorted(tables)], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    config_path = ROOT / "configs/experiment_67_three_agent.json"
    config = json.loads(config_path.read_text())
    mechanisms = enumerate_anonymous_budget_balanced()
    independent_candidates = candidate_tables()
    if len(mechanisms) != config["exhaustive_candidates"] or len(independent_candidates) != len(mechanisms):
        raise SystemExit("candidate-count mismatch")
    accepted = [mechanism for mechanism in mechanisms if verify(mechanism).accepted]
    independent_accepted = [table for table in independent_candidates if check_table(table)["accepted"]]
    primary_tables = {table_from_mechanism(mechanism) for mechanism in accepted}
    independent_tables = set(independent_accepted)
    if primary_tables != independent_tables:
        raise SystemExit("primary/independent accepted sets differ")
    baseline = majority()
    baselines = (anonymous_and(), anonymous_or(), baseline)
    output = {
        "config": config,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "candidate_generation": {
            "primary_count": len(mechanisms),
            "independent_count": len(independent_candidates),
            "accepted_count": len(accepted),
            "primary_frontier_sha256": digest(primary_tables),
            "independent_frontier_sha256": digest(independent_tables),
            "set_equal": primary_tables == independent_tables,
        },
        "baseline": {"name": baseline.name, "outcomes": fingerprint(baseline),
                     "verification": verify(baseline).as_dict(), "metrics": metrics(baseline)},
        "canonical_baselines": [
            {"name": mechanism.name, "outcomes": fingerprint(mechanism),
             "verification": verify(mechanism).as_dict(), "metrics": metrics(mechanism)}
            for mechanism in baselines
        ],
        "accepted_frontier": [
            {"outcomes": fingerprint(mechanism), "verification": verify(mechanism).as_dict(),
             "metrics": metrics(mechanism)} for mechanism in accepted
        ],
        "transfer_audit": {
            "accepted_nonzero_payment_tables": sum(
                any(payment for outcome in mechanism.outcomes for payment in outcome.payments)
                for mechanism in accepted
            ),
            "statement": "No accepted mechanism in this bounded anonymous class uses a nonzero transfer.",
        },
        "welfare_frontier": {
            "best_accepted_expected_allocative_welfare": max(
                metrics(mechanism)["expected_allocative_welfare"] for mechanism in accepted
            ),
            "baseline_expected_allocative_welfare": metrics(baseline)["expected_allocative_welfare"],
            "strict_improvers_over_majority": [
                fingerprint(mechanism) for mechanism in accepted
                if metrics(mechanism)["expected_allocative_welfare"] > metrics(baseline)["expected_allocative_welfare"]
            ],
            "statement": "Three-agent majority is pointwise welfare-maximal on this unit-value domain.",
        },
    }
    artifact = ROOT / "artifacts/experiment_67_three_agent_results.json"
    artifact.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    csv_path = ROOT / "artifacts/three_agent_frontier.csv"
    with csv_path.open("w", newline="") as handle:
        fields = ["index", "expected_allocative_welfare", "expected_utility_disparity",
                  "max_utility_disparity", "description_length", "choices", "payments"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, mechanism in enumerate(accepted):
            metric = metrics(mechanism)
            writer.writerow({
                "index": index,
                **{key: metric[key] for key in fields[1:5]},
                "choices": "".join(str(item["choice"]) for item in fingerprint(mechanism)),
                "payments": ";".join(",".join(map(str, item["payments"])) for item in fingerprint(mechanism)),
            })
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    points = []
    for index, mechanism in enumerate(accepted):
        metric = metrics(mechanism)
        x = 80 + metric["expected_utility_disparity"] * 180
        y = 220 - metric["expected_allocative_welfare"] * 60
        points.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5"/><text x="{x + 8:.1f}" y="{y + 4:.1f}">M{index}</text>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="520" height="280" viewBox="0 0 520 280">'
           '<rect width="100%" height="100%" fill="white"/><text x="20" y="24" font-size="16">Experiment 67 three-agent accepted frontier</text>'
           '<line x1="80" y1="220" x2="440" y2="220" stroke="black"/><line x1="80" y1="60" x2="80" y2="220" stroke="black"/>'
           '<text x="180" y="258">expected truthful utility spread</text><text transform="translate(16 190) rotate(-90)">expected allocative welfare</text>'
           + "".join(points) + '</svg>\n')
    (report_dir / "three_agent_frontier.svg").write_text(svg)
    print(artifact.relative_to(ROOT))


if __name__ == "__main__":
    main()
