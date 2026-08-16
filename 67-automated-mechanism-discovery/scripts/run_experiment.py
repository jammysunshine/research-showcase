#!/usr/bin/env python3
"""Run the frozen Experiment 67 baseline, exact enumeration, and proposal loop."""
import json
import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mechanism_discovery.independent_verifier import check
from mechanism_discovery.model import anonymous_or, canonical_baselines, priority_majority
from mechanism_discovery.search import evolutionary_search, exhaustive_search
from mechanism_discovery.verifier import metrics, verify


def fingerprint(mechanism):
    return [{"profile": list(p), "choice": mechanism.outcome(p).choice,
             "payments": list(mechanism.outcome(p).payments)}
            for p in ((0, 0), (0, 1), (1, 0), (1, 1))]


def main():
    config = json.loads((ROOT / "configs/experiment_67.json").read_text())
    # Frozen accepted baseline.  The priority rule remains in the canonical
    # comparator catalogue as a deliberately rejecting fairness diagnostic.
    baseline = anonymous_or()
    baselines = canonical_baselines()
    exact = exhaustive_search()
    frontier = []
    for row in exact:
        frontier.append({"outcomes": fingerprint(row["mechanism"]), "metrics": row["metrics"]})
    output = {
        "config": config,
        "baseline": {"name": baseline.name, "outcomes": fingerprint(baseline),
                     "primary_verifier": verify(baseline).as_dict(),
                     "independent_verifier": check(baseline), "metrics": metrics(baseline)},
        "diagnostic_priority_baseline": {
            "name": priority_majority().name,
            "primary_verifier": verify(priority_majority()).as_dict(),
            "independent_verifier": check(priority_majority()),
            "metrics": metrics(priority_majority()),
        },
        "canonical_baselines": [
            {"name": mechanism.name, "outcomes": fingerprint(mechanism),
             "primary_verifier": verify(mechanism).as_dict(),
             "independent_verifier": check(mechanism), "metrics": metrics(mechanism)}
            for mechanism in baselines
        ],
        "exhaustive_search": {"candidate_count": config["exhaustive_candidates"], "accepted_count": len(exact),
                              "frontier": frontier},
        "evolutionary_search": evolutionary_search(**config["evolutionary"]),
    }
    artifact = ROOT / "artifacts/experiment_67_results.json"
    artifact.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    csv_path = ROOT / "artifacts/frontier.csv"
    with csv_path.open("w", newline="") as handle:
        fields = ["index", "expected_allocative_welfare", "expected_utility_disparity",
                  "max_utility_disparity", "worst_case_regret", "description_length", "choices"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(frontier):
            writer.writerow({"index": index, **{key: row["metrics"][key] for key in fields[1:-1]},
                             "choices": "".join(str(item["choice"]) for item in row["outcomes"])})
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    # Dependency-free SVG: the four certified tables plotted by welfare and
    # disparity, with labels kept in the artifact CSV for exact reuse.
    points = []
    for index, row in enumerate(frontier):
        x = 80 + row["metrics"]["expected_utility_disparity"] * 240
        y = 220 - row["metrics"]["expected_allocative_welfare"] * 100
        points.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5"/><text x="{x + 8:.1f}" y="{y + 4:.1f}">M{index}</text>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="520" height="280" viewBox="0 0 520 280">'
           '<rect width="100%" height="100%" fill="white"/><text x="20" y="24" font-size="16">Experiment 67 accepted frontier</text>'
           '<line x1="80" y1="220" x2="440" y2="220" stroke="black"/><line x1="80" y1="60" x2="80" y2="220" stroke="black"/>'
           '<text x="180" y="258">expected utility disparity</text><text transform="translate(16 190) rotate(-90)">expected allocative welfare</text>'
           + ''.join(points) + '</svg>\n')
    (report_dir / "frontier.svg").write_text(svg)
    print(artifact.relative_to(ROOT))


if __name__ == "__main__":
    main()
