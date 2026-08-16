#!/usr/bin/env python3
"""Run the serious finite public-project study and write machine-readable evidence."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechanism_discovery.public_project import (  # noqa: E402
    PublicProjectSpec,
    enumerate_anonymous_monotone,
    frontier,
    public_project_metrics,
    sum_threshold_mechanism,
    verify_public_project,
)


def serialise_mechanism(mechanism):
    return {
        "name": mechanism.name,
        "n_agents": mechanism.spec.n_agents,
        "max_value": mechanism.spec.max_value,
        "cost": mechanism.spec.cost,
        "allocation_by_state": [[list(state), allocation] for state, allocation in mechanism.allocation_by_state],
    }


def evolutionary_probe(spec: PublicProjectSpec, *, seed: int = 6701, population: int = 64, generations: int = 40) -> dict:
    """A seeded proposal loop; every proposal is still checked exactly."""
    rng = random.Random(seed)
    states = spec.states
    proposals = []
    for _ in range(population * generations):
        threshold = rng.randrange(0, spec.n_agents * spec.max_value + 2)
        proposal = sum_threshold_mechanism(spec, threshold, name=f"proposal_threshold_{threshold}")
        report = verify_public_project(proposal)
        proposals.append({"threshold": threshold, "accepted": report["accepted"], "metrics": public_project_metrics(proposal)})
    accepted = [p for p in proposals if p["accepted"]]
    best = max(accepted, key=lambda p: p["metrics"]["expected_welfare"], default=None)
    return {"seed": seed, "population": population, "generations": generations, "evaluated": len(proposals), "accepted": len(accepted), "best": best}


def main() -> None:
    config = json.loads((ROOT / "configs" / "public_project_study.json").read_text())
    study_spec = PublicProjectSpec(n_agents=config["n_agents"], max_value=config["max_value"], cost=3)
    all_rules = list(enumerate_anonymous_monotone(study_spec))
    rows = []
    for mechanism in all_rules:
        verification = verify_public_project(mechanism)
        if verification["accepted"] and mechanism.allocation((study_spec.max_value,) * study_spec.n_agents):
            rows.append({"mechanism": serialise_mechanism(mechanism), "verification": verification, "metrics": public_project_metrics(mechanism)})

    by_cost = {}
    for cost in config["costs"]:
        spec = PublicProjectSpec(study_spec.n_agents, study_spec.max_value, cost)
        candidates = frontier(spec)
        by_cost[str(cost)] = {
            "candidate_count": len(list(enumerate_anonymous_monotone(spec))),
            "accepted_count": len(candidates),
            "best_worst_case_regret": candidates[0]["metrics"]["worst_case_regret"] if candidates else None,
            "best_expected_welfare": max((r["metrics"]["expected_welfare"] for r in candidates), default=None),
            "best_mechanism": serialise_mechanism(candidates[0]["mechanism"]) if candidates else None,
        }

    scale = []
    for n_agents in range(3, 6):
        spec = PublicProjectSpec(n_agents=n_agents, max_value=2, cost=n_agents)
        feasible = []
        for threshold in range(0, n_agents * spec.max_value + 2):
            mechanism = sum_threshold_mechanism(spec, threshold)
            verification = verify_public_project(mechanism, check_anonymity=False)
            if verification["accepted"] and mechanism.allocation((spec.max_value,) * n_agents):
                feasible.append({"threshold": threshold, "metrics": public_project_metrics(mechanism)})
        best = min(feasible, key=lambda row: (row["metrics"]["worst_case_regret"], -row["metrics"]["expected_welfare"])) if feasible else None
        scale.append({"n_agents": n_agents, "cost": n_agents, "feasible_sum_thresholds": feasible, "best": best})

    # The optimized antichain enumerator makes an exact cross-n confirmation
    # affordable through five agents.  This is the generalization test for
    # the three-agent headline, not a sampled scaling curve.
    exact_scale = []
    for n_agents in range(3, 6):
        cost_frontier = []
        for cost in range(1, 2 * n_agents + 1):
            spec = PublicProjectSpec(n_agents=n_agents, max_value=2, cost=cost)
            candidates = frontier(spec, check_anonymity=False)
            cost_frontier.append({
                "cost": cost,
                "candidate_count": len(list(enumerate_anonymous_monotone(spec))),
                "accepted_count": len(candidates),
                "best_worst_case_regret": candidates[0]["metrics"]["worst_case_regret"] if candidates else None,
                "best_expected_welfare": max((r["metrics"]["expected_welfare"] for r in candidates), default=None),
                "accepted": [
                    {"mechanism": serialise_mechanism(r["mechanism"]), "metrics": r["metrics"]}
                    for r in candidates
                ],
            })
        exact_scale.append({"n_agents": n_agents, "max_value": 2, "cost_frontier": cost_frontier})

    payload = {
        "study": "public_project_exact_frontier",
        "question": "How much welfare is lost when deterministic anonymous DSIC/EPIR public-project rules must cover a known cost without deficit?",
        "configuration": config,
        "search_domain": {"n_agents": study_spec.n_agents, "max_value": study_spec.max_value, "costs": config["costs"], "allocation_class": "all anonymous monotone Boolean rules", "payment_rule": "normalized discrete critical value"},
        "enumerated_rule_count": len(all_rules),
        "accepted_at_cost_3_count": len(rows),
        "accepted_at_cost_3": rows,
        "cost_frontier": by_cost,
        "sum_threshold_scale": scale,
        "exact_cross_n_scale": exact_scale,
        "proposal_probe": evolutionary_probe(study_spec),
        "baseline_efficient": serialise_mechanism(sum_threshold_mechanism(study_spec, study_spec.cost, name="efficient_baseline")),
        "baseline_efficient_verification": verify_public_project(sum_threshold_mechanism(study_spec, study_spec.cost)),
    }
    artifact = ROOT / "artifacts" / "public_project_study.json"
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    csv = ROOT / "artifacts" / "public_project_frontier.csv"
    with csv.open("w", encoding="utf-8", newline="") as handle:
        handle.write("cost,accepted_count,best_worst_case_regret,best_expected_welfare,best_mechanism\n")
        for cost, row in by_cost.items():
            handle.write(f"{cost},{row['accepted_count']},{row['best_worst_case_regret']},{row['best_expected_welfare']},{row['best_mechanism']['name'] if row['best_mechanism'] else ''}\n")

    scaling_csv = ROOT / "artifacts" / "public_project_scaling.csv"
    with scaling_csv.open("w", encoding="utf-8", newline="") as handle:
        handle.write("n_agents,cost,candidate_count,accepted_count,best_worst_case_regret,best_expected_welfare\n")
        for result in exact_scale:
            for row in result["cost_frontier"]:
                handle.write(
                    f"{result['n_agents']},{row['cost']},{row['candidate_count']},"
                    f"{row['accepted_count']},{row['best_worst_case_regret']},{row['best_expected_welfare']}\n"
                )

    svg = ROOT / "reports" / "public_project_frontier.svg"
    svg.parent.mkdir(exist_ok=True)
    points = []
    for i, (cost, row) in enumerate(by_cost.items()):
        if row["best_worst_case_regret"] is not None:
            points.append((60 + i * 90, 230 - 28 * row["best_worst_case_regret"], cost))
    polyline = " ".join(f"{x},{y}" for x, y, _ in points)
    labels = "".join(f'<text x="{x - 4}" y="255" font-size="12">{c}</text>' for x, _, c in points)
    svg.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="680" height="290" viewBox="0 0 680 290">
<rect width="100%" height="100%" fill="white"/><text x="20" y="25" font-family="sans-serif" font-size="16">Exact BB frontier: worst-case welfare regret by cost</text>
<line x1="45" y1="230" x2="640" y2="230" stroke="#333"/><line x1="45" y1="40" x2="45" y2="230" stroke="#333"/>
<polyline points="{polyline}" fill="none" stroke="#1769aa" stroke-width="3"/>{''.join(f'<circle cx="{x}" cy="{y}" r="4" fill="#1769aa"/>' for x,y,_ in points)}{labels}
<text x="20" y="52" font-size="11">regret</text><text x="620" y="270" font-size="11">cost</text></svg>\n''')
    print(json.dumps({"artifact": str(artifact), "enumerated_rule_count": len(all_rules), "accepted_at_cost_3": len(rows), "cost_frontier": by_cost}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
