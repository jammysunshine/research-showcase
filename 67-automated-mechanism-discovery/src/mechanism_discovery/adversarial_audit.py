"""Exact held-out and adversarial audits for the frozen two-agent domain."""

from fractions import Fraction
from itertools import product

from .independent_verifier import ROWS, ROW_INDEX, table_from_mechanism


def _utility(preference: int, magnitude: int, row: tuple[int, int, int], agent: int) -> int:
    return magnitude * int(row[0] == preference) - row[agent + 1]


def welfare_by_profile(table: tuple[tuple[int, int, int], ...]) -> tuple[int, ...]:
    return tuple(
        sum(_utility(profile[agent], 1, row, agent) for agent in (0, 1))
        for profile, row in zip(ROWS, table)
    )


def parse_distributions(config: dict) -> dict[str, tuple[Fraction, ...]]:
    parsed = {}
    for name, weights in config["distributions"].items():
        vector = tuple(Fraction(weight[0], weight[1]) for weight in weights)
        if len(vector) != len(ROWS) or sum(vector) != 1 or any(weight < 0 for weight in vector):
            raise ValueError(f"invalid confirmation distribution: {name}")
        parsed[name] = vector
    return parsed


def distributional_welfare(table, distributions: dict[str, tuple[Fraction, ...]]) -> dict[str, str]:
    welfare = welfare_by_profile(table)
    return {name: str(sum((weight * value for weight, value in zip(weights, welfare)), Fraction(0)))
            for name, weights in distributions.items()}


def pointwise_no_welfare_improvement(candidate, baseline) -> dict:
    candidate_welfare = welfare_by_profile(candidate)
    baseline_welfare = welfare_by_profile(baseline)
    deltas = tuple(candidate_value - baseline_value
                   for candidate_value, baseline_value in zip(candidate_welfare, baseline_welfare))
    return {"candidate_welfare": list(candidate_welfare), "baseline_welfare": list(baseline_welfare),
            "deltas": list(deltas), "all_nonpositive": all(delta <= 0 for delta in deltas)}


def coalition_witnesses(table: tuple[tuple[int, int, int], ...]) -> list[dict]:
    """Find Pareto-improving joint report deviations for every nonempty coalition."""
    witnesses = []
    for true_profile in ROWS:
        honest = table[ROW_INDEX[true_profile]]
        for coalition in ((0,), (1,), (0, 1)):
            for reports in ROWS:
                if any(reports[agent] != true_profile[agent] for agent in (0, 1) if agent not in coalition):
                    continue
                proposed = table[ROW_INDEX[reports]]
                gains = [_utility(true_profile[agent], 1, proposed, agent) -
                         _utility(true_profile[agent], 1, honest, agent) for agent in coalition]
                if all(gain >= 0 for gain in gains) and any(gain > 0 for gain in gains):
                    witnesses.append({"true_profile": list(true_profile), "coalition": list(coalition),
                                      "reports": list(reports), "gains": gains})
    return witnesses


def magnitude_perturbation_witnesses(table: tuple[tuple[int, int, int], ...]) -> list[dict]:
    """Audit DSIC and ex-post IR when values are independently in {0,1,2}."""
    witnesses = []
    for true_profile in ROWS:
        for magnitudes in product((0, 1, 2), repeat=2):
            honest = table[ROW_INDEX[true_profile]]
            if honest[1] + honest[2] != 0:
                witnesses.append({"property": "budget_balance", "profile": list(true_profile)})
            for agent in (0, 1):
                honest_utility = _utility(true_profile[agent], magnitudes[agent], honest, agent)
                if honest_utility < 0:
                    witnesses.append({"property": "individual_rationality", "profile": list(true_profile),
                                      "magnitudes": list(magnitudes), "agent": agent})
                lie = list(true_profile)
                lie[agent] = 1 - true_profile[agent]
                lie_utility = _utility(true_profile[agent], magnitudes[agent], table[ROW_INDEX[tuple(lie)]], agent)
                if lie_utility > honest_utility:
                    witnesses.append({"property": "dsic", "profile": list(true_profile),
                                      "magnitudes": list(magnitudes), "agent": agent,
                                      "gain": lie_utility - honest_utility})
    return witnesses


def audit_baseline(mechanism) -> dict:
    table = table_from_mechanism(mechanism)
    coalition = coalition_witnesses(table)
    perturbations = magnitude_perturbation_witnesses(table)
    return {
        "coalition_pareto_deviation_count": len(coalition),
        "coalition_witnesses": coalition,
        "magnitude_perturbation_failure_count": len(perturbations),
        "magnitude_perturbation_witnesses": perturbations,
    }
