"""Certified exhaustive search and a seeded evolutionary proposal loop."""

from itertools import product
import random
from .model import Mechanism, Outcome, PROFILES, valid_outcomes
from .verifier import metrics, verify


def exhaustive_search() -> list[dict]:
    rows = []
    for outcomes in product(tuple(valid_outcomes()), repeat=len(PROFILES)):
        mechanism = Mechanism(outcomes, "enumerated")
        report = verify(mechanism)
        if report.accepted:
            rows.append({"mechanism": mechanism, "verification": report, "metrics": metrics(mechanism)})
    return rows


def evolutionary_search(seed: int, population_size: int, generations: int) -> dict:
    """Proposal-only search. Candidates retain zero transfers and are always verified."""
    rng = random.Random(seed)
    population = [tuple(rng.randint(0, 1) for _ in PROFILES) for _ in range(population_size)]
    evaluated = 0
    accepted = 0
    best = None
    for _ in range(generations):
        scored = []
        for choices in population:
            candidate = Mechanism(tuple(Outcome(c, (0, 0)) for c in choices), "evolutionary")
            report = verify(candidate)
            evaluated += 1
            if report.accepted:
                accepted += 1
                score = (metrics(candidate)["expected_welfare"], -metrics(candidate)["expected_utility_disparity"])
                scored.append((score, choices, candidate))
                if best is None or score > best[0]:
                    best = (score, choices)
        scored.sort(reverse=True)
        parents = [x[1] for x in scored[: max(1, population_size // 4)]] or population[:1]
        population = []
        while len(population) < population_size:
            parent = list(rng.choice(parents))
            parent[rng.randrange(len(parent))] ^= 1
            population.append(tuple(parent))
    return {"seed": seed, "population_size": population_size, "generations": generations,
            "evaluated": evaluated, "accepted": accepted, "best_choices": list(best[1]) if best else None}
