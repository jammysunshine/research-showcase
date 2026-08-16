"""Finite, deterministic binary social-choice mechanisms with transfers.

The experiment deliberately keeps the domain finite so every certificate can be
recomputed without a solver or external data.
"""

from dataclasses import dataclass
from itertools import product
from typing import Iterable

Type = int
Profile = tuple[int, int]
PAYMENT_GRID = (-1, 0, 1)  # positive means paid to the mechanism
PROFILES: tuple[Profile, ...] = tuple(product((0, 1), repeat=2))


@dataclass(frozen=True)
class Outcome:
    choice: int
    payments: tuple[int, int]


@dataclass(frozen=True)
class Mechanism:
    """A total direct-revelation rule, represented by its four profile outcomes."""

    outcomes: tuple[Outcome, Outcome, Outcome, Outcome]
    name: str = "unnamed"

    def outcome(self, reports: Profile) -> Outcome:
        return self.outcomes[PROFILES.index(reports)]


def value(true_type: Type, choice: int) -> int:
    """Unit value for the agent's preferred binary alternative, zero otherwise."""
    return int(true_type == choice)


def utility(true_type: Type, outcome: Outcome, agent: int) -> int:
    return value(true_type, outcome.choice) - outcome.payments[agent]


def priority_majority() -> Mechanism:
    """Legacy name for serial dictatorship with agent 0 as dictator."""
    return serial_dictatorship(0, name="priority_majority_agent_0")


def serial_dictatorship(priority_agent: int, *, name: str | None = None) -> Mechanism:
    """Canonical deterministic serial dictatorship for this one-choice domain."""
    if priority_agent not in (0, 1):
        raise ValueError("priority_agent must be 0 or 1")
    return Mechanism(
        tuple(Outcome(choice=profile[priority_agent], payments=(0, 0)) for profile in PROFILES),
        name=name or f"serial_dictatorship_agent_{priority_agent}",
    )


def anonymous_or() -> Mechanism:
    """Anonymous monotone rule: choose 1 if either report is 1."""
    return Mechanism(
        tuple(Outcome(choice=int(any(profile)), payments=(0, 0)) for profile in PROFILES),
        name="anonymous_or",
    )


def anonymous_and() -> Mechanism:
    """Anonymous monotone rule: choose 1 only if both reports are 1."""
    return Mechanism(
        tuple(Outcome(choice=int(all(profile)), payments=(0, 0)) for profile in PROFILES),
        name="anonymous_and",
    )


def majority_with_tie_break(tie_choice: int) -> Mechanism:
    """Utilitarian binary rule with a deterministic, report-independent tie break."""
    if tie_choice not in (0, 1):
        raise ValueError("tie_choice must be 0 or 1")

    def choice(profile: Profile) -> int:
        votes_for_one = sum(profile)
        if votes_for_one == 1:
            return tie_choice
        return int(votes_for_one == 2)

    return Mechanism(
        tuple(Outcome(choice(profile), (0, 0)) for profile in PROFILES),
        name=f"majority_tie_{tie_choice}",
    )


def vcg_pivot(tie_choice: int) -> Mechanism:
    """Clarke-pivot VCG for the binary public-decision valuation model.

    It maximizes reported total value with a fixed tie break.  The transfer of
    agent i is the other agent's foregone reported value, so it is DSIC and
    ex-post IR here but deliberately fails exact budget balance on disagreement.
    """
    if tie_choice not in (0, 1):
        raise ValueError("tie_choice must be 0 or 1")
    efficient = majority_with_tie_break(tie_choice)
    outcomes = []
    for profile in PROFILES:
        selected = efficient.outcome(profile).choice
        payments = tuple(
            1 - int(selected == profile[1 - agent])
            for agent in (0, 1)
        )
        outcomes.append(Outcome(selected, payments))
    return Mechanism(tuple(outcomes), name=f"vcg_pivot_tie_{tie_choice}")


def canonical_baselines() -> tuple[Mechanism, ...]:
    """Named canonical comparators representable on the frozen finite domain."""
    return (
        serial_dictatorship(0),
        serial_dictatorship(1),
        anonymous_and(),
        anonymous_or(),
        majority_with_tie_break(0),
        majority_with_tie_break(1),
        vcg_pivot(0),
        vcg_pivot(1),
        constant(0),
        constant(1),
    )


def constant(choice: int) -> Mechanism:
    return Mechanism(tuple(Outcome(choice, (0, 0)) for _ in PROFILES), name=f"constant_{choice}")


def valid_outcomes() -> Iterable[Outcome]:
    for choice in (0, 1):
        for payments in product(PAYMENT_GRID, repeat=2):
            if sum(payments) == 0:
                yield Outcome(choice, payments)
