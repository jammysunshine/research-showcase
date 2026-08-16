"""Primary exhaustive verifier; every predicate returns concrete witnesses."""

from dataclasses import asdict, dataclass
from .model import Mechanism, PROFILES, utility


@dataclass(frozen=True)
class Witness:
    property: str
    profile: tuple[int, int]
    agent: int | None = None
    deviation: int | None = None
    detail: str = ""


@dataclass(frozen=True)
class Verification:
    dsic: bool
    ir: bool
    budget_balance: bool
    feasibility: bool
    anonymity: bool
    fairness: bool
    coalition_strategyproof: bool
    neutrality: bool
    witnesses: tuple[Witness, ...]

    @property
    def accepted(self) -> bool:
        # Fairness is frozen as anonymity plus max truthful utility disparity <= 1.
        # The bounded robustness threat model is two-agent coalition deviations.
        return (
            self.dsic and self.ir and self.budget_balance and self.feasibility
            and self.fairness and self.coalition_strategyproof
        )

    def as_dict(self) -> dict:
        data = asdict(self)
        data["accepted"] = self.accepted
        return data


def verify(mechanism: Mechanism) -> Verification:
    witnesses: list[Witness] = []
    for reports in PROFILES:
        outcome = mechanism.outcome(reports)
        if outcome.choice not in (0, 1) or len(outcome.payments) != 2:
            witnesses.append(Witness("feasibility", reports, detail="invalid outcome"))
        if sum(outcome.payments) != 0:
            witnesses.append(Witness("budget_balance", reports, detail=str(outcome.payments)))
        for agent in (0, 1):
            true_type = reports[agent]
            truthful = utility(true_type, outcome, agent)
            if truthful < 0:
                witnesses.append(Witness("individual_rationality", reports, agent, detail=str(truthful)))
            for deviation in (0, 1):
                if deviation == true_type:
                    continue
                deviating_reports = list(reports)
                deviating_reports[agent] = deviation
                deviating = utility(true_type, mechanism.outcome(tuple(deviating_reports)), agent)
                if deviating > truthful:
                    witnesses.append(Witness("dsic", reports, agent, deviation, f"{truthful}->{deviating}"))

    # Exact anonymity: swapping agents swaps payments and leaves the decision unchanged.
    for reports in PROFILES:
        swapped = (reports[1], reports[0])
        left, right = mechanism.outcome(reports), mechanism.outcome(swapped)
        if left.choice != right.choice or left.payments != (right.payments[1], right.payments[0]):
            witnesses.append(Witness("anonymity", reports, detail=f"vs {swapped}"))

    # Fairness is an exact finite predicate: anonymity and max truthful utility
    # disparity no greater than one unit on every profile.
    max_disparity = 0
    for reports in PROFILES:
        outcome = mechanism.outcome(reports)
        utilities = [utility(reports[i], outcome, i) for i in (0, 1)]
        max_disparity = max(max_disparity, abs(utilities[0] - utilities[1]))
    if max_disparity > 1:
        witnesses.append(Witness("fairness", (0, 0), detail=f"max_disparity={max_disparity}"))

    # Bounded robustness threat model: no coalition can strictly improve both
    # members by jointly changing reports.
    for truthful in PROFILES:
        truthful_outcome = mechanism.outcome(truthful)
        truthful_utilities = [utility(truthful[i], truthful_outcome, i) for i in (0, 1)]
        for joint_report in PROFILES:
            if joint_report == truthful:
                continue
            deviating_outcome = mechanism.outcome(joint_report)
            deviating_utilities = [utility(truthful[i], deviating_outcome, i) for i in (0, 1)]
            if all(deviating_utilities[i] > truthful_utilities[i] for i in (0, 1)):
                witnesses.append(Witness("coalition_strategyproof", truthful,
                                         detail=f"joint_report={joint_report}"))

    # Neutrality is reported for the equivalence/impossibility audit, but is not
    # part of the primary acceptance predicate.
    for reports in PROFILES:
        complement = (1 - reports[0], 1 - reports[1])
        if mechanism.outcome(complement).choice != 1 - mechanism.outcome(reports).choice:
            witnesses.append(Witness("neutrality", reports, detail=f"vs {complement}"))

    properties = {w.property for w in witnesses}
    return Verification(
        dsic="dsic" not in properties,
        ir="individual_rationality" not in properties,
        budget_balance="budget_balance" not in properties,
        feasibility="feasibility" not in properties,
        anonymity="anonymity" not in properties,
        fairness="fairness" not in properties and "anonymity" not in properties,
        coalition_strategyproof="coalition_strategyproof" not in properties,
        neutrality="neutrality" not in properties,
        witnesses=tuple(witnesses),
    )


def metrics(mechanism: Mechanism) -> dict[str, float]:
    """Uniform-profile welfare, revenue, fairness, regret, and simplicity."""
    welfare = 0
    allocative_welfare = 0
    revenue = 0
    disparity = 0
    max_disparity = 0
    regret = 0
    for profile in PROFILES:
        outcome = mechanism.outcome(profile)
        utilities = [utility(profile[i], outcome, i) for i in (0, 1)]
        welfare += sum(utilities)
        allocative_welfare += sum(int(profile[i] == outcome.choice) for i in (0, 1))
        revenue += sum(outcome.payments)
        disparity += abs(utilities[0] - utilities[1])
        max_disparity = max(max_disparity, abs(utilities[0] - utilities[1]))
        for agent, true_type in enumerate(profile):
            truthful = utilities[agent]
            other = profile[1 - agent]
            deviation = 1 - true_type
            alternate = (deviation, other) if agent == 0 else (other, deviation)
            regret = max(regret, utility(true_type, mechanism.outcome(alternate), agent) - truthful)
    choices = [mechanism.outcome(p).choice for p in PROFILES]
    return {
        "expected_welfare": welfare / len(PROFILES),
        "expected_allocative_welfare": allocative_welfare / len(PROFILES),
        "expected_revenue": revenue / len(PROFILES),
        "expected_utility_disparity": disparity / len(PROFILES),
        "max_utility_disparity": float(max_disparity),
        "worst_case_regret": float(regret),
        "description_length": float(sum(c != 0 for c in choices) + sum(abs(x) for p in PROFILES for x in mechanism.outcome(p).payments)),
        "evaluation_profiles": float(len(PROFILES)),
    }
