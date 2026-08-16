"""Exact three-agent extension of Experiment 67.

The extension is deliberately finite and auditable: three binary reports, two
alternatives, deterministic direct mechanisms, and integer transfers in
``{-2,-1,0,1,2}``.  We enumerate every anonymous, pointwise exactly
budget-balanced table in that bounded domain.  An anonymous table is determined
by its choice for each report-count and by the type-symmetric payment row for
counts one and two.
"""

from dataclasses import asdict, dataclass
from itertools import combinations, product

Type = int
Profile = tuple[int, int, int]
PAYMENT_GRID = (-2, -1, 0, 1, 2)
PROFILES: tuple[Profile, ...] = tuple(product((0, 1), repeat=3))


@dataclass(frozen=True)
class Outcome:
    choice: int
    payments: tuple[int, int, int]


@dataclass(frozen=True)
class Mechanism:
    outcomes: tuple[Outcome, ...]
    name: str = "unnamed"

    def outcome(self, reports: Profile) -> Outcome:
        return self.outcomes[PROFILES.index(reports)]


def value(true_type: Type, choice: int) -> int:
    return int(true_type == choice)


def utility(true_type: Type, outcome: Outcome, agent: int) -> int:
    return value(true_type, outcome.choice) - outcome.payments[agent]


def majority() -> Mechanism:
    return Mechanism(
        tuple(Outcome(int(sum(profile) >= 2), (0, 0, 0)) for profile in PROFILES),
        name="three_agent_majority",
    )


def anonymous_or() -> Mechanism:
    return Mechanism(
        tuple(Outcome(int(any(profile)), (0, 0, 0)) for profile in PROFILES),
        name="three_agent_anonymous_or",
    )


def anonymous_and() -> Mechanism:
    return Mechanism(
        tuple(Outcome(int(all(profile)), (0, 0, 0)) for profile in PROFILES),
        name="three_agent_anonymous_and",
    )


def _payment_patterns() -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Return type-1/type-0 payment pairs for count-one and count-two rows."""
    count_one = tuple((p_one, p_zero) for p_one in PAYMENT_GRID for p_zero in PAYMENT_GRID
                      if p_one + 2 * p_zero == 0)
    count_two = tuple((p_one, p_zero) for p_one in PAYMENT_GRID for p_zero in PAYMENT_GRID
                      if 2 * p_one + p_zero == 0)
    return count_one, count_two


COUNT_ONE_PAYMENTS, COUNT_TWO_PAYMENTS = _payment_patterns()


def enumerate_anonymous_budget_balanced() -> tuple[Mechanism, ...]:
    """Enumerate all 2^4 * 3 * 3 = 144 bounded anonymous BB mechanisms."""
    mechanisms: list[Mechanism] = []
    for choices in product((0, 1), repeat=4):
        for count_one in COUNT_ONE_PAYMENTS:
            for count_two in COUNT_TWO_PAYMENTS:
                outcomes = []
                for profile in PROFILES:
                    count = sum(profile)
                    choice = choices[count]
                    if count == 1:
                        p_one, p_zero = count_one
                        payments = tuple(p_one if bit else p_zero for bit in profile)
                    elif count == 2:
                        p_one, p_zero = count_two
                        payments = tuple(p_one if bit else p_zero for bit in profile)
                    else:
                        payments = (0, 0, 0)
                    outcomes.append(Outcome(choice, payments))
                mechanisms.append(Mechanism(tuple(outcomes)))
    return tuple(mechanisms)


@dataclass(frozen=True)
class Witness:
    property: str
    profile: Profile
    agent: int | None = None
    deviation: Profile | None = None
    coalition: tuple[int, ...] = ()
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
        return (self.dsic and self.ir and self.budget_balance and self.feasibility
                and self.anonymity and self.fairness and self.coalition_strategyproof)

    def as_dict(self) -> dict:
        result = asdict(self)
        result["accepted"] = self.accepted
        return result


def verify(mechanism: Mechanism) -> Verification:
    witnesses: list[Witness] = []
    for profile in PROFILES:
        outcome = mechanism.outcome(profile)
        if outcome.choice not in (0, 1) or len(outcome.payments) != 3:
            witnesses.append(Witness("feasibility", profile, detail="invalid outcome"))
        if sum(outcome.payments) != 0:
            witnesses.append(Witness("budget_balance", profile, detail=str(outcome.payments)))
        for agent in range(3):
            truthful = utility(profile[agent], outcome, agent)
            if truthful < 0:
                witnesses.append(Witness("individual_rationality", profile, agent, detail=str(truthful)))
            for report in (0, 1):
                if report == profile[agent]:
                    continue
                alternate = list(profile)
                alternate[agent] = report
                deviation = tuple(alternate)
                gained = utility(profile[agent], mechanism.outcome(deviation), agent)
                if gained > truthful:
                    witnesses.append(Witness("dsic", profile, agent, deviation, detail=f"{truthful}->{gained}"))

    # Exact anonymity: every permutation leaves choice invariant and permutes payments.
    for profile in PROFILES:
        for permutation in ((1, 0, 2), (2, 1, 0), (0, 2, 1)):
            permuted = tuple(profile[index] for index in permutation)
            left, right = mechanism.outcome(profile), mechanism.outcome(permuted)
            expected = tuple(left.payments[index] for index in permutation)
            if left.choice != right.choice or right.payments != expected:
                witnesses.append(Witness("anonymity", profile, detail=f"permutation={permutation}"))

    max_disparity = 0
    for profile in PROFILES:
        outcome = mechanism.outcome(profile)
        utilities = [utility(profile[i], outcome, i) for i in range(3)]
        max_disparity = max(max_disparity, max(utilities) - min(utilities))
    if max_disparity > 1:
        witnesses.append(Witness("fairness", (0, 0, 0), detail=f"max_disparity={max_disparity}"))

    # Coalition robustness: no nonempty coalition can make every member strictly better.
    agents = (0, 1, 2)
    for truthful in PROFILES:
        honest = mechanism.outcome(truthful)
        honest_utilities = tuple(utility(truthful[i], honest, i) for i in agents)
        for size in range(1, 4):
            for coalition in combinations(agents, size):
                outsiders = tuple(i for i in agents if i not in coalition)
                for reports in product((0, 1), repeat=size):
                    joint = list(truthful)
                    for agent, report in zip(coalition, reports):
                        joint[agent] = report
                    joint_report = tuple(joint)
                    if joint_report == truthful:
                        continue
                    proposed = mechanism.outcome(joint_report)
                    proposed_utilities = tuple(utility(truthful[i], proposed, i) for i in agents)
                    if all(proposed_utilities[i] > honest_utilities[i] for i in coalition):
                        witnesses.append(Witness("coalition_strategyproof", truthful,
                                                 coalition=coalition, deviation=joint_report))

    for profile in PROFILES:
        complement = tuple(1 - bit for bit in profile)
        if mechanism.outcome(complement).choice != 1 - mechanism.outcome(profile).choice:
            witnesses.append(Witness("neutrality", profile, deviation=complement))

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
    welfare = allocative_welfare = revenue = disparity = 0
    max_disparity = 0
    regret = 0
    for profile in PROFILES:
        outcome = mechanism.outcome(profile)
        utilities = [utility(profile[i], outcome, i) for i in range(3)]
        welfare += sum(utilities)
        allocative_welfare += sum(int(profile[i] == outcome.choice) for i in range(3))
        revenue += sum(outcome.payments)
        spread = max(utilities) - min(utilities)
        disparity += spread
        max_disparity = max(max_disparity, spread)
        for agent in range(3):
            alternate = list(profile)
            alternate[agent] = 1 - alternate[agent]
            alternate_utility = utility(profile[agent], mechanism.outcome(tuple(alternate)), agent)
            regret = max(regret, alternate_utility - utilities[agent])
    choices = [mechanism.outcome(profile).choice for profile in PROFILES]
    return {
        "expected_welfare": welfare / len(PROFILES),
        "expected_allocative_welfare": allocative_welfare / len(PROFILES),
        "expected_revenue": revenue / len(PROFILES),
        "expected_utility_disparity": disparity / len(PROFILES),
        "max_utility_disparity": float(max_disparity),
        "worst_case_regret": float(regret),
        "description_length": float(sum(c != 0 for c in choices) +
                                    sum(abs(x) for p in PROFILES for x in mechanism.outcome(p).payments)),
        "evaluation_profiles": float(len(PROFILES)),
    }


def fingerprint(mechanism: Mechanism) -> list[dict]:
    return [{"profile": list(profile), "choice": mechanism.outcome(profile).choice,
             "payments": list(mechanism.outcome(profile).payments)} for profile in PROFILES]


def table_from_mechanism(mechanism: Mechanism) -> tuple[tuple[int, int, int, int], ...]:
    return tuple((item["choice"], *item["payments"]) for item in fingerprint(mechanism))
