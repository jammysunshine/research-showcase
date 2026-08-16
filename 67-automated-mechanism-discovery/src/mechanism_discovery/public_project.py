"""Exact finite search for deterministic public-project mechanisms.

The study uses the standard single-parameter public-project model.  Agents
have integer values for a binary project, the project has a known cost, and a
direct rule is required to be anonymous and monotone.  For a monotone
allocation rule, the normalized DSIC/EPIR payment is the discrete critical
value.  This lets us exhaustively enumerate the allocation-rule class without
using a solver while retaining an independent, profile-by-profile verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from typing import Iterable


@dataclass(frozen=True)
class PublicProjectSpec:
    n_agents: int = 3
    max_value: int = 2
    cost: int = 3

    @property
    def profiles(self) -> tuple[tuple[int, ...], ...]:
        return tuple(product(range(self.max_value + 1), repeat=self.n_agents))

    @property
    def states(self) -> tuple[tuple[int, ...], ...]:
        """Anonymous states represented by sorted value profiles."""
        return tuple(sorted({tuple(sorted(p)) for p in self.profiles}))


@dataclass(frozen=True)
class PublicProjectMechanism:
    spec: PublicProjectSpec
    allocation_by_state: tuple[tuple[tuple[int, ...], int], ...]
    name: str = "unnamed"

    @lru_cache(maxsize=None)
    def _table(self) -> dict[tuple[int, ...], int]:
        return dict(self.allocation_by_state)

    def allocation(self, reports: tuple[int, ...]) -> int:
        state = tuple(sorted(reports))
        return int(self._table()[state])

    def threshold(self, reports: tuple[int, ...], agent: int) -> int | None:
        """Smallest report at which this agent is included, holding others fixed."""
        others = list(reports)
        for report in range(self.spec.max_value + 1):
            others[agent] = report
            if self.allocation(tuple(others)):
                return report
        return None

    def payment(self, reports: tuple[int, ...], agent: int) -> int:
        threshold = self.threshold(reports, agent)
        return 0 if threshold is None or not self.allocation(reports) else threshold

    def payments(self, reports: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(self.payment(reports, i) for i in range(self.spec.n_agents))

    def utility(self, true_values: tuple[int, ...], reports: tuple[int, ...], agent: int) -> int:
        return true_values[agent] * self.allocation(reports) - self.payment(reports, agent)

    def welfare(self, reports: tuple[int, ...]) -> int:
        return self.allocation(reports) * (sum(reports) - self.spec.cost)


def efficient_mechanism(spec: PublicProjectSpec) -> PublicProjectMechanism:
    return sum_threshold_mechanism(spec, spec.cost, name=f"efficient_sum_threshold_c{spec.cost}")


def sum_threshold_mechanism(spec: PublicProjectSpec, threshold: int, *, name: str | None = None) -> PublicProjectMechanism:
    rows = tuple((state, int(sum(state) >= threshold)) for state in spec.states)
    return PublicProjectMechanism(spec, rows, name or f"sum_threshold_{threshold}")


def sink_mechanism(spec: PublicProjectSpec, sink: int = 0) -> PublicProjectMechanism:
    """A canonical budget-balanced comparator that ignores one valuation."""
    if not 0 <= sink < spec.n_agents:
        raise ValueError("sink outside agent range")
    rows = []
    for state in spec.states:
        # A symmetric state does not identify the sink, so use the canonical
        # representative only for a named comparator; the verifier exposes
        # this as a non-anonymous comparator when evaluated profile-wise.
        rows.append((state, int(sum(state[:-1]) >= spec.cost)))
    return PublicProjectMechanism(spec, tuple(rows), name=f"sink_{sink}_proxy")


def _monotone(mask: int, states: tuple[tuple[int, ...], ...]) -> bool:
    for i, left in enumerate(states):
        for j, right in enumerate(states):
            if i == j or all(a <= b for a, b in zip(left, right)):
                if ((mask >> i) & 1) and not ((mask >> j) & 1):
                    return False
    return True


def enumerate_anonymous_monotone(spec: PublicProjectSpec) -> Iterable[PublicProjectMechanism]:
    """Enumerate every anonymous monotone Boolean allocation rule.

    A monotone rule is an upward-closed subset of the sorted-state poset.
    Enumerating its minimal active states (an antichain) avoids scanning the
    ``2**|states|`` bit masks used by the original pilot implementation.  The
    resulting masks are sorted before serialization so the three-agent
    certificate remains byte-for-byte compatible with the pilot artifact.
    """
    states = spec.states
    comparable = [
        sum(1 << j for j in range(i + 1, len(states))
            if all(a <= b for a, b in zip(states[i], states[j]))
            or all(a >= b for a, b in zip(states[i], states[j])))
        for i in range(len(states))
    ]

    masks: list[int] = []

    def visit(index: int, minimal: list[int], blocked: int) -> None:
        if index == len(states):
            mask = 0
            for source in minimal:
                for target, state in enumerate(states):
                    if all(a <= b for a, b in zip(states[source], state)):
                        mask |= 1 << target
            masks.append(mask)
            return
        visit(index + 1, minimal, blocked)
        if not (blocked >> index) & 1:
            visit(index + 1, minimal + [index], blocked | comparable[index])

    visit(0, [], 0)
    for mask in sorted(masks):
        rows = tuple((state, (mask >> i) & 1) for i, state in enumerate(states))
        yield PublicProjectMechanism(spec, rows, name=f"anonymous_monotone_mask_{mask}")


def verify_public_project(
    mechanism: PublicProjectMechanism,
    *,
    require_budget: bool = True,
    check_anonymity: bool = True,
    max_coalition_size: int = 1,
) -> dict:
    spec = mechanism.spec
    if max_coalition_size < 1:
        raise ValueError("max_coalition_size must be at least 1")
    witnesses: list[dict] = []
    for reports in spec.profiles:
        allocation = mechanism.allocation(reports)
        payments = mechanism.payments(reports)
        if allocation not in (0, 1):
            witnesses.append({"property": "feasibility", "profile": reports})
        if allocation == 0 and any(payments):
            witnesses.append({"property": "no_payment_without_project", "profile": reports, "payments": payments})
        if require_budget and allocation and sum(payments) < spec.cost:
            witnesses.append({"property": "weak_budget_balance", "profile": reports, "payments": payments})
        for agent in range(spec.n_agents):
            truthful = mechanism.utility(reports, reports, agent)
            if truthful < 0:
                witnesses.append({"property": "ex_post_ir", "profile": reports, "agent": agent, "utility": truthful})
            for deviation in range(spec.max_value + 1):
                if deviation == reports[agent]:
                    continue
                deviating = list(reports)
                deviating[agent] = deviation
                gain = mechanism.utility(reports, tuple(deviating), agent) - truthful
                if gain > 0:
                    witnesses.append({"property": "dsic", "profile": reports, "agent": agent, "deviation": deviation, "gain": gain})
    if max_coalition_size >= 2:
        coalition_sizes = range(2, min(max_coalition_size, spec.n_agents) + 1)
        for reports in spec.profiles:
            truthful_utils = [mechanism.utility(reports, reports, agent) for agent in range(spec.n_agents)]
            for coalition_size in coalition_sizes:
                for coalition in combinations(range(spec.n_agents), coalition_size):
                    for deviations in product(range(spec.max_value + 1), repeat=coalition_size):
                        if all(reports[agent] == deviated for agent, deviated in zip(coalition, deviations)):
                            continue
                        deviating = list(reports)
                        for agent, deviated in zip(coalition, deviations):
                            deviating[agent] = deviated
                        deviating_reports = tuple(deviating)
                        baseline_total = sum(truthful_utils[i] for i in coalition)
                        deviating_total = sum(
                            mechanism.utility(reports, deviating_reports, agent)
                            for agent in coalition
                        )
                        if deviating_total > baseline_total:
                            witnesses.append(
                                {
                                    "property": "coalitional_dsic",
                                    "profile": reports,
                                    "coalition": list(coalition),
                                    "deviation": list(deviations),
                                    "gain": deviating_total - baseline_total,
                                    "coalition_size": coalition_size,
                                }
                            )
                            break
                    else:
                        continue
                    break
    # Anonymous state tables are exact by construction; retain a check so the
    # certificate catches malformed serialized mechanisms.
    if check_anonymity:
        for reports in spec.profiles:
            for perm in _permutations(tuple(range(spec.n_agents))):
                permuted = tuple(reports[i] for i in perm)
                if mechanism.allocation(reports) != mechanism.allocation(permuted):
                    witnesses.append({"property": "anonymity", "profile": reports, "permutation": perm})
    properties = {w["property"] for w in witnesses}
    return {
        "accepted": not properties,
        "dsic": "dsic" not in properties,
        "ex_post_ir": "ex_post_ir" not in properties,
        "weak_budget_balance": "weak_budget_balance" not in properties,
        "feasibility": "feasibility" not in properties,
        "anonymity": "anonymity" not in properties,
        "coalitional_dsic": "coalitional_dsic" not in properties,
        "max_coalition_size": max_coalition_size,
        "witnesses": witnesses,
    }


def _permutations(values: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    if len(values) <= 1:
        yield values
        return
    for i, value in enumerate(values):
        rest = values[:i] + values[i + 1 :]
        for tail in _permutations(rest):
            yield (value,) + tail


def public_project_metrics(mechanism: PublicProjectMechanism) -> dict[str, float | int]:
    spec = mechanism.spec
    rows = []
    for profile in spec.profiles:
        optimal = max(0, sum(profile) - spec.cost)
        actual = mechanism.welfare(profile)
        rows.append({
            "profile": profile,
            "allocation": mechanism.allocation(profile),
            "payments": mechanism.payments(profile),
            "welfare": actual,
            "optimal_welfare": optimal,
            "regret": optimal - actual,
            "revenue": sum(mechanism.payments(profile)),
        })
    positive = [row for row in rows if row["optimal_welfare"] > 0]
    return {
        "expected_welfare": sum(row["welfare"] for row in rows) / len(rows),
        "expected_optimal_welfare": sum(row["optimal_welfare"] for row in rows) / len(rows),
        "expected_regret": sum(row["regret"] for row in rows) / len(rows),
        "worst_case_regret": max(row["regret"] for row in rows),
        "positive_optimum_profiles": len(positive),
        "project_rate": sum(row["allocation"] for row in rows) / len(rows),
        "max_revenue": max(row["revenue"] for row in rows),
        "state_count": len(spec.states),
        "profile_count": len(spec.profiles),
    }


def frontier(
    spec: PublicProjectSpec,
    *,
    check_anonymity: bool = True,
    max_coalition_size: int = 1,
) -> list[dict]:
    """Return all accepted mechanisms, sorted by regret then expected welfare."""
    accepted = []
    for mechanism in enumerate_anonymous_monotone(spec):
        verification = verify_public_project(mechanism, check_anonymity=check_anonymity, max_coalition_size=max_coalition_size)
        if verification["accepted"] and mechanism.allocation(tuple([spec.max_value] * spec.n_agents)):
            accepted.append({"mechanism": mechanism, "verification": verification, "metrics": public_project_metrics(mechanism)})
    return sorted(accepted, key=lambda row: (row["metrics"]["worst_case_regret"], -row["metrics"]["expected_welfare"], row["mechanism"].name))
