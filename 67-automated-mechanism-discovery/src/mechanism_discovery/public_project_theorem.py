"""Closed-form frontier for the finite ternary public-project model.

The theorem in ``PUBLIC_PROJECT_THEOREM.md`` characterizes the accepted
allocation rules for every agent count. This module contains the constructive
side of that result without using the antichain search.
"""

from __future__ import annotations

from .public_project import PublicProjectMechanism, PublicProjectSpec, enumerate_anonymous_monotone


def theorem_frontier_count(n_agents: int, cost: int) -> int:
    """Number of accepted rules in the theorem's declared range."""
    if n_agents < 1 or cost < 1:
        raise ValueError("n_agents and cost must be positive")
    if cost > 2 * n_agents:
        return 0
    return n_agents + 1 if cost <= n_agents else 1


def theorem_mechanisms(spec: PublicProjectSpec) -> tuple[PublicProjectMechanism, ...]:
    """Construct exactly the theorem-predicted mechanisms for ``spec``."""
    if spec.n_agents < 1 or spec.max_value != 2 or spec.cost < 1:
        raise ValueError("the theorem requires n>=1, max_value=2, and cost>=1")
    n, cost = spec.n_agents, spec.cost
    if cost > 2 * n:
        return ()
    minimum_k = 0 if cost <= n else n
    mechanisms = []
    for k in range(minimum_k, n + 1):
        boundary = (1,) * (n - k) + (2,) * k
        rows = tuple(
            (state, int(all(left <= right for left, right in zip(boundary, state))))
            for state in spec.states
        )
        mechanisms.append(PublicProjectMechanism(spec, rows, name=f"theorem_suffix_k{k}_c{cost}"))
    return tuple(mechanisms)


def theorem_statement() -> str:
    return (
        "For every n>=1 and integer 1<=c<=2n, accepted rules are exactly "
        "q_k(v)=1 iff every value is at least 1 and at least k values equal 2, "
        "for k=0,...,n when c<=n, and only k=n when n<c<=2n."
    )


def value_lattice_minimum(cost: int, n_agents: int) -> int:
    """Least critical value capable of covering ``cost`` at the all-max profile."""
    if n_agents < 1 or cost < 1:
        raise ValueError("n_agents and cost must be positive")
    return (cost + n_agents - 1) // n_agents


def value_lattice_mechanisms(spec: PublicProjectSpec) -> tuple[PublicProjectMechanism, ...]:
    """Construct every rule in the arbitrary integer-value lattice theorem.

    The construction enumerates upper sets only in the restricted sublattice
    ``{ceil(c/n), ..., max_value}^n`` and embeds them in the full state table.
    It deliberately does not call the full-domain frontier or verifier.
    """
    if spec.n_agents < 1 or spec.max_value < 1 or spec.cost < 1:
        raise ValueError("the value-lattice theorem requires n, max_value, and cost to be positive")
    minimum = value_lattice_minimum(spec.cost, spec.n_agents)
    if minimum > spec.max_value:
        return ()
    restricted = PublicProjectSpec(
        n_agents=spec.n_agents,
        max_value=spec.max_value - minimum,
        cost=spec.cost,
    )
    mechanisms = []
    for source in enumerate_anonymous_monotone(restricted):
        if not source.allocation((restricted.max_value,) * restricted.n_agents):
            continue
        rows = tuple(
            (
                state,
                int(
                    min(state) >= minimum
                    and source.allocation(tuple(value - minimum for value in state))
                ),
            )
            for state in spec.states
        )
        mechanisms.append(
            PublicProjectMechanism(
                spec,
                rows,
                name=f"value_lattice_k{minimum}_{source.name}",
            )
        )
    return tuple(mechanisms)


def value_lattice_frontier_count(n_agents: int, max_value: int, cost: int) -> int:
    """Exact count in the arbitrary integer-value lattice theorem."""
    return len(value_lattice_mechanisms(PublicProjectSpec(n_agents, max_value, cost)))
