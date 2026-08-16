"""Exact audit of the three-agent rule printed in Guo (AAAI 2024).

The cited paper gives a two-ReLU anonymous Groves term for three agents.  Its
printed decimal coefficients are treated here as exact terminating decimals,
not as recovered network weights.  This module verifies that *printed rule*
on the continuous ordered cube ``0 <= a <= b <= c <= 1``.

The verifier is finite because the rule and first-best value are piecewise
affine.  Their break planes partition the ordered cube into polytopes, and an
affine-over-positive-affine ratio attains its extrema at a polytope vertex.
We enumerate all intersections of three break/facet planes with exact
``Fraction`` arithmetic, then check every resulting arrangement vertex.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Sequence


Point = tuple[Fraction, Fraction, Fraction]
Plane = tuple[Fraction, Fraction, Fraction, Fraction]


@dataclass(frozen=True)
class ContinuousAudit:
    """Certificate summary for the printed three-agent mechanism."""

    vertices_examined: int
    minimum_charge_ratio: Fraction
    minimum_witness: Point
    maximum_charge_ratio: Fraction
    maximum_witness: Point
    worst_case_efficiency: Fraction


@dataclass(frozen=True)
class FourAgentAudit:
    """Exact continuous audit of the printed four-agent, five-node formula."""

    vertices_examined: int
    minimum_charge_ratio: Fraction
    minimum_deficit: Fraction
    minimum_witness: tuple[Fraction, Fraction, Fraction, Fraction]
    maximum_charge_ratio: Fraction
    maximum_witness: tuple[Fraction, Fraction, Fraction, Fraction]
    worst_case_efficiency: Fraction


def relu(value: Fraction) -> Fraction:
    return max(value, Fraction(0))


def published_h(left: Fraction, right: Fraction) -> Fraction:
    """The 3-agent formula on p. 9742 of Guo (2024), exactly as printed."""
    if left > right:
        left, right = right, left
    return (
        Fraction(2, 3) * relu(left + right - 1)
        + Fraction(1, 6) * relu(5 * left + 3 * right - 2)
        + Fraction(2, 3)
    )


def first_best(profile: Point) -> Fraction:
    return max(sum(profile, Fraction(0)), Fraction(1))


def total_charge(profile: Point) -> Fraction:
    a, b, c = profile
    return published_h(b, c) + published_h(a, c) + published_h(a, b)


def _solve_three(planes: Sequence[Plane]) -> Point | None:
    """Exact solution of three affine equations; ``None`` if singular."""
    matrix = [list(plane) for plane in planes]
    for column in range(3):
        pivot = next((row for row in range(column, 3) if matrix[row][column]), None)
        if pivot is None:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        scale = matrix[column][column]
        matrix[column] = [item / scale for item in matrix[column]]
        for row in range(3):
            if row == column:
                continue
            scale = matrix[row][column]
            if scale:
                matrix[row] = [item - scale * base for item, base in zip(matrix[row], matrix[column])]
    return tuple(row[3] for row in matrix)  # type: ignore[return-value]


def break_planes() -> tuple[Plane, ...]:
    """Ordered-cube facets and every change-of-affinity plane in the audit."""
    planes: list[Plane] = [
        (Fraction(1), Fraction(0), Fraction(0), Fraction(0)),  # a = 0
        (Fraction(0), Fraction(0), Fraction(1), Fraction(1)),  # c = 1
        (Fraction(1), Fraction(-1), Fraction(0), Fraction(0)),  # a = b
        (Fraction(0), Fraction(1), Fraction(-1), Fraction(0)),  # b = c
        (Fraction(1), Fraction(1), Fraction(1), Fraction(1)),  # sum = 1
    ]
    for first, second in ((0, 1), (0, 2), (1, 2)):
        coefficients = [Fraction(0), Fraction(0), Fraction(0)]
        coefficients[first] = Fraction(1)
        coefficients[second] = Fraction(1)
        planes.append((*coefficients, Fraction(1)))
        coefficients = [Fraction(0), Fraction(0), Fraction(0)]
        coefficients[first] = Fraction(5)
        coefficients[second] = Fraction(3)
        planes.append((*coefficients, Fraction(2)))
    return tuple(planes)


def arrangement_vertices() -> tuple[Point, ...]:
    """All ordered-cube vertices induced by the piecewise-affine break planes."""
    candidates = set()
    for planes in combinations(break_planes(), 3):
        candidate = _solve_three(planes)
        if candidate is not None and Fraction(0) <= candidate[0] <= candidate[1] <= candidate[2] <= Fraction(1):
            candidates.add(candidate)
    return tuple(sorted(candidates))


def audit_printed_rule() -> ContinuousAudit:
    """Return exact extrema and worst-case efficiency for the printed formula."""
    vertices = arrangement_vertices()
    ratios = [(total_charge(point) / first_best(point), point) for point in vertices]
    minimum_ratio, minimum_witness = min(ratios)
    maximum_ratio, maximum_witness = max(ratios)
    return ContinuousAudit(
        vertices_examined=len(vertices),
        minimum_charge_ratio=minimum_ratio,
        minimum_witness=minimum_witness,
        maximum_charge_ratio=maximum_ratio,
        maximum_witness=maximum_witness,
        worst_case_efficiency=Fraction(3) - maximum_ratio,
    )


def published_h_four(first: Fraction, second: Fraction, third: Fraction) -> Fraction:
    """The 4-agent five-ReLU formula printed on p. 9742 of Guo (2024)."""
    first, second, third = sorted((first, second, third))
    return (
        relu(-Fraction(7220, 10000) * first - Fraction(5927, 10000) * second
             - Fraction(5925, 10000) * third + Fraction(5926, 10000))
        + relu(-Fraction(4485, 10000) * first - Fraction(5939, 10000) * second
               - Fraction(3858, 10000) * third + Fraction(3856, 10000))
        + relu(Fraction(1925, 10000) * first + Fraction(4570, 10000) * second
               + Fraction(4436, 10000) * third - Fraction(2218, 10000))
        - relu(-Fraction(4820, 10000) * first - Fraction(3097, 10000) * second
               - Fraction(915, 10000) * third + Fraction(3667, 10000))
        + Fraction(9197, 10000) * first + Fraction(6558, 10000) * second
        + Fraction(6646, 10000) * third + Fraction(2218, 10000)
    )


def _solve_square(planes: Sequence[Sequence[Fraction]]) -> tuple[Fraction, ...] | None:
    """Exact solution of a square system represented as ``coefficients, rhs``."""
    dimension = len(planes)
    matrix = [list(plane) for plane in planes]
    for column in range(dimension):
        pivot = next((row for row in range(column, dimension) if matrix[row][column]), None)
        if pivot is None:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        scale = matrix[column][column]
        matrix[column] = [item / scale for item in matrix[column]]
        for row in range(dimension):
            if row == column:
                continue
            scale = matrix[row][column]
            if scale:
                matrix[row] = [item - scale * base for item, base in zip(matrix[row], matrix[column])]
    return tuple(row[-1] for row in matrix)


def break_planes_four() -> tuple[tuple[Fraction, ...], ...]:
    """All facets and ReLU/first-best break planes for four ordered reports."""
    planes: list[tuple[Fraction, ...]] = [
        (Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(1)),
        (Fraction(1), Fraction(-1), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(-1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(-1), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(1), Fraction(1), Fraction(1)),
    ]
    # Each tuple is affine weights followed by its constant term.  A break is
    # weight · input = -constant, hence the sign reversal in the final entry.
    relu_forms = (
        (-7220, -5927, -5925, 5926),
        (-4485, -5939, -3858, 3856),
        (1925, 4570, 4436, -2218),
        (-4820, -3097, -915, 3667),
    )
    for omitted in range(4):
        kept = [index for index in range(4) if index != omitted]
        for first, second, third, constant in relu_forms:
            coefficients = [Fraction(0)] * 4
            for index, weight in zip(kept, (first, second, third)):
                coefficients[index] = Fraction(weight, 10000)
            planes.append(tuple(coefficients + [-Fraction(constant, 10000)]))
    return tuple(planes)


def arrangement_vertices_four() -> tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...]:
    """Vertices of the four-dimensional arrangement for the printed formula."""
    candidates = set()
    for planes in combinations(break_planes_four(), 4):
        candidate = _solve_square(planes)
        if candidate is not None and Fraction(0) <= candidate[0] <= candidate[1] <= candidate[2] <= candidate[3] <= Fraction(1):
            candidates.add(candidate)
    return tuple(sorted(candidates))  # type: ignore[return-value]


def audit_printed_four_agent_rule(charge_offset: Fraction = Fraction(0)) -> FourAgentAudit:
    """Audit the printed 4-agent rule, optionally after the paper's constant repair.

    The paper's correction construction adds ``epsilon_L / n`` to each Groves
    term.  ``charge_offset`` expresses that quantity exactly; it does not
    recover the unreported neural-network weights.
    """
    vertices = arrangement_vertices_four()
    ratios = []
    for point in vertices:
        charge = sum((published_h_four(*(point[:index] + point[index + 1:])) + charge_offset
                      for index in range(4)), Fraction(0))
        s_value = max(sum(point, Fraction(0)), Fraction(1))
        ratios.append((charge / s_value, charge - 3 * s_value, point))
    minimum_ratio, minimum_deficit, minimum_witness = min(ratios)
    maximum_ratio, _, maximum_witness = max(ratios)
    return FourAgentAudit(
        vertices_examined=len(vertices),
        minimum_charge_ratio=minimum_ratio,
        minimum_deficit=minimum_deficit,
        minimum_witness=minimum_witness,
        maximum_charge_ratio=maximum_ratio,
        maximum_witness=maximum_witness,
        worst_case_efficiency=Fraction(4) - maximum_ratio,
    )
