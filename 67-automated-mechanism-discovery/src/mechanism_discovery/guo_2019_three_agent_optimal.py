"""Exact continuous audit of Guo (IJCAI 2019), Equation (2).

The paper reproduces an optimal three-agent VCG redistribution rule credited
to Guo and Shen (2017).  The rule is evaluated exactly on the continuous
ordered cube.  The arrangement is complete because every term is affine
between the listed max-function break planes.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations


Point = tuple[Fraction, Fraction, Fraction]
Plane = tuple[Fraction, Fraction, Fraction, Fraction]


@dataclass(frozen=True)
class OptimalThreeAgentAudit:
    vertices_examined: int
    minimum_charge_ratio: Fraction
    minimum_witness: Point
    maximum_charge_ratio: Fraction
    maximum_witness: Point
    worst_case_efficiency: Fraction
    distinct_from_aaai_2024: bool
    distinctness_witness: Point
    equation_two_charge: Fraction
    aaai_2024_charge: Fraction


def h(left: Fraction, right: Fraction) -> Fraction:
    """Equation (2), with both definitions of ``T`` expanded exactly."""
    return (max(left + right, Fraction(2, 3))
            + max(left + right, Fraction(1)) / 2
            - max(left, right, Fraction(2, 3)) / 2
            - Fraction(1, 6))


def first_best(profile: Point) -> Fraction:
    return max(sum(profile, Fraction(0)), Fraction(1))


def total_charge(profile: Point) -> Fraction:
    a, b, c = profile
    return h(b, c) + h(a, c) + h(a, b)


def _solve(planes: tuple[Plane, Plane, Plane]) -> Point | None:
    matrix = [list(plane) for plane in planes]
    for column in range(3):
        pivot = next((row for row in range(column, 3) if matrix[row][column]), None)
        if pivot is None:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        divisor = matrix[column][column]
        matrix[column] = [value / divisor for value in matrix[column]]
        for row in range(3):
            if row != column and matrix[row][column]:
                factor = matrix[row][column]
                matrix[row] = [value - factor * base for value, base in zip(matrix[row], matrix[column])]
    return tuple(row[3] for row in matrix)  # type: ignore[return-value]


def break_planes() -> tuple[Plane, ...]:
    """Cube, ordering, first-best, pair-sum, and pair-maximum break planes."""
    planes: list[Plane] = [
        (Fraction(1), 0, 0, 0), (0, 0, Fraction(1), Fraction(1)),
        (Fraction(1), Fraction(-1), 0, 0), (0, Fraction(1), Fraction(-1), 0),
        (Fraction(1), Fraction(1), Fraction(1), Fraction(1)),
    ]
    for left, right in ((0, 1), (0, 2), (1, 2)):
        for threshold in (Fraction(2, 3), Fraction(1)):
            coefficients = [Fraction(0), Fraction(0), Fraction(0)]
            coefficients[left] = coefficients[right] = Fraction(1)
            planes.append((*coefficients, threshold))
    for index in range(3):
        coefficients = [Fraction(0), Fraction(0), Fraction(0)]
        coefficients[index] = Fraction(1)
        planes.append((*coefficients, Fraction(2, 3)))
    return tuple(planes)


def arrangement_vertices() -> tuple[Point, ...]:
    candidates = set()
    for basis in combinations(break_planes(), 3):
        point = _solve(basis)
        if point is not None and Fraction(0) <= point[0] <= point[1] <= point[2] <= Fraction(1):
            candidates.add(point)
    return tuple(sorted(candidates))


def audit() -> OptimalThreeAgentAudit:
    from src.mechanism_discovery.published_rule_audit import total_charge as aaai_charge

    vertices = arrangement_vertices()
    ratios = [(total_charge(point) / first_best(point), point) for point in vertices]
    minimum_ratio, minimum_witness = min(ratios)
    maximum_ratio, maximum_witness = max(ratios)
    differences = [(point, total_charge(point), aaai_charge(point)) for point in vertices
                   if total_charge(point) != aaai_charge(point)]
    witness, equation_two_charge, aaai_2024_charge = differences[0]
    return OptimalThreeAgentAudit(
        vertices_examined=len(vertices),
        minimum_charge_ratio=minimum_ratio,
        minimum_witness=minimum_witness,
        maximum_charge_ratio=maximum_ratio,
        maximum_witness=maximum_witness,
        worst_case_efficiency=Fraction(3) - maximum_ratio,
        distinct_from_aaai_2024=bool(differences),
        distinctness_witness=witness,
        equation_two_charge=equation_two_charge,
        aaai_2024_charge=aaai_2024_charge,
    )
