"""Exact certificates for shallow max-affine public-project rules.

The intentionally small language is an executable specification for formulas
made from rational affine forms by addition, scaling, and ``max``.  It is not a
general neural-network verifier.  For a represented formula, every branch
boundary is affine; enumerating intersections with the ordered-cube facets
therefore yields a complete certificate for extrema of charge/first-best.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Iterable, Sequence


Affine = tuple[Fraction, ...]  # coefficients followed by constant
Point = tuple[Fraction, ...]


def affine(*coefficients: Fraction | int, constant: Fraction | int = 0) -> Affine:
    return tuple(Fraction(value) for value in coefficients) + (Fraction(constant),)


def _add(left: Affine, right: Affine) -> Affine:
    return tuple(x + y for x, y in zip(left, right))


def _scale(value: Affine, scalar: Fraction | int) -> Affine:
    return tuple(Fraction(scalar) * item for item in value)


@dataclass(frozen=True)
class Expr:
    """A shallow max-affine expression with exact rational coefficients."""

    affine_terms: tuple[Affine, ...] = ()
    maxima: tuple[tuple[Affine, ...], ...] = ()
    minima: tuple[tuple[Affine, ...], ...] = ()

    @classmethod
    def from_affine(cls, value: Affine) -> "Expr":
        return cls((value,))

    @classmethod
    def maximum(cls, *branches: Affine) -> "Expr":
        if len(branches) < 2:
            raise ValueError("a maximum needs at least two affine branches")
        return cls(maxima=(tuple(branches),))

    def __add__(self, other: "Expr") -> "Expr":
        return Expr(self.affine_terms + other.affine_terms, self.maxima + other.maxima,
                    self.minima + other.minima)

    def scale(self, scalar: Fraction | int) -> "Expr":
        factor = Fraction(scalar)
        maxima = tuple(tuple(_scale(branch, factor) for branch in maximum) for maximum in self.maxima)
        minima = tuple(tuple(_scale(branch, factor) for branch in minimum) for minimum in self.minima)
        if factor < 0:
            maxima, minima = minima, maxima
        return Expr(tuple(_scale(term, factor) for term in self.affine_terms), maxima, minima)

    def evaluate(self, point: Point) -> Fraction:
        def evaluate_affine(form: Affine) -> Fraction:
            return sum((coefficient * value for coefficient, value in zip(form[:-1], point)), form[-1])
        return (sum((evaluate_affine(term) for term in self.affine_terms), Fraction(0))
                + sum((max(evaluate_affine(branch) for branch in maximum)
                       for maximum in self.maxima), Fraction(0))
                + sum((min(evaluate_affine(branch) for branch in minimum)
                       for minimum in self.minima), Fraction(0)))

    def break_planes(self) -> tuple[Affine, ...]:
        planes = []
        for maximum in self.maxima:
            planes.extend(tuple(x - y for x, y in zip(left, right))
                          for left, right in combinations(maximum, 2))
        for minimum in self.minima:
            planes.extend(tuple(x - y for x, y in zip(left, right))
                          for left, right in combinations(minimum, 2))
        return tuple(planes)


def constant(dimension: int, value: Fraction | int) -> Affine:
    return affine(*([0] * dimension), constant=value)


def _solve(rows: Sequence[Affine]) -> Point | None:
    dimension = len(rows)
    table = [list(row[:-1]) + [-row[-1]] for row in rows]
    for column in range(dimension):
        pivot = next((row for row in range(column, dimension) if table[row][column]), None)
        if pivot is None:
            return None
        table[column], table[pivot] = table[pivot], table[column]
        divisor = table[column][column]
        table[column] = [value / divisor for value in table[column]]
        for row in range(dimension):
            if row != column and table[row][column]:
                factor = table[row][column]
                table[row] = [value - factor * base for value, base in zip(table[row], table[column])]
    return tuple(row[-1] for row in table)


def ordered_cube_facets(dimension: int) -> tuple[Affine, ...]:
    result = [affine(1, *([0] * (dimension - 1)), constant=0),
              affine(*([0] * (dimension - 1)), 1, constant=-1)]
    for index in range(dimension - 1):
        coefficients = [0] * dimension
        coefficients[index], coefficients[index + 1] = 1, -1
        result.append(affine(*coefficients))
    return tuple(result)


@dataclass(frozen=True)
class Certificate:
    arrangement_planes: int
    candidate_bases_examined: int
    vertices: tuple[Point, ...]
    minimum_charge_ratio: Fraction
    minimum_witness: Point
    maximum_charge_ratio: Fraction
    maximum_witness: Point
    worst_case_efficiency: Fraction
    minimum_budget_slack: Fraction
    minimum_slack_witness: Point


@dataclass(frozen=True)
class ExtremumCertificate:
    """Exact range of an expression on the declared ordered cube."""

    arrangement_planes: int
    candidate_bases_examined: int
    vertices: tuple[Point, ...]
    minimum: Fraction
    minimum_witness: Point
    maximum: Fraction
    maximum_witness: Point


@dataclass(frozen=True)
class GrovesUtilityCertificate:
    """Worst truthful utility across agents for an efficient Groves rule."""

    minimum_utility: Fraction
    minimum_witness: Point
    minimum_agent: int
    per_agent: tuple[ExtremumCertificate, ...]


def certify_ordered_cube_extrema(expression: Expr, dimension: int) -> ExtremumCertificate:
    """Certify exact extrema of a piecewise-affine expression on the cube."""
    planes = ordered_cube_facets(dimension) + expression.break_planes()
    vertices = set()
    for basis in combinations(planes, dimension):
        point = _solve(basis)
        if point is not None and all(Fraction(0) <= point[index] <= Fraction(1)
                                     for index in range(dimension)) and all(
                                         point[index] <= point[index + 1] for index in range(dimension - 1)):
            vertices.add(point)
    ordered_vertices = tuple(sorted(vertices))
    if not ordered_vertices:
        raise ValueError("empty arrangement")
    values = tuple((expression.evaluate(point), point) for point in ordered_vertices)
    minimum, minimum_witness = min(values)
    maximum, maximum_witness = max(values)
    return ExtremumCertificate(len(planes), sum(1 for _ in combinations(planes, dimension)),
                               ordered_vertices, minimum, minimum_witness, maximum, maximum_witness)


def certify_minimum_groves_utility(terms: Sequence[Expr], dimension: int) -> GrovesUtilityCertificate:
    """Certify ``min_i,theta S(theta)-h(theta_-i)`` in the stated Groves model."""
    if len(terms) != dimension:
        raise ValueError("one deleted-input term is required per agent")
    total = affine(*([1] * dimension), constant=0)
    first_best = Expr.maximum(total, constant(dimension, 1))
    certificates = tuple(certify_ordered_cube_extrema(first_best + term.scale(-1), dimension)
                         for term in terms)
    minimum, agent, witness = min((certificate.minimum, index, certificate.minimum_witness)
                                  for index, certificate in enumerate(certificates))
    return GrovesUtilityCertificate(minimum, witness, agent, certificates)


def certify_ordered_public_project_charge(charge: Expr, dimension: int) -> Certificate:
    """Certify a charge formula on ``0 <= x_1 <= ... <= x_n <= 1`` exactly."""
    total = affine(*([1] * dimension), constant=0)
    first_best = Expr.maximum(total, constant(dimension, 1))
    planes = ordered_cube_facets(dimension) + charge.break_planes() + first_best.break_planes()
    vertices = set()
    for basis in combinations(planes, dimension):
        point = _solve(basis)
        if point is not None and all(Fraction(0) <= point[index] <= Fraction(1)
                                     for index in range(dimension)) and all(
                                         point[index] <= point[index + 1] for index in range(dimension - 1)):
            vertices.add(point)
    ordered_vertices = tuple(sorted(vertices))
    if not ordered_vertices:
        raise ValueError("empty arrangement")
    rows = []
    for point in ordered_vertices:
        denominator = first_best.evaluate(point)
        total_charge = charge.evaluate(point)
        rows.append((total_charge / denominator, total_charge - Fraction(dimension - 1) * denominator, point))
    low_ratio, _, low_witness = min(rows)
    high_ratio, _, high_witness = max(rows)
    _, slack, slack_witness = min(rows, key=lambda row: (row[1], row[2]))
    return Certificate(len(planes), sum(1 for _ in combinations(planes, dimension)),
                       ordered_vertices, low_ratio, low_witness, high_ratio, high_witness,
                       Fraction(dimension) - high_ratio, slack, slack_witness)
