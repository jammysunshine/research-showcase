"""Standalone exact replay for the printed Guo (2024) four-agent formula.

This deliberately does not import ``published_rule_audit``.  It reconstructs
the decimal rule, every affine break plane, and the arrangement-vertex audit
from scratch so agreement is a cross-check rather than shared implementation.
"""

from fractions import Fraction
from itertools import combinations


def _relu(value):
    return max(value, Fraction(0))


def _h(x, y, z):
    x, y, z = sorted((x, y, z))
    return (
        _relu(-Fraction(7220, 10000) * x - Fraction(5927, 10000) * y - Fraction(5925, 10000) * z + Fraction(5926, 10000))
        + _relu(-Fraction(4485, 10000) * x - Fraction(5939, 10000) * y - Fraction(3858, 10000) * z + Fraction(3856, 10000))
        + _relu(Fraction(1925, 10000) * x + Fraction(4570, 10000) * y + Fraction(4436, 10000) * z - Fraction(2218, 10000))
        - _relu(-Fraction(4820, 10000) * x - Fraction(3097, 10000) * y - Fraction(915, 10000) * z + Fraction(3667, 10000))
        + Fraction(9197, 10000) * x + Fraction(6558, 10000) * y + Fraction(6646, 10000) * z + Fraction(2218, 10000)
    )


def _solve(rows):
    table = [list(row) for row in rows]
    for column in range(4):
        pivot = next((row for row in range(column, 4) if table[row][column]), None)
        if pivot is None:
            return None
        table[column], table[pivot] = table[pivot], table[column]
        divisor = table[column][column]
        table[column] = [value / divisor for value in table[column]]
        for row in range(4):
            if row != column and table[row][column]:
                divisor = table[row][column]
                table[row] = [value - divisor * pivot_value for value, pivot_value in zip(table[row], table[column])]
    return tuple(row[-1] for row in table)


def _planes():
    planes = [
        (Fraction(1), 0, 0, 0, 0), (0, 0, 0, Fraction(1), Fraction(1)),
        (Fraction(1), Fraction(-1), 0, 0, 0), (0, Fraction(1), Fraction(-1), 0, 0),
        (0, 0, Fraction(1), Fraction(-1), 0), (Fraction(1), Fraction(1), Fraction(1), Fraction(1), Fraction(1)),
    ]
    forms = ((-7220, -5927, -5925, 5926), (-4485, -5939, -3858, 3856),
             (1925, 4570, 4436, -2218), (-4820, -3097, -915, 3667))
    for omitted in range(4):
        remaining = [index for index in range(4) if index != omitted]
        for *weights, constant in forms:
            row = [Fraction(0)] * 4
            for index, weight in zip(remaining, weights):
                row[index] = Fraction(weight, 10000)
            planes.append(tuple(row + [-Fraction(constant, 10000)]))
    return planes


def replay(charge_offset=Fraction(0)):
    """Return the independently recomputed certificate fields."""
    vertices = set()
    for basis in combinations(_planes(), 4):
        point = _solve(basis)
        if point is not None and Fraction(0) <= point[0] <= point[1] <= point[2] <= point[3] <= Fraction(1):
            vertices.add(point)
    checks = []
    for point in sorted(vertices):
        charge = sum((_h(*(point[:index] + point[index + 1:])) + charge_offset for index in range(4)), Fraction(0))
        first_best = max(sum(point, Fraction(0)), Fraction(1))
        checks.append((charge / first_best, charge - 3 * first_best, point))
    lower = min(checks)
    upper = max(checks)
    return {
        "vertices_examined": len(vertices),
        "minimum_charge_ratio": lower[0],
        "minimum_deficit": lower[1],
        "minimum_witness": lower[2],
        "maximum_charge_ratio": upper[0],
        "maximum_witness": upper[2],
        "worst_case_efficiency": Fraction(4) - upper[0],
    }
