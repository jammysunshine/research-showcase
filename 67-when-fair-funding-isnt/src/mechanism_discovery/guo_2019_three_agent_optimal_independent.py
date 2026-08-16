"""Standalone exact replay of IJCAI-2019 Equation (2), without shared code."""

from fractions import Fraction
from itertools import combinations


def _h(x, y):
    return (max(x + y, Fraction(2, 3)) + max(x + y, Fraction(1)) / 2
            - max(x, y, Fraction(2, 3)) / 2 - Fraction(1, 6))


def _solve(rows):
    table = [list(row) for row in rows]
    for column in range(3):
        pivot = next((row for row in range(column, 3) if table[row][column]), None)
        if pivot is None:
            return None
        table[column], table[pivot] = table[pivot], table[column]
        divisor = table[column][column]
        table[column] = [value / divisor for value in table[column]]
        for row in range(3):
            if row != column and table[row][column]:
                factor = table[row][column]
                table[row] = [value - factor * base for value, base in zip(table[row], table[column])]
    return tuple(row[3] for row in table)


def replay():
    planes = [(Fraction(1), 0, 0, 0), (0, 0, Fraction(1), Fraction(1)),
              (Fraction(1), Fraction(-1), 0, 0), (0, Fraction(1), Fraction(-1), 0),
              (Fraction(1), Fraction(1), Fraction(1), Fraction(1))]
    for left, right in ((0, 1), (0, 2), (1, 2)):
        for threshold in (Fraction(2, 3), Fraction(1)):
            row = [Fraction(0), Fraction(0), Fraction(0)]
            row[left] = row[right] = Fraction(1)
            planes.append(tuple(row + [threshold]))
    for index in range(3):
        row = [Fraction(0), Fraction(0), Fraction(0)]
        row[index] = Fraction(1)
        planes.append(tuple(row + [Fraction(2, 3)]))
    vertices = set()
    for basis in combinations(planes, 3):
        point = _solve(basis)
        if point is not None and Fraction(0) <= point[0] <= point[1] <= point[2] <= Fraction(1):
            vertices.add(point)
    rows = []
    for a, b, c in sorted(vertices):
        charge = _h(b, c) + _h(a, c) + _h(a, b)
        first_best = max(a + b + c, Fraction(1))
        rows.append((charge / first_best, (a, b, c)))
    lower = min(rows)
    upper = max(rows)
    return {"vertices_examined": len(vertices), "minimum_charge_ratio": lower[0],
            "minimum_witness": lower[1], "maximum_charge_ratio": upper[0],
            "maximum_witness": upper[1], "worst_case_efficiency": Fraction(3) - upper[0]}
