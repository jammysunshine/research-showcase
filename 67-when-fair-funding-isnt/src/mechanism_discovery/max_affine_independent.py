"""Standalone replay for serialized shallow max-affine certificates.

This module deliberately imports neither ``piecewise_affine`` nor formula
transcriptions.  It interprets the certificate's rational expression payload,
derives its arrangement, and recomputes every reported extremum.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Any


def _fraction(value: str | int) -> Fraction:
    return Fraction(value)


def _forms(items: list[list[str]]) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(_fraction(value) for value in item) for item in items)


def _evaluate(form: tuple[Fraction, ...], point: tuple[Fraction, ...]) -> Fraction:
    return sum((weight * value for weight, value in zip(form[:-1], point)), form[-1])


def _expression_value(specification: dict[str, Any], point: tuple[Fraction, ...]) -> Fraction:
    total = sum((_evaluate(form, point) for form in _forms(specification["affine_terms"])), Fraction(0))
    total += sum((max(_evaluate(branch, point) for branch in branches)
                  for branches in (_forms(group) for group in specification["maxima"])), Fraction(0))
    return total + sum((min(_evaluate(branch, point) for branch in branches)
                        for branches in (_forms(group) for group in specification["minima"])), Fraction(0))


def _network_value(specification: dict[str, Any], inputs: tuple[Fraction, ...]) -> Fraction:
    """Evaluate a serialized rational affine--ReLU--affine network directly."""
    output = sum((_fraction(weight) * value
                  for weight, value in zip(specification["output_weights"], inputs)),
                 _fraction(specification["output_bias"]))
    for unit in specification["hidden"]:
        preactivation = sum((_fraction(weight) * value
                             for weight, value in zip(unit["weights"], inputs)),
                            _fraction(unit["bias"]))
        output += _fraction(unit["output_weight"]) * max(Fraction(0), preactivation)
    return output


def _deleted_input_network_charge(specification: dict[str, Any], point: tuple[Fraction, ...]) -> Fraction:
    return sum((_network_value(specification, point[:index] + point[index + 1:])
                for index in range(len(point))), Fraction(0))


def _network_break_planes(specification: dict[str, Any], dimension: int) -> tuple[tuple[Fraction, ...], ...]:
    """Lift each deleted-input ReLU preactivation boundary to report space."""
    planes = []
    for deleted in range(dimension):
        for unit in specification["hidden"]:
            # A zero output coefficient makes the activation semantically
            # irrelevant. The compiler eliminates it, so the direct route
            # must certify the same minimal function arrangement.
            if _fraction(unit["output_weight"]) == 0:
                continue
            weights = tuple(_fraction(weight) for weight in unit["weights"])
            if len(weights) != dimension - 1:
                raise ValueError("source network input dimension disagrees with certificate")
            iterator = iter(weights)
            planes.append(tuple(Fraction(0) if coordinate == deleted else next(iterator)
                                for coordinate in range(dimension))
                          + (_fraction(unit["bias"]),))
    return tuple(planes)


def _deleted_network_break_planes(specification: dict[str, Any], dimension: int,
                                  deleted: int) -> tuple[tuple[Fraction, ...], ...]:
    """Lift activation boundaries for one specified deleted-input term."""
    planes = []
    for unit in specification["hidden"]:
        if _fraction(unit["output_weight"]) == 0:
            continue
        weights = tuple(_fraction(weight) for weight in unit["weights"])
        if len(weights) != dimension - 1:
            raise ValueError("source network input dimension disagrees with certificate")
        iterator = iter(weights)
        planes.append(tuple(Fraction(0) if coordinate == deleted else next(iterator)
                            for coordinate in range(dimension)) + (_fraction(unit["bias"]),))
    return tuple(planes)


def _canonical_plane(plane: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    pivot = next((value for value in plane if value != 0), None)
    if pivot is None:
        return plane
    return tuple(value / pivot for value in plane)


def _feasible_vertices(planes: tuple[tuple[Fraction, ...], ...], dimension: int) -> tuple[tuple[Fraction, ...], ...]:
    planes = tuple(dict.fromkeys(_canonical_plane(plane) for plane in planes))
    vertices = set()
    for basis in combinations(planes, dimension):
        point = _solve(basis)
        if point is not None and all(Fraction(0) <= value <= Fraction(1) for value in point) and all(
            point[index] <= point[index + 1] for index in range(dimension - 1)
        ):
            vertices.add(point)
    return tuple(sorted(vertices))


def _difference(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    return tuple(a - b for a, b in zip(left, right))


def _base_planes(dimension: int) -> tuple[tuple[Fraction, ...], ...]:
    """Domain and first-best boundaries shared by all public-project audits."""
    planes = [[Fraction(1)] + [Fraction(0)] * (dimension - 1) + [Fraction(0)]]
    planes.append([Fraction(0)] * (dimension - 1) + [Fraction(1), Fraction(-1)])
    for index in range(dimension - 1):
        form = [Fraction(0)] * (dimension + 1)
        form[index], form[index + 1] = Fraction(1), Fraction(-1)
        planes.append(form)
    planes.append([Fraction(1)] * dimension + [Fraction(-1)])
    return tuple(tuple(plane) for plane in planes)


def _planes(specification: dict[str, Any], dimension: int) -> tuple[tuple[Fraction, ...], ...]:
    planes = list(_base_planes(dimension))
    for key in ("maxima", "minima"):
        for group in specification[key]:
            forms = _forms(group)
            planes.extend(_difference(left, right) for left, right in combinations(forms, 2))
    return tuple(tuple(plane) for plane in planes)


def _solve(rows: tuple[tuple[Fraction, ...], ...]) -> tuple[Fraction, ...] | None:
    dimension = len(rows)
    table = [list(row[:-1]) + [-row[-1]] for row in rows]
    for column in range(dimension):
        pivot = next((row for row in range(column, dimension) if table[row][column] != 0), None)
        if pivot is None:
            return None
        table[column], table[pivot] = table[pivot], table[column]
        divisor = table[column][column]
        table[column] = [value / divisor for value in table[column]]
        for row in range(dimension):
            if row != column and table[row][column] != 0:
                multiplier = table[row][column]
                table[row] = [value - multiplier * pivot_value
                              for value, pivot_value in zip(table[row], table[column])]
    return tuple(row[-1] for row in table)


def replay_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Recompute a certificate entry from its serialized expression only."""
    dimension = int(entry["dimension"])
    specification = entry["specification"]
    planes = _planes(specification, dimension)
    bases_examined = 0
    for basis in combinations(planes, dimension):
        bases_examined += 1
    ordered = _feasible_vertices(planes, dimension)
    rows = []
    for point in ordered:
        first_best = max(sum(point, Fraction(0)), Fraction(1))
        charge = _expression_value(specification, point)
        rows.append((charge / first_best, charge - Fraction(dimension - 1) * first_best, point))
    low, _, low_witness = min(rows)
    high, _, high_witness = max(rows)
    _, slack, slack_witness = min(rows, key=lambda row: (row[1], row[2]))
    return {
        "arrangement_planes": len(planes),
        "candidate_bases_examined": bases_examined,
        "vertices": [[f"{value.numerator}/{value.denominator}" for value in point] for point in ordered],
        "minimum_charge_ratio": f"{low.numerator}/{low.denominator}",
        "minimum_witness": [f"{value.numerator}/{value.denominator}" for value in low_witness],
        "maximum_charge_ratio": f"{high.numerator}/{high.denominator}",
        "maximum_witness": [f"{value.numerator}/{value.denominator}" for value in high_witness],
        "worst_case_efficiency": f"{(Fraction(dimension) - high).numerator}/{(Fraction(dimension) - high).denominator}",
        "minimum_budget_slack": f"{slack.numerator}/{slack.denominator}",
        "minimum_slack_witness": [f"{value.numerator}/{value.denominator}" for value in slack_witness],
    }


def replay_deleted_input_network(specification: dict[str, Any], dimension: int) -> dict[str, Any]:
    """Certify a serialized ReLU source directly, without compiled expressions.

    The candidate set is formed from ordered-cube, first-best, and ReLU
    activation boundaries.  On each resulting cell the direct source charge
    is affine, so its extrema occur at these exact vertices.  This is a
    second verifier route: it consumes only source-network coefficients and
    never reads the producer's max/min expression serialization.
    """
    planes = _base_planes(dimension) + _network_break_planes(specification, dimension)
    bases_examined = sum(1 for _ in combinations(planes, dimension))
    ordered = _feasible_vertices(planes, dimension)
    rows = []
    for point in ordered:
        first_best = max(sum(point, Fraction(0)), Fraction(1))
        charge = _deleted_input_network_charge(specification, point)
        rows.append((charge / first_best, charge - Fraction(dimension - 1) * first_best, point))
    low, _, low_witness = min(rows)
    high, _, high_witness = max(rows)
    _, slack, slack_witness = min(rows, key=lambda row: (row[1], row[2]))
    return {
        "arrangement_planes": len(planes),
        "candidate_bases_examined": bases_examined,
        "vertices": [[f"{value.numerator}/{value.denominator}" for value in point] for point in ordered],
        "minimum_charge_ratio": f"{low.numerator}/{low.denominator}",
        "minimum_witness": [f"{value.numerator}/{value.denominator}" for value in low_witness],
        "maximum_charge_ratio": f"{high.numerator}/{high.denominator}",
        "maximum_witness": [f"{value.numerator}/{value.denominator}" for value in high_witness],
        "worst_case_efficiency": f"{(Fraction(dimension) - high).numerator}/{(Fraction(dimension) - high).denominator}",
        "minimum_budget_slack": f"{slack.numerator}/{slack.denominator}",
        "minimum_slack_witness": [f"{value.numerator}/{value.denominator}" for value in slack_witness],
    }


def replay_deleted_input_network_utility_margin(specification: dict[str, Any], dimension: int) -> dict[str, Any]:
    """Directly certify the smallest efficient-Groves truthful utility.

    This consumes source-network coefficients only.  For each deleted report,
    the network and ``max(sum(theta), 1)`` are affine on every enumerated cell.
    """
    candidates = []
    per_agent = []
    for deleted in range(dimension):
        planes = _base_planes(dimension) + _deleted_network_break_planes(specification, dimension, deleted)
        ordered = _feasible_vertices(planes, dimension)
        rows = []
        for point in ordered:
            first_best = max(sum(point, Fraction(0)), Fraction(1))
            utility = first_best - _network_value(specification, point[:deleted] + point[deleted + 1:])
            rows.append((utility, point))
        utility, witness = min(rows)
        candidates.append((utility, deleted, witness))
        per_agent.append({
            "arrangement_planes": len(planes),
            "candidate_bases_examined": sum(1 for _ in combinations(planes, dimension)),
            "minimum_utility": f"{utility.numerator}/{utility.denominator}",
            "minimum_witness": [f"{value.numerator}/{value.denominator}" for value in witness],
        })
    minimum, agent, witness = min(candidates)
    return {
        "minimum_utility": f"{minimum.numerator}/{minimum.denominator}",
        "minimum_witness": [f"{value.numerator}/{value.denominator}" for value in witness],
        "minimum_agent": agent,
        "per_agent": per_agent,
    }


def replay_payload(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Replay certificates and bind declared source networks to each formula.

    A source network, when present, is evaluated on every vertex of the common
    refinement of its ReLU breakplanes and the compiled expression's planes.
    Since both representations are affine within each refinement cell, this
    checks semantic equality throughout the declared ordered cube without
    reusing the producer's compiler.
    """
    replay = {name: replay_entry(entry) for name, entry in payload["entries"].items()}
    for name, network in payload.get("source_networks", {}).items():
        if name not in payload["entries"]:
            raise ValueError(f"source network has no certificate entry: {name}")
        entry = payload["entries"][name]
        dimension = int(entry["dimension"])
        if len(network["output_weights"]) != dimension - 1:
            raise ValueError("source network input dimension disagrees with certificate")
        source_vertices = _feasible_vertices(
            _planes(entry["specification"], dimension) + _network_break_planes(network, dimension), dimension
        )
        for point in source_vertices:
            compiled_value = _expression_value(entry["specification"], point)
            source_value = _deleted_input_network_charge(network, point)
            if source_value != compiled_value:
                raise ValueError(
                    f"source network disagrees with compiled expression for {name} at {point}"
                )
    return replay
