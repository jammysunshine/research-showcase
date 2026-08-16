"""Frozen deterministic fixtures for the rational-ReLU verifier crosscheck."""

from __future__ import annotations

import hashlib
from fractions import Fraction
from typing import Any

from .piecewise_affine import Expr, affine
from .rational_relu import compile_one_hidden_layer


def _draw(seed: int, position: int, denominator: int) -> str:
    digest = hashlib.sha256(f"{seed}:{position}".encode()).digest()
    numerator = int.from_bytes(digest[:4], "big") % 11 - 5
    return str(Fraction(numerator, denominator))


def deterministic_network(seed: int, input_dimension: int, width: int, denominator: int = 7) -> dict[str, Any]:
    """Return an exact public network from the frozen SHA-256 fixture rule."""
    position = 0

    def draw() -> str:
        nonlocal position
        value = _draw(seed, position, denominator)
        position += 1
        return value

    return {
        "output_weights": tuple(draw() for _ in range(input_dimension)),
        "output_bias": draw(),
        "hidden": tuple({
            "weights": tuple(draw() for _ in range(input_dimension)),
            "bias": draw(),
            "output_weight": draw(),
        } for _ in range(width)),
    }


def deleted_input_terms(specification: dict[str, Any], agents: int) -> tuple[Expr, ...]:
    """Compile one source term for each omitted report."""
    variables = tuple(affine(*(1 if index == coordinate else 0 for index in range(agents)))
                      for coordinate in range(agents))
    return tuple(compile_one_hidden_layer(specification, variables[:deleted] + variables[deleted + 1:])
                 for deleted in range(agents))


def deleted_input_charge(specification: dict[str, Any], agents: int) -> Expr:
    """Compile the sum of one source term after each agent report is deleted."""
    return sum(deleted_input_terms(specification, agents), Expr())
