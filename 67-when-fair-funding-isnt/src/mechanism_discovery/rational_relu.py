"""Exact compiler for declared one-hidden-layer rational ReLU formulas.

This is intentionally an input-format adapter, not a general neural-network
verifier.  It accepts a public, rational network specification and lowers it
to the max/min-affine certificate language.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping, Sequence

from .piecewise_affine import Affine, Expr


def _fraction(value: str | int | Fraction) -> Fraction:
    if isinstance(value, float):
        raise ValueError("floating-point coefficients are not an exact network specification")
    return Fraction(value)


def _combine(weights: Sequence[str | int | Fraction], forms: Sequence[Affine], bias=0) -> Affine:
    if len(weights) != len(forms):
        raise ValueError("network weight/input dimension mismatch")
    dimension = len(forms[0]) - 1
    if any(len(form) != dimension + 1 for form in forms):
        raise ValueError("inconsistent affine input dimensions")
    return tuple(sum((_fraction(weight) * form[index] for weight, form in zip(weights, forms)), Fraction(0))
                 for index in range(dimension + 1 - 1)) + (
                     sum((_fraction(weight) * form[-1] for weight, form in zip(weights, forms)),
                         _fraction(bias)),
                 )


def compile_one_hidden_layer(specification: Mapping[str, object], inputs: Sequence[Affine]) -> Expr:
    """Lower a serialized rational affine--ReLU--affine network to ``Expr``.

    Required keys are ``output_weights``, ``output_bias``, and ``hidden``;
    each hidden unit has ``weights``, ``bias``, and ``output_weight``. Values
    may be integers, ``Fraction`` instances, or strings accepted by
    ``Fraction`` (including ``"p/q"``).
    """
    if not inputs:
        raise ValueError("network needs at least one input")
    try:
        output_weights = specification["output_weights"]
        output_bias = specification["output_bias"]
        hidden = specification["hidden"]
    except KeyError as error:
        raise ValueError(f"missing network key: {error.args[0]}") from error
    if not isinstance(output_weights, Sequence) or isinstance(output_weights, str):
        raise ValueError("output_weights must be a sequence")
    if not isinstance(hidden, Sequence) or isinstance(hidden, (str, bytes)):
        raise ValueError("hidden must be a sequence")
    zero = tuple(Fraction(0) for _ in range(len(inputs[0])))
    expression = Expr.from_affine(_combine(output_weights, inputs, output_bias))
    for unit in hidden:
        if not isinstance(unit, Mapping):
            raise ValueError("hidden unit must be a mapping")
        try:
            preactivation = _combine(unit["weights"], inputs, unit["bias"])
            output_weight = _fraction(unit["output_weight"])
        except KeyError as error:
            raise ValueError(f"missing hidden-unit key: {error.args[0]}") from error
        # Preserve function semantics while avoiding a spurious arrangement
        # boundary for a ReLU that contributes exactly zero to the output.
        if output_weight != 0:
            expression = expression + Expr.maximum(zero, preactivation).scale(output_weight)
    return expression
