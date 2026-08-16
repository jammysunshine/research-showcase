"""Typed transcriptions of fixed public-project formulas for Phase IV."""

from fractions import Fraction

from .piecewise_affine import Expr, affine
from .rational_relu import compile_one_hidden_layer


def _sum(*expressions: Expr) -> Expr:
    result = Expr()
    for expression in expressions:
        result = result + expression
    return result


def guo_2019_equation_two_charge() -> Expr:
    """Total charge of Guo (IJCAI 2019), Eq. (2), on ordered 3-agent reports."""
    a, b, c = affine(1, 0, 0), affine(0, 1, 0), affine(0, 0, 1)
    const = lambda value: affine(0, 0, 0, constant=value)

    def h(left, right):
        pair = tuple(x + y for x, y in zip(left, right))
        return _sum(Expr.maximum(pair, const(Fraction(2, 3))),
                    Expr.maximum(pair, const(1)).scale(Fraction(1, 2)),
                    Expr.maximum(left, right, const(Fraction(2, 3))).scale(Fraction(-1, 2)),
                    Expr.from_affine(const(Fraction(-1, 6))))
    return _sum(h(b, c), h(a, c), h(a, b))


def guo_2016_equation_three_charge() -> Expr:
    """Groves-term equivalent of Guo (PRIMA 2016), Eq. (3), for three agents.

    Equation (3) specifies total redistribution rather than Groves charges.
    If ``R`` is that total redistribution and ``E`` is the sum of the three
    VCG externality terms, then ``E - R`` is the charge representation needed
    by the certificate: subtracting ``2 * first_best`` recovers exactly the
    source convention's budget slack.
    """
    a, b, c = affine(1, 0, 0), affine(0, 1, 0), affine(0, 0, 1)
    const = lambda value: affine(0, 0, 0, constant=value)
    threshold = Fraction(2, 3)
    correction = (Fraction(1, 2) + Fraction(1, 6) + Fraction(64, 81)) / 3

    def externality(left, right):
        return Expr.maximum(tuple(x + y for x, y in zip(left, right)), const(threshold))

    def redistribution(left, right):
        pair = tuple(x + y for x, y in zip(left, right))
        raw = (Expr.maximum(left, const(threshold))
               + Expr.maximum(right, const(threshold))
               + Expr.maximum(pair, const(1)).scale(-1)).scale(Fraction(1, 2))
        return raw + Expr.from_affine(const(-correction))

    pairs = ((b, c), (a, c), (a, b))
    return _sum(*(externality(*pair) + redistribution(*pair).scale(-1) for pair in pairs))


def guo_2024_three_agent_charge() -> Expr:
    """Total charge of Guo (AAAI 2024)'s printed 3-agent formula."""
    a, b, c = affine(1, 0, 0), affine(0, 1, 0), affine(0, 0, 1)
    zero = affine(0, 0, 0)
    const = lambda value: affine(0, 0, 0, constant=value)

    def h(left, right):
        pair = tuple(x + y for x, y in zip(left, right))
        skew = tuple(5 * x + 3 * y for x, y in zip(left, right))
        return _sum(Expr.maximum(zero, tuple(x - y for x, y in zip(pair, const(1)))).scale(Fraction(2, 3)),
                    Expr.maximum(zero, tuple(x - y for x, y in zip(skew, const(2)))).scale(Fraction(1, 6)),
                    Expr.from_affine(const(Fraction(2, 3))))
    return _sum(h(b, c), h(a, c), h(a, b))


def guo_2024_four_agent_charge(charge_offset: Fraction = Fraction(0)) -> Expr:
    """Total charge of Guo (AAAI 2024)'s printed 4-agent decimal formula.

    The input is ordered.  Consequently, deleting any coordinate leaves the
    remaining coordinates ordered, so this is a direct typed transcription of
    the public formula rather than a reconstruction of unpublished weights.
    ``charge_offset`` is the per-term constant in the paper's repair family.
    """
    variables = tuple(affine(*(1 if index == coordinate else 0 for index in range(4)))
                      for coordinate in range(4))

    def h(first, second, third):
        return compile_one_hidden_layer(guo_2024_four_agent_network_spec(charge_offset),
                                        (first, second, third))

    return _sum(*(h(*(variables[:index] + variables[index + 1:])) for index in range(4)))


def guo_2024_four_agent_network_spec(charge_offset: Fraction = Fraction(0)) -> dict:
    """Serialized coefficient source for the printed 4-agent shallow ReLU rule."""
    return {
        "output_weights": ("9197/10000", "6558/10000", "6646/10000"),
        "output_bias": str(Fraction(2218, 10000) + charge_offset),
        "hidden": (
            {"weights": ("-7220/10000", "-5927/10000", "-5925/10000"),
             "bias": "5926/10000", "output_weight": "1"},
            {"weights": ("-4485/10000", "-5939/10000", "-3858/10000"),
             "bias": "3856/10000", "output_weight": "1"},
            {"weights": ("1925/10000", "4570/10000", "4436/10000"),
             "bias": "-2218/10000", "output_weight": "1"},
            {"weights": ("-4820/10000", "-3097/10000", "-915/10000"),
             "bias": "3667/10000", "output_weight": "-1"},
        ),
    }
