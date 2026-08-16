"""Exact uniform output-bias repair for declared deleted-input networks."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping

from .piecewise_affine import Certificate


@dataclass(frozen=True)
class UniformRepair:
    """The minimal nonnegative per-deleted-input offset in this family."""

    agents: int
    baseline_slack: Fraction
    per_term_offset: Fraction

    @property
    def repaired_slack(self) -> Fraction:
        return self.baseline_slack + self.agents * self.per_term_offset


def synthesize_minimal_uniform_repair(certificate: Certificate, agents: int) -> UniformRepair:
    """Return the exact smallest bias offset making certified slack nonnegative.

    A bias offset is added independently to every deleted-input term, so the
    total charge at every profile changes by ``agents * offset``.  The
    certificate's minimum-slack witness proves necessity as well as
    sufficiency within this scalar repair family.
    """
    if agents < 1:
        raise ValueError("agents must be positive")
    slack = certificate.minimum_budget_slack
    return UniformRepair(agents, slack, max(Fraction(0), -slack / agents))


def add_output_bias_offset(specification: Mapping[str, Any], offset: Fraction) -> dict[str, Any]:
    """Copy a serialized rational network with an exact output-bias offset."""
    if "output_bias" not in specification:
        raise ValueError("network has no output_bias")
    return {
        "output_weights": tuple(specification["output_weights"]),
        "output_bias": str(Fraction(specification["output_bias"]) + Fraction(offset)),
        "hidden": tuple(dict(unit) for unit in specification["hidden"]),
    }
