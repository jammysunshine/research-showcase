#!/usr/bin/env python3
"""Emit a canonical JSON certificate for the exact published-rule audit."""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mechanism_discovery.published_rule_audit import (
    audit_printed_four_agent_rule,
    audit_printed_rule,
)
from src.mechanism_discovery.published_rule_audit_independent import replay


def encode(value):
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, tuple):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {key: encode(item) for key, item in value.__dict__.items()}
    return value


def main():
    three_agent = audit_printed_rule()
    four_agent = audit_printed_four_agent_rule()
    repair = Fraction(1, 20000)
    repaired = audit_printed_four_agent_rule(repair)
    independent = replay()
    if encode(four_agent) != encode(independent):
        raise SystemExit("independent four-agent replay disagrees")
    payload = {
        "source": "Guo (AAAI 2024), printed formulas on p. 9742",
        "interpretation": "all displayed decimal coefficients are exact terminating decimals; this does not recover unreported trained weights",
        "three_agent_printed_rule": encode(three_agent),
        "four_agent_printed_rule": encode(four_agent),
        "constant_repair": {
            "per_groves_term_offset": encode(repair),
            "four_agent_repaired_rule": encode(repaired),
        },
        "independent_replay": encode(independent),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    destination = ROOT / "artifacts/published_rule_audit.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(serialized)
    print(destination)
    print(hashlib.sha256(serialized.encode()).hexdigest())


if __name__ == "__main__":
    main()
