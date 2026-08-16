#!/usr/bin/env python3
"""Emit a certificate for the continuous IJCAI-2019 Equation (2) audit."""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mechanism_discovery.guo_2019_three_agent_optimal import audit
from src.mechanism_discovery.guo_2019_three_agent_optimal_independent import replay


def encode(value):
    if isinstance(value, Fraction): return f"{value.numerator}/{value.denominator}"
    if isinstance(value, tuple): return [encode(item) for item in value]
    if isinstance(value, dict): return {key: encode(item) for key, item in value.items()}
    if hasattr(value, "__dict__"): return {key: encode(item) for key, item in value.__dict__.items()}
    return value


def main():
    result = audit()
    independent = replay()
    primary = {key: value for key, value in result.__dict__.items()
               if key not in {"distinct_from_aaai_2024", "distinctness_witness", "equation_two_charge", "aaai_2024_charge"}}
    if encode(primary) != encode(independent):
        raise SystemExit("independent Equation (2) replay disagrees")
    payload = {"source": "Guo (IJCAI 2019), Equation (2), p. 315",
               "scope": "exact continuous ordered cube; formula reproduced there and credited to Guo and Shen (2017)",
               "audit": encode(result), "independent_replay": encode(independent)}
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output = ROOT / "artifacts" / "guo_2019_three_agent_optimal_audit.json"
    output.write_text(serialized)
    print(output)
    print(hashlib.sha256(serialized.encode()).hexdigest())


if __name__ == "__main__": main()
