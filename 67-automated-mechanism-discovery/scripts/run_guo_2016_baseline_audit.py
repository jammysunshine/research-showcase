#!/usr/bin/env python3
"""Write the exact PRIMA-2016 Equation (3) grid-audit certificate."""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mechanism_discovery.guo_2016_baseline import audit_grid
from src.mechanism_discovery.guo_2016_baseline_independent import replay


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


def main() -> None:
    audits = [audit_grid(agents) for agents in (3, 4, 5, 6)]
    if [audit.__dict__ for audit in audits] != [replay(agents) for agents in (3, 4, 5, 6)]:
        raise SystemExit("independent PRIMA-2016 replay disagrees")
    payload = {
        "source": "Guo (PRIMA 2016), Equation (3), Proposition 1, and Theorem 1",
        "scope": "exhaustive rational grid {0,1/4,1/2,3/4,1}^n; not a continuous proof",
        "audits": [encode(audit) for audit in audits],
        "independent_replay": [encode(replay(agents)) for agents in (3, 4, 5, 6)],
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output = ROOT / "artifacts" / "guo_2016_grid_audit.json"
    output.write_text(serialized)
    print(output)
    print(hashlib.sha256(serialized.encode()).hexdigest())


if __name__ == "__main__":
    main()
