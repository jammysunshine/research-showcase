#!/usr/bin/env python3
"""Independently replay the generic max-affine certificate artifact."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mechanism_discovery.max_affine_independent import replay_payload


def main():
    source = ROOT / "artifacts" / "max_affine_certification.json"
    payload = json.loads(source.read_text())
    replay = replay_payload(payload)
    expected = {name: entry["certificate"] for name, entry in payload["entries"].items()}
    if replay != expected:
        raise SystemExit("independent replay does not match the serialized certificate")
    output_payload = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "entries": replay}
    serialized = json.dumps(output_payload, indent=2, sort_keys=True) + "\n"
    output = ROOT / "artifacts" / "max_affine_independent_certificate.json"
    output.write_text(serialized)
    print(output)
    print(hashlib.sha256(serialized.encode()).hexdigest())


if __name__ == "__main__":
    main()
