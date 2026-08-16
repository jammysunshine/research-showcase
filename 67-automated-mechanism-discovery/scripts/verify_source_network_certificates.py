#!/usr/bin/env python3
"""Independently certify serialized public ReLU sources without compilation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mechanism_discovery.max_affine_independent import replay_deleted_input_network


def main() -> None:
    source = ROOT / "artifacts" / "max_affine_certification.json"
    payload = json.loads(source.read_text())
    sources = payload.get("source_networks", {})
    if not sources:
        raise SystemExit("certificate contains no public source networks")
    replay = {}
    for name, network in sources.items():
        entry = payload["entries"].get(name)
        if entry is None:
            raise SystemExit(f"source network has no certificate entry: {name}")
        direct = replay_deleted_input_network(network, int(entry["dimension"]))
        if direct != entry["certificate"]:
            raise SystemExit(f"direct source replay disagrees with certificate: {name}")
        replay[name] = direct
    output_payload = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "entries": replay}
    serialized = json.dumps(output_payload, indent=2, sort_keys=True) + "\n"
    output = ROOT / "artifacts" / "max_affine_source_network_certificate.json"
    output.write_text(serialized)
    print(output)
    print(hashlib.sha256(serialized.encode()).hexdigest())


if __name__ == "__main__":
    main()
