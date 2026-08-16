"""Exact, closed-form characterization of coalition-manipulability.

The primary verifier (`verify_public_project` in `public_project.py`) decides
coalition-cap-K DSIC by brute-force enumeration: for every profile, every
coalition, and every joint deviation, check whether the coalition's summed
utility strictly increases. That is exponential in the number of report
levels and requires bounding both `n` and the coalition-size cap.

This script replaces that search with an exact formula for the sum-threshold/
critical-value mechanism, derived from two facts proven by convexity:

1. For a coalition T of size k with true report levels ranging over
   [0, max_value], and fixed outsider truthful-value sum S_O, the deviation
   that minimizes T's total critical-value payment while still building the
   project is "every member of T reports max_value" (the all-max
   construction used in Section 4.11 of the paper). Its payment is
       min_payment(k, S_O, cost) = k * max(0, (cost - S_O) - (k-1) * max_value)
   whenever (cost - S_O) <= k * max_value (else T cannot build the project by
   any deviation). This is exact, not a bound: max(0, x - D) is convex, so
   for a fixed report sum the minimum of a sum of convex terms is attained at
   equal split, and among equal splits the payment-minimizing split is the
   one with the largest feasible sum, i.e. every report at max_value.

2. For a truthful profile with coalition value-sum V_T and outsider sum S_O
   such that the project already builds truthfully (V_T + S_O >= cost), the
   coalition's truthful total payment is minimized (i.e. hardest to beat,
   which is what we search for) by an extremal ("bang-bang") distribution of
   V_T among the k members: push as many members to max_value as possible,
   one member to the remainder, the rest to 0. This is the same convexity
   argument in the opposite direction (maximizing a convex sum with a fixed
   total is achieved by extreme/majorizing points).

Given these two closed-form pieces, whether *any* profile makes coalition
size k profitable reduces to a small integer sweep over (V_T, S_O) instead of
enumerating profiles and deviations, and works for arbitrary n, max_value,
cost -- not only domains small enough to brute-force. This script computes
that existence check and cross-checks it against every row of the existing
brute-force baseline-audit artifact, aiming for an exact match, not merely a
one-directional (no-false-positive) match as in
`verify_public_project_coalition_lemma.py`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_AUDIT = REPO_ROOT / "artifacts" / "public_project_coalition_baseline_audit.json"
OUTPUT = REPO_ROOT / "artifacts" / "public_project_coalition_characterization_certificate.json"


def _min_payment(k: int, s_o: int, m: int, cost: int) -> int | None:
    r0 = cost - s_o
    if r0 <= 0:
        return 0
    if r0 > k * m:
        return None
    return k * max(0, r0 - (k - 1) * m)


def _max_sum_p(v_t: int, s_o: int, k: int, m: int, cost: int) -> int:
    e = v_t + s_o - cost
    if e < 0:
        return 0
    num_m = min(v_t // m, k)
    remainder = v_t - num_m * m
    total = num_m * max(0, m - e)
    if remainder > 0 and num_m < k:
        total += max(0, remainder - e)
    return total


def _coalition_size_manipulable(k: int, n: int, m: int, cost: int) -> bool:
    for s_o in range(0, (n - k) * m + 1):
        mp = _min_payment(k, s_o, m, cost)
        if mp is None:
            continue  # T can never build the project via any deviation at this s_o
        # Branch 1: truthful profile does not build (V_T + s_o < cost).
        v_t_max = min(k * m, cost - s_o - 1)
        if v_t_max >= 0 and v_t_max - mp > 0:
            return True
        # Branch 2: truthful profile already builds (V_T + s_o >= cost).
        for v_t in range(max(0, cost - s_o), k * m + 1):
            max_sum_p = _max_sum_p(v_t, s_o, k, m, cost)
            if max_sum_p - mp > 0:
                return True
    return False


def _min_failing_coalition_size(n: int, m: int, cost: int) -> int | None:
    for k in range(2, n + 1):
        if _coalition_size_manipulable(k, n, m, cost):
            return k
    return None


def _rows_from_domain(domain: dict[str, Any]) -> list[dict[str, Any]]:
    if "cost_rows" in domain:
        return domain["cost_rows"]
    return [row for block in domain["by_n"] for row in block["cost_rows"]]


def main() -> None:
    data = json.loads(BASELINE_AUDIT.read_text())
    checked = 0
    exact_matches = 0
    mismatches: list[dict[str, Any]] = []
    for domain in data["domains"]:
        for row in _rows_from_domain(domain):
            n = row["n_agents"]
            m = row["max_value"]
            cost = row["cost"]
            searched = row["min_failing_coalition_size"]
            predicted = _min_failing_coalition_size(n, m, cost)
            checked += 1
            if predicted == searched:
                exact_matches += 1
            else:
                mismatches.append(
                    {
                        "domain": domain["domain"],
                        "n_agents": n,
                        "max_value": m,
                        "cost": cost,
                        "searched_min_failing_coalition_size": searched,
                        "predicted_min_failing_coalition_size": predicted,
                    }
                )

    payload = {
        "characterization": (
            "min_payment(k, S_O, cost) = k * max(0, (cost - S_O) - (k-1) * max_value) "
            "is the exact minimum achievable coalition payment; existence of a "
            "profitable size-k coalition deviation is decided by a bounded sweep "
            "over (V_T, S_O) rather than full profile/report enumeration."
        ),
        "rows_checked": checked,
        "exact_matches": exact_matches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    digest_source = json.dumps(
        {"rows_checked": checked, "exact_matches": exact_matches, "mismatch_count": len(mismatches)},
        sort_keys=True,
    ).encode("utf-8")
    payload["digest"] = hashlib.sha256(digest_source).hexdigest()

    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"rows_checked={checked} exact_matches={exact_matches} mismatches={len(mismatches)}")


if __name__ == "__main__":
    main()
