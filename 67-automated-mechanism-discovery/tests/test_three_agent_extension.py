import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechanism_discovery.three_agent_extension import (  # noqa: E402
    PROFILES, anonymous_and, anonymous_or, enumerate_anonymous_budget_balanced,
    majority, metrics, verify,
)
from mechanism_discovery.three_agent_independent import (  # noqa: E402
    candidate_tables, check_table, independent_frontier,
)


class ThreeAgentExtensionTests(unittest.TestCase):
    def test_frozen_candidate_count(self):
        self.assertEqual(len(PROFILES), 8)
        self.assertEqual(len(enumerate_anonymous_budget_balanced()), 144)
        self.assertEqual(len(candidate_tables()), 144)

    def test_independent_frontier_matches_primary(self):
        primary = {
            tuple((outcome.choice, *outcome.payments) for outcome in mechanism.outcomes)
            for mechanism in enumerate_anonymous_budget_balanced() if verify(mechanism).accepted
        }
        independent = set(independent_frontier())
        self.assertEqual(primary, independent)
        self.assertEqual(len(primary), 5)

    def test_canonical_rules_and_zero_transfers(self):
        for mechanism in (anonymous_and(), anonymous_or(), majority()):
            result = verify(mechanism)
            self.assertTrue(result.accepted, result)
            self.assertEqual(metrics(mechanism)["expected_revenue"], 0.0)
        self.assertTrue(verify(majority()).neutrality)

    def test_majority_is_pointwise_welfare_maximal(self):
        baseline = metrics(majority())["expected_allocative_welfare"]
        for mechanism in enumerate_anonymous_budget_balanced():
            self.assertLessEqual(metrics(mechanism)["expected_allocative_welfare"], baseline)

    def test_nonzero_transfer_candidates_do_not_pass(self):
        for mechanism in enumerate_anonymous_budget_balanced():
            nonzero = any(payment for outcome in mechanism.outcomes for payment in outcome.payments)
            if nonzero:
                self.assertFalse(verify(mechanism).accepted)

    def test_frozen_config_records_boundary(self):
        config = json.loads((ROOT / "configs/experiment_67_three_agent.json").read_text())
        self.assertEqual(config["agents"], 3)
        self.assertEqual(config["payment_grid"], [-2, -1, 0, 1, 2])
        self.assertEqual(config["exhaustive_candidates"], 144)

    def test_independent_checker_rejects_a_nonanonymous_table(self):
        table = next(iter(candidate_tables()))
        mutated = list(table)
        mutated[1] = (mutated[1][0], mutated[1][1] + 1, mutated[1][2] - 1, mutated[1][3])
        self.assertFalse(check_table(tuple(mutated))["accepted"])


if __name__ == "__main__":
    unittest.main()
