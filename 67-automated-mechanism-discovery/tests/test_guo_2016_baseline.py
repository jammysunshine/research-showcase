from fractions import Fraction
import unittest

from src.mechanism_discovery.guo_2016_baseline import audit_grid, raw_redistribution, upper_bound
from src.mechanism_discovery.guo_2016_baseline_independent import replay


class Guo2016BaselineTests(unittest.TestCase):
    def test_equation_three_and_proposition_one_are_exact(self):
        self.assertEqual(raw_redistribution((Fraction(0), Fraction(0))), Fraction(1, 6))
        self.assertEqual(upper_bound(3), Fraction(118, 81))

    def test_frozen_grid_is_nondeficit_and_independently_replayed(self):
        for agents in (3, 4, 5, 6):
            audit = audit_grid(agents)
            self.assertEqual(audit.profiles_examined, 5 ** agents)
            self.assertGreaterEqual(audit.minimum_budget_slack, 0)
            self.assertEqual(replay(agents), audit.__dict__)


if __name__ == "__main__":
    unittest.main()
