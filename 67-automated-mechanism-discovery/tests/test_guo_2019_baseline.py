from fractions import Fraction
import unittest

from src.mechanism_discovery.guo_2019_baseline import audit_grid, groves_term, local_f
from src.mechanism_discovery.guo_2019_baseline_independent import replay


class Guo2019BaselineTests(unittest.TestCase):
    def test_equation_six_branches_are_exact(self):
        self.assertEqual(local_f(Fraction(1, 2), Fraction(1, 4), Fraction(1)), Fraction(17, 24))
        self.assertEqual(local_f(Fraction(0), Fraction(0), Fraction(0)), Fraction(1, 3))

    def test_three_agent_symmetrisation_is_a_distinct_published_baseline(self):
        self.assertEqual(groves_term((Fraction(0), Fraction(0))), Fraction(2, 3))
        # Equation (6) is the asymptotic construction, not Equation (2)'s
        # separately optimal three-agent rule.
        self.assertEqual(groves_term((Fraction(1), Fraction(1))), Fraction(8, 3))

    def test_exhaustive_frozen_grid_is_nondeficit(self):
        for agents in (3, 4, 5):
            audit = audit_grid(agents)
            self.assertEqual(audit.profiles_examined, 5 ** agents)
            self.assertGreaterEqual(audit.minimum_budget_slack, Fraction(0))

    def test_independent_replay_agrees_on_every_frozen_agent_count(self):
        for agents in (3, 4, 5, 6):
            self.assertEqual(replay(agents), audit_grid(agents).__dict__)


if __name__ == "__main__":
    unittest.main()
