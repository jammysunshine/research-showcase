from fractions import Fraction
import unittest

from src.mechanism_discovery.guo_2019_three_agent_optimal import arrangement_vertices, audit, h
from src.mechanism_discovery.guo_2019_three_agent_optimal_independent import replay


class Guo2019ThreeAgentOptimalTests(unittest.TestCase):
    def test_equation_two_is_transcribed_exactly(self):
        self.assertEqual(h(Fraction(0), Fraction(0)), Fraction(2, 3))
        self.assertEqual(h(Fraction(1), Fraction(1)), Fraction(7, 3))

    def test_continuous_arrangement_certificate_matches_independent_replay(self):
        result = audit()
        self.assertEqual(len(arrangement_vertices()), 23)
        self.assertEqual(result.vertices_examined, replay()["vertices_examined"])
        self.assertEqual(result.minimum_charge_ratio, replay()["minimum_charge_ratio"])
        self.assertEqual(result.maximum_charge_ratio, replay()["maximum_charge_ratio"])

    def test_known_optimum_has_the_claimed_two_thirds_efficiency(self):
        result = audit()
        self.assertEqual(result.minimum_charge_ratio, Fraction(2))
        self.assertEqual(result.maximum_charge_ratio, Fraction(7, 3))
        self.assertEqual(result.worst_case_efficiency, Fraction(2, 3))

    def test_equation_two_is_not_the_later_printed_neural_formula(self):
        result = audit()
        self.assertTrue(result.distinct_from_aaai_2024)
        self.assertNotEqual(result.equation_two_charge, result.aaai_2024_charge)


if __name__ == "__main__": unittest.main()
