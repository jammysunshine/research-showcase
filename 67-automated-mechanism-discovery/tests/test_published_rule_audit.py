from fractions import Fraction
import unittest

from src.mechanism_discovery.published_rule_audit import (
    arrangement_vertices,
    arrangement_vertices_four,
    audit_printed_rule,
    audit_printed_four_agent_rule,
    first_best,
    published_h,
    total_charge,
)
from src.mechanism_discovery.published_rule_audit_independent import replay


class PublishedRuleAuditTests(unittest.TestCase):
    def test_printed_formula_is_evaluated_as_exact_terminating_decimals(self):
        self.assertEqual(published_h(Fraction(0), Fraction(0)), Fraction(2, 3))
        self.assertEqual(published_h(Fraction(1), Fraction(1)), Fraction(7, 3))

    def test_arrangement_is_nontrivial_and_bounded(self):
        vertices = arrangement_vertices()
        self.assertEqual(len(vertices), 19)
        self.assertIn((Fraction(0), Fraction(0), Fraction(0)), vertices)
        self.assertIn((Fraction(1), Fraction(1), Fraction(1)), vertices)

    def test_exact_continuous_audit_reproduces_two_thirds(self):
        result = audit_printed_rule()
        self.assertEqual(result.minimum_charge_ratio, Fraction(2))
        self.assertEqual(result.minimum_witness, (Fraction(0), Fraction(0), Fraction(0)))
        self.assertEqual(result.maximum_charge_ratio, Fraction(7, 3))
        self.assertEqual(result.maximum_witness, (Fraction(1), Fraction(1), Fraction(1)))
        self.assertEqual(result.worst_case_efficiency, Fraction(2, 3))

    def test_witnesses_satisfy_both_sides_of_the_published_constraint(self):
        result = audit_printed_rule()
        for witness in (result.minimum_witness, result.maximum_witness):
            charge = total_charge(witness)
            s_value = first_best(witness)
            self.assertGreaterEqual(charge, 2 * s_value)
            self.assertLessEqual(charge, (3 - result.worst_case_efficiency) * s_value)

    def test_printed_four_agent_decimals_have_an_exact_deficit_witness(self):
        result = audit_printed_four_agent_rule()
        self.assertEqual(len(arrangement_vertices_four()), 116)
        self.assertEqual(result.minimum_deficit, -Fraction(1, 5000))
        self.assertEqual(result.minimum_witness,
                         (Fraction(0), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)))

    def test_constant_repair_eliminates_the_printed_rule_deficit(self):
        repaired = audit_printed_four_agent_rule(Fraction(1, 20000))
        self.assertGreaterEqual(repaired.minimum_deficit, Fraction(0))

    def test_constant_repair_is_minimal_within_the_uniform_offset_family(self):
        # The witnessed shortfall is 1/5000 and four terms receive the offset.
        almost_repaired = audit_printed_four_agent_rule(Fraction(1, 20001))
        self.assertLess(almost_repaired.minimum_deficit, Fraction(0))
        self.assertEqual(audit_printed_four_agent_rule(Fraction(1, 20000)).minimum_deficit,
                         Fraction(0))

    def test_standalone_replay_agrees_on_the_exact_printed_rule_witness(self):
        primary = audit_printed_four_agent_rule()
        independent = replay()
        self.assertEqual(independent["minimum_deficit"], primary.minimum_deficit)
        self.assertEqual(independent["minimum_witness"], primary.minimum_witness)
        self.assertEqual(independent["worst_case_efficiency"], primary.worst_case_efficiency)


if __name__ == "__main__":
    unittest.main()
