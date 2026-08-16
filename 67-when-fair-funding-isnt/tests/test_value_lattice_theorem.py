import unittest

from src.mechanism_discovery.public_project import PublicProjectSpec, verify_public_project
from src.mechanism_discovery.public_project_theorem import (
    theorem_mechanisms,
    value_lattice_frontier_count,
    value_lattice_mechanisms,
)


def rule_set(mechanisms):
    return {mechanism.allocation_by_state for mechanism in mechanisms}


class ValueLatticeTheoremTests(unittest.TestCase):
    def test_extends_the_ternary_theorem_exactly(self):
        for agents in range(1, 5):
            for cost in range(1, 2 * agents + 1):
                spec = PublicProjectSpec(agents, 2, cost)
                self.assertEqual(rule_set(value_lattice_mechanisms(spec)), rule_set(theorem_mechanisms(spec)))

    def test_three_agent_four_value_count_blocks(self):
        self.assertEqual(
            [value_lattice_frontier_count(3, 4, cost) for cost in range(1, 13)],
            [65, 65, 65, 15, 15, 15, 4, 4, 4, 1, 1, 1],
        )

    def test_constructed_rules_pass_primary_verifier(self):
        for cost in (1, 4, 7, 10):
            spec = PublicProjectSpec(3, 4, cost)
            for mechanism in value_lattice_mechanisms(spec):
                self.assertTrue(verify_public_project(mechanism)["accepted"])
