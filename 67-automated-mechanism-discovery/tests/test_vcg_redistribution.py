from fractions import Fraction
import unittest

from src.mechanism_discovery.vcg_redistribution import (
    expected_charge_coefficients,
    solve_by_vertex_enumeration,
    solve_counterexample_guided,
    synthesis_constraints,
    verify_solution,
)
from src.mechanism_discovery.vcg_redistribution_independent import replay
from src.mechanism_discovery.guo_2016_baseline import pivot_groves_term, vcg_revenue


class ExactRedistributionSynthesisTests(unittest.TestCase):
    def setUp(self):
        self.values = (Fraction(0), Fraction(1, 2), Fraction(1))
        self.inputs, self.constraints = synthesis_constraints(self.values, 3)
        self.uniform = expected_charge_coefficients(
            self.values, (Fraction(1, 3),) * 3, 3, self.inputs
        )

    def test_constraint_system_has_expected_anonymous_dimension(self):
        self.assertEqual(len(self.inputs), 6)
        self.assertEqual(len(self.constraints), 22)
        self.assertTrue(any(item.label == "no_deficit:(Fraction(0, 1), Fraction(0, 1), Fraction(0, 1))"
                            for item in self.constraints))

    def test_full_exact_vertex_oracle_returns_feasible_uniform_optimum(self):
        result = solve_by_vertex_enumeration(self.constraints, self.uniform)
        self.assertEqual(result.objective, Fraction(10, 3))
        self.assertEqual(verify_solution(self.constraints, self.uniform, result), [])
        self.assertGreater(result.feasible_vertices, 0)

    def test_counterexample_guided_search_matches_complete_oracle(self):
        oracle = solve_by_vertex_enumeration(self.constraints, self.uniform)
        cegis = solve_counterexample_guided(self.constraints, self.uniform)
        self.assertEqual(cegis.solution.objective, oracle.objective)
        self.assertEqual(verify_solution(self.constraints, self.uniform, cegis.solution), [])
        self.assertGreaterEqual(cegis.rounds, 1)

    def test_standalone_replay_confirms_global_optimum(self):
        result = solve_by_vertex_enumeration(self.constraints, self.uniform)
        check = replay(self.values, (Fraction(1, 3),) * 3, result.values)
        self.assertTrue(check["accepted"], check)
        self.assertEqual(check["candidate_objective"], result.objective)

    def test_prior_shift_preserves_exact_feasibility(self):
        low = expected_charge_coefficients(
            self.values, (Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)), 3, self.inputs
        )
        result = solve_by_vertex_enumeration(self.constraints, low)
        self.assertEqual(verify_solution(self.constraints, low, result), [])
        # This grid happens to have the same lexicographically selected optimum
        # under both priors.  The test guards against treating an absent shift
        # effect as evidence of a distribution-specific discovery.
        self.assertEqual(result.values, solve_by_vertex_enumeration(self.constraints, self.uniform).values)

    def test_pivot_offset_matches_normalized_vcg_revenue(self):
        profile = (Fraction(0), Fraction(1, 3), Fraction(1))
        total_offsets = sum(
            (pivot_groves_term(profile[:agent] + profile[agent + 1:]) for agent in range(3)),
            Fraction(0),
        )
        self.assertEqual(total_offsets - 2 * max(sum(profile), Fraction(1)), vcg_revenue(profile))
        self.assertEqual(pivot_groves_term((Fraction(0), Fraction(1, 3))), Fraction(2, 3))


if __name__ == "__main__":
    unittest.main()
