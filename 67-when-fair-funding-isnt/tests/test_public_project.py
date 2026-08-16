import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mechanism_discovery.public_project import (  # noqa: E402
    PublicProjectSpec,
    enumerate_anonymous_monotone,
    frontier,
    sum_threshold_mechanism,
    verify_public_project,
)
from mechanism_discovery.public_project_independent import check  # noqa: E402
from mechanism_discovery.public_project_theorem import theorem_frontier_count, theorem_mechanisms  # noqa: E402


class PublicProjectTests(unittest.TestCase):
    def test_all_agent_theorem_construction(self):
        for n in range(1, 9):
            for cost in range(1, 2 * n + 1):
                spec = PublicProjectSpec(n_agents=n, max_value=2, cost=cost)
                mechanisms = theorem_mechanisms(spec)
                self.assertEqual(len(mechanisms), theorem_frontier_count(n, cost))
                if n <= 5:
                    self.assertTrue(all(verify_public_project(m, check_anonymity=False)["accepted"] for m in mechanisms))
            self.assertEqual(theorem_frontier_count(n, 2 * n + 1), 0)
            self.assertEqual(theorem_mechanisms(PublicProjectSpec(n_agents=n, max_value=2, cost=2 * n + 1)), ())

    def test_exhaustive_monotone_count_is_finite_and_reproducible(self):
        spec = PublicProjectSpec(3, 2, 3)
        self.assertEqual(len(list(enumerate_anonymous_monotone(spec))), 16)
        self.assertEqual(len(frontier(spec)), 4)
        self.assertEqual(len(frontier(spec, max_coalition_size=1)), 4)
        self.assertEqual(len(frontier(spec, max_coalition_size=2)), 2)

    def test_antichain_enumerator_scales_exactly_through_five_agents(self):
        counts = [len(list(enumerate_anonymous_monotone(PublicProjectSpec(n, 2, n)))) for n in range(3, 6)]
        self.assertEqual(counts, [16, 32, 64])
        self.assertEqual(len(frontier(PublicProjectSpec(4, 2, 4))), 5)

    def test_antichain_enumerator_reaches_six_agents_exactly(self):
        spec = PublicProjectSpec(6, 2, 6)
        self.assertEqual(len(spec.states), 28)
        self.assertEqual(len(list(enumerate_anonymous_monotone(spec))), 128)

    def test_value_lattice_extension_is_exact(self):
        spec = PublicProjectSpec(3, 3, 3)
        self.assertEqual(len(spec.states), 20)
        self.assertEqual(len(list(enumerate_anonymous_monotone(spec))), 66)
        counts = [len(frontier(PublicProjectSpec(3, 3, cost))) for cost in range(1, 10)]
        self.assertEqual(counts, [15, 15, 15, 4, 4, 4, 1, 1, 1])

    def test_efficient_rule_has_a_budget_counterexample(self):
        spec = PublicProjectSpec(3, 2, 3)
        report = verify_public_project(sum_threshold_mechanism(spec, 3))
        self.assertTrue(report["dsic"])
        self.assertTrue(report["ex_post_ir"])
        self.assertFalse(report["weak_budget_balance"])
        self.assertTrue(any(w["property"] == "weak_budget_balance" for w in report["witnesses"]))

    def test_serialized_accepted_rule_passes_independent_checker(self):
        row = frontier(PublicProjectSpec(3, 2, 3))[0]["mechanism"]
        serialized = {
            "name": row.name,
            "n_agents": row.spec.n_agents,
            "max_value": row.spec.max_value,
            "cost": row.spec.cost,
            "allocation_by_state": [[list(s), q] for s, q in row.allocation_by_state],
        }
        self.assertTrue(check(serialized)["accepted"])

    def test_sum_threshold_is_dsic_under_heldout_values(self):
        spec = PublicProjectSpec(3, 2, 3)
        report = verify_public_project(sum_threshold_mechanism(spec, 4))
        self.assertTrue(report["dsic"])
        self.assertTrue(report["ex_post_ir"])

    def test_coalitional_dsic_shrinks_frontier_at_cost_three(self):
        spec = PublicProjectSpec(3, 2, 3)
        base = frontier(spec)
        coalitional = frontier(spec, max_coalition_size=2)
        self.assertEqual(len(base), 4)
        self.assertEqual(len(coalitional), 2)
        names = {row["mechanism"].name for row in coalitional}
        self.assertEqual(names, {"anonymous_monotone_mask_960", "anonymous_monotone_mask_512"})

    def test_independent_checker_rejects_noncoalition_robust_frontier_row(self):
        spec = PublicProjectSpec(3, 2, 3)
        base = frontier(spec)
        fragile = next(row["mechanism"] for row in base if row["mechanism"].name in {"anonymous_monotone_mask_896", "anonymous_monotone_mask_768"})
        fragile_report = check({
            "name": fragile.name,
            "n_agents": fragile.spec.n_agents,
            "max_value": fragile.spec.max_value,
            "cost": fragile.spec.cost,
            "allocation_by_state": [[list(s), q] for s, q in fragile.allocation_by_state],
        }, max_coalition_size=2)
        # The first serialized rule in the unbounded frontier is not robust to
        # coalition deviations for the bounded model.
        self.assertFalse(fragile_report["coalitional_dsic"] or fragile_report["accepted"])

    def test_coalitional_and_universal_independent_checker_align(self):
        spec = PublicProjectSpec(3, 2, 3)
        for row in frontier(spec, max_coalition_size=2):
            mechanism = row["mechanism"]
            serialised = {
                "name": mechanism.name,
                "n_agents": mechanism.spec.n_agents,
                "max_value": mechanism.spec.max_value,
                "cost": mechanism.spec.cost,
                "allocation_by_state": [[list(s), q] for s, q in mechanism.allocation_by_state],
            }
            self.assertTrue(check(serialised, max_coalition_size=2)["accepted"])


if __name__ == "__main__":
    unittest.main()
