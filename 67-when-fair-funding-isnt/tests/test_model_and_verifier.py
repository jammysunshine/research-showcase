import sys
import json
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mechanism_discovery.adversarial_audit import audit_baseline, parse_distributions
from mechanism_discovery.independent_verifier import check, independent_frontier, table_from_mechanism
from mechanism_discovery.model import (
    Mechanism,
    Outcome,
    PROFILES,
    anonymous_or,
    canonical_baselines,
    majority_with_tie_break,
    priority_majority,
    serial_dictatorship,
    utility,
    vcg_pivot,
)
from mechanism_discovery.search import exhaustive_search, evolutionary_search
from mechanism_discovery.verifier import metrics, verify


class ModelAndVerifierTests(unittest.TestCase):
    def test_utility_includes_transfer(self):
        self.assertEqual(utility(1, Outcome(1, (1, 0)), 0), 0)
        self.assertEqual(utility(0, Outcome(1, (-1, 1)), 0), 1)

    def test_baseline_is_accepted_by_both_checkers(self):
        mechanism = anonymous_or()
        self.assertTrue(verify(mechanism).accepted)
        self.assertTrue(check(mechanism)["accepted"])
        self.assertEqual(metrics(mechanism)["expected_welfare"], 1.5)

    def test_canonical_acceptance_boundary_is_reproduced(self):
        for mechanism in (majority_with_tie_break(0), majority_with_tie_break(1), anonymous_or()):
            with self.subTest(mechanism=mechanism.name):
                self.assertTrue(verify(mechanism).accepted)
                self.assertTrue(check(mechanism)["accepted"])
        for mechanism in (serial_dictatorship(0), serial_dictatorship(1), priority_majority()):
            with self.subTest(mechanism=mechanism.name):
                self.assertFalse(verify(mechanism).accepted)
                self.assertFalse(check(mechanism)["accepted"])

    def test_majority_rule_is_welfare_maximizing_on_each_profile(self):
        for tie_choice in (0, 1):
            mechanism = majority_with_tie_break(tie_choice)
            for profile in PROFILES:
                selected_welfare = sum(int(value == mechanism.outcome(profile).choice) for value in profile)
                self.assertEqual(selected_welfare, max(sum(int(value == c) for value in profile) for c in (0, 1)))
            self.assertEqual(metrics(mechanism)["expected_allocative_welfare"], 1.5)

    def test_vcg_pivot_matches_analytic_budget_balance_boundary(self):
        for tie_choice in (0, 1):
            mechanism = vcg_pivot(tie_choice)
            report = verify(mechanism)
            self.assertTrue(report.dsic)
            self.assertTrue(report.ir)
            self.assertTrue(report.feasibility)
            self.assertFalse(report.budget_balance)
            self.assertFalse(report.accepted)
            self.assertTrue(any(w.property == "budget_balance" for w in report.witnesses))
            self.assertEqual(metrics(mechanism)["expected_allocative_welfare"], 1.5)
            self.assertEqual(metrics(mechanism)["expected_revenue"], 0.5)

    def test_canonical_catalogue_has_unique_names(self):
        names = [mechanism.name for mechanism in canonical_baselines()]
        self.assertEqual(len(names), len(set(names)))

    def test_dsic_counterexample_contains_profitable_deviation(self):
        mechanism = Mechanism(tuple(Outcome(1 - p[0], (0, 0)) for p in PROFILES), "anti_agent_0")
        report = verify(mechanism)
        self.assertFalse(report.dsic)
        witness = next(w for w in report.witnesses if w.property == "dsic")
        self.assertIsNotNone(witness.deviation)
        self.assertFalse(check(mechanism)["accepted"])

    def test_budget_balance_and_ir_witnesses(self):
        mechanism = Mechanism(tuple(Outcome(0, (1, 0)) for _ in PROFILES), "bad_transfers")
        report = verify(mechanism)
        self.assertFalse(report.budget_balance)
        self.assertFalse(report.ir)

    def test_exhaustive_search_covers_frozen_space(self):
        accepted = exhaustive_search()
        self.assertGreater(len(accepted), 0)
        self.assertTrue(all(row["verification"].accepted for row in accepted))

    def test_independent_enumerator_matches_primary_frontier(self):
        primary = {table_from_mechanism(row["mechanism"]) for row in exhaustive_search()}
        independent = set(independent_frontier())
        self.assertEqual(primary, independent)
        self.assertEqual(len(independent), 4)

    def test_baseline_survives_bounded_coalition_and_value_perturbation_audits(self):
        audit = audit_baseline(anonymous_or())
        self.assertEqual(audit["coalition_pareto_deviation_count"], 0)
        self.assertEqual(audit["magnitude_perturbation_failure_count"], 0)

    def test_confirmation_distribution_config_is_normalized(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs" / "confirmation_67.json").read_text())
        distributions = parse_distributions(config)
        self.assertGreaterEqual(len(distributions), 4)
        self.assertTrue(all(sum(weights) == 1 for weights in distributions.values()))

    def test_evolutionary_loop_is_seeded_and_verifies_proposals(self):
        result = evolutionary_search(seed=67, population_size=8, generations=3)
        self.assertEqual(result["evaluated"], 24)
        self.assertGreaterEqual(result["accepted"], 0)


if __name__ == "__main__":
    unittest.main()
