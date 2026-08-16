from fractions import Fraction
from copy import deepcopy
import json
from pathlib import Path
import unittest

from src.mechanism_discovery.guo_2019_three_agent_optimal import audit as equation_two_audit
from src.mechanism_discovery.guo_2016_baseline import (
    efficiency_ratio as prima_efficiency_ratio,
    total_corrected_redistribution,
    vcg_revenue,
)
from src.mechanism_discovery.max_affine_corpus import (
    guo_2016_equation_three_charge,
    guo_2019_equation_two_charge,
    guo_2024_three_agent_charge,
    guo_2024_four_agent_charge,
    guo_2024_four_agent_network_spec,
)
from src.mechanism_discovery.piecewise_affine import certify_minimum_groves_utility, certify_ordered_public_project_charge
from src.mechanism_discovery.rational_relu import compile_one_hidden_layer
from src.mechanism_discovery.max_affine_independent import replay_deleted_input_network, replay_deleted_input_network_utility_margin, replay_payload
from src.mechanism_discovery.relu_benchmark import deleted_input_charge, deleted_input_terms, deterministic_network
from src.mechanism_discovery.published_rule_audit import audit_printed_four_agent_rule, audit_printed_rule
from src.mechanism_discovery.uniform_repair import add_output_bias_offset, synthesize_minimal_uniform_repair


class PiecewiseAffineCertificateTest(unittest.TestCase):
    def test_rational_relu_compiler_accepts_serialized_coefficients(self):
        a, b = (Fraction(2), Fraction(3))
        from src.mechanism_discovery.piecewise_affine import affine
        expression = compile_one_hidden_layer({
            "output_weights": ("1/2", "-1/3"), "output_bias": "1/7",
            "hidden": ({"weights": ("2", "-1"), "bias": "-1", "output_weight": "3/5"},),
        }, (affine(1, 0), affine(0, 1)))
        self.assertEqual(expression.evaluate((a, b)), Fraction(1, 7) + Fraction(1, 2) * a - b / 3)

    def test_rational_relu_compiler_rejects_floating_point_source(self):
        from src.mechanism_discovery.piecewise_affine import affine
        with self.assertRaises(ValueError):
            compile_one_hidden_layer({"output_weights": (0.5,), "output_bias": 0,
                                      "hidden": ()}, (affine(1),))

    def test_rational_relu_compiler_elides_zero_output_activation_boundary(self):
        from src.mechanism_discovery.piecewise_affine import affine
        source = {
            "output_weights": ("0",), "output_bias": "0",
            "hidden": ({"weights": ("1",), "bias": "-1/2", "output_weight": "0"},),
        }
        expression = compile_one_hidden_layer(source, (affine(1),))
        self.assertEqual(expression.evaluate((Fraction(3, 4),)), Fraction(0))
        self.assertEqual(expression.break_planes(), ())

    def test_published_relu_source_specification_compiles_to_four_agent_rule(self):
        compiled = guo_2024_four_agent_charge()
        specification = guo_2024_four_agent_network_spec()
        self.assertEqual(specification["output_bias"], "1109/5000")
        self.assertEqual(certify_ordered_public_project_charge(compiled, 4).minimum_budget_slack,
                         Fraction(-1, 5000))

    def test_reproduces_prima_2016_source_convention_at_every_certificate_vertex(self):
        generic = certify_ordered_public_project_charge(guo_2016_equation_three_charge(), 3)
        source_rows = [
            (vcg_revenue(point) - total_corrected_redistribution(point), prima_efficiency_ratio(point), point)
            for point in generic.vertices
        ]
        self.assertEqual(generic.minimum_budget_slack, min(source_rows)[0])
        self.assertEqual(generic.worst_case_efficiency, min(source_rows, key=lambda row: row[1])[1])

    def test_reproduces_known_three_agent_optimum(self):
        generic = certify_ordered_public_project_charge(guo_2019_equation_two_charge(), 3)
        known = equation_two_audit()
        self.assertEqual(len(generic.vertices), known.vertices_examined)
        self.assertEqual(generic.minimum_charge_ratio, known.minimum_charge_ratio)
        self.assertEqual(generic.maximum_charge_ratio, known.maximum_charge_ratio)
        self.assertEqual(generic.worst_case_efficiency, known.worst_case_efficiency)

    def test_reproduces_printed_aaai_rule(self):
        generic = certify_ordered_public_project_charge(guo_2024_three_agent_charge(), 3)
        known = audit_printed_rule()
        self.assertEqual(len(generic.vertices), known.vertices_examined)
        self.assertEqual(generic.minimum_charge_ratio, known.minimum_charge_ratio)
        self.assertEqual(generic.maximum_charge_ratio, known.maximum_charge_ratio)
        self.assertEqual(generic.worst_case_efficiency, known.worst_case_efficiency)

    def test_reproduces_four_agent_decimal_rule_and_repair(self):
        for offset in (Fraction(0), Fraction(1, 20000)):
            generic = certify_ordered_public_project_charge(guo_2024_four_agent_charge(offset), 4)
            known = audit_printed_four_agent_rule(offset)
            self.assertEqual(len(generic.vertices), known.vertices_examined)
            self.assertEqual(generic.minimum_charge_ratio, known.minimum_charge_ratio)
            self.assertEqual(generic.minimum_budget_slack, known.minimum_deficit)
            self.assertEqual(generic.maximum_charge_ratio, known.maximum_charge_ratio)
            self.assertEqual(generic.worst_case_efficiency, known.worst_case_efficiency)

    def test_uniform_repair_synthesizes_known_four_agent_offset_and_is_minimal(self):
        source = guo_2024_four_agent_network_spec()
        baseline = certify_ordered_public_project_charge(deleted_input_charge(source, 4), 4)
        repair = synthesize_minimal_uniform_repair(baseline, 4)
        self.assertEqual(repair.per_term_offset, Fraction(1, 20000))
        repaired = certify_ordered_public_project_charge(
            deleted_input_charge(add_output_bias_offset(source, repair.per_term_offset), 4), 4
        )
        self.assertEqual(repaired.minimum_budget_slack, 0)
        under_repaired = deleted_input_charge(add_output_bias_offset(source, repair.per_term_offset / 2), 4)
        point = baseline.minimum_slack_witness
        self.assertLess(under_repaired.evaluate(point) - 3 * max(sum(point), Fraction(1)), 0)

    def test_uniform_budget_repair_has_exact_ir_tradeoff(self):
        source = guo_2024_four_agent_network_spec()
        budget = certify_ordered_public_project_charge(deleted_input_charge(source, 4), 4)
        repair = synthesize_minimal_uniform_repair(budget, 4)
        baseline = certify_minimum_groves_utility(deleted_input_terms(source, 4), 4)
        repaired_source = add_output_bias_offset(source, repair.per_term_offset)
        repaired = certify_minimum_groves_utility(deleted_input_terms(repaired_source, 4), 4)
        direct = replay_deleted_input_network_utility_margin(repaired_source, 4)
        self.assertEqual(repaired.minimum_utility, baseline.minimum_utility - repair.per_term_offset)
        self.assertEqual(direct["minimum_utility"], f"{repaired.minimum_utility.numerator}/{repaired.minimum_utility.denominator}")

    def test_certificate_reports_exact_arrangement_work(self):
        certificate = certify_ordered_public_project_charge(guo_2024_four_agent_charge(), 4)
        self.assertGreater(certificate.arrangement_planes, 4)
        self.assertGreater(certificate.candidate_bases_examined, len(certificate.vertices))

    def test_serialized_certificate_replays_without_primary_engine(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "artifacts" / "max_affine_certification.json").read_text())
        expected = {name: entry["certificate"] for name, entry in payload["entries"].items()}
        self.assertEqual(replay_payload(payload), expected)

    def test_independent_replay_detects_tampered_result(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "artifacts" / "max_affine_certification.json").read_text())
        corrupted = deepcopy(payload)
        corrupted["entries"]["guo_aaai_2024_printed_4_agent"]["certificate"]["minimum_budget_slack"] = "0/1"
        expected = {name: entry["certificate"] for name, entry in corrupted["entries"].items()}
        self.assertNotEqual(replay_payload(corrupted), expected)

    def test_independent_replay_detects_tampered_formula(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "artifacts" / "max_affine_certification.json").read_text())
        corrupted = deepcopy(payload)
        specification = corrupted["entries"]["guo_aaai_2024_printed_4_agent"]["specification"]
        specification["affine_terms"][0][-1] = "0/1"
        with self.assertRaises(ValueError):
            replay_payload(corrupted)

    def test_independent_replay_detects_tampered_source_network(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "artifacts" / "max_affine_certification.json").read_text())
        corrupted = deepcopy(payload)
        corrupted["source_networks"]["guo_aaai_2024_printed_4_agent"]["output_bias"] = "0/1"
        with self.assertRaises(ValueError):
            replay_payload(corrupted)

    def test_direct_source_network_certificate_matches_compiled_certificate(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "artifacts" / "max_affine_certification.json").read_text())
        for name, network in payload["source_networks"].items():
            self.assertEqual(
                replay_deleted_input_network(network, int(payload["entries"][name]["dimension"])),
                payload["entries"][name]["certificate"],
            )

    def test_direct_source_network_certificate_detects_changed_coefficient(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "artifacts" / "max_affine_certification.json").read_text())
        name = "guo_aaai_2024_printed_4_agent"
        corrupted = deepcopy(payload["source_networks"][name])
        corrupted["hidden"][0]["output_weight"] = "0/1"
        self.assertNotEqual(
            replay_deleted_input_network(corrupted, 4),
            payload["entries"][name]["certificate"],
        )

    def test_frozen_synthetic_relu_fixture_crosschecks_two_certificate_routes(self):
        source = deterministic_network(670202, input_dimension=3, width=3)
        compiled = certify_ordered_public_project_charge(deleted_input_charge(source, 4), 4)
        def encode(value):
            if isinstance(value, Fraction):
                return f"{value.numerator}/{value.denominator}"
            if isinstance(value, tuple):
                return [encode(item) for item in value]
            if hasattr(value, "__dict__"):
                return {key: encode(item) for key, item in value.__dict__.items()}
            return value
        self.assertEqual(replay_deleted_input_network(source, 4), encode(compiled))

    def test_source_only_certificate_is_dimension_generic(self):
        source = deterministic_network(671102, input_dimension=6, width=2)
        compiled = certify_ordered_public_project_charge(deleted_input_charge(source, 7), 7)

        def encode(value):
            if isinstance(value, Fraction):
                return f"{value.numerator}/{value.denominator}"
            if isinstance(value, tuple):
                return [encode(item) for item in value]
            if hasattr(value, "__dict__"):
                return {key: encode(item) for key, item in value.__dict__.items()}
            return value

        self.assertEqual(replay_deleted_input_network(source, 7), encode(compiled))


if __name__ == "__main__":
    unittest.main()
