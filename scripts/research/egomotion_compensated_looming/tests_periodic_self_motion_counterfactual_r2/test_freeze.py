from __future__ import annotations

import copy
import json
import unittest

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_freeze,
)


class FreezeValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(
            validate_freeze.DEFAULT_CONTRACT.read_text(encoding="utf-8")
        )
        cls.geometry = json.loads(
            validate_freeze.DEFAULT_GEOMETRY.read_text(encoding="utf-8")
        )
        cls.budget = json.loads(
            validate_freeze.DEFAULT_BUDGET.read_text(encoding="utf-8")
        )

    def validate(self, contract=None, geometry=None, budget=None):
        return validate_freeze.validate_bundle(
            contract or self.contract,
            geometry or self.geometry,
            budget or self.budget,
            verify_dependencies=False,
        )

    def test_canonical_bundle(self) -> None:
        self.assertEqual([], self.validate())

    def test_rejects_premature_formal_authorization(self) -> None:
        budget = copy.deepcopy(self.budget)
        budget["formal_execution_authorized"] = True
        self.assertTrue(
            any(
                error.startswith("FORMAL_AUTHORIZATION_NOT_FALSE:")
                for error in self.validate(budget=budget)
            )
        )

    def test_rejects_frame_level_sample_inflation(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["factorial_design"]["analysis_cluster_count"] = 480
        self.assertIn(
            "ANALYSIS_CLUSTER_COUNT", self.validate(contract=contract)
        )

    def test_rejects_algorithm_threshold_drift(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["unchanged_algorithm_lock"]["threshold_per_s"] = 0.009
        self.assertIn("R3_THRESHOLD", self.validate(contract=contract))

    def test_rejects_algorithm_identity_drift(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["unchanged_algorithm_lock"]["implementation"] = "R4"
        self.assertIn("R3_IMPLEMENTATION", self.validate(contract=contract))

    def test_rejects_response_field_drift(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["unchanged_algorithm_lock"]["response_field"] = "raw_response"
        self.assertIn("R3_RESPONSE_FIELD", self.validate(contract=contract))

    def test_rejects_reset_or_pair_state_drift(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["unchanged_algorithm_lock"]["reset_rule"] = "never reset"
        contract["unchanged_algorithm_lock"]["pair_state"] = "shared across arms"
        errors = self.validate(contract=contract)
        self.assertIn("R3_RESET_RULE", errors)
        self.assertIn("R3_PAIR_STATE", errors)

    def test_rejects_motion_block_hash_drift(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["trajectory_blocks"]["ADVIO_13"]["pose_csv_sha256"] = (
            "0" * 64
        )
        self.assertIn(
            "TRAJECTORY_POSE_HASH:ADVIO_13",
            self.validate(contract=contract),
        )

    def test_rejects_authority_ceiling_drift(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["non_goals"] = [
            item
            for item in contract["non_goals"]
            if not any(
                token in item.lower()
                for token in ("sequence16", "cotracker", "android")
            )
        ]
        errors = self.validate(contract=contract)
        self.assertIn("SEQUENCE16_NOT_CLOSED", errors)
        self.assertIn("COTRACKER_NOT_CLOSED", errors)
        self.assertIn("ANDROID_NOT_CLOSED", errors)

    def test_rejects_claim_or_successor_authority_drift(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["claim_ceiling"] = "NATURAL_DATA_CAUSAL"
        contract["successor_policy"]["automatic_android_authority"] = True
        errors = self.validate(contract=contract)
        self.assertIn("CLAIM_CEILING", errors)
        self.assertIn(
            "SUCCESSOR_AUTHORITY:automatic_android_authority", errors
        )

    def test_rejects_main_low_texture_check_drift(self) -> None:
        contract = copy.deepcopy(self.contract)
        quality = contract["quality_interventions"]
        quality["formal_main_manipulation_check"][
            "low_texture_sequence_pass"
        ] = "Require source-known edge-spread in every main scene."
        quality["formal_main_manipulation_check"].pop(
            "low_texture_no_blur_identity"
        )
        errors = self.validate(contract=contract)
        self.assertIn("MAIN_LOW_TEXTURE_METRIC", errors)
        self.assertIn("MAIN_LOW_TEXTURE_IDENTITY", errors)

    def test_rejects_invalid_metric_or_edge_scope_drift(self) -> None:
        contract = copy.deepcopy(self.contract)
        definitions = contract["quality_interventions"][
            "response_blind_metric_definitions"
        ]
        definitions["source_known_edge_spread"] = "Use any visible edge."
        definitions["invalid_metric_rule"] = "Omit invalid values."
        errors = self.validate(contract=contract)
        self.assertIn("EDGE_SPREAD_SCOPE", errors)
        self.assertIn("INVALID_METRIC_FAIL_CLOSED", errors)

    def test_rejects_terminal_family_or_formula_drift(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["estimands"]["terminal_driving_family"][8] = "POSTHOC"
        contract["statistical_plan"]["bootstrap"][
            "simultaneous_interval"
        ] = "Use pointwise intervals."
        errors = self.validate(contract=contract)
        self.assertIn("TERMINAL_DRIVING_FAMILY", errors)
        self.assertIn("SIMULTANEOUS_INTERVAL_FORMULA", errors)

    def test_rejects_terminal_logic_drift(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["decision_rules"]["MOTION_SUPPORTED"] = (
            "MOTION_CLEAN is SUPPORTED."
        )
        self.assertIn(
            "TERMINAL_RULE:MOTION_SUPPORTED",
            self.validate(contract=contract),
        )

    def test_rejects_guarded_launcher_drift(self) -> None:
        budget = copy.deepcopy(self.budget)
        budget["host_policy"]["launcher_required"] = "run.py"
        self.assertIn("GUARDED_LAUNCHER", self.validate(budget=budget))

    def test_rejects_unexercisable_preflight_concurrency(self) -> None:
        budget = copy.deepcopy(self.budget)
        budget["host_policy"]["candidate_profiles"]["balanced"] = 12
        budget["required_preflight"]["scheduling_comparison"] = (
            "Compare 8 and 12 workers."
        )
        errors = self.validate(budget=budget)
        self.assertIn("PREFLIGHT_WORKER_PROFILES", errors)
        self.assertIn("PREFLIGHT_SCHEDULING_COMPARISON", errors)

    def test_rejects_missing_geometry_gate(self) -> None:
        geometry = copy.deepcopy(self.geometry)
        geometry["required_gates"].pop()
        errors = self.validate(geometry=geometry)
        self.assertIn("GEOMETRY_GATE_COUNT", errors)
        self.assertIn("GEOMETRY_GATE_MISSING:G14_", errors)

    def test_rejects_linked_spec_hash_drift(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["linked_specs"]["run_budget_sha256"] = "0" * 64
        self.assertIn(
            "LINKED_SPEC_HASH:run_budget_sha256",
            self.validate(contract=contract),
        )

    def test_rejects_future_phase_activation(self) -> None:
        budget = copy.deepcopy(self.budget)
        budget["phase_budget"][1]["allowed_now"] = True
        self.assertIn("FUTURE_PHASE_ALLOWED", self.validate(budget=budget))


if __name__ == "__main__":
    unittest.main()
