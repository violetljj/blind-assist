#!/usr/bin/env python3
"""Mutation tests for the non-execution TARO O0M protocol lock."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.research.taro_o0m.validate_taro_o0m_protocol import (
    expected_records,
    validate_runtime_absence,
    validate_static_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = REPO_ROOT / "docs/research/taro/TARO_O0M_SYNTHETIC_IDENTIFIABILITY_AND_FACTORIAL_MECHANICS_PROTOCOL_LOCK_2026-08-10.json"
FIXTURE = REPO_ROOT / "docs/research/taro/TARO_O0M_EXECUTION_FIXTURE_SPEC_2026-08-10.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TaroO0MProtocolValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load(PROTOCOL)
        cls.fixture = load(FIXTURE)

    def validate(self, *, protocol: dict | None = None, fixture: dict | None = None) -> list[str]:
        return validate_static_contract(
            copy.deepcopy(self.protocol if protocol is None else protocol),
            copy.deepcopy(self.fixture if fixture is None else fixture),
        )

    def test_frozen_contract_is_valid(self) -> None:
        self.assertEqual(self.validate(), [])

    def test_implementation_authority_cannot_drift(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["execution_authority"]["o0m_implementation"] = True
        self.assertIn("AUTHORITY_EXCEEDED", self.validate(protocol=protocol))

    def test_execution_authority_cannot_drift(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["execution_authority"]["o0m_execution"] = True
        self.assertIn("AUTHORITY_EXCEEDED", self.validate(protocol=protocol))

    def test_execution_case_id_cannot_reuse_p0_id(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        case = fixture["identifiability_cases"][0]
        case["id"] = case["source_p0_case_id"]
        self.assertTrue(any(error.startswith("EXECUTION_FAMILY_ID:") for error in self.validate(fixture=fixture)))

    def test_identifiability_truth_is_recomputed(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        case = fixture["identifiability_cases"][0]
        case["measurement_jacobian_whitened"] = [[0.0] * 4 for _ in range(4)]
        self.assertIn(f"IDENTIFIABILITY_TRUTH:{case['id']}", self.validate(fixture=fixture))

    def test_numeric_contract_drift_is_rejected(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        fixture["numeric_contract"]["sigma_factor_oracle_m"] = 0.0
        self.assertIn("NUMERIC_CONTRACT", self.validate(fixture=fixture))

    def test_per_arm_output_hash_is_recomputed(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        scene = fixture["factorial_scenes"][0]
        key = "NONE|VALUE_ONLY_COMMON_SUPPORT"
        scene["expected_records"][key]["output_sha256"] = "0" * 64
        self.assertIn(f"RECORD_TRUTH:{scene['id']}", self.validate(fixture=fixture))

    def test_value_only_common_support_must_be_constant(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        scene = fixture["factorial_scenes"][0]
        key = "SCALE|VALUE_ONLY_COMMON_SUPPORT"
        scene["expected_records"][key]["common_support_sha256"] = "F" * 64
        errors = self.validate(fixture=fixture)
        self.assertIn(f"RECORD_TRUTH:{scene['id']}", errors)

    def test_boundary_validity_value_only_cannot_become_clear(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        scene = next(item for item in fixture["factorial_scenes"] if item["id"] == "o0m_exec_boundary_validity")
        key = "BOUNDARY|VALUE_ONLY_COMMON_SUPPORT"
        scene["expected_records"][key]["output"]["query_state"] = "CLEAR_OBSERVED"
        self.assertIn(f"RECORD_TRUTH:{scene['id']}", self.validate(fixture=fixture))

    def test_body_motion_action_must_remain_forbidden(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        action = next(item for item in fixture["action_filter_cases"] if item["requires_body_motion"])
        action["expected_allowed"] = True
        self.assertIn("BODY_MOTION_FILTER", self.validate(fixture=fixture))

    def test_successor_must_remain_non_execution_implementation_lock(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["unique_successor"]["execution_authority"] = True
        self.assertIn("SUCCESSOR", self.validate(protocol=protocol))

    def test_real_data_and_network_budgets_remain_false(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["resource_budget"]["real_data"] = True
        protocol["resource_budget"]["network"] = True
        errors = self.validate(protocol=protocol)
        self.assertIn("REAL_DATA_BUDGET", errors)
        self.assertIn("NETWORK_BUDGET", errors)

    def test_runtime_presence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "scripts/research/taro_o0m_runtime"
            runtime.mkdir(parents=True)
            (runtime / "run_o0m_canary.py").write_text("pass\n", encoding="utf-8")
            self.assertIn(
                "PROHIBITED_RUNTIME_PRESENT:scripts/research/taro_o0m_runtime/run_o0m_canary.py",
                validate_runtime_absence(Path(temporary)),
            )

    def test_renamed_runtime_presence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "scripts/research/taro_o0m_runtime"
            runtime.mkdir(parents=True)
            (runtime / "renamed_solver.py").write_text("pass\n", encoding="utf-8")
            self.assertIn(
                "PROHIBITED_RUNTIME_PRESENT:scripts/research/taro_o0m_runtime/renamed_solver.py",
                validate_runtime_absence(Path(temporary)),
            )

    def test_freeze_and_claim_ceiling_drift_are_rejected(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["freeze"]["unresolved_before_o0m_execution"] = []
        protocol["claim_ceiling"] = "TBD"
        errors = self.validate(protocol=protocol)
        self.assertIn("FREEZE_DRIFT", errors)
        self.assertIn("CLAIM_CEILING_DRIFT", errors)

    def test_nonempty_gate_body_drift_is_rejected(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["o0m_gates"][0]["condition"] = "TBD"
        self.assertIn("GATE_CONTRACT_DRIFT", self.validate(protocol=protocol))

    def test_authority_key_deletion_is_rejected(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        del protocol["execution_authority"]["o0m_execution"]
        self.assertIn("AUTHORITY_EXCEEDED", self.validate(protocol=protocol))

    def test_scene_truth_label_cannot_drift_with_recomputed_records(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        scene = next(item for item in fixture["factorial_scenes"] if item["id"] == "o0m_exec_scale_isolated")
        scene["truth_clearance_m"] = 0.2
        self.assertIn(f"SCENE_CONTRACT:{scene['id']}", self.validate(fixture=fixture))

    def test_o0r_cannot_be_marked_ready_or_executable(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["o0r_admission"]["current_terminal"] = "READY"
        protocol["o0r_admission"]["execution_authority"] = True
        self.assertIn("O0R_ADMISSION_DRIFT", self.validate(protocol=protocol))

    def test_status_and_presence_scope_cannot_claim_execution(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["status"] = ["SCIENTIFIC_PASS", "EXECUTION_AUTHORIZED"]
        protocol["o0m_protocol_scope"]["implementation_present"] = True
        protocol["o0m_protocol_scope"]["runner_present"] = True
        protocol["o0m_protocol_scope"]["artifact_present"] = True
        errors = self.validate(protocol=protocol)
        self.assertIn("STATUS_DRIFT", errors)
        self.assertIn("SCOPE_DRIFT", errors)

    def test_fixture_outcome_and_claim_ceiling_cannot_expand(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        fixture["outcome_access"] = "FULL_REAL_OUTCOME"
        fixture["claim_ceiling"] = "REAL HEADROOM ESTABLISHED"
        errors = self.validate(fixture=fixture)
        self.assertIn("FIXTURE_OUTCOME_ACCESS", errors)
        self.assertIn("FIXTURE_CLAIM_CEILING", errors)

    def test_identifiability_rule_cannot_admit_prior_rank(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        fixture["identifiability_rule"]["measurement_matrix"] = "Whitened Hessian with prior, damping and regularizer."
        self.assertIn("IDENTIFIABILITY_RULE_DRIFT", self.validate(fixture=fixture))

    def test_positive_camera_only_action_cannot_be_rejected(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        action = next(item for item in fixture["action_filter_cases"] if not item["requires_body_motion"])
        action["expected_allowed"] = False
        self.assertIn("ACTION_FILTER_DRIFT", self.validate(fixture=fixture))

    def test_successor_policy_cannot_drift_or_pre_authorize(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["successor_policy"]["unique_successor"] = "OTHER"
        protocol["successor_policy"]["successor_currently_authorized"] = True
        self.assertIn("SUCCESSOR_POLICY_DRIFT", self.validate(protocol=protocol))

    def test_claims_and_result_model_cannot_coordinate_drift(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["claims_allowed"].append("FACTOR_CAUSAL_HEADROOM")
        protocol["claims_forbidden"].remove("FACTOR_CAUSAL_HEADROOM")
        protocol["result_model"]["scientific_status"] = "PASS"
        errors = self.validate(protocol=protocol)
        self.assertIn("CLAIMS_ALLOWED_DRIFT", errors)
        self.assertIn("CLAIMS_FORBIDDEN_DRIFT", errors)
        self.assertIn("RESULT_MODEL_DRIFT", errors)
        self.assertIn("PROTOCOL_SEMANTIC_CORE_DRIFT", errors)

    def test_fixture_additional_fields_are_rejected(self) -> None:
        mutations = (
            ("top", lambda fixture: fixture.__setitem__("b1_consumed_outcome", "LEAK"), "FIXTURE_TOP_LEVEL_KEYS"),
            ("scene", lambda fixture: fixture["factorial_scenes"][0].__setitem__("future_oracle_outcome", "LEAK"), "FACTORIAL_SCENE_KEYS:"),
            ("case", lambda fixture: fixture["identifiability_cases"][0].__setitem__("future_frame_result", "LEAK"), "IDENTIFIABILITY_CASE_KEYS:"),
        )
        for name, mutate, expected_prefix in mutations:
            with self.subTest(name=name):
                fixture = copy.deepcopy(self.fixture)
                mutate(fixture)
                errors = self.validate(fixture=fixture)
                self.assertIn("FIXTURE_CANONICAL_DRIFT", errors)
                self.assertTrue(any(error.startswith(expected_prefix) for error in errors))

    def test_scene_scientific_inputs_are_canonically_locked(self) -> None:
        mutations = (
            ("truth", lambda scene: scene.__setitem__("truth_clearance_m", 0.2)),
            ("oracle_valid", lambda scene: scene["oracle_factor_valid"].__setitem__("SCALE", False)),
            ("provenance", lambda scene: scene["oracle_provenance"].__setitem__("SCALE", scene["factor_provenance"]["SCALE"])),
            ("identity", lambda scene: scene.__setitem__("factor_identity_sha256", "0" * 64)),
            ("anchor", lambda scene: scene.__setitem__("anchor_identity", "other-anchor")),
            ("timestamp", lambda scene: scene.__setitem__("max_source_timestamp_ns", 2_000_000_001)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                fixture = copy.deepcopy(self.fixture)
                scene = fixture["factorial_scenes"][0]
                records_before = copy.deepcopy(scene["expected_records"])
                mutate(scene)
                scene["expected_records"] = expected_records(scene, fixture["numeric_contract"])
                if name == "truth":
                    self.assertEqual(records_before, scene["expected_records"])
                self.assertIn(f"SCENE_INPUT_DRIFT:{scene['id']}", self.validate(fixture=fixture))

    def test_scene_cardinality_and_execution_mapping_are_exact(self) -> None:
        duplicate_scene = copy.deepcopy(self.fixture)
        duplicate_scene["factorial_scenes"].append(copy.deepcopy(duplicate_scene["factorial_scenes"][0]))
        self.assertIn("FACTORIAL_SCENE_COUNT", self.validate(fixture=duplicate_scene))

        duplicate_case = copy.deepcopy(self.fixture)
        duplicate_case["identifiability_cases"][1]["id"] = duplicate_case["identifiability_cases"][0]["id"]
        self.assertTrue(any(error.startswith("DUPLICATE_EXECUTION_CASE:") for error in self.validate(fixture=duplicate_case)))

        swapped_mapping = copy.deepcopy(self.fixture)
        first, second = swapped_mapping["identifiability_cases"][:2]
        first["source_p0_case_id"], second["source_p0_case_id"] = second["source_p0_case_id"], first["source_p0_case_id"]
        self.assertTrue(any(error.startswith("EXECUTION_SOURCE_MAPPING:") for error in self.validate(fixture=swapped_mapping)))

    def test_required_binding_roles_cannot_be_deleted(self) -> None:
        for role in ("O0M_EXECUTION_FIXTURE", "P0_NUMERIC_EVALUATOR", "O0M_STATIC_TESTS"):
            with self.subTest(role=role):
                protocol = copy.deepcopy(self.protocol)
                protocol["bindings"] = [binding for binding in protocol["bindings"] if binding["role"] != role]
                self.assertIn("BINDING_ROLE_SET", self.validate(protocol=protocol))

    def test_hidden_static_solver_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            static_root = Path(temporary) / "scripts/research/taro_o0m"
            static_root.mkdir(parents=True)
            (static_root / "hidden_solver.py").write_text("pass\n", encoding="utf-8")
            self.assertIn(
                "PROHIBITED_STATIC_MODULE_PRESENT:scripts/research/taro_o0m/hidden_solver.py",
                validate_runtime_absence(Path(temporary)),
            )

    def test_preexisting_exclusive_artifact_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact_root = Path(temporary) / "artifacts.local/evidence/taro/o0m-analytic-mechanics-r0"
            artifact_root.mkdir(parents=True)
            (artifact_root / "result.json").write_text("{}\n", encoding="utf-8")
            self.assertIn(
                "PROHIBITED_ARTIFACT_ROOT_PRESENT:artifacts.local/evidence/taro/o0m-analytic-mechanics-r0",
                validate_runtime_absence(Path(temporary)),
            )

    def test_protocol_semantic_core_cannot_coordinate_drift(self) -> None:
        mutations = (
            ("primary", lambda protocol: protocol["factorial_contract"].update(primary_mode="FULL_BLOCK_VALUE_VALIDITY_UNCERTAINTY", post_outcome_supported_combination_selection_forbidden=False)),
            ("prior_rank", lambda protocol: protocol["identifiability_contract"].__setitem__("prior_damping_regularizer_may_add_rank", True)),
            ("unknown_negative", lambda protocol: protocol["schema_contract"].__setitem__("unknown_is_negative", True)),
            ("routing", lambda protocol: protocol["routing_contract"].__setitem__("o0m_pass", "ESTABLISHES_REAL_HEADROOM")),
            ("constraints", lambda protocol: protocol.__setitem__("constraints", [])),
            ("execute_now", lambda protocol: protocol["experiment_design"].__setitem__("stop_conditions", "EXECUTE_NOW")),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                protocol = copy.deepcopy(self.protocol)
                mutate(protocol)
                self.assertIn("PROTOCOL_SEMANTIC_CORE_DRIFT", self.validate(protocol=protocol))

    def test_reparameterization_and_leakage_tests_remain_required(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["experiment_design"]["implementation_lock_required_tests"] = []
        protocol["freeze"]["unresolved_before_o0m_execution"].remove(
            "non-axis-aligned projector reparameterization and future/outcome leakage mutation tests"
        )
        errors = self.validate(protocol=protocol)
        self.assertIn("PROTOCOL_SEMANTIC_CORE_DRIFT", errors)
        self.assertIn("FREEZE_DRIFT", errors)


if __name__ == "__main__":
    unittest.main()
