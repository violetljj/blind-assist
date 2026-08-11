"""Mutation tests for the TARO R5 amendment validator."""

from __future__ import annotations

import copy
import json
import unittest

from . import validate_r5_amendment as validator


class R5AmendmentValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lock = json.loads(validator.DEFAULT_LOCK_PATH.read_text(encoding="utf-8"))

    def errors_for(self, mutation) -> list[str]:
        payload = copy.deepcopy(self.lock)
        mutation(payload)
        return validator.validate_payload(payload, verify_files=False)

    def assert_rejected(self, mutation, code_fragment: str) -> None:
        errors = self.errors_for(mutation)
        self.assertTrue(any(code_fragment in error for error in errors), errors)

    def test_exact_lock_and_bound_files_pass(self) -> None:
        self.assertEqual([], validator.validate_payload(self.lock, verify_files=True))

    def test_parent_or_sequence_mutation_is_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value["exact_cohort"]["parent_order"][0].__setitem__("physical_frame_count", 24),
            "R5_PARENT_ORDER_DRIFT",
        )
        self.assert_rejected(
            lambda value: value["exact_cohort"].__setitem__("canonical_frame_identity_sequence_sha256", "0" * 64),
            "R5_LOCKED_SEQUENCE_HASH_DRIFT",
        )

    def test_policy_or_selection_mutation_is_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value["frozen_algorithm"].__setitem__("selection_field_allowlist", ["direct_extraction_evaluable"]),
            "R5_SELECTION_ALLOWLIST_DRIFT",
        )
        self.assert_rejected(
            lambda value: value["frozen_algorithm"].__setitem__("direct_extraction_failure_after_selection_action", "FALL_BACK"),
            "R5_POST_OUTCOME_FALLBACK_ALLOWED",
        )

    def test_phase_firewall_mutation_is_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value["phase_contract"]["phase_a"].__setitem__("required_zero_read_counts", ["FARO"]),
            "R5_PHASE_A_ZERO_READ_SET_DRIFT",
        )
        self.assert_rejected(
            lambda value: value["phase_contract"]["phase_b"].__setitem__("prior_eval_truth_access_forbidden", False),
            "R5_PHASE_B_PRIOR_EVAL_ACCESS_ALLOWED",
        )

    def test_gate_relaxation_is_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value["confirmation_gates"].__setitem__("parents_jointly_positive_height_and_normal_required", 7),
            "R5_JOINT_PARENT_GATE_DRIFT",
        )
        self.assert_rejected(
            lambda value: value["confirmation_gates"].__setitem__("unknown_is_negative", True),
            "R5_UNKNOWN_BECAME_NEGATIVE",
        )

    def test_execution_authority_cannot_expand(self) -> None:
        self.assert_rejected(
            lambda value: value["execution_authority"].__setitem__("depthart_inference", True),
            "R5_EXECUTION_AUTHORITY_DRIFT",
        )
        self.assert_rejected(
            lambda value: value["execution_contract_skeleton"].__setitem__("rerun", True),
            "R5_DESTRUCTIVE_RULE_DRIFT",
        )

    def test_predecessor_binding_and_successor_are_frozen(self) -> None:
        self.assert_rejected(
            lambda value: value["predecessor_bindings"][0].__setitem__("sha256", "F" * 64),
            "R5_PREDECESSOR_BINDING_SET_DRIFT",
        )
        self.assert_rejected(
            lambda value: value.__setitem__("unique_successor", "RUN_R5"),
            "R5_SUCCESSOR_DRIFT",
        )


if __name__ == "__main__":
    unittest.main()
