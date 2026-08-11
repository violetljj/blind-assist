#!/usr/bin/env python3
"""Mutation tests for the TARO R5 implementation lock validator."""

from __future__ import annotations

import copy
import json
import unittest

from scripts.research.taro_o0r_candidate_scale_runtime import validate_r5_implementation_lock as validator


class R5ImplementationLockValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lock = json.loads(validator.DEFAULT_LOCK_PATH.read_text(encoding="utf-8"))

    def errors_for(self, mutation) -> list[str]:
        value = copy.deepcopy(self.lock)
        mutation(value)
        return validator.validate_payload(value, verify_files=False)

    def assert_rejected(self, mutation, fragment: str) -> None:
        errors = self.errors_for(mutation)
        self.assertTrue(any(fragment in error for error in errors), errors)

    def test_exact_lock_and_bound_files_pass(self) -> None:
        self.assertEqual([], validator.validate_payload(self.lock, verify_files=True))

    def test_transform_or_code_binding_mutation_is_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value["candidate_identity"].__setitem__("postprocess_id", "ALIGN_CORNERS_FALSE"),
            "R5_IMPL_POSTPROCESS_DRIFT",
        )
        self.assert_rejected(
            lambda value: value["implementation_bindings"][0].__setitem__("sha256", "0" * 64),
            "R5_IMPL_CODE_BINDING_SET_DRIFT",
        )

    def test_phase_firewall_or_policy_mutation_is_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value["phase_firewall"].__setitem__("phase_a_payload_allowlist", ["color", "highres_depth"]),
            "R5_IMPL_PHASE_A_ALLOWLIST_DRIFT",
        )
        self.assert_rejected(
            lambda value: value["frozen_algorithm"].__setitem__("direct_failure_after_selection", "FALL_BACK"),
            "R5_IMPL_DIRECT_FALLBACK_DRIFT",
        )

    def test_authority_or_root_reuse_cannot_expand(self) -> None:
        self.assert_rejected(
            lambda value: value["execution_authority"].__setitem__("depthart_inference", True),
            "R5_IMPL_AUTHORITY_DRIFT",
        )
        self.assert_rejected(
            lambda value: value["evidence_contract"].__setitem__("rerun", True),
            "R5_IMPL_EVIDENCE_RULE_DRIFT",
        )


if __name__ == "__main__":
    unittest.main()
