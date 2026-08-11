#!/usr/bin/env python3
"""Mutation tests for the TARO R6 implementation-lock validator."""

from __future__ import annotations

import copy
import json
import unittest

from scripts.research.taro_o0r_candidate_scale_runtime import validate_r6_implementation_lock as validator


class R6ImplementationLockValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lock = json.loads(validator.DEFAULT_LOCK_PATH.read_text(encoding="utf-8"))

    def assert_rejected(self, mutation, fragment: str) -> None:
        value = copy.deepcopy(self.lock)
        mutation(value)
        errors = validator.validate_payload(value, verify_files=False)
        self.assertTrue(any(fragment in error for error in errors), errors)

    def test_exact_lock_and_bound_files_pass(self) -> None:
        self.assertEqual([], validator.validate_payload(self.lock, verify_files=True))

    def test_factor_owner_or_selection_read_mutation_is_rejected(self) -> None:
        self.assert_rejected(lambda value: value["frozen_algorithm"].__setitem__("query_clearance_owner", "SELECTED"), "R6_IMPL_QUERY_OWNER_DRIFT")
        self.assert_rejected(lambda value: value["frozen_algorithm"]["owner_selection_fields_read"]["QUERY_CLEARANCE"].append("KNOWNNESS_OUTCOME"), "R6_IMPL_SELECTION_FIELD_DRIFT")

    def test_code_binding_or_replay_claim_mutation_is_rejected(self) -> None:
        self.assert_rejected(lambda value: value["implementation_bindings"][0].__setitem__("sha256", "0" * 64), "R6_IMPL_CODE_BINDING_SET_DRIFT")
        self.assert_rejected(lambda value: value["formation_replay_receipt"].__setitem__("promotion_allowed", True), "R6_IMPL_REPLAY_CLAIM_DRIFT")

    def test_authority_or_successor_cannot_expand(self) -> None:
        self.assert_rejected(lambda value: value["execution_authority"].__setitem__("untouched_truth_scoring", True), "R6_IMPL_AUTHORITY_DRIFT")
        self.assert_rejected(lambda value: value.__setitem__("unique_successor", "EXECUTE_NOW"), "R6_IMPL_SUCCESSOR_DRIFT")


if __name__ == "__main__":
    unittest.main()
