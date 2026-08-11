#!/usr/bin/env python3
"""Mutation tests for the TARO R6 prospective-runtime implementation lock."""

from __future__ import annotations

import copy
import json
import unittest

from scripts.research.taro_o0r_candidate_scale_runtime import validate_r6_prospective_runtime_implementation_lock as validator


class ProspectiveRuntimeImplementationLockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(validator.DEFAULT_LOCK.read_text(encoding="utf-8"))

    def errors(self, mutation) -> list[str]:
        payload = copy.deepcopy(self.payload)
        mutation(payload)
        return validator.validate_payload(payload, verify_files=False)

    def test_frozen_lock_and_bound_files_pass(self) -> None:
        self.assertEqual([], validator.validate_file())

    def test_result_side_argument_claim_cannot_flip(self) -> None:
        errors = self.errors(lambda row: row["frozen_implementation"].__setitem__("public_builder_has_faro_truth_task_metric_or_outcome_argument", True))
        self.assertIn("R6_RUNTIME_IMPL_PUBLIC_API_DRIFT", errors)

    def test_query_owner_cannot_change(self) -> None:
        errors = self.errors(lambda row: row["frozen_implementation"].__setitem__("query_clearance_owner", "DIRECT_APPLE_SUPPORT"))
        self.assertIn("R6_RUNTIME_IMPL_FACTOR_OWNER_DRIFT", errors)

    def test_synthetic_count_cannot_be_inflated(self) -> None:
        errors = self.errors(lambda row: row["synthetic_test_receipt"].__setitem__("query_clearance_evaluable_query_count", 9))
        self.assertIn("R6_RUNTIME_IMPL_SYNTHETIC_RESULT_DRIFT", errors)

    def test_real_execution_cannot_be_smuggled_in(self) -> None:
        errors = self.errors(lambda row: row["execution_authority"].__setitem__("formation_replay", True))
        self.assertIn("R6_RUNTIME_IMPL_EXECUTION_AUTHORITY_DRIFT:formation_replay", errors)


if __name__ == "__main__":
    unittest.main()
