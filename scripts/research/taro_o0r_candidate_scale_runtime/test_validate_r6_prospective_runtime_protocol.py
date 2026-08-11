#!/usr/bin/env python3
"""Mutation tests for the TARO R6 prospective-runtime protocol lock."""

from __future__ import annotations

import copy
import json
import unittest

from scripts.research.taro_o0r_candidate_scale_runtime import validate_r6_prospective_runtime_protocol as validator


class ProspectiveRuntimeProtocolValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(validator.DEFAULT_LOCK.read_text(encoding="utf-8"))

    def errors(self, mutation) -> list[str]:
        payload = copy.deepcopy(self.payload)
        mutation(payload)
        return validator.validate_payload(payload, verify_files=False)

    def test_frozen_lock_and_bound_files_pass(self) -> None:
        self.assertEqual([], validator.validate_file())

    def test_truth_surface_cannot_be_runtime_eligible(self) -> None:
        errors = self.errors(lambda row: row["interface_seam_closed_by_this_protocol"].__setitem__("truth_defined_pixel_ids_allowed", True))
        self.assertIn("R6_RUNTIME_PROTOCOL_TRUTH_SURFACE_DRIFT:truth_defined_pixel_ids_allowed", errors)

    def test_query_clearance_owner_cannot_change(self) -> None:
        errors = self.errors(lambda row: row["adopted_policy"].__setitem__("query_clearance_owner", "PHASE_A_SELECTED_SUPPORT_BOUNDARY_COMPONENT"))
        self.assertIn("R6_RUNTIME_PROTOCOL_QUERY_OWNER_DRIFT", errors)

    def test_untouched_parents_cannot_be_removed_from_forbidden_roster(self) -> None:
        errors = self.errors(lambda row: row["data_roles"]["r6_untouched_parent_ids"].pop())
        self.assertIn("R6_RUNTIME_PROTOCOL_DATA_ROLE_DRIFT", errors)

    def test_execution_cannot_be_smuggled_into_protocol(self) -> None:
        errors = self.errors(lambda row: row["execution_authority"].__setitem__("truth_scoring", True))
        self.assertIn("R6_RUNTIME_PROTOCOL_EXECUTION_AUTHORITY_DRIFT:truth_scoring", errors)


if __name__ == "__main__":
    unittest.main()
