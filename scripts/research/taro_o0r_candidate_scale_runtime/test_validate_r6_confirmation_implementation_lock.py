#!/usr/bin/env python3
"""Mutation tests for the TARO R6 confirmation implementation lock."""

from __future__ import annotations

import copy
import json
import unittest

from scripts.research.taro_o0r_candidate_scale_runtime import validate_r6_confirmation_implementation_lock as validator


class R6ConfirmationImplementationLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(validator.DEFAULT_LOCK_PATH.read_text(encoding="utf-8"))

    def test_lock_and_bound_files_validate(self) -> None:
        self.assertEqual([], validator.validate_payload(self.payload, verify_files=True))

    def test_faro_in_phase_a_is_rejected(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["phase_firewall"]["phase_a_payload_allowlist"].append("highres_depth")
        self.assertIn("R6_CONFIRM_IMPL_PHASE_A_ALLOWLIST_DRIFT", validator.validate_payload(changed, verify_files=False))

    def test_formation_parent_or_frame_count_is_rejected(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["exact_cohort"]["roster"][0]["visit_id"] = "470974"
        self.assertIn("R6_CONFIRM_IMPL_COHORT_DRIFT", validator.validate_payload(changed, verify_files=False))

    def test_candidate_transform_drift_is_rejected(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["frozen_algorithm"]["postprocess_id"] = "ALIGN_CORNERS_FALSE"
        self.assertIn("R6_CONFIRM_IMPL_TRANSFORM_DRIFT", validator.validate_payload(changed, verify_files=False))

    def test_execution_authority_cannot_be_smuggled_into_implementation_lock(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["execution_authority"]["model_execution"] = True
        self.assertIn("R6_CONFIRM_IMPL_AUTHORITY_DRIFT:model_execution", validator.validate_payload(changed, verify_files=False))


if __name__ == "__main__":
    unittest.main()
