#!/usr/bin/env python3
"""Mutation tests for the TARO R6 one-shot execution lock."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.research.taro_o0r_candidate_scale_runtime import validate_r6_confirmation_execution_lock as validator


class R6ConfirmationExecutionLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(validator.DEFAULT_LOCK_PATH.read_text(encoding="utf-8"))

    def _validate_mutation(self, mutate) -> str:
        changed = copy.deepcopy(self.payload)
        mutate(changed)
        temp_root = validator.REPO_ROOT / "docs" / "research" / "taro"
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            path = Path(directory) / "lock.json"
            changed["unique_argv"][-1] = path.relative_to(validator.REPO_ROOT).as_posix()
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(validator.R6ExecutionLockError) as caught:
                validator.validate_execution_lock(path)
        return caught.exception.code

    def test_exact_lock_validates_before_root_creation(self) -> None:
        self.assertEqual(validator.LOCK_ID, validator.validate_execution_lock()["lock_id"])

    def test_faro_cannot_move_into_phase_a(self) -> None:
        code = self._validate_mutation(lambda value: value["phase_firewall"]["phase_a_payload_allowlist"].append("highres_depth"))
        self.assertEqual("R6_EXEC_PHASE_FIREWALL_DRIFT", code)

    def test_candidate_postprocess_identity_cannot_change(self) -> None:
        code = self._validate_mutation(lambda value: value["candidate_identity"].__setitem__("postprocess_id", "ALIGN_CORNERS_FALSE"))
        self.assertEqual("R6_EXEC_CANDIDATE_IDENTITY_DRIFT", code)

    def test_user_authority_cannot_be_removed(self) -> None:
        code = self._validate_mutation(lambda value: value["user_authority"].__setitem__("explicit_model_and_truth_execution_authority", False))
        self.assertEqual("R6_EXEC_USER_AUTHORITY_DRIFT", code)


if __name__ == "__main__":
    unittest.main()
