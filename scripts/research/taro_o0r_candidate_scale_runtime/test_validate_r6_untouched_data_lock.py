#!/usr/bin/env python3
"""Mutation tests for the TARO R6 untouched data-use lock."""

from __future__ import annotations

import copy
import json
import unittest

from scripts.research.taro_o0r_candidate_scale_runtime import validate_r6_untouched_data_lock as validator


class R6UntouchedDataLockTests(unittest.TestCase):
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

    def test_roster_or_request_mutation_is_rejected(self) -> None:
        self.assert_rejected(lambda value: value["selection"]["roster"][0].__setitem__("visit_id", "410690"), "R6_DATA_LOCK_ROSTER_DRIFT")
        self.assert_rejected(lambda value: value["asset_plan"].__setitem__("expanded_requests_sha256", "0" * 64), "R6_DATA_LOCK_REQUEST_HASH_DRIFT")

    def test_authority_or_successor_cannot_expand(self) -> None:
        self.assert_rejected(lambda value: value["authority"].__setitem__("head_requests", True), "R6_DATA_LOCK_AUTHORITY_DRIFT")
        self.assert_rejected(lambda value: value.__setitem__("unique_successor", "DOWNLOAD_NOW"), "R6_DATA_LOCK_SUCCESSOR_DRIFT")


if __name__ == "__main__":
    unittest.main()
