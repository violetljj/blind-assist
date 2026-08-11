#!/usr/bin/env python3
"""Focused tests for metadata-only TARO R6 untouched cohort planning."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from scripts.research.taro_o0r_candidate_scale_runtime import r6_factor_split as r6
from scripts.research.taro_o0r_candidate_scale_runtime import r6_untouched_cohort as cohort


class R6UntouchedCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[3]
        cls.plan = cohort.build_plan(cls.repo)

    def test_exact_metadata_roster_is_reproducible_and_formation_disjoint(self) -> None:
        self.assertEqual(8, self.plan["invariants"]["parent_count"])
        self.assertEqual(1757, self.plan["selection"]["eligible_row_count"])
        self.assertEqual(186, self.plan["selection"]["matched_official_identity_count"])
        self.assertEqual(24, self.plan["request_plan"]["request_count"])
        self.assertEqual(0, self.plan["request_plan"]["response_body_bytes_allowed"])
        self.assertFalse({row["visit_id"] for row in self.plan["selection"]["roster"]} & r6.FORBIDDEN_FORMATION_PARENTS)
        self.assertEqual(self.plan, cohort.build_plan(self.repo))

    def test_metadata_lock_has_zero_media_model_and_truth_authority(self) -> None:
        self.assertEqual(
            {"metadata_selection": True, "head_requests": False, "source_download": False, "source_decode": False, "model_execution": False, "truth_scoring": False, "training": False},
            self.plan["authority"],
        )
        self.assertFalse(self.plan["invariants"]["media_body_bytes_read"])
        self.assertFalse(self.plan["invariants"]["model_outputs_read"])
        self.assertFalse(self.plan["invariants"]["truth_payload_read"])

    def test_roster_or_authority_mutation_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.plan)
        mutated["selection"]["roster"][0]["visit_id"] = next(iter(r6.FORBIDDEN_FORMATION_PARENTS))
        with self.assertRaises(cohort.R6CohortError):
            cohort.validate_plan(mutated, repo_root=self.repo, recompute=False)
        mutated = copy.deepcopy(self.plan)
        mutated["authority"]["source_download"] = True
        with self.assertRaisesRegex(cohort.R6CohortError, "AUTHORITY_DRIFT"):
            cohort.validate_plan(mutated, repo_root=self.repo, recompute=False)


if __name__ == "__main__":
    unittest.main()
