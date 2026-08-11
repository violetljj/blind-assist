from __future__ import annotations

import unittest
from pathlib import Path

from scripts.research.taro_o1r_r7_canary_runtime import fresh_confirmation_cohort as cohort


class FreshConfirmationCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[3]
        cls.plan = cohort.build_plan(cls.root)

    def test_exact_roster_and_parent_disjointness(self) -> None:
        roster = self.plan["selection"]["roster"]
        self.assertEqual([(row["visit_id"], row["video_id"], row["selection_rank_sha256"]) for row in roster], cohort.EXPECTED_ROSTER)
        self.assertEqual(self.plan["invariants"]["prior_taro_parent_overlap"], 0)

    def test_request_plan_is_head_only_and_zero_body(self) -> None:
        request = self.plan["request_plan"]
        self.assertEqual(request["request_count"], 24)
        self.assertEqual(request["method"], "HEAD")
        self.assertEqual(request["response_body_bytes_allowed"], 0)

    def test_plan_roundtrip_recomputes(self) -> None:
        self.assertEqual(cohort.validate_plan(self.plan, repo_root=self.root), self.plan)

    def test_plan_has_no_execution_authority(self) -> None:
        authority = self.plan["authority"]
        self.assertTrue(authority["metadata_selection"])
        self.assertFalse(any(value for key, value in authority.items() if key != "metadata_selection"))


if __name__ == "__main__":
    unittest.main()
