#!/usr/bin/env python3
"""Focused tests for the R6 zero-body HEAD receipt mechanics."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.research.taro_o0r_candidate_scale_runtime import r6_untouched_cohort as cohort
from scripts.research.taro_o0r_candidate_scale_runtime import run_r6_untouched_head as head


class R6UntouchedHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = cohort.build_plan(Path(__file__).resolve().parents[3])

    def build(self, fake):
        return head.build_head_receipt(
            self.plan,
            execution_lock_sha256="A" * 64,
            data_lock_sha256="B" * 64,
            head_fn=fake,
            timeout_seconds=1.0,
            maximum_attempts=2,
            maximum_compressed_source_bytes=2_147_483_648,
        )

    def test_all_exact_requests_pass_with_zero_body_bytes(self) -> None:
        receipt = self.build(lambda row, timeout: {"http_status": 200, "content_length_bytes": 100, "etag": "E", "last_modified": "L", "redirect_chain": [], "transport_errors": []})
        self.assertTrue(receipt["passed"])
        self.assertEqual(24, receipt["available_asset_count"])
        self.assertEqual(2400, receipt["total_content_length_bytes"])
        self.assertEqual(0, receipt["response_body_bytes_read"])

    def test_transport_failure_is_retried_and_retained_as_unavailable(self) -> None:
        calls = {}

        def fake(row, timeout):
            calls[row["url"]] = calls.get(row["url"], 0) + 1
            if row["asset"] == "lowres_wide.traj":
                raise RuntimeError("blocked")
            return {"http_status": 200, "content_length_bytes": 100, "etag": None, "last_modified": None, "redirect_chain": [], "transport_errors": []}

        receipt = self.build(fake)
        self.assertFalse(receipt["passed"])
        self.assertEqual(16, receipt["available_asset_count"])
        self.assertEqual(32, receipt["request_attempt_count"])
        self.assertTrue(all(row["attempt_count"] == 2 for row in receipt["assets"] if row["asset"] == "lowres_wide.traj"))

    def test_redirect_or_budget_overrun_cannot_pass(self) -> None:
        def fake(row, timeout):
            return {"http_status": 200, "content_length_bytes": 100, "etag": None, "last_modified": None, "redirect_chain": ["https://other.example"], "transport_errors": []}

        receipt = self.build(fake)
        self.assertFalse(receipt["passed"])
        self.assertEqual(0, receipt["available_asset_count"])


if __name__ == "__main__":
    unittest.main()
