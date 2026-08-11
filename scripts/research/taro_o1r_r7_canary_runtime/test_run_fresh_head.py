from __future__ import annotations

import unittest
from pathlib import Path

from scripts.research.taro_o1r_r7_canary_runtime import fresh_confirmation_cohort as cohort
from scripts.research.taro_o1r_r7_canary_runtime import run_fresh_head as runner


class FreshHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = cohort.build_plan(Path(__file__).resolve().parents[3])

    def test_all_bound_headers_pass_without_body(self) -> None:
        receipt = runner.build_head_receipt(
            self.plan,
            execution_lock_sha256="A" * 64,
            data_lock_sha256="B" * 64,
            head_fn=lambda request, timeout: {"http_status": 200, "content_length_bytes": 1024, "etag": None, "last_modified": None, "redirect_chain": [], "transport_errors": []},
            timeout_seconds=1.0,
            maximum_attempts=2,
            maximum_compressed_source_bytes=100000,
        )
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["response_body_bytes_read"], 0)
        self.assertEqual(receipt["terminal"], runner.PASS_TERMINAL)

    def test_unavailable_header_fails_without_replacement(self) -> None:
        missing_video = self.plan["selection"]["roster"][0]["video_id"]

        def fake(request, timeout):
            status = 403 if request["video_id"] == missing_video and request["asset"] == "lowres_wide.traj" else 200
            return {"http_status": status, "content_length_bytes": None if status != 200 else 1024, "etag": None, "last_modified": None, "redirect_chain": [], "transport_errors": []}

        receipt = runner.build_head_receipt(
            self.plan,
            execution_lock_sha256="A" * 64,
            data_lock_sha256="B" * 64,
            head_fn=fake,
            timeout_seconds=1.0,
            maximum_attempts=2,
            maximum_compressed_source_bytes=100000,
        )
        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["available_asset_count"], 23)
        self.assertEqual(receipt["terminal"], runner.UNAVAILABLE_TERMINAL)
        self.assertFalse(receipt["replacement_allowed"])


if __name__ == "__main__":
    unittest.main()
