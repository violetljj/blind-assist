from __future__ import annotations

import unittest
from pathlib import Path

from scripts.research.taro_o1r_r8_clear_runtime import pool_cohort
from scripts.research.taro_o1r_r8_clear_runtime import run_pool_head as runner


class PoolHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = pool_cohort.build_pool(Path(__file__).resolve().parents[3])

    def test_all_72_headers_pass_without_body(self) -> None:
        receipt = runner.build_head_receipt(self.plan, execution_lock_sha256="A" * 64, protocol_sha256="B" * 64, head_fn=lambda request, timeout: {"http_status": 200, "content_length_bytes": 1024, "etag": None, "last_modified": None, "redirect_chain": [], "transport_errors": []}, timeout_seconds=1.0, maximum_attempts=2, maximum_compressed_source_bytes=100000)
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["available_asset_count"], 72)
        self.assertEqual(receipt["response_body_bytes_read"], 0)

    def test_one_unavailable_asset_fails_without_replacement(self) -> None:
        missing = self.plan["request_plan"]["requests"][0]["url"]

        def fake(request, timeout):
            status = 403 if request["url"] == missing else 200
            return {"http_status": status, "content_length_bytes": None if status != 200 else 1024, "etag": None, "last_modified": None, "redirect_chain": [], "transport_errors": []}

        receipt = runner.build_head_receipt(self.plan, execution_lock_sha256="A" * 64, protocol_sha256="B" * 64, head_fn=fake, timeout_seconds=1.0, maximum_attempts=2, maximum_compressed_source_bytes=100000)
        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["available_asset_count"], 71)
        self.assertFalse(receipt["replacement_allowed"])

    def test_draft_execution_lock_cannot_run_without_user_authority(self) -> None:
        lock = Path(__file__).resolve().parents[3] / "docs/research/taro/TARO_O1R_R8_CLEAR_NEGATIVE_CONTROL_POOL_HEAD_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json"
        with self.assertRaises(runner.PoolHeadError) as caught:
            runner.validate_execution_lock(lock)
        self.assertEqual(caught.exception.code, "R8_HEAD_LOCK_IDENTITY")


if __name__ == "__main__":
    unittest.main()
