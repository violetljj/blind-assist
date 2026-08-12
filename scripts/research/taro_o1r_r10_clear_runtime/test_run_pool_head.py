from __future__ import annotations

import unittest
from pathlib import Path

from scripts.research.taro_o1r_r10_clear_runtime import fresh_pool
from scripts.research.taro_o1r_r10_clear_runtime import run_pool_head as runner


class PoolHeadTests(unittest.TestCase):
    def test_zero_body_success_receipt(self) -> None:
        plan = fresh_pool.build_pool(Path(__file__).resolve().parents[3])
        receipt = runner.build_head_receipt(plan, lock_sha256="A" * 64, protocol_sha256="B" * 64, head_fn=lambda _request, _timeout: {"http_status": 200, "content_length_bytes": 1, "etag": None, "last_modified": None, "redirect_chain": [], "transport_errors": []}, timeout_seconds=1, maximum_attempts=2, maximum_bytes=1000)
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["response_body_bytes_read"], 0)
        self.assertEqual(receipt["asset_count"], 96)


if __name__ == "__main__":
    unittest.main()
