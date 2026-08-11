#!/usr/bin/env python3
"""Focused non-network tests for the exact TARO R6 download plan."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.research.taro_o0r_candidate_scale_runtime import r6_untouched_cohort as cohort
from scripts.research.taro_o0r_candidate_scale_runtime import run_r6_untouched_download as download
from scripts.research.taro_o0r_candidate_scale_runtime import run_r6_untouched_head as head


class R6UntouchedDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[3]
        cls.plan = cohort.build_plan(cls.repo)

    def test_exact_download_paths_cover_all_24_head_requests(self) -> None:
        rows = download.expanded_download_plan(self.plan)
        self.assertEqual(24, len(rows))
        self.assertEqual(24, len({row["relative_path"] for row in rows}))
        self.assertEqual(self.plan["request_plan"]["requests"], [{key: row[key] for key in ("visit_id", "video_id", "asset", "url")} for row in rows])

    def test_bound_head_receipt_is_an_exact_zero_body_pass(self) -> None:
        value = json.loads((self.repo / download.EXPECTED_BINDING_PATHS["R6_HEAD_RECEIPT"]).read_text(encoding="utf-8"))
        receipt = head.validate_head_receipt(self.plan, value, maximum_attempts=2)
        self.assertTrue(receipt["passed"])
        self.assertEqual(24, receipt["available_asset_count"])
        self.assertEqual(0, receipt["response_body_bytes_read"])


if __name__ == "__main__":
    unittest.main()
