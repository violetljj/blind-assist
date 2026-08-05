#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("preflight_p3_r0_2_1_arkit_validation_assets.py")
SPEC = importlib.util.spec_from_file_location("asset_preflight", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AssetPreflightTest(unittest.TestCase):
    def test_generates_five_validation_urls_per_parent(self) -> None:
        roster = {"selected": [{"visit_id": "v", "video_id": "1"}, {"visit_id": "w", "video_id": "2"}]}
        rows = MODULE.requests_for(roster, "https://example.test/raw")
        self.assertEqual(10, len(rows))
        self.assertTrue(all("/Validation/" in row["url"] for row in rows))

    def test_disposition_fails_closed(self) -> None:
        self.assertIn("INCOMPLETE", MODULE.disposition([{"http_status": None, "content_length_bytes": None}]))
        self.assertIn("NOT_AVAILABLE", MODULE.disposition([{"http_status": 404, "content_length_bytes": 1}]))
        self.assertTrue(MODULE.disposition([{"http_status": 200, "content_length_bytes": 1}]).endswith("AVAILABLE_MEDIA_UNOPENED"))


if __name__ == "__main__":
    unittest.main()
