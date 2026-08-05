#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("download_p3_r0_2_1_arkit_validation_assets.py")
SPEC = importlib.util.spec_from_file_location("download_validation", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DownloadValidationTest(unittest.TestCase):
    def test_earliest_common_stems_are_contiguous_and_deterministic(self) -> None:
        stems = [f"1_{1 + index * 0.1:.1f}" for index in range(5)]
        maps = {name: {stem: f"{name}/{stem}.png" for stem in stems} for name in ("rgb", "depth", "confidence")}
        self.assertEqual(stems[:4], MODULE.earliest_common_stems(maps, 4))

    def test_gap_fails_closed(self) -> None:
        stems = ["1_1.0", "1_1.1", "1_2.0", "1_2.1"]
        maps = {name: {stem: stem for stem in stems} for name in ("rgb", "depth", "confidence")}
        with self.assertRaisesRegex(ValueError, "500 ms gap"):
            MODULE.earliest_common_stems(maps, 4)

    def test_unavailable_preflight_asset_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "asset unavailable"):
            MODULE.lookup_preflight({"assets": [{"video_id": "1", "asset": "x", "http_status": 404, "content_length_bytes": 1}]})


if __name__ == "__main__":
    unittest.main()
