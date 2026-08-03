#!/usr/bin/env python3

import json
import unittest

from preflight_asset_headers import cohort_header_requests, disposition
from validate_protocol import REPO_ROOT


class AssetHeaderPreflightTest(unittest.TestCase):
    def test_development_assets_and_sealed_rgb_only_are_requested(self) -> None:
        path = REPO_ROOT / "docs/research/hftf/SPATIAL_CALIBRATION_HEAD_R1_COHORT_ROSTER_LOCK_2026-08-04.json"
        lock = json.loads(path.read_text(encoding="utf-8"))
        rows = cohort_header_requests(lock)
        self.assertEqual(len(rows), 104)
        self.assertEqual(len({row["video_id"] for row in rows}), 24)
        sealed = {row["video_id"] for row in lock["roles"]["sealed"]}
        sealed_rows = [row for row in rows if row["video_id"] in sealed]
        self.assertEqual(len(sealed_rows), 4)
        self.assertTrue(all(row["asset"] == "lowres_wide.zip" for row in sealed_rows))
        self.assertTrue(all("/Validation/" in row["url"] for row in sealed_rows))

    def test_transport_error_is_not_misclassified_as_missing_asset(self) -> None:
        self.assertEqual(
            disposition([{"http_status": None, "content_length_bytes": None}]),
            "SPATIAL_CALIBRATION_HEAD_R1_ASSET_HEADER_PREFLIGHT_INCOMPLETE",
        )
        self.assertEqual(
            disposition([{"http_status": 404, "content_length_bytes": 0}]),
            "SPATIAL_CALIBRATION_HEAD_R1_COHORT_NOT_EVALUABLE_NO_REPLACEMENT",
        )


if __name__ == "__main__":
    unittest.main()
