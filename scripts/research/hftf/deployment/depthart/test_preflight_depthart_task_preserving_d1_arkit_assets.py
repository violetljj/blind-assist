from __future__ import annotations

import unittest

from scripts.research.hftf.deployment.depthart.preflight_depthart_task_preserving_d1_arkit_assets import (
    ASSETS,
    disposition,
    requests_for,
    roster_rows,
)


def fixture_roster() -> dict:
    return {
        "primary": [
            {"visit_id": f"p{i}", "video_id": f"pv{i}", "fold": "Training"}
            for i in range(8)
        ],
        "reserve": [
            {"visit_id": f"r{i}", "video_id": f"rv{i}", "fold": "Training"}
            for i in range(8)
        ],
    }


class DepthArtD1ArkitPreflightTest(unittest.TestCase):
    def test_roster_and_request_order_are_frozen(self) -> None:
        rows = roster_rows(fixture_roster())
        self.assertEqual(16, len(rows))
        self.assertEqual(("PRIMARY", 1, "pv0"), (rows[0]["role"], rows[0]["frozen_order"], rows[0]["video_id"]))
        self.assertEqual(("RESERVE", 8, "rv7"), (rows[-1]["role"], rows[-1]["frozen_order"], rows[-1]["video_id"]))
        requests = requests_for(fixture_roster(), "https://example.invalid/raw")
        self.assertEqual(80, len(requests))
        self.assertEqual(list(ASSETS), [row["asset"] for row in requests[:5]])
        self.assertEqual(
            "https://example.invalid/raw/Training/pv0/lowres_wide.zip",
            requests[0]["url"],
        )

    def test_disposition_requires_every_asset(self) -> None:
        rows = [{"http_status": 200, "content_length_bytes": 1} for _ in range(80)]
        self.assertEqual("D1_ARKIT_ASSET_HEADERS_AVAILABLE_MEDIA_BODY_UNOPENED", disposition(rows))
        rows[0]["http_status"] = 404
        self.assertEqual("D1_ARKIT_ASSETS_NOT_AVAILABLE_MEDIA_BODY_UNOPENED", disposition(rows))
        rows[0]["http_status"] = None
        self.assertEqual("D1_ARKIT_ASSET_HEADER_PREFLIGHT_INCOMPLETE_MEDIA_BODY_UNOPENED", disposition(rows))

    def test_duplicate_identity_fails_closed(self) -> None:
        roster = fixture_roster()
        roster["reserve"][0]["visit_id"] = roster["primary"][0]["visit_id"]
        with self.assertRaisesRegex(ValueError, "visit overlap"):
            roster_rows(roster)


if __name__ == "__main__":
    unittest.main()
