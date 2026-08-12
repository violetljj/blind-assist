import unittest

from scripts.research.hftf.deployment.depthart.preflight_depthart_task_preserving_d3_phase_a_assets import (
    ASSETS,
    disposition,
    requests_for,
    roster_rows,
)


def fixture_roster() -> dict:
    return {
        "pool": [
            {
                "pool_order": index + 1,
                "visit_id": f"visit{index}",
                "video_id": f"video{index}",
                "fold": "Training",
                "role": "D3_METADATA_CANDIDATE_POOL_ONLY",
            }
            for index in range(48)
        ]
    }


class D3PhaseAHeadPreflightTest(unittest.TestCase):
    def test_request_plan_is_exact(self) -> None:
        rows = roster_rows(fixture_roster())
        self.assertEqual(48, len(rows))
        requests = requests_for(fixture_roster(), "https://example.invalid/raw")
        self.assertEqual(96, len(requests))
        self.assertEqual(list(ASSETS), [row["asset"] for row in requests[:2]])
        self.assertEqual(
            "https://example.invalid/raw/Training/video0/lowres_wide_intrinsics.zip",
            requests[0]["url"],
        )

    def test_disposition_fails_closed(self) -> None:
        rows = [{"http_status": 200, "content_length_bytes": 1} for _ in range(96)]
        self.assertEqual(
            "D3_PHASE_A_ASSET_HEADERS_96_OF_96_AVAILABLE_MEDIA_BODY_UNOPENED",
            disposition(rows),
        )
        rows[0]["http_status"] = 404
        self.assertEqual("D3_PHASE_A_ASSETS_NOT_AVAILABLE_MEDIA_BODY_UNOPENED", disposition(rows))
        rows[0]["http_status"] = None
        self.assertEqual(
            "D3_PHASE_A_HEADER_PREFLIGHT_INCOMPLETE_MEDIA_BODY_UNOPENED",
            disposition(rows),
        )

    def test_pool_order_drift_fails(self) -> None:
        roster = fixture_roster()
        roster["pool"][0]["pool_order"] = 2
        with self.assertRaisesRegex(ValueError, "pool order drift"):
            roster_rows(roster)

    def test_role_drift_fails(self) -> None:
        roster = fixture_roster()
        roster["pool"][0]["role"] = "D3_TRAIN"
        with self.assertRaisesRegex(ValueError, "role drift"):
            roster_rows(roster)


if __name__ == "__main__":
    unittest.main()
