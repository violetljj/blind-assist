import unittest

from scripts.research.hftf.deployment.depthart.preflight_depthart_task_preserving_d2_phase_b_assets import (
    ASSETS,
    disposition,
    requests_for,
    selected_rows,
)


def fixture_manifest() -> dict:
    return {
        "selected_phase_b": [
            {
                "pool_order": index + 1,
                "visit_id": f"visit{index}",
                "video_id": f"video{index}",
                "fold": "Training",
            }
            for index in range(16)
        ]
    }


class D2PhaseBHeadPreflightTest(unittest.TestCase):
    def test_request_plan_is_exact(self) -> None:
        self.assertEqual(16, len(selected_rows(fixture_manifest())))
        requests = requests_for(fixture_manifest(), "https://example.invalid/raw")
        self.assertEqual(32, len(requests))
        self.assertEqual(list(ASSETS), [row["asset"] for row in requests[:2]])

    def test_disposition_fails_closed(self) -> None:
        rows = [{"http_status": 200, "content_length_bytes": 1} for _ in range(32)]
        self.assertEqual("D2_PHASE_B_ASSET_HEADERS_AVAILABLE_MEDIA_BODY_UNOPENED", disposition(rows))
        rows[0]["http_status"] = 404
        self.assertEqual("D2_PHASE_B_ASSETS_NOT_AVAILABLE_MEDIA_BODY_UNOPENED", disposition(rows))
        rows[0]["http_status"] = None
        self.assertEqual("D2_PHASE_B_HEADER_PREFLIGHT_INCOMPLETE_MEDIA_BODY_UNOPENED", disposition(rows))

    def test_duplicate_parent_fails(self) -> None:
        manifest = fixture_manifest()
        manifest["selected_phase_b"][1]["visit_id"] = manifest["selected_phase_b"][0]["visit_id"]
        with self.assertRaisesRegex(ValueError, "visit overlap"):
            selected_rows(manifest)


if __name__ == "__main__":
    unittest.main()
