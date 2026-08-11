import unittest

from scripts.research.hftf.deployment.depthart.preflight_depthart_task_preserving_d2r1_assets import (
    ASSETS,
    disposition,
    requests_for,
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


class D2R1HeadPreflightTest(unittest.TestCase):
    def test_request_plan_is_exact(self) -> None:
        rows = requests_for(fixture_manifest(), "https://example.invalid/raw")
        self.assertEqual(64, len(rows))
        self.assertEqual(list(ASSETS), [row["asset"] for row in rows[:4]])

    def test_disposition_enforces_availability_and_bound(self) -> None:
        rows = [{"http_status": 200, "content_length_bytes": 1} for _ in range(64)]
        self.assertEqual("D2R1_ASSET_HEADERS_AVAILABLE_BODY_UNOPENED", disposition(rows, 64, 64))
        self.assertEqual(
            "D2R1_BODY_SIZE_EXCEEDS_AUTHORIZED_BOUND_BODY_UNOPENED", disposition(rows, 65, 64)
        )
        rows[0]["http_status"] = 404
        self.assertEqual("D2R1_ASSETS_NOT_AVAILABLE_BODY_UNOPENED", disposition(rows, 64, 64))

    def test_duplicate_identity_fails(self) -> None:
        manifest = fixture_manifest()
        manifest["selected_phase_b"][1]["video_id"] = manifest["selected_phase_b"][0]["video_id"]
        with self.assertRaisesRegex(ValueError, "video overlap"):
            requests_for(manifest, "https://example.invalid/raw")


if __name__ == "__main__":
    unittest.main()
