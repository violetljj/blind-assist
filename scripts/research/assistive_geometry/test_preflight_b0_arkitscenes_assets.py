import unittest

from scripts.research.assistive_geometry.preflight_b0_arkitscenes_assets import (
    disposition,
    requests_for,
)


class PreflightB0ArkitScenesAssetsTest(unittest.TestCase):
    def test_generates_five_fold_aware_urls_per_parent(self) -> None:
        roster = {
            "roles": {
                "TRAIN": [
                    {"visit_id": "v", "video_id": "1", "official_fold": "Training"}
                ],
                "CONFIRMATION": [
                    {"visit_id": "w", "video_id": "2", "official_fold": "Validation"}
                ],
            }
        }
        rows = requests_for(roster, "https://example.test/raw")
        self.assertEqual(10, len(rows))
        self.assertEqual(5, sum("/Training/1/" in row["url"] for row in rows))
        self.assertEqual(5, sum("/Validation/2/" in row["url"] for row in rows))

    def test_disposition_fails_closed(self) -> None:
        self.assertIn("INCOMPLETE", disposition([{"http_status": None, "content_length_bytes": None}]))
        self.assertIn("NOT_AVAILABLE", disposition([{"http_status": 404, "content_length_bytes": 1}]))
        self.assertTrue(
            disposition([{"http_status": 200, "content_length_bytes": 1}]).endswith(
                "AVAILABLE_MEDIA_UNOPENED"
            )
        )


if __name__ == "__main__":
    unittest.main()
