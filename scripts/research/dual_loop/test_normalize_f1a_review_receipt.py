import unittest

from normalize_f1a_review_receipt import normalize_review


class NormalizeF1aReviewReceiptTest(unittest.TestCase):
    def test_normalizes_aliases_without_changing_label_values(self) -> None:
        result = normalize_review(
            {
                "reviewer": "a",
                "negative_windows": [
                    {
                        "negative_category": "STATIC_SCENE",
                        "interval": {"start": 1.0, "end": 3.0},
                    }
                ],
                "timeline_coverage": {
                    "input_id": "x",
                    "complete_sampled_timeline_reviewed": True,
                    "contact_sheets_reviewed": [{"path": "sheet.jpg"}],
                    "dense_frames_reviewed": [{"path": "dense.jpg"}],
                },
            },
            "a" * 64,
        )
        self.assertEqual("a", result["reviewer_id"])
        self.assertEqual(
            "STATIC_SCENE", result["negative_windows"][0]["negative_type"]
        )
        self.assertEqual(
            {"start": 1.0, "end": 3.0},
            result["negative_windows"][0]["window_interval_seconds"],
        )
        self.assertTrue(result["timeline_coverage"][0]["full_timeline_coverage"])
        self.assertEqual(
            ["sheet.jpg"],
            result["timeline_coverage"][0]["contact_sheets_reviewed"],
        )


if __name__ == "__main__":
    unittest.main()
