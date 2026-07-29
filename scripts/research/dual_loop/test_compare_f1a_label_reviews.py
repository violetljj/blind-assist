import unittest

from compare_f1a_label_reviews import compare_reviews


def positive(item_id: str, start: float, region: str = "CENTER") -> dict:
    return {
        "item_id": item_id,
        "input_id": "input",
        "event_type": "CAMERA_APPROACH_STATIC_OBSTACLE",
        "onset_interval_seconds": {"start": start, "end": start + 0.5},
        "alertable_start_interval_seconds": {"start": start + 1.0, "end": start + 1.5},
        "end_or_clear_interval_seconds": {"start": start + 3.0, "end": start + 3.5},
        "region": region,
    }


def negative(item_id: str, start: float) -> dict:
    return {
        "item_id": item_id,
        "input_id": "input",
        "negative_type": "STATIC_SCENE",
        "window_interval_seconds": {"start": start, "end": start + 4.0},
    }


class CompareF1aLabelReviewsTest(unittest.TestCase):
    def test_matches_close_events_and_overlapping_windows(self) -> None:
        result = compare_reviews(
            {"events": [positive("a", 10.0)], "negative_windows": [negative("na", 20.0)]},
            {"events": [positive("b", 10.5)], "negative_windows": [negative("nb", 21.0)]},
        )
        self.assertEqual("MODEL_CONSENSUS", result["status"])
        self.assertEqual(2, len(result["agreements"]))

    def test_routes_unmatched_and_region_disagreement(self) -> None:
        result = compare_reviews(
            {"events": [positive("a", 10.0, "LEFT")], "negative_windows": []},
            {
                "events": [
                    positive("b", 10.5, "RIGHT"),
                    positive("c", 30.0, "CENTER"),
                ],
                "negative_windows": [],
            },
        )
        self.assertEqual("INDEPENDENT_AI_ADJUDICATION_REQUIRED", result["status"])
        self.assertEqual(
            {"POSITIVE_REGION_DISAGREEMENT", "POSITIVE_B_ONLY"},
            {item["kind"] for item in result["disagreements"]},
        )


if __name__ == "__main__":
    unittest.main()
