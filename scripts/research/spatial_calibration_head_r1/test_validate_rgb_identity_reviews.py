#!/usr/bin/env python3

import unittest

from validate_rgb_identity_reviews import adjudicate


class IdentityReviewValidationTest(unittest.TestCase):
    def review(self, reviewer: str, labels: list[str]) -> dict:
        return {
            "schema": "blindassist_spatial_calibration_head_r1_rgb_identity_review",
            "candidate_sha256": "hash",
            "reviewer_id": reviewer,
            "edges": [{"edge_id": f"RGBID-{index:06d}", "label": label} for index, label in enumerate(labels)],
        }

    def test_two_complete_distinct_reviews_admit(self) -> None:
        candidates = {"candidate_edge_count": 2, "edges": [{"edge_id": "RGBID-000000"}, {"edge_id": "RGBID-000001"}]}
        result = adjudicate(candidates, self.review("a", ["DISTINCT_CAPTURE", "DISTINCT_CAPTURE"]), self.review("b", ["DISTINCT_CAPTURE", "DISTINCT_CAPTURE"]), "hash")
        self.assertEqual(result["terminal"], "SPATIAL_CALIBRATION_HEAD_R1_RGB_IDENTITY_REVIEW_ADMITTED")

    def test_unknown_disagreement_or_missing_holds(self) -> None:
        candidates = {"candidate_edge_count": 2, "edges": [{"edge_id": "RGBID-000000"}, {"edge_id": "RGBID-000001"}]}
        result = adjudicate(candidates, self.review("a", ["DISTINCT_CAPTURE", "UNKNOWN"]), self.review("b", ["SAME_CAPTURE"]), "hash")
        self.assertEqual(result["terminal"], "HOLD_COHORT_INDEPENDENCE")
        self.assertFalse(result["complete"])

    def test_same_reviewer_is_rejected(self) -> None:
        candidates = {"candidate_edge_count": 1, "edges": [{"edge_id": "RGBID-000000"}]}
        with self.assertRaises(ValueError):
            adjudicate(candidates, self.review("a", ["DISTINCT_CAPTURE"]), self.review("a", ["DISTINCT_CAPTURE"]), "hash")


if __name__ == "__main__":
    unittest.main()
