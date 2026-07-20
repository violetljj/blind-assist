import unittest

import extract_public_video_explicit_turn_review_material as subject


class ExplicitTurnReviewExtractionTest(unittest.TestCase):
    def test_padding_clamps_at_video_start(self) -> None:
        self.assertEqual((0, 5000), subject.padded_window([1000, 2000], 3000))

    def test_padding_extends_both_sides(self) -> None:
        self.assertEqual((7000, 15000), subject.padded_window([10000, 12000], 3000))

    def test_selects_top_candidate_per_direction_and_source(self) -> None:
        candidates = [
            {"candidate_id": "a", "direction": "LEFT", "parent_source_id": "s", "run_length": 2,
             "mean_absolute_median_dx_norm": 0.2},
            {"candidate_id": "b", "direction": "LEFT", "parent_source_id": "s", "run_length": 3,
             "mean_absolute_median_dx_norm": 0.1},
            {"candidate_id": "c", "direction": "RIGHT", "parent_source_id": "s", "run_length": 2,
             "mean_absolute_median_dx_norm": 0.3},
        ]
        selected = subject.select_candidates(candidates, {
            "mode": "top_per_direction_per_source", "maximum_per_direction_per_source": 1
        })
        self.assertEqual({"b", "c"}, {row["candidate_id"] for row in selected})

    def test_selects_explicit_candidate_ids_in_requested_order(self) -> None:
        candidates = [{"candidate_id": "a"}, {"candidate_id": "b"}]
        selected = subject.select_candidates(candidates, {
            "mode": "explicit_candidate_ids", "candidate_ids": ["b", "a"]
        })
        self.assertEqual(["b", "a"], [row["candidate_id"] for row in selected])


if __name__ == "__main__":
    unittest.main()
