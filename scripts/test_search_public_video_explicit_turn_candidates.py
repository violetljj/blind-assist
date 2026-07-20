import unittest

import search_public_video_explicit_turn_candidates as subject


class ExplicitTurnCandidateSearchTest(unittest.TestCase):
    def row(self, timestamp, x, hit, source="s"):
        return {"source_id": source, "timestamp_ms": timestamp,
                "future_route_anchors": [{"point_xy_norm": [x, 0.9]}] * 3,
                "teacher_marker_hit_fraction_diagnostic_only": hit, "item_id": f"i{timestamp}",
                "local_video_path": "v", "source_video_sha256": "h"}

    def test_find_runs_requires_same_direction_and_consecutive_time(self) -> None:
        spec = {"left_if_mean_anchor_x_below": 0.47, "right_if_mean_anchor_x_above": 0.53,
                "minimum_teacher_marker_hit_fraction": 1 / 3, "required_consecutive_samples": 2,
                "expected_step_ms": 1000, "exclude_r789_window_padding_ms": 0,
                "maximum_candidates_per_direction": 20}
        rows = [self.row(0, 0.4, 1.0), self.row(1000, 0.4, 1.0),
                self.row(2000, 0.6, 1.0), self.row(4000, 0.6, 1.0)]
        result = subject.find_runs(rows, spec, {})
        self.assertEqual(1, len(result))
        self.assertEqual("LEFT", result[0]["direction"])

    def test_excluded_window_padding_rejects_overlap(self) -> None:
        self.assertTrue(subject.overlaps_excluded("s", 1000, 2000, {"s": [(2500, 3000)]}, 1000))


if __name__ == "__main__":
    unittest.main()
