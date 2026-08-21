from __future__ import annotations

import unittest

import numpy as np

import run_p1_a4_online_correspondence as a4


class AggregationTest(unittest.TestCase):
    def test_identity_affine_preserves_bbox(self):
        _, points, _, cells, scale_x, scale_y = a4._query_geometry(
            512, 384, [100.0, 80.0, 300.0, 280.0]
        )
        candidate, bbox, diagnostic = a4.aggregate_points(
            points,
            points.copy(),
            np.full(25, 0.9, dtype="float32"),
            np.ones(25, dtype=bool),
            cells,
            [100 * scale_x, 80 * scale_y, 300 * scale_x, 280 * scale_y],
            512,
            384,
            scale_x,
            scale_y,
        )
        self.assertIsNotNone(candidate)
        np.testing.assert_allclose(bbox, [100, 80, 300, 280], atol=1e-3)
        self.assertEqual(diagnostic["coarse_coverage"], 9)
        self.assertAlmostEqual(candidate["identity_support"], 0.9, places=5)

    def test_visibility_floor_returns_null(self):
        _, points, _, cells, scale_x, scale_y = a4._query_geometry(
            512, 384, [100.0, 80.0, 300.0, 280.0]
        )
        candidate, bbox, diagnostic = a4.aggregate_points(
            points,
            points,
            np.ones(25, dtype="float32"),
            np.arange(25) < 5,
            cells,
            [50, 53, 150, 187],
            512,
            384,
            scale_x,
            scale_y,
        )
        self.assertIsNone(candidate)
        self.assertIsNone(bbox)
        self.assertEqual(diagnostic["rejection"], "VISIBLE_LT_6")


class TerminalTest(unittest.TestCase):
    def fixture(self):
        result = {
            "evaluation": {"aggregate": {
                "correct_identity_coverage": {"numerator": 120},
                "wrong_instance_asserted_frames": 0,
                "identity_switches": 0,
                "wrong_lock_persistence_max_duration_ms": 0,
                "false_reacquisitions": 0,
                "false_loss_frames": 100,
                "state_expectation_violations": 0,
                "event_expectation_violations": 0,
            }},
            "post_outcome_descriptive_failure_attribution": {"wrong_background_asserted_frames": 0},
            "by_temporal_mode": {
                "TEMP_OCCLUSION": {"recovery_successes": 1},
                "OUT_OF_VIEW_RETURN": {"recovery_successes": 0},
            },
        }
        prediction = {"authority_receipt": {
            "future_frame_reads": 0,
            "gt_oracle_resets": 0,
            "object_uid_or_visibility_gt_reads": 0,
            "semantic_detector_reid_vlm_reads": 0,
            "global_target_searches": 0,
            "online_query_feature_replacements": 0,
            "causal_violations": 0,
            "frames": 1724,
            "point_rows": 43100,
        }}
        return result, prediction

    def test_full_gate_establishes_signal(self):
        result, prediction = self.fixture()
        self.assertEqual(
            a4._gate_result(result, prediction)[0],
            "STRONG_TEMPORAL_CORRESPONDENCE_SIGNAL_ESTABLISHED",
        )

    def test_future_read_is_not_evaluable(self):
        result, prediction = self.fixture()
        prediction["authority_receipt"]["future_frame_reads"] = 1
        self.assertEqual(a4._gate_result(result, prediction)[0], "NOT_EVALUABLE_ONLINE_INTERFACE")

    def test_coverage_gain_with_safety_failure(self):
        result, prediction = self.fixture()
        result["evaluation"]["aggregate"]["wrong_instance_asserted_frames"] = 446
        self.assertEqual(
            a4._gate_result(result, prediction)[0],
            "CORRESPONDENCE_COVERAGE_GAIN_WITH_IDENTITY_SAFETY_FAILURE",
        )


if __name__ == "__main__":
    unittest.main()
