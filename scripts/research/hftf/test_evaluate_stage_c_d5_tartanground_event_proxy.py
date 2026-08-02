import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_stage_c_d5_tartanground_event_proxy import (
    aggregate_trace_metrics,
    apply_decision_confirmation,
    causal_confirmation,
    decision_policy_spec,
    lane_truth_state,
    predict,
    raw_lane_active,
    trace_metrics,
)


class TartanGroundEventProxyTest(unittest.TestCase):
    def test_predict_rejects_unknown_input_arm_before_io(self):
        with self.assertRaisesRegex(ValueError, "Unknown input arm"):
            predict(
                [],
                Path("missing-checkpoint.pt"),
                Path("missing-pretrained.pt"),
                "cpu",
                input_arm="not-an-arm",
            )

    def test_lane_truth_requires_positive_or_complete_negative(self):
        self.assertTrue(
            lane_truth_state(
                np.asarray([0.0, 1.0, 0.0]),
                np.asarray([1, 1, 0]),
            )
        )
        self.assertFalse(
            lane_truth_state(
                np.asarray([0.0, 0.0, 0.0]),
                np.asarray([1, 1, 1]),
            )
        )
        self.assertIsNone(
            lane_truth_state(
                np.asarray([0.0, 0.0, 0.0]),
                np.asarray([1, 1, 0]),
            )
        )

    def test_trace_metrics_count_events_false_alerts_and_clearance(self):
        rows = [
            {"truth": False, "active": True},
            {"truth": True, "active": False},
            {"truth": True, "active": True},
            {"truth": False, "active": True},
            {"truth": False, "active": False},
            {"truth": None, "active": True},
            {"truth": True, "active": False},
        ]

        result = trace_metrics(rows)

        self.assertEqual(result["positive_event_count"], 2)
        self.assertEqual(result["hit_event_count"], 1)
        self.assertEqual(result["false_alert_event_count"], 2)
        self.assertEqual(result["clearance_eligible_event_count"], 1)
        self.assertEqual(result["cleared_event_count"], 1)
        self.assertEqual(result["clearance_delay_anchor_steps"], [2])

    def test_aggregate_uses_event_and_exposure_denominators(self):
        traces = [
            [
                {"truth": True, "active": True},
                {"truth": False, "active": False},
            ],
            [
                {"truth": True, "active": False},
                {"truth": False, "active": True},
            ],
        ]

        result = aggregate_trace_metrics(traces)

        self.assertEqual(result["positive_event_count"], 2)
        self.assertEqual(result["event_recall"], 0.5)
        self.assertEqual(result["false_active_lane_frame_rate"], 0.5)
        self.assertEqual(result["clearance_rate"], 0.5)

    def test_causal_confirmation_requires_consecutive_observations(self):
        values = [True, True, False, True, True, True]

        self.assertEqual(
            causal_confirmation(values, 2),
            [False, True, False, False, True, True],
        )
        self.assertEqual(
            causal_confirmation(values, 3),
            [False, False, False, False, False, True],
        )

    def test_height_selective_policy_separates_observability(self):
        risk = np.asarray([0.6, 0.2])
        unknown = np.asarray([0.1, 0.1])

        self.assertTrue(
            raw_lane_active(
                risk,
                unknown,
                "body",
                "height_temporal_selective_v0",
            )
        )
        self.assertFalse(
            raw_lane_active(
                risk,
                unknown,
                "head",
                "height_temporal_selective_v0",
            )
        )
        self.assertTrue(
            raw_lane_active(
                np.asarray([0.95, 0.2]),
                unknown,
                "head",
                "height_temporal_selective_v0",
            )
        )

    def test_v1_high_confidence_head_override_is_immediate(self):
        spec = decision_policy_spec("height_temporal_selective_v1")[
            "head"
        ]

        self.assertEqual(
            apply_decision_confirmation(
                [False, True, True],
                [True, False, False],
                spec,
            ),
            [True, False, True],
        )

    def test_v2_body_requires_spatial_support_or_override(self):
        policy = "height_spatiotemporal_selective_v2"
        unknown = np.zeros(6)

        self.assertFalse(
            raw_lane_active(
                np.asarray([0.6, 0.6, 0.1, 0.1, 0.1, 0.1]),
                unknown,
                "body",
                policy,
            )
        )
        self.assertTrue(
            raw_lane_active(
                np.asarray([0.6, 0.6, 0.6, 0.1, 0.1, 0.1]),
                unknown,
                "body",
                policy,
            )
        )
        self.assertTrue(
            raw_lane_active(
                np.asarray([0.85, 0.1, 0.1, 0.1, 0.1, 0.1]),
                unknown,
                "body",
                policy,
            )
        )


if __name__ == "__main__":
    unittest.main()
