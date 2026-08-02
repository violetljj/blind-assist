import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_stage_c_d5_tartanground_event_proxy import (
    aggregate_trace_metrics,
    lane_truth_state,
    trace_metrics,
)


class TartanGroundEventProxyTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
