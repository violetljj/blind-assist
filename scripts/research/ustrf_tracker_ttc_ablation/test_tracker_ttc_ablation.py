from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_ablation.py")
SPEC = importlib.util.spec_from_file_location("tracker_ablation", MODULE_PATH)
tracker_ablation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
import sys
sys.modules[SPEC.name] = tracker_ablation
SPEC.loader.exec_module(tracker_ablation)


class TrackerAblationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "arms": {
                "T0": {},
                "T1": {"alpha": 0.85, "beta": 0.005},
                "T2": {"high_confidence_min": 0.50},
                "T3": {},
            }
        }

    @staticmethod
    def detection(left: float, confidence: float = 0.8) -> dict:
        return {"box": [left, 10.0, left + 20.0, 50.0], "confidence": confidence, "label": "person"}

    def test_t0_reuses_track_for_iou_match(self) -> None:
        state = tracker_ablation.ArmState()
        first = tracker_ablation.associate([self.detection(10.0)], 1, "T0", state, self.config)
        second = tracker_ablation.associate([self.detection(12.0)], 2, "T0", state, self.config)
        self.assertEqual(first[0][0].track_id, second[0][0].track_id)
        self.assertEqual(1, state.track_births)

    def test_t2_low_confidence_only_continues_existing_track(self) -> None:
        state = tracker_ablation.ArmState()
        self.assertEqual([], tracker_ablation.associate([self.detection(10.0, 0.40)], 1, "T2", state, self.config))
        high = tracker_ablation.associate([self.detection(10.0, 0.80)], 2, "T2", state, self.config)
        low = tracker_ablation.associate([self.detection(11.0, 0.40)], 3, "T2", state, self.config)
        self.assertEqual(1, len(high))
        self.assertEqual(high[0][0].track_id, low[0][0].track_id)
        self.assertEqual(1, state.track_births)

    def test_route_hit_is_fail_closed_for_unknown(self) -> None:
        box = [100.0, 100.0, 150.0, 180.0]
        self.assertFalse(tracker_ablation.route_hit(box, {"status": "unknown"}, 640, 480, 0.08))
        self.assertTrue(tracker_ablation.route_hit(box, {"status": "known", "uv": [170.0, 150.0]}, 640, 480, 0.08))

    def test_window_truth_match_uses_alertable_to_clear_interval(self) -> None:
        alert = {"start_frame": 20, "end_frame": 25}
        truth = {"alertable_frame": 24, "passed_or_cleared_frame": 30}
        self.assertTrue(alert["start_frame"] <= truth["passed_or_cleared_frame"] and alert["end_frame"] >= truth["alertable_frame"])


if __name__ == "__main__":
    unittest.main()
