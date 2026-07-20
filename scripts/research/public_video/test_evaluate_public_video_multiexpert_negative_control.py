import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_public_video_multiexpert_negative_control as subject


class MultiExpertNegativeControlTest(unittest.TestCase):
    def test_chromatic_event_causes_negative_control_failure(self) -> None:
        dino = [{"timestamp_ms": t, "vector": [0.0, 0.0]} for t in range(0, 15000, 1000)]
        chromatic = []
        for t in range(0, 15000, 1000):
            detections = []
            if 5000 <= t < 9000:
                detections = [{"class_name": "traffic cone", "features": {"high_saturation_fraction": 0.8, "dark_fraction": 0.1}}]
            chromatic.append({"timestamp_ms": t, "detections": detections})
        contract = {
            "risk_evidence_policy": {
                "policy_id": "chromatic_construction_marker_v1",
                "target_classes": ["barricade", "traffic cone"],
                "detection_acceptance": "high_saturation_fraction > dark_fraction",
                "minimum_accepted_detections_per_active_frame": 1,
                "absolute_color_threshold_used": False,
                "geometry_gate_used": False,
            },
            "lifecycle": {"selected_groups": ["barrier_structure"], "entry_window_samples": 3, "entry_min_active_samples": 2, "clear_absent_samples": 5},
        }
        result = subject.evaluate_channels(
            dino_samples=dino,
            dino_direction=np.asarray([1.0, 0.0]),
            chromatic_samples=chromatic,
            chromatic_contract=contract,
            windows={"pre_clear": [0, 4000], "negative_challenge": [5000, 9000], "post_clear": [10000, 14000]},
            minimum_samples=3,
        )
        self.assertEqual(["chromatic_construction_marker"], result["fusion"]["positive_channels"])
        self.assertFalse(result["fusion"]["negative_control_passed"])

    def test_no_channel_open_passes(self) -> None:
        rows = [{"timestamp_ms": t, "vector": [0.0], "detections": []} for t in range(0, 15000, 1000)]
        contract = {
            "risk_evidence_policy": {
                "policy_id": "chromatic_construction_marker_v1", "target_classes": ["barricade", "traffic cone"],
                "detection_acceptance": "high_saturation_fraction > dark_fraction", "minimum_accepted_detections_per_active_frame": 1,
                "absolute_color_threshold_used": False, "geometry_gate_used": False,
            },
            "lifecycle": {"selected_groups": ["barrier_structure"], "entry_window_samples": 3, "entry_min_active_samples": 2, "clear_absent_samples": 5},
        }
        result = subject.evaluate_channels(
            dino_samples=rows, dino_direction=np.asarray([1.0]), chromatic_samples=rows,
            chromatic_contract=contract, windows={"pre_clear": [0, 4000], "negative_challenge": [5000, 9000], "post_clear": [10000, 14000]}, minimum_samples=3,
        )
        self.assertTrue(result["fusion"]["negative_control_passed"])


if __name__ == "__main__":
    unittest.main()
