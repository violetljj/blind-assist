import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_public_video_radial_lifecycle_gap_bridge_probe as subject


POLICY = {
    "policy_id": "chromatic_construction_marker_v1",
    "target_classes": ["barricade", "traffic cone"],
    "detection_acceptance": "high_saturation_fraction > dark_fraction",
    "minimum_accepted_detections_per_active_frame": 1,
    "absolute_color_threshold_used": False,
    "geometry_gate_used": False,
}


def sample(timestamp: int, active: bool) -> dict:
    detections = []
    if active:
        detections = [{
            "class_name": "traffic cone",
            "features": {"high_saturation_fraction": 0.8, "dark_fraction": 0.1},
        }]
    return {"timestamp_ms": timestamp, "detections": detections}


class RadialLifecycleGapBridgeProbeTest(unittest.TestCase):
    def test_short_gap_is_uncertain_and_does_not_realert(self) -> None:
        samples = [sample(i * 1000, i in {0, 1, 2, 8, 9}) for i in range(20)]
        candidates = [{
            "event_entry_timestamp_ms": 0,
            "last_active_timestamp_ms": 2_000,
            "radial_approach_passed": True,
        }]
        state = subject.radial_entry_lifecycle(samples, POLICY, candidates, clear_absent_samples=7)
        self.assertEqual([0], state["reminder_timestamps_ms"])
        self.assertEqual(16_000, state["intervals"][0]["confirmed_clear_timestamp_ms"])

    def test_too_short_persistence_false_clears_and_does_not_open_without_radial(self) -> None:
        samples = [sample(i * 1000, i in {0, 1, 2, 8, 9}) for i in range(15)]
        candidates = [{
            "event_entry_timestamp_ms": 0,
            "last_active_timestamp_ms": 2_000,
            "radial_approach_passed": True,
        }]
        state = subject.radial_entry_lifecycle(samples, POLICY, candidates, clear_absent_samples=5)
        self.assertEqual(1, len(state["intervals"]))
        self.assertEqual(7_000, state["intervals"][0]["confirmed_clear_timestamp_ms"])
        self.assertEqual([0], state["reminder_timestamps_ms"])


if __name__ == "__main__":
    unittest.main()
