import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_public_video_marker_radial_approach_probe as subject


POLICY = {"target_classes": ["barricade", "traffic cone"]}


def sample(t, bottom, center, area):
    return {"timestamp_ms": t, "detections": [{"class_name": "traffic cone", "features": {
        "high_saturation_fraction": 0.8, "dark_fraction": 0.1,
        "bottom_y_norm": bottom, "center_x_norm": center, "area_ratio": area,
    }}]}


class MarkerRadialApproachProbeTest(unittest.TestCase):
    def test_radial_growth_passes(self):
        rows = [sample(i * 1000, 0.2 + i * 0.06, 0.55 + i * 0.01, 0.001 * (i + 1)) for i in range(7)]
        result = subject.event_diagnostics(rows, {"event_entry_timestamp_ms": 0, "last_active_timestamp_ms": 6000}, POLICY)
        self.assertTrue(result["radial_approach_passed"])

    def test_lateral_sweep_fails(self):
        rows = [sample(i * 1000, 0.55 + i * 0.002, 0.2 + i * 0.1, 0.001 * (1 + i * 0.1)) for i in range(7)]
        result = subject.event_diagnostics(rows, {"event_entry_timestamp_ms": 0, "last_active_timestamp_ms": 6000}, POLICY)
        self.assertFalse(result["radial_approach_passed"])

    def test_too_few_samples_fails(self):
        rows = [sample(i * 1000, 0.2 + i * 0.1, 0.5, 0.001 * (i + 1)) for i in range(3)]
        result = subject.event_diagnostics(rows, {"event_entry_timestamp_ms": 0, "last_active_timestamp_ms": 2000}, POLICY)
        self.assertFalse(result["radial_approach_passed"])


if __name__ == "__main__":
    unittest.main()
