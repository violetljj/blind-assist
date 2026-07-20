import unittest

import run_public_video_marker_local_approach_probe as probe


def sample(timestamp_ms: int, bottom: float, center: float, area: float) -> dict:
    return {
        "timestamp_ms": timestamp_ms,
        "detections": [{
            "class_name": "traffic cone",
            "features": {
                "high_saturation_fraction": 0.8,
                "dark_fraction": 0.1,
                "bottom_y_norm": bottom,
                "center_x_norm": center,
                "area_ratio": area,
            },
        }],
    }


class LocalApproachProbeTest(unittest.TestCase):
    def test_local_window_reuses_exact_radial_gate(self) -> None:
        samples = [sample(i * 1000, 0.20 + i * 0.04, 0.50, 0.001 * (i + 1)) for i in range(5)]
        event = {"event_entry_timestamp_ms": 0, "last_active_timestamp_ms": 4000}
        policy = {"target_classes": ["traffic cone"]}
        self.assertEqual(1, len(probe.local_windows(samples, event, policy, 5)))

    def test_horizontal_sweep_is_rejected(self) -> None:
        samples = [sample(i * 1000, 0.20 + i * 0.02, 0.10 + i * 0.15, 0.001 * (i + 1)) for i in range(5)]
        event = {"event_entry_timestamp_ms": 0, "last_active_timestamp_ms": 4000}
        policy = {"target_classes": ["traffic cone"]}
        self.assertEqual([], probe.local_windows(samples, event, policy, 5))


if __name__ == "__main__":
    unittest.main()
