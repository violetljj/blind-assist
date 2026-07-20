import unittest

import numpy as np

import run_public_video_actionability_profile_lifecycle_probe as subject


class ActionabilityProfileLifecycleProbeTest(unittest.TestCase):
    def test_frame_targets_follow_reentry_transitions(self) -> None:
        timestamps = [0, 1000, 2000, 3000, 4000]
        transitions = [
            {"state": "intervention_needed", "timestamp_ms": 1000},
            {"state": "route_clear", "timestamp_ms": 3000},
        ]
        np.testing.assert_array_equal([0, 1, 1, 0, 0], subject.frame_targets(timestamps, transitions))

    def test_event_open_requires_consecutive_seconds(self) -> None:
        rows = subject.event_lifecycle_predictions(
            np.asarray([0.9, 0.1, 0.9]), np.asarray([1, 1, 1]),
            np.asarray(["e", "e", "e"]), np.asarray(["s", "s", "s"]),
            np.asarray([0, 1000, 2000]), 0.5,
        )
        self.assertEqual(0, rows[0]["predicted_label"])

    def test_event_open_reports_second_timestamp(self) -> None:
        rows = subject.event_lifecycle_predictions(
            np.asarray([0.1, 0.8, 0.9]), np.asarray([1, 1, 1]),
            np.asarray(["e", "e", "e"]), np.asarray(["s", "s", "s"]),
            np.asarray([0, 1000, 2000]), 0.5,
        )
        self.assertEqual(1, rows[0]["predicted_label"])
        self.assertEqual(2000, rows[0]["first_open_timestamp_ms"])


if __name__ == "__main__":
    unittest.main()
