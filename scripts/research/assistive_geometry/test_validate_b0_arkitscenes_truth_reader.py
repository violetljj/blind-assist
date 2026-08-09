import unittest

from scripts.research.assistive_geometry.validate_b0_arkitscenes_truth_reader import (
    _entry_map,
    _judge,
    _nearest_entry,
)


class ValidateB0ArkitScenesTruthReaderTest(unittest.TestCase):
    def test_entry_map_rejects_duplicate_stems(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _entry_map([{"path": "a/1_1.0.png"}, {"path": "b/1_1.0.png"}])

    def test_nearest_intrinsics_is_bounded(self) -> None:
        entry, gap = _nearest_entry([{"path": "1_1.00.pincam"}, {"path": "1_1.10.pincam"}], "1_1.02")
        self.assertEqual("1_1.00.pincam", entry["path"])
        self.assertAlmostEqual(0.02, gap)
        with self.assertRaisesRegex(ValueError, "50 ms"):
            _nearest_entry([{"path": "1_2.00.pincam"}], "1_1.00")

    def test_judge_fails_one_broken_gate(self) -> None:
        metrics = {
            "upsampling_train": {
                "video_count": 1,
                "frame_count": 10,
                "orientation_agreement_fraction": 1.0,
                "frame_gate_pass_fraction": 1.0,
                "maximum_intrinsics_timestamp_gap_seconds": 0.0,
                "ground_height_absolute_difference_m": {"count": 5, "median": 0.01},
                "clearance_absolute_difference_m": {"count": 5, "median": 0.01},
                "occupied_decision_agreement_fraction": 1.0,
            },
            "main_train_capability": {
                "evaluated_frame_count": 10,
                "ground_available_fraction": 1.0,
                "all_bands_known_fraction": 1.0,
                "videos_with_ground_count": 1,
                "maximum_pose_bracketing_gap_seconds": 0.1,
                "unknown_clearance_leak_count": 1,
            },
        }
        gates = {
            "upsampling": {
                "expected_video_count": 1,
                "minimum_frame_count": 10,
                "minimum_orientation_agreement_fraction": 1.0,
                "minimum_frame_gate_pass_fraction": 1.0,
                "maximum_intrinsics_timestamp_gap_seconds": 0.05,
                "minimum_dual_ground_frame_count": 1,
                "maximum_ground_height_median_absolute_difference_m": 0.1,
                "minimum_clearance_pair_count": 1,
                "maximum_clearance_median_absolute_difference_m": 0.1,
                "minimum_occupied_decision_agreement_fraction": 0.9,
            },
            "main_train_capability": {
                "expected_frame_count": 10,
                "minimum_ground_available_fraction": 0.5,
                "minimum_all_bands_known_fraction": 0.5,
                "expected_videos_with_ground_count": 1,
                "maximum_pose_bracketing_gap_seconds": 0.25,
            },
        }
        checks, passed = _judge(metrics, gates)
        self.assertFalse(checks["unknown_fail_closed"])
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
