from __future__ import annotations

import unittest

import capture_dtr_carla_c2_rich_scene as capture


class FinalReckoningCaptureExtensionTest(unittest.TestCase):
    def test_optional_wearer_yaw_rate_integrates_without_affecting_legacy(self) -> None:
        legacy = {"segments": [{"start_s": 0.0}]}
        scripted = {
            "yaw_segments": [
                {"start_s": 0.0, "yaw_rate_degrees_per_second": 0.0},
                {"start_s": 2.0, "yaw_rate_degrees_per_second": 30.0},
                {"start_s": 4.0, "yaw_rate_degrees_per_second": 0.0},
            ]
        }
        self.assertEqual(0.0, capture.trajectory_yaw_offset_degrees(legacy, 9.0))
        self.assertEqual(0.0, capture.trajectory_yaw_offset_degrees(scripted, 2.0))
        self.assertEqual(30.0, capture.trajectory_yaw_offset_degrees(scripted, 3.0))
        self.assertEqual(60.0, capture.trajectory_yaw_offset_degrees(scripted, 9.0))

    def test_explicit_asset_key_override_precedes_template_trajectory(self) -> None:
        protocol = {
            "trajectory_library": {
                "template_path": {"name": "template"},
                "scenario_path": {"name": "scenario"},
            }
        }
        asset = {"asset_key": "shell", "trajectory": "template_path"}
        scenario = {"asset_trajectories": {"shell": "scenario_path"}}
        self.assertEqual(
            {"name": "scenario"},
            capture.trajectory_for_asset(asset, scenario, protocol),
        )


if __name__ == "__main__":
    unittest.main()
