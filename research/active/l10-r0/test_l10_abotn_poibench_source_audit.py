from __future__ import annotations

import json
import unittest

from l10_abotn_poibench_source_audit import inspect_episode


class AbotnPoiBenchSourceAuditTest(unittest.TestCase):
    def test_complete_episode_is_pose_complete_but_has_no_facade_truth(self) -> None:
        payload = json.dumps(
            {
                "instruction": "go to target",
                "trajectory": [
                    {"x": 0, "y": 0, "z": 1, "pitch": 0, "roll": 0, "yaw": 0},
                    {"x": 1, "y": 0, "z": 1, "pitch": 0, "roll": 0, "yaw": 0},
                ],
                "label": {
                    "extend": {
                        "goal_label": "target",
                        "el_unique_id": "7",
                        "start_point": [0, 0],
                        "end_point": [1, 0],
                    }
                },
            }
        ).encode()
        row = inspect_episode("annotations/scene/traj_0.json", payload)
        self.assertEqual(row["missing_episode_fields"], [])
        self.assertEqual(row["missing_extend_fields"], [])
        self.assertEqual(row["missing_pose_frames"], 0)
        self.assertNotIn("facade_id", row)

    def test_missing_pose_field_is_counted(self) -> None:
        payload = json.dumps(
            {
                "instruction": "go to target",
                "trajectory": [{"x": 0, "y": 0, "z": 1, "pitch": 0, "roll": 0}],
                "label": {"extend": {}},
            }
        ).encode()
        row = inspect_episode("annotations/scene/traj_0.json", payload)
        self.assertEqual(row["missing_pose_frames"], 1)
        self.assertIn("goal_label", row["missing_extend_fields"])


if __name__ == "__main__":
    unittest.main()
