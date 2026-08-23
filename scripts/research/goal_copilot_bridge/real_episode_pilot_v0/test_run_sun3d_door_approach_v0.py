from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.real_episode_pilot_v0 import run_sun3d_door_approach_v0 as sut


class Sun3dDoorApproachTest(unittest.TestCase):
    def test_uniform_indices_preserve_endpoints_and_count(self) -> None:
        indices = sut._uniform_indices(94, 200, 15)
        self.assertEqual(indices[0], 94)
        self.assertEqual(indices[-1], 200)
        self.assertEqual(len(indices), 15)
        self.assertEqual(len(set(indices)), 15)

    def test_lost_is_only_derived_after_visible(self) -> None:
        truth = [
            {"observation_id": "a", "visibility": "NOT_VISIBLE", "arrival_truth": False, "map_trajectory_range_m": 5.0, "map_trajectory_bearing_deg": 0.0},
            {"observation_id": "b", "visibility": "VISIBLE", "arrival_truth": False, "map_trajectory_range_m": 4.0, "map_trajectory_bearing_deg": 0.0, "native_bbox_xyxy": [0, 0, 50, 50], "width": 100, "height": 100},
            {"observation_id": "c", "visibility": "NOT_VISIBLE", "arrival_truth": False, "map_trajectory_range_m": 3.0, "map_trajectory_bearing_deg": 0.0},
        ]
        episodes = [
            {"episode_id": item["observation_id"], "candidates": []}
            for item in truth
        ]
        decisions = [
            {"episode_id": item["observation_id"], "action": "ABSTAIN", "selected_candidate_ids": []}
            for item in truth
        ]
        result = sut._evaluate(truth, episodes, decisions)
        states = [item["episode_truth_state"] for item in result["observations"]]
        self.assertEqual(states, ["NOT_VISIBLE", "VISIBLE", "LOST"])
        self.assertEqual(result["lost_after_visible_count"], 1)


if __name__ == "__main__":
    unittest.main()
