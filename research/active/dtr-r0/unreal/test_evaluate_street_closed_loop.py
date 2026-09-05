import unittest

import evaluate_street_closed_loop as evaluation
from street_scenarios import contacts_for_step, scenario_catalog


class ActualTrajectoryEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.spec = next(s for s in scenario_catalog() if s["id"] == "low_obstacle_collision")

    def episode(self, positions, completed=True, reached=False):
        frames = []
        for i, (x, y) in enumerate(positions):
            ego = {"x_m": x, "y_m": y}
            contacts = contacts_for_step(self.spec, i-1, i, frames[-1]["ego"], ego) if frames else []
            velocity = {"vx_mps": x-positions[i-1][0], "vy_mps": y-positions[i-1][1]} if i else {"vx_mps": 1, "vy_mps": 0}
            frames.append({"sample_index": i, "time_s": i, "ego": ego,
                           "contacts_since_previous": contacts, "applied_command": velocity,
                           "response": {"prediction": {"route_risk": False},
                                        "command": {"action": "WALK", "vx_mps": 1, "vy_mps": 0}, "elapsed_s": .1}})
        return {"scenario_id": self.spec["id"], "episode_id": "test", "frames": frames,
                "completed": completed, "goal_reached": reached, "goal_forward_m": 8}

    def test_stopped_without_contact_is_not_success(self):
        ep = self.episode([(0, 0), (1, 0), (2, 0), (2, 0)])
        result = evaluation.evaluate_episode(self.spec, ep)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertFalse(result["contact"])
        self.assertFalse(result["success"])

    def test_incomplete_goal_is_not_success(self):
        ep = self.episode([(0, 0), (8, 1)], completed=False, reached=True)
        result = evaluation.evaluate_episode(self.spec, ep)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertFalse(result["success"])

    def test_unreported_low_obstacle_contact_invalidates_receipt(self):
        ep = self.episode([(0, 0), (8, 0)], reached=True)
        ep["frames"][-1]["contacts_since_previous"] = []
        result = evaluation.evaluate_episode(self.spec, ep)
        self.assertEqual(result["contact_types"], ["FOOT_TRIP_PROXY"])
        self.assertEqual(result["status"], "INVALID")
        self.assertFalse(result["success"])

    def test_alert_must_precede_motion_and_command_must_be_applied(self):
        baseline = self.episode([(0, 0), (1, 0), (2, 0)])
        assisted = self.episode([(0, 0), (1, 0), (1.5, .5)])
        command = assisted["frames"][1]["response"]["command"]
        command.update(vx_mps=.5, vy_mps=.5, depth_near_risk=True, action="SIDESTEP")
        result = evaluation.causal_trajectory_check(self.spec, baseline, assisted)
        self.assertTrue(result["trajectory_changed_after_sensor_alert"])
        self.assertEqual(result["trigger_sources"], ["OBSERVED_DEPTH"])
        command["depth_near_risk"] = False
        assisted["frames"][2]["response"]["command"]["depth_near_risk"] = True
        self.assertFalse(evaluation.causal_trajectory_check(self.spec, baseline, assisted)["trajectory_changed_after_sensor_alert"])


if __name__ == "__main__":
    unittest.main()
