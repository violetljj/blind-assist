from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.pilot.task_api import Observation
from scripts.research.goal_copilot_2_audit.audit import CORRUPTIONS, corrupt_moderate_subset
from scripts.research.goal_copilot_2a.noise import corrupt_observation


class GoalCopilot2AuditTest(unittest.TestCase):
    def test_full_subset_matches_frozen_combined_moderate(self) -> None:
        observations = [
            Observation(False, None, None, 0.0, True, True, True, None, None, 0.0, 0.7, False),
            Observation(True, -8.0, 0.3, 0.8, True, True, True, 0.4, 0.1, 0.8, 0.9, False),
            Observation(True, 2.0, 0.7, 0.95, True, True, True, 0.8, 0.2, 0.9, 0.95, True),
        ]
        for scenario_index in range(12):
            history: list[Observation] = []
            for action_index in range(12):
                current = observations[action_index % len(observations)]
                expected = corrupt_observation(
                    current,
                    history,
                    condition="COMBINED_MODERATE",
                    scenario_index=scenario_index,
                    action_index=action_index,
                )
                actual = corrupt_moderate_subset(
                    current,
                    history,
                    active=CORRUPTIONS,
                    scenario_index=scenario_index,
                    action_index=action_index,
                )
                self.assertEqual(expected, actual)
                history.append(current)

    def test_ablation_rejects_unknown_corruption(self) -> None:
        observation = Observation(False, None, None, 0.0, True, True, True, None, None, 0.0, 0.7, False)
        with self.assertRaises(ValueError):
            corrupt_moderate_subset(
                observation,
                [],
                active=("UNKNOWN",),
                scenario_index=0,
                action_index=0,
            )


if __name__ == "__main__":
    unittest.main()
