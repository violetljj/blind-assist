from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_groundbench_candidate_zoom_development import (
    development_decision,
)


def metrics(correct: int, wrong: int) -> dict:
    return {
        "given_usable_proposal": {"correct_grounding": correct},
        "wrong_confident_guidance_all_observations": {"numerator": wrong},
    }


class CandidateZoomDevelopmentTest(unittest.TestCase):
    def test_promising_requires_more_correct_without_more_wrong(self) -> None:
        self.assertTrue(development_decision(metrics(10, 2), metrics(11, 2))["confirmation_authorized"])
        self.assertFalse(development_decision(metrics(10, 2), metrics(10, 1))["confirmation_authorized"])
        self.assertFalse(development_decision(metrics(10, 2), metrics(11, 3))["confirmation_authorized"])


if __name__ == "__main__":
    unittest.main()
