from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.train_candidate_verifier import balanced_accuracy


class TrainContextCandidateVerifierTest(unittest.TestCase):
    def test_balanced_accuracy(self) -> None:
        self.assertEqual(balanced_accuracy([8, 6], [10, 10]), 0.7)


if __name__ == "__main__":
    unittest.main()
