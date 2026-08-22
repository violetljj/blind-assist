from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.train_context_candidate_verifier_fixed import DECISION_THRESHOLD, EPOCHS


class TrainContextCandidateVerifierFixedTest(unittest.TestCase):
    def test_training_is_fixed_before_future_validation(self) -> None:
        self.assertEqual(EPOCHS, 5)
        self.assertEqual(DECISION_THRESHOLD, 0.5)


if __name__ == "__main__":
    unittest.main()
