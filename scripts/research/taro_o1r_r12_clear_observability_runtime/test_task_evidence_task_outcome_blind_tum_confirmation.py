import unittest

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_task_outcome_blind_tum_confirmation as subject


class TaskOutcomeBlindTumConfirmationTest(unittest.TestCase):
    def test_frozen_policy_exact(self) -> None:
        self.assertEqual("NORMALIZED_POSE_TASK_BLEND", subject.FROZEN_POLICY["family"])
        self.assertEqual("visible_unknown", subject.FROZEN_POLICY["task_term"])
        self.assertEqual(0.8, subject.FROZEN_POLICY["task_weight"])
        self.assertEqual(0.05, subject.FROZEN_POLICY["rotation_weight"])

    def test_confirmation_thresholds_match_lock(self) -> None:
        self.assertEqual(16, subject.MIN_REFERENCES)
        self.assertEqual(4, subject.MIN_PARENTS)
        self.assertEqual(3, subject.MIN_STRICT_WIN_PARENTS)


if __name__ == "__main__":
    unittest.main()
