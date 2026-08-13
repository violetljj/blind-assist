import unittest

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_arkitscenes_opportunity_confirmation as subject


class ArkitScenesOpportunityConfirmationTest(unittest.TestCase):
    def test_opportunity_gate_uses_denominator(self) -> None:
        self.assertFalse(subject.opportunity_gate(3, 3))
        self.assertTrue(subject.opportunity_gate(4, 3))
        self.assertFalse(subject.opportunity_gate(8, 3))
        self.assertTrue(subject.opportunity_gate(8, 4))

    def test_frozen_policy_exact(self) -> None:
        self.assertEqual(0.8, subject.FROZEN_POLICY["task_weight"])
        self.assertEqual(0.05, subject.FROZEN_POLICY["rotation_weight"])


if __name__ == "__main__":
    unittest.main()
