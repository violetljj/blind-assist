import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_stage_c_d6_veto_eligibility_ranking import (
    VetoEligibilityStudent,
    compose_ranking_logits,
    eligibility_targets,
    masked_veto_loss,
)


class VetoEligibilityRankingTest(unittest.TestCase):
    def test_model_outputs_field_logits(self):
        model = VetoEligibilityStudent()

        output = model(torch.zeros(2, 5, 3, 128, 224))

        self.assertEqual((2, 3, 3, 6, 6), tuple(output.shape))
        self.assertLess(
            sum(parameter.numel() for parameter in model.parameters()),
            25_000,
        )

    def test_eligibility_requires_critical_known_baseline_active(self):
        shape = (1, 3, 3, 2, 2)
        reference_risk = torch.full(shape, 10.0)
        reference_known = torch.full(shape, 10.0)
        risk = torch.zeros(shape)
        known = torch.ones(shape)
        reference_risk[0, 1, 1, 0, 1] = -10.0
        reference_known[0, 1, 1, 1, 0] = -10.0
        known[0, 1, 1, 1, 1] = 0.0
        risk[0, 2, 2, 0, 0] = 1.0

        target, eligible = eligibility_targets(
            reference_risk,
            reference_known,
            risk,
            known,
        )

        self.assertFalse(bool(eligible[0, 0, 1, 0, 0]))
        self.assertFalse(bool(eligible[0, 1, 0, 0, 0]))
        self.assertFalse(bool(eligible[0, 1, 1, 0, 1]))
        self.assertFalse(bool(eligible[0, 1, 1, 1, 0]))
        self.assertFalse(bool(eligible[0, 1, 1, 1, 1]))
        self.assertTrue(bool(eligible[0, 1, 1, 0, 0]))
        self.assertEqual(float(target[0, 1, 1, 0, 0]), 1.0)
        self.assertEqual(float(target[0, 2, 2, 0, 0]), 0.0)

    def test_masked_loss_ignores_ineligible_cells(self):
        logits = torch.zeros((1, 3, 3, 1, 2), requires_grad=True)
        target = torch.zeros_like(logits)
        eligible = torch.zeros_like(logits, dtype=torch.bool)
        eligible[0, 1, 1, 0, 0] = True
        target[0, 1, 1, 0, 0] = 1.0
        logits.data[0, 1, 1, 0, 1] = 100.0

        loss = masked_veto_loss(
            logits,
            target,
            eligible,
            torch.tensor(1.0),
        )
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), 0.693147, places=5)
        self.assertEqual(float(logits.grad[0, 1, 1, 0, 1]), 0.0)

    def test_zero_head_confidence_residual_exactly_matches_comparator(
        self,
    ):
        model = VetoEligibilityStudent(zero_head=True)
        frames = torch.randn(2, 5, 3, 128, 224)
        reference_risk = torch.randn(2, 3, 3, 6, 6)

        logits = compose_ranking_logits(
            model(frames),
            reference_risk,
            "confidence_residual",
        )

        torch.testing.assert_close(logits, -reference_risk)


if __name__ == "__main__":
    unittest.main()
