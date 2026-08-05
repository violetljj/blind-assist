from __future__ import annotations

import unittest

import torch
from train_dav2_392_distilled_student_r0 import distillation_loss


class DistillationLossTest(unittest.TestCase):
    def test_identity_has_zero_loss(self) -> None:
        depth = torch.ones((2, 4, 4), dtype=torch.float32)
        loss, components = distillation_loss(depth, depth, 0.05, 0.5, 0.25, 0.1, 20.0)
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(all(value == 0.0 for value in components.values()))

    def test_global_scale_is_penalized(self) -> None:
        teacher = torch.ones((1, 4, 4), dtype=torch.float32)
        candidate = teacher * 2.0
        loss, components = distillation_loss(
            candidate, teacher, 0.05, 0.5, 0.25, 0.1, 20.0
        )
        self.assertGreater(float(loss), 0.0)
        self.assertGreater(components["depth"], 0.0)
        self.assertEqual(components["gradient"], 0.0)
        self.assertGreater(components["scale"], 0.0)


if __name__ == "__main__":
    unittest.main()
