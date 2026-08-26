from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).with_name("fusion.py")
SPEC = importlib.util.spec_from_file_location("ulr_fusion", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FusionContractTest(unittest.TestCase):
    def test_fixed_fusion_averages_only_available_evidence(self):
        scores = torch.tensor([[[0.9, 0.3, 0.6]]])
        available = torch.tensor([[[True, False, True]]])
        output = MODULE.fixed_equal_available_fusion(scores, available)
        self.assertAlmostEqual(0.75, output.candidate_scores.item())
        self.assertEqual([0.5, 0.0, 0.5], output.modality_weights[0, 0].tolist())

    def test_router_assigns_zero_weight_to_missing_modality(self):
        router = MODULE.QualityConditionedEvidenceRouter(modalities=3, quality_dim=2)
        scores = torch.rand(2, 4, 3)
        available = torch.ones_like(scores, dtype=torch.bool)
        available[:, :, 1] = False
        quality = torch.rand(2, 4, 3, 2)
        output = router(scores, available, quality)
        self.assertEqual((2, 4), tuple(output.candidate_scores.shape))
        self.assertTrue(torch.all(output.modality_weights[:, :, 1] == 0))
        self.assertTrue(torch.allclose(output.modality_weights.sum(dim=-1), torch.ones(2, 4)))

    def test_static_fusion_has_no_quality_input(self):
        model = MODULE.StaticLearnedFusion(modalities=3)
        scores = torch.rand(2, 5, 3, requires_grad=True)
        available = torch.ones_like(scores, dtype=torch.bool)
        output = model(scores, available)
        output.candidate_scores.sum().backward()
        self.assertIsNotNone(scores.grad)
        self.assertEqual((2, 5), tuple(output.candidate_scores.shape))

    def test_router_rejects_candidate_with_no_evidence(self):
        router = MODULE.QualityConditionedEvidenceRouter(modalities=3, quality_dim=2)
        scores = torch.rand(1, 1, 3)
        available = torch.zeros_like(scores, dtype=torch.bool)
        quality = torch.rand(1, 1, 3, 2)
        with self.assertRaisesRegex(ValueError, "at least one"):
            router(scores, available, quality)


if __name__ == "__main__":
    unittest.main()
