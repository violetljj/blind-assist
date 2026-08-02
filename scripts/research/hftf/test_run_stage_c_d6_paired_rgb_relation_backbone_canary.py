import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d6_paired_rgb_relation_backbone_canary import (
    PairedRgbRelationModel,
)
from run_stage_c_d6_source_centered_relation_encoder_canary import (
    RelationEncoder,
)


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            *[torch.nn.Conv2d(3, 3, 1) for _ in range(13)]
        )
        self.pointwise = torch.nn.Conv2d(3, 128, 1)
        self.temporal_depthwise = torch.nn.Conv3d(
            3,
            3,
            (1, 1, 1),
            groups=3,
        )


class PairedRgbRelationBackboneCanaryTest(unittest.TestCase):
    def test_only_tail_and_pointwise_are_trainable(self):
        model = PairedRgbRelationModel(FakeBackbone())
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in model.backbone.encoder[:9].parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.backbone.encoder[9:].parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.backbone.pointwise.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.relation.parameters()
            )
        )

    def test_relation_head_accepts_paired_grid(self):
        head = RelationEncoder()
        delta = torch.zeros(2, 128, 3, 6)
        output = head(torch.cat((delta, delta.abs()), dim=1))
        self.assertEqual((2,), tuple(output.shape))


if __name__ == "__main__":
    unittest.main()
