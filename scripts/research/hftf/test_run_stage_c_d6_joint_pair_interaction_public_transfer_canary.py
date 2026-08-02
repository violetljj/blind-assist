import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d6_joint_pair_interaction_public_transfer_canary import (
    JointPairInteractionModel,
    build_source_baselines,
)


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 1)
        )
        self.temporal_depthwise = torch.nn.Conv3d(
            3,
            3,
            (5, 1, 1),
            groups=3,
            bias=False,
        )
        self.pointwise = torch.nn.Conv2d(3, 128, 1)


class JointPairInteractionPublicTransferCanaryTest(
    unittest.TestCase
):
    def test_joint_pair_model_shape_and_trainable_boundary(self):
        model = JointPairInteractionModel(FakeBackbone())
        current = torch.zeros(2, 3, 128, 224)
        baseline = torch.ones(2, 3, 128, 224)
        self.assertEqual((2,), tuple(model(current, baseline).shape))
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in model.backbone.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.pair_stem.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.relation.parameters()
            )
        )

    def test_source_baselines_average_reference_images(self):
        tensors = torch.arange(12, dtype=torch.float32).reshape(
            4,
            3,
            1,
            1,
        )
        baselines = build_source_baselines(
            tensors,
            {"a": [0, 2], "b": [1]},
        )
        torch.testing.assert_close(
            baselines["a"],
            (tensors[0] + tensors[2]) / 2,
        )
        torch.testing.assert_close(baselines["b"], tensors[1])


if __name__ == "__main__":
    unittest.main()
