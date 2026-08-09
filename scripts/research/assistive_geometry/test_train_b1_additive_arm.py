from __future__ import annotations

import unittest

import torch.nn as nn

from scripts.research.assistive_geometry.assistive_geometry_model import AssistiveTaskHeads
from scripts.research.assistive_geometry.train_b1_additive_arm import (
    ARM_SPECS,
    arm_slug,
    configure_trainable,
)


class DummyGeometryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.metric_depthart = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
        self.assistive_heads = AssistiveTaskHeads()


class AdditiveArmTrainingTests(unittest.TestCase):
    def test_losses_are_strictly_additive(self) -> None:
        order = list(ARM_SPECS)
        for previous, current in zip(order, order[1:]):
            previous_losses = set(ARM_SPECS[previous]["losses"])
            current_losses = set(ARM_SPECS[current]["losses"])
            self.assertTrue(previous_losses < current_losses)

    def test_a1_trains_encoder_and_ground_only(self) -> None:
        model = DummyGeometryModel()
        groups, trainable = configure_trainable(model, "A1_PLUS_GROUND")
        self.assertEqual([group["group_name"] for group in groups], ["depthart_encoder_decoder", "active_assistive_heads"])
        self.assertTrue(trainable)
        self.assertTrue(all(parameter.requires_grad for parameter in model.metric_depthart.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.assistive_heads.ground_pre.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.assistive_heads.ground_out.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.assistive_heads.band_mlp.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.assistive_heads.confidence_out.parameters()))

    def test_a4_trains_all_heads(self) -> None:
        model = DummyGeometryModel()
        _, trainable = configure_trainable(model, "A4_PLUS_CONFIDENCE")
        self.assertEqual(len(trainable), len(list(model.parameters())))
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_arm_slug_is_path_stable(self) -> None:
        self.assertEqual(arm_slug("A3_PLUS_FALSE_CLEAR"), "a3-plus-false-clear")


if __name__ == "__main__":
    unittest.main()
