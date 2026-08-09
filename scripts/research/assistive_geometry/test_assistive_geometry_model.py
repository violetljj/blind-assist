import unittest
import copy

import torch

from scripts.research.assistive_geometry.assistive_geometry_model import (
    AssistiveTaskHeads,
    compute_b1_losses,
    confidence_correctness,
    horizontal_flip_batch,
)


def synthetic_targets(height: int, width: int) -> dict[str, torch.Tensor]:
    depth = torch.full((1, 1, height, width), 1.5)
    valid = torch.ones_like(depth, dtype=torch.bool)
    return {
        "dense_depth_m": depth,
        "depth_valid": valid,
        "ground_probability": torch.zeros_like(depth),
        "ground_label_valid": valid,
        "ground_plane_valid": torch.tensor([True]),
        "camera_height_m": torch.tensor([1.5]),
        "up_camera": torch.tensor([[0.0, -1.0, 0.0]]),
        "intrinsics_tensor": torch.tensor([[[100.0, 0.0, width / 2], [0.0, 100.0, height / 2], [0.0, 0.0, 1.0]]]),
        "clearance_m": torch.tensor([[0.8, 1.2, 1.8]]),
        "clearance_valid": torch.tensor([[True, True, True]]),
        "occupancy": torch.tensor([[[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]]),
        "occupancy_valid": torch.ones((1, 3, 3), dtype=torch.bool),
    }


class AssistiveGeometryModelTest(unittest.TestCase):
    def test_heads_accept_both_orientation_shapes(self) -> None:
        heads = AssistiveTaskHeads()
        for height, width in ((608, 448), (448, 608)):
            output = heads(torch.randn(2, 48, height // 4, width // 4), (height, width))
            self.assertEqual((2, 1, height, width), tuple(output["ground_logits"].shape))
            self.assertEqual((2, 3), tuple(output["clearance_m"].shape))
            self.assertEqual((2, 3, 3), tuple(output["occupancy_logits"].shape))
            self.assertEqual((2, 3), tuple(output["confidence_logits"].shape))

    def test_all_frozen_losses_are_finite_and_backward(self) -> None:
        height, width = 16, 12
        targets = synthetic_targets(height, width)
        outputs = {
            "dense_depth_m": torch.full((1, 1, height, width), 1.4, requires_grad=True),
            "ground_logits": torch.zeros((1, 1, height, width), requires_grad=True),
            "clearance_m": torch.tensor([[0.9, 1.1, 1.7]], requires_grad=True),
            "occupancy_logits": torch.zeros((1, 3, 3), requires_grad=True),
            "confidence_logits": torch.zeros((1, 3), requires_grad=True),
        }
        active = [
            "masked_log_depth", "valid_neighbor_log_gradient", "ground_bce", "ground_plane_depth",
            "clearance_huber", "occupancy_bce", "false_clear_extra", "confidence_bce",
        ]
        losses = compute_b1_losses(outputs, targets, active)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(outputs["dense_depth_m"].grad)
        self.assertIsNotNone(outputs["occupancy_logits"].grad)

    def test_censored_clear_confidence_is_not_unknown_or_negative(self) -> None:
        targets = synthetic_targets(4, 4)
        targets["clearance_valid"][:] = False
        targets["occupancy"][:] = 0.0
        outputs = {
            "clearance_m": torch.zeros((1, 3)),
            "occupancy_logits": torch.full((1, 3, 3), -4.0),
            "confidence_logits": torch.zeros((1, 3)),
        }
        correct, valid = confidence_correctness(outputs, targets)
        self.assertTrue(torch.all(valid))
        self.assertTrue(torch.all(correct == 1))
        targets["occupancy_valid"][:, 0, :] = False
        _, valid = confidence_correctness(outputs, targets)
        self.assertFalse(bool(valid[0, 0]))

    def test_horizontal_flip_is_involution_and_swaps_bands_and_k(self) -> None:
        targets = synthetic_targets(4, 6)
        once = horizontal_flip_batch(targets)
        self.assertEqual(5.0 - float(targets["intrinsics_tensor"][0, 0, 2]), float(once["intrinsics_tensor"][0, 0, 2]))
        self.assertTrue(torch.equal(once["clearance_m"][:, 0], targets["clearance_m"][:, 2]))
        twice = horizontal_flip_batch(once)
        for key in targets:
            self.assertTrue(torch.equal(targets[key], twice[key]), key)

    def test_adamw_resume_matches_uninterrupted_heads(self) -> None:
        torch.manual_seed(20260809)
        initial = AssistiveTaskHeads()
        uninterrupted = copy.deepcopy(initial)
        first_leg = copy.deepcopy(initial)
        feature = torch.randn(1, 48, 4, 3)

        def step(model: AssistiveTaskHeads, optimizer: torch.optim.Optimizer) -> None:
            optimizer.zero_grad(set_to_none=True)
            output = model(feature, (16, 12))
            loss = sum(value.float().mean() for value in output.values())
            loss.backward()
            optimizer.step()

        optimizer_a = torch.optim.AdamW(uninterrupted.parameters(), lr=1e-4, weight_decay=0.01)
        step(uninterrupted, optimizer_a)
        step(uninterrupted, optimizer_a)

        optimizer_b = torch.optim.AdamW(first_leg.parameters(), lr=1e-4, weight_decay=0.01)
        step(first_leg, optimizer_b)
        resumed = AssistiveTaskHeads()
        resumed.load_state_dict(copy.deepcopy(first_leg.state_dict()))
        optimizer_c = torch.optim.AdamW(resumed.parameters(), lr=1e-4, weight_decay=0.01)
        optimizer_c.load_state_dict(copy.deepcopy(optimizer_b.state_dict()))
        step(resumed, optimizer_c)
        for left, right in zip(uninterrupted.state_dict().values(), resumed.state_dict().values(), strict=True):
            self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()
