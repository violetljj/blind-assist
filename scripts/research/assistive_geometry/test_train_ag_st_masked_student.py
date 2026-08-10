from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_ag_st_masked_student import (  # noqa: E402
    MaskedFactorStudent,
    TIER_A_SOURCE,
    TIER_C_TEACHER,
    TIER_UNKNOWN,
    compute_boundary_only_losses,
    compute_depth_support_losses,
    compute_depth_support_precision_losses,
    compute_student_losses,
    compute_unified_continuous_boundary_losses,
    fit_scalar_support_calibration,
    select_parent_split,
    tier_weights,
)


class AgStMaskedStudentTest(unittest.TestCase):
    def test_parent_split_is_deterministic_shape_stratified_and_disjoint(self) -> None:
        shapes = {
            **{f"p{index}": (336, 252) for index in range(8)},
            **{f"s{index}": (252, 336) for index in range(6)},
            **{f"t{index}": (294, 336) for index in range(2)},
        }
        first = select_parent_split(shapes)
        second = select_parent_split(dict(reversed(tuple(shapes.items()))))
        self.assertEqual(first, second)
        train, selection, canary, receipt = first
        self.assertEqual(12, len(train))
        self.assertEqual(2, len(selection))
        self.assertEqual(2, len(canary))
        self.assertEqual(16, len(set(train) | set(selection) | set(canary)))
        self.assertEqual(1, sum(parent.startswith("p") for parent in selection))
        self.assertEqual(1, sum(not parent.startswith("p") for parent in selection))
        self.assertEqual(1, sum(parent.startswith("p") for parent in canary))
        self.assertEqual(1, sum(not parent.startswith("p") for parent in canary))
        self.assertIn("WITHOUT_LABEL_CONTENT", receipt["method"])

    def test_zero_residual_initialization_preserves_base_depth_and_priors(self) -> None:
        model = MaskedFactorStudent()
        priors = {
            "support_probability": 0.25,
            "boundary_probability": 0.10,
            "obstacle_probability": 0.40,
            "boundary_distance_px": 8.0,
        }
        model.initialize_priors(priors)
        feature = torch.randn(1, 48, 4, 5)
        base_depth = torch.full((1, 1, 16, 20), 2.0)
        outputs = model(feature, base_depth, (16, 20))
        torch.testing.assert_close(outputs["depth_m"], base_depth)
        self.assertAlmostEqual(
            priors["support_probability"],
            float(torch.sigmoid(outputs["support_logits"]).mean().detach()),
            places=6,
        )
        self.assertAlmostEqual(
            priors["boundary_distance_px"],
            float(outputs["boundary_distance_px"].mean().detach()),
            places=5,
        )

    def test_dilated_pyramid_head_accepts_multiscale_and_base_depth_features(self) -> None:
        model = MaskedFactorStudent(
            channels=192,
            hidden=64,
            head_profile="dilated_pyramid",
            use_base_depth_feature=True,
        )
        model.initialize_priors(
            {
                "support_probability": 0.25,
                "boundary_probability": 0.10,
                "obstacle_probability": 0.40,
                "boundary_distance_px": 8.0,
            }
        )
        feature = torch.randn(1, 192, 12, 16)
        base_depth = torch.full((1, 1, 24, 32), 2.0)
        outputs = model(feature, base_depth, (24, 32))
        self.assertEqual((1, 1, 24, 32), tuple(outputs["depth_m"].shape))
        torch.testing.assert_close(outputs["depth_m"], base_depth)
        self.assertGreater(
            sum(parameter.numel() for parameter in model.parameters()),
            100_000,
        )

    def test_factor_split_routes_boundary_gradient_away_from_shared_trunk(self) -> None:
        model = MaskedFactorStudent(
            channels=192,
            hidden=64,
            head_profile="factor_split_dilated",
            use_base_depth_feature=True,
        )
        feature = torch.randn(1, 192, 12, 16)
        base_depth = torch.full((1, 1, 24, 32), 2.0)
        outputs = model(feature, base_depth, (24, 32))
        outputs["boundary_logits"].sum().backward()
        self.assertIsNotNone(model.boundary_trunk)
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.boundary_trunk.parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.trunk.parameters())
        )

    def test_identity_gate_starts_near_closed_and_preserves_base_depth(self) -> None:
        model = MaskedFactorStudent(
            depth_gate_profile="identity_sigmoid",
            use_base_depth_feature=True,
        )
        feature = torch.randn(1, 48, 4, 5)
        base_depth = torch.full((1, 1, 16, 20), 2.0)
        outputs = model(feature, base_depth, (16, 20))
        torch.testing.assert_close(outputs["depth_m"], base_depth)
        self.assertAlmostEqual(
            0.05,
            float(outputs["depth_identity_gate"].mean().detach()),
            places=6,
        )
        with torch.no_grad():
            model.depth_residual.bias.fill_(1.0)
        corrected = model(feature, base_depth, (16, 20))["depth_m"]
        self.assertGreater(float(corrected.mean().detach()), 2.0)
        self.assertLess(float(corrected.mean().detach()), 2.5)

    def test_parent_split_accepts_ten_six_orientation_mix(self) -> None:
        shapes = {
            **{f"p{index}": (336, 252) for index in range(10)},
            **{f"l{index}": (252, 336) for index in range(6)},
        }
        train, selection, canary, _ = select_parent_split(shapes)
        self.assertEqual((12, 2, 2), (len(train), len(selection), len(canary)))
        self.assertEqual(1, sum(parent.startswith("p") for parent in selection))
        self.assertEqual(1, sum(parent.startswith("p") for parent in canary))

    def test_parent_split_scales_to_combined_32_parent_fit(self) -> None:
        shapes = {
            **{f"p{index}": (336, 252) for index in range(18)},
            **{f"l{index}": (252, 336) for index in range(14)},
        }
        train, selection, canary, receipt = select_parent_split(
            shapes,
            split_token="AG_ST_COMBINED_DEPTH_SUPPORT_R0",
        )
        self.assertEqual((28, 2, 2), (len(train), len(selection), len(canary)))
        self.assertEqual(32, len(set(train) | set(selection) | set(canary)))
        self.assertIn("28_2_2", receipt["method"])

    def test_unknown_pixels_cannot_change_masked_losses(self) -> None:
        outputs = {
            "depth_m": torch.full((1, 1, 2, 2), 1.2),
            "support_logits": torch.zeros((1, 1, 2, 2)),
            "boundary_logits": torch.zeros((1, 1, 2, 2)),
            "boundary_distance_px": torch.full((1, 1, 2, 2), 4.0),
            "obstacle_logits": torch.zeros((1, 1, 2, 2)),
        }
        targets = {
            "metric_depth_m": torch.tensor([[[[1.0, 5.0], [6.0, 7.0]]]]),
            "metric_valid": torch.tensor([[[[True, False], [False, False]]]]),
            "metric_tier": torch.tensor([[[[TIER_A_SOURCE, TIER_UNKNOWN], [TIER_UNKNOWN, TIER_UNKNOWN]]]]),
            "support": torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]]),
            "support_valid": torch.tensor([[[[True, False], [False, False]]]]),
            "support_tier": torch.tensor([[[[TIER_A_SOURCE, TIER_UNKNOWN], [TIER_UNKNOWN, TIER_UNKNOWN]]]]),
            "boundary": torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]]),
            "boundary_distance_px": torch.tensor([[[[0.0, 32.0], [0.0, 32.0]]]]),
            "obstacle": torch.tensor([[[[0.5, 0.0], [1.0, 0.0]]]]),
            "evidence_valid": torch.tensor([[[[True, False], [False, False]]]]),
            "evidence_tier": torch.tensor([[[[TIER_A_SOURCE, TIER_UNKNOWN], [TIER_UNKNOWN, TIER_UNKNOWN]]]]),
        }
        changed = copy.deepcopy(targets)
        for key in ("metric_depth_m", "support", "boundary", "boundary_distance_px", "obstacle"):
            changed[key][..., 0, 1:] = 999.0
            changed[key][..., 1, :] = -999.0
        weights = {"support_pos_weight": 1.0, "boundary_pos_weight": 1.0}
        first = compute_student_losses(outputs, targets, weights)
        second = compute_student_losses(outputs, changed, weights)
        for key in first:
            torch.testing.assert_close(first[key], second[key])

    def test_tier_weight_orders_source_teacher_and_unknown(self) -> None:
        weights = tier_weights(
            torch.tensor([TIER_A_SOURCE, TIER_C_TEACHER, TIER_UNKNOWN])
        )
        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertGreater(float(weights[1]), float(weights[2]))
        self.assertEqual(0.0, float(weights[2]))

    def test_multifactor_loss_keeps_boundary_and_obstacle_masks_independent(self) -> None:
        outputs = {
            "depth_m": torch.ones((1, 1, 2, 2)),
            "support_logits": torch.zeros((1, 1, 2, 2)),
            "boundary_logits": torch.zeros((1, 1, 2, 2)),
            "boundary_distance_px": torch.full((1, 1, 2, 2), 4.0),
            "obstacle_logits": torch.zeros((1, 1, 2, 2)),
        }
        targets = {
            "metric_depth_m": torch.ones((1, 1, 2, 2)),
            "metric_valid": torch.tensor([[[[True, False], [False, False]]]]),
            "metric_tier": torch.tensor([[[[TIER_A_SOURCE, 0], [0, 0]]]]),
            "support": torch.zeros((1, 1, 2, 2)),
            "support_valid": torch.tensor([[[[True, False], [False, False]]]]),
            "support_tier": torch.tensor([[[[TIER_A_SOURCE, 0], [0, 0]]]]),
            "boundary": torch.zeros((1, 1, 2, 2)),
            "boundary_soft": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
            "boundary_distance_px": torch.tensor([[[[0.0, 32.0], [32.0, 32.0]]]]),
            "boundary_valid": torch.tensor([[[[True, False], [False, False]]]]),
            "boundary_tier": torch.tensor([[[[TIER_A_SOURCE, 0], [0, 0]]]]),
            "obstacle": torch.zeros((1, 1, 2, 2)),
            "evidence_valid": torch.tensor([[[[False, True], [False, False]]]]),
            "evidence_tier": torch.tensor([[[[0, TIER_A_SOURCE], [0, 0]]]]),
        }
        changed = copy.deepcopy(targets)
        changed["boundary_soft"][..., 0, 1] = 1.0
        changed["boundary_distance_px"][..., 0, 1] = 0.0
        changed["obstacle"][..., 0, 0] = 1.0
        weights = {"support_pos_weight": 1.0, "boundary_pos_weight": 1.0}
        first = compute_student_losses(outputs, targets, weights)
        second = compute_student_losses(outputs, changed, weights)
        for key in first:
            torch.testing.assert_close(first[key], second[key])

    def test_unified_continuous_profile_uses_boundary_specific_objective(self) -> None:
        outputs = {
            "depth_m": torch.ones((1, 1, 2, 2), requires_grad=True),
            "support_logits": torch.zeros((1, 1, 2, 2), requires_grad=True),
            "boundary_logits": torch.zeros((1, 1, 2, 2), requires_grad=True),
            "boundary_distance_px": torch.full(
                (1, 1, 2, 2), 8.0, requires_grad=True
            ),
            "obstacle_logits": torch.zeros((1, 1, 2, 2), requires_grad=True),
        }
        valid = torch.tensor([[[[True, True], [False, False]]]])
        tiers = torch.tensor(
            [[[[TIER_A_SOURCE, TIER_A_SOURCE], [TIER_UNKNOWN, TIER_UNKNOWN]]]]
        )
        targets = {
            "metric_depth_m": torch.ones((1, 1, 2, 2)),
            "metric_valid": valid,
            "metric_tier": tiers,
            "support": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
            "support_valid": valid,
            "support_tier": tiers,
            "boundary": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
            "boundary_soft": torch.tensor([[[[1.0, 0.4], [0.0, 0.0]]]]),
            "boundary_distance_px": torch.tensor([[[[0.0, 3.0], [32.0, 32.0]]]]),
            "boundary_valid": valid,
            "boundary_tier": tiers,
            "obstacle": torch.zeros((1, 1, 2, 2)),
            "evidence_valid": valid,
            "evidence_tier": tiers,
        }
        losses = compute_unified_continuous_boundary_losses(
            outputs,
            targets,
            {"support_pos_weight": 1.0, "boundary_pos_weight": 1.0},
        )
        self.assertIn("raw/boundary_soft_heat_bce", losses)
        self.assertIn("raw/boundary_near_distance", losses)
        self.assertNotIn("raw/boundary_bce", losses)
        losses["total"].backward()
        for key in (
            "depth_m",
            "support_logits",
            "boundary_logits",
            "boundary_distance_px",
            "obstacle_logits",
        ):
            self.assertIsNotNone(outputs[key].grad)

    def test_depth_support_profile_ignores_unknown_and_noncore_outputs(self) -> None:
        outputs = {
            "depth_m": torch.full((1, 1, 2, 2), 1.2, requires_grad=True),
            "support_logits": torch.zeros((1, 1, 2, 2), requires_grad=True),
            "boundary_logits": torch.full((1, 1, 2, 2), 999.0),
            "obstacle_logits": torch.full((1, 1, 2, 2), -999.0),
        }
        targets = {
            "metric_depth_m": torch.tensor([[[[1.0, 9.0], [9.0, 9.0]]]]),
            "metric_valid": torch.tensor([[[[True, False], [False, False]]]]),
            "metric_tier": torch.tensor(
                [[[[TIER_A_SOURCE, TIER_UNKNOWN], [TIER_UNKNOWN, TIER_UNKNOWN]]]]
            ),
            "support": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
            "support_valid": torch.tensor([[[[True, False], [False, False]]]]),
            "support_tier": torch.tensor(
                [[[[TIER_A_SOURCE, TIER_UNKNOWN], [TIER_UNKNOWN, TIER_UNKNOWN]]]]
            ),
        }
        changed = copy.deepcopy(targets)
        changed["metric_depth_m"][..., 0, 1:] = -999.0
        changed["metric_depth_m"][..., 1, :] = -999.0
        changed["support"][..., 0, 1:] = 999.0
        changed["support"][..., 1, :] = 999.0
        class_weights = {"support_pos_weight": 1.0}
        first = compute_depth_support_losses(outputs, targets, class_weights)
        second = compute_depth_support_losses(outputs, changed, class_weights)
        for key in first:
            torch.testing.assert_close(first[key], second[key])
        self.assertNotIn("raw/boundary_bce", first)
        self.assertNotIn("raw/obstacle_bce", first)

    def test_precision_profile_masks_unknown_and_backpropagates_both_factors(self) -> None:
        outputs = {
            "depth_m": torch.full((1, 1, 2, 2), 1.2, requires_grad=True),
            "base_depth_m": torch.ones((1, 1, 2, 2)),
            "depth_identity_gate": torch.full(
                (1, 1, 2, 2), 0.05, requires_grad=True
            ),
            "support_logits": torch.zeros((1, 1, 2, 2), requires_grad=True),
        }
        targets = {
            "metric_depth_m": torch.tensor([[[[1.0, float("nan")], [9.0, 9.0]]]]),
            "metric_valid": torch.tensor([[[[True, False], [False, False]]]]),
            "metric_tier": torch.tensor(
                [[[[TIER_A_SOURCE, TIER_UNKNOWN], [TIER_UNKNOWN, TIER_UNKNOWN]]]]
            ),
            "support": torch.tensor([[[[1.0, float("nan")], [0.0, 0.0]]]]),
            "support_valid": torch.tensor([[[[True, False], [False, False]]]]),
            "support_tier": torch.tensor(
                [[[[TIER_A_SOURCE, TIER_UNKNOWN], [TIER_UNKNOWN, TIER_UNKNOWN]]]]
            ),
        }
        changed = copy.deepcopy(targets)
        changed["metric_depth_m"][..., 0, 1:] = -999.0
        changed["metric_depth_m"][..., 1, :] = -999.0
        changed["support"][..., 0, 1:] = 999.0
        changed["support"][..., 1, :] = 999.0
        first = compute_depth_support_precision_losses(outputs, targets)
        second = compute_depth_support_precision_losses(outputs, changed)
        for key in first:
            torch.testing.assert_close(first[key], second[key])
        first["total"].backward()
        self.assertIn("raw/depth_identity_gate_bce", first)
        self.assertTrue(torch.isfinite(outputs["depth_m"].grad).all())
        self.assertTrue(torch.isfinite(outputs["depth_identity_gate"].grad).all())
        self.assertTrue(torch.isfinite(outputs["support_logits"].grad).all())

    def test_train_only_scalar_support_calibration_reduces_bce(self) -> None:
        fitted = fit_scalar_support_calibration(
            torch.tensor([2.0, 3.0, 5.0, 6.0]),
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            torch.ones(4),
            steps=120,
        )
        self.assertLess(
            fitted["train_weighted_bce_after"],
            fitted["train_weighted_bce_before"],
        )
        self.assertGreater(fitted["temperature"], 0.0)

    def test_boundary_only_loss_ignores_unknown_pixels(self) -> None:
        outputs = {
            "boundary_logits": torch.zeros((1, 1, 2, 2), requires_grad=True),
            "boundary_distance_px": torch.full(
                (1, 1, 2, 2), 8.0, requires_grad=True
            ),
        }
        targets = {
            "boundary_distance_px": torch.tensor(
                [[[[0.0, 32.0], [32.0, 32.0]]]]
            ),
            "evidence_valid": torch.tensor(
                [[[[True, False], [False, False]]]]
            ),
            "evidence_tier": torch.tensor(
                [[[[TIER_A_SOURCE, TIER_UNKNOWN], [TIER_UNKNOWN, TIER_UNKNOWN]]]]
            ),
        }
        changed = copy.deepcopy(targets)
        changed["boundary_distance_px"][..., 0, 1:] = 0.0
        changed["boundary_distance_px"][..., 1, :] = 0.0
        first = compute_boundary_only_losses(outputs, targets)
        second = compute_boundary_only_losses(outputs, changed)
        for key in first:
            torch.testing.assert_close(first[key], second[key])
        first["total"].backward()
        self.assertTrue(torch.isfinite(outputs["boundary_logits"].grad).all())
        self.assertTrue(torch.isfinite(outputs["boundary_distance_px"].grad).all())


if __name__ == "__main__":
    unittest.main()
