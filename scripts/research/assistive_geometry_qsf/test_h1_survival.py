from __future__ import annotations

import io
import unittest

import torch

from scripts.research.assistive_geometry.assistive_geometry_model import (
    AssistiveTaskHeads,
    horizontal_flip_batch,
)
from scripts.research.assistive_geometry_qsf.h1_survival import (
    HAZARD_BINS,
    QsfH1TaskHeads,
    compile_h1_targets,
    decode_h1_outputs,
    discrete_survival_nll,
    h1_parameter_budget,
)


def _targets() -> dict[str, torch.Tensor]:
    return {
        "dense_depth_m": torch.ones(2, 1, 2, 6),
        "depth_valid": torch.ones(2, 1, 2, 6, dtype=torch.bool),
        "ground_probability": torch.ones(2, 1, 2, 6),
        "ground_label_valid": torch.ones(2, 1, 2, 6, dtype=torch.bool),
        "ground_plane_valid": torch.ones(2, dtype=torch.bool),
        "camera_height_m": torch.ones(2),
        "up_camera": torch.tensor([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]]),
        "intrinsics_tensor": torch.eye(3).repeat(2, 1, 1),
        "clearance_m": torch.tensor([[0.4, 1.2, 0.0], [2.4, 0.0, 1.8]]),
        "clearance_valid": torch.tensor([[True, True, False], [True, False, True]]),
        "occupancy": torch.tensor(
            [
                [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        ),
        "occupancy_valid": torch.tensor(
            [
                [[True, True, True], [True, True, True], [True, True, True]],
                [[True, True, True], [False, False, False], [True, True, True]],
            ]
        ),
    }


class H1SurvivalTest(unittest.TestCase):
    def test_target_compiler_separates_event_censor_and_unknown(self) -> None:
        compiled = compile_h1_targets(_targets())
        self.assertEqual([[0, 2, -1], [-1, -1, 3]], compiled["event_bin"].tolist())
        self.assertEqual(
            [[False, False, True], [True, False, False]],
            compiled["right_censored"].tolist(),
        )
        self.assertEqual(
            [[True, True, True], [True, False, True]],
            compiled["distribution_valid"].tolist(),
        )

    def test_unknown_is_not_silently_compiled_as_right_censor(self) -> None:
        targets = _targets()
        targets["occupancy_valid"][0, 2, 2] = False
        compiled = compile_h1_targets(targets)
        self.assertTrue(bool(compiled["unknown"][0, 2]))
        self.assertFalse(bool(compiled["right_censored"][0, 2]))

    def test_hazard_decoder_is_monotone_and_normalized(self) -> None:
        logits = torch.randn(5, 3, HAZARD_BINS)
        decoded = decode_h1_outputs(logits)
        differences = torch.diff(decoded["occupancy_cdf"], dim=-1)
        self.assertTrue(bool((differences >= 0.0).all()))
        total = decoded["event_probability"].sum(dim=-1) + decoded["tail_probability"]
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1e-6))
        self.assertTrue(bool((decoded["clearance_m"] >= 0.0).all()))
        self.assertTrue(bool((decoded["clearance_m"] <= 2.0).all()))

    def test_right_censor_and_event_gradients_have_correct_direction(self) -> None:
        compiled = compile_h1_targets(_targets())
        logits = torch.zeros(2, 3, HAZARD_BINS, requires_grad=True)
        loss, _ = discrete_survival_nll(logits, compiled)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertLess(float(logits.grad[0, 0, 0]), 0.0)
        self.assertGreater(float(logits.grad[0, 2].sum()), 0.0)
        self.assertEqual(0.0, float(logits.grad[1, 1].abs().sum()))

    def test_zero_support_has_zero_loss_and_gradient(self) -> None:
        targets = _targets()
        targets["clearance_m"].zero_()
        targets["clearance_valid"].zero_()
        targets["occupancy"].zero_()
        targets["occupancy_valid"].zero_()
        compiled = compile_h1_targets(targets)
        logits = torch.randn(2, 3, HAZARD_BINS, requires_grad=True)
        loss, _ = discrete_survival_nll(logits, compiled)
        self.assertEqual(0.0, float(loss.detach()))
        loss.backward()
        self.assertEqual(0.0, float(logits.grad.abs().sum()))

    def test_horizontal_flip_commutes_with_target_compilation(self) -> None:
        original = compile_h1_targets(_targets())
        flipped = compile_h1_targets(horizontal_flip_batch(_targets()))
        for key in (
            "event_bin",
            "event_observed",
            "right_censored",
            "distribution_valid",
            "unknown",
        ):
            self.assertTrue(torch.equal(flipped[key], torch.flip(original[key], dims=(1,))))

    def test_task_head_parameter_budget_matches_direct_head_exactly(self) -> None:
        budget = h1_parameter_budget()
        self.assertTrue(budget["exact_match"])
        self.assertEqual(
            sum(value.numel() for value in AssistiveTaskHeads().parameters()),
            sum(value.numel() for value in QsfH1TaskHeads().parameters()),
        )

    def test_checkpoint_round_trip_and_output_schema(self) -> None:
        torch.manual_seed(17)
        source = QsfH1TaskHeads()
        feature = torch.randn(2, 48, 3, 6)
        expected = source(feature, (12, 24))
        buffer = io.BytesIO()
        torch.save(source.state_dict(), buffer)
        buffer.seek(0)
        restored = QsfH1TaskHeads()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        actual = restored(feature, (12, 24))
        self.assertEqual(set(expected), set(actual))
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_invalid_or_contradictory_payloads_fail_closed(self) -> None:
        targets = _targets()
        targets["occupancy"][0, 0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "disagree"):
            compile_h1_targets(targets)
        targets = _targets()
        targets["occupancy"][0, 2] = torch.tensor([0.0, 1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "monotone"):
            compile_h1_targets(targets)


if __name__ == "__main__":
    unittest.main()
