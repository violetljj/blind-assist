from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_attempt05_uncertainty_calibrators import (  # noqa: E402
    BOUNDARY_FEATURE_NAMES,
    DEPTH_FEATURE_NAMES,
    PixelScaleCalibrator,
    calibration_features,
)
from train_ag_r2_f1_factor_learnability_attempt05 import calibrated_outputs  # noqa: E402


def member(seed: int, height: int = 6, width: int = 8) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    probability = lambda: torch.rand((1, 1, height, width), generator=generator).clamp(0.02, 0.98)
    return {
        "predicted_log_depth": torch.randn((1, 1, height, width), generator=generator) * 0.2,
        "depth_log_sigma": torch.randn((1, 1, height, width), generator=generator) * 0.1 - 0.5,
        "depth_valid_probability": probability(),
        "support_probability": probability(),
        "obstacle_probability": probability(),
        "boundary_probability": probability(),
        "boundary_sigma_px": probability() * 10.0 + 0.5,
        "evidence_valid_probability": probability(),
        "support_plane_normal_camera_xyz": torch.tensor([[0.0, 1.0, 0.0]]),
        "camera_height_m": torch.tensor([1.2]),
        "support_residual_sigma_m": torch.tensor([0.5]),
        "support_valid_probability": torch.tensor([0.9]),
        "depth_gate": torch.tensor([0.2]),
    }


class Attempt05CalibratorTest(unittest.TestCase):
    def test_feature_shapes_and_finiteness(self) -> None:
        members = [member(seed) for seed in (17, 29, 43)]
        base = torch.ones((1, 1, 6, 8)) * 1.5
        depth, boundary = calibration_features(members, base)
        self.assertEqual(depth.shape, (1, len(DEPTH_FEATURE_NAMES), 6, 8))
        self.assertEqual(boundary.shape, (1, len(BOUNDARY_FEATURE_NAMES), 6, 8))
        self.assertTrue(bool(torch.isfinite(depth).all()))
        self.assertTrue(bool(torch.isfinite(boundary).all()))

    def test_pointwise_calibrator_2d_and_4d_agree(self) -> None:
        model = PixelScaleCalibrator(torch.zeros(4), torch.ones(4), 8, 0.01, 3.0)
        features = torch.randn((1, 4, 5, 7))
        image = model(features)
        flat = model(features.permute(0, 2, 3, 1).reshape(-1, 4)).reshape(1, 1, 5, 7)
        torch.testing.assert_close(image, flat)

    def test_attempt05_changes_only_two_sigma_fields(self) -> None:
        members = [member(seed) for seed in (17, 29, 43)]
        depth = PixelScaleCalibrator(
            torch.zeros(len(DEPTH_FEATURE_NAMES)),
            torch.ones(len(DEPTH_FEATURE_NAMES)),
            8,
            0.01,
            3.0,
        )
        boundary = PixelScaleCalibrator(
            torch.zeros(len(BOUNDARY_FEATURE_NAMES)),
            torch.ones(len(BOUNDARY_FEATURE_NAMES)),
            8,
            0.25,
            32.0,
        )
        sample = SimpleNamespace(base_depth_feature=torch.ones((1, 6, 8)), native_hw=(6, 8))
        result = calibrated_outputs(
            [[members[0]], [members[1]], [members[2]]],
            [{"sample": sample}],
            depth,
            boundary,
            torch.device("cpu"),
        )[0]
        self.assertEqual(result["depth_log_sigma"].shape, members[0]["depth_log_sigma"].shape)
        self.assertEqual(result["boundary_sigma_px"].shape, members[0]["boundary_sigma_px"].shape)
        for key in members[0]:
            if key not in {"depth_log_sigma", "boundary_sigma_px"}:
                self.assertTrue(torch.equal(result[key], members[0][key]), key)


if __name__ == "__main__":
    unittest.main()
