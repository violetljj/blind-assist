from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_attempt06_uncertainty_calibrators import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    SUPPORT_FEATURE_NAMES,
    _load_calibration_rows,
    load_attempt06_calibrators,
    support_features,
)


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
    }


class Attempt06CalibratorTest(unittest.TestCase):
    def test_support_feature_shape_and_finiteness(self) -> None:
        members = [member(seed) for seed in (17, 29, 43)]
        base = torch.ones((1, 1, 6, 8)) * 1.5
        factors = {
            "support_residual_sigma_m": torch.tensor([0.7]),
            "camera_height_m": torch.tensor([1.2]),
        }
        features = support_features(
            members,
            base,
            factors,
            {"geometry_pixel_count": 24},
            "PORTRAIT_ROT90_CLOCKWISE",
        )
        self.assertEqual(features.shape, (1, len(SUPPORT_FEATURE_NAMES)))
        self.assertTrue(bool(torch.isfinite(features).all()))
        self.assertEqual(float(features[0, -1]), 1.0)

    def test_calibration_roster_excludes_new_selection_and_preserved_canary(self) -> None:
        rows, receipt = _load_calibration_rows()
        parents = {row["parent_id"] for row in rows}
        self.assertEqual(len(rows), 69)
        self.assertEqual(len(parents), 23)
        self.assertTrue(parents.isdisjoint(receipt["attempt06_selection_excluded"]["parents"]))
        self.assertTrue(parents.isdisjoint(receipt["preserved_canary_excluded"]["parents"]))

    def test_checkpoint_loads_three_calibrators(self) -> None:
        checkpoint = DEFAULT_OUTPUT_DIR / "uncertainty_calibrators.pt"
        depth, boundary, support, metadata = load_attempt06_calibrators(checkpoint, torch.device("cpu"))
        self.assertEqual(depth.feature_mean.numel(), 14)
        self.assertEqual(boundary.feature_mean.numel(), 16)
        self.assertEqual(support.feature_mean.numel(), len(SUPPORT_FEATURE_NAMES))
        self.assertIn("data_receipt", metadata)


if __name__ == "__main__":
    unittest.main()
