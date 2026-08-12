#!/usr/bin/env python3
"""Focused deterministic tests for Attempt-03 support geometry."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_ag_r2_f1_factor_learnability_attempt03 import (
    MAX_HEIGHT_M,
    MIN_HEIGHT_M,
    geometry_height_and_sigma,
    height_candidates,
    sigma_candidates,
    weighted_quantile,
)


class Attempt03GeometryTest(unittest.TestCase):
    def test_weighted_quantile_respects_weight_mass(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        weights = torch.tensor([1.0, 1.0, 8.0, 1.0])
        self.assertEqual(float(weighted_quantile(values, weights, 0.5)), 3.0)

    def test_candidate_grid_is_deterministic_and_physical(self) -> None:
        height = height_candidates()
        sigma = sigma_candidates()
        self.assertEqual(len(height), 256)
        self.assertEqual(len(sigma), 24)
        self.assertEqual(len({str(row) for row in height}), len(height))
        self.assertEqual(len({str(row) for row in sigma}), len(sigma))
        self.assertGreater(MIN_HEIGHT_M, 0.0)
        self.assertGreater(MAX_HEIGHT_M, MIN_HEIGHT_M)

    def test_geometry_recovers_horizontal_plane_in_both_display_bases(self) -> None:
        height, width = 40, 50
        true_height = 1.25
        k = np.asarray([[80.0, 0.0, 24.5], [0.0, 80.0, 19.5], [0.0, 0.0, 1.0]], dtype=np.float32)
        height_config = {
            "depth_source": "predicted",
            "support_threshold": 0.15,
            "support_power": 1.0,
            "estimator": "weighted_quantile",
            "quantile": 0.5,
            "mode_bin_m": 0.04,
            "mode_radius_m": 0.12,
            "scope": "frame_camera_plane",
            "metric_scale_calibration": "none",
        }
        sigma_config = {"source": "geometry_mad", "multiplier": 1.0}
        for gravity, horizontal_axis in ((np.asarray([0.0, -1.0, 0.0], np.float32), "y"), (np.asarray([-1.0, 0.0, 0.0], np.float32), "x")):
            yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
            denominator = (yy - k[1, 2]) / k[1, 1] if horizontal_axis == "y" else (xx - k[0, 2]) / k[0, 0]
            valid = denominator > 0.05
            depth = np.ones((height, width), dtype=np.float32)
            depth[valid] = true_height / denominator[valid]
            support = np.zeros_like(depth)
            support[valid] = 0.95
            with tempfile.TemporaryDirectory() as directory:
                label = Path(directory) / "input.npz"
                np.savez_compressed(label, intrinsics_output=k, gravity_up_camera_xyz=gravity)
                sample = SimpleNamespace(
                    label_path=label,
                    native_hw=(height, width),
                    base_depth_feature=torch.from_numpy(depth)[None],
                )
                outputs = {
                    "predicted_log_depth": torch.from_numpy(np.log(depth))[None, None],
                    "support_probability": torch.from_numpy(support)[None, None],
                    "depth_valid_probability": torch.from_numpy(valid.astype(np.float32))[None, None],
                    "support_plane_normal_camera_xyz": torch.tensor([[0.0, -1.0, 0.0]]),
                    "camera_height_m": torch.tensor([0.8]),
                    "support_residual_sigma_m": torch.tensor([0.5]),
                }
                normal, estimated, sigma, receipt = geometry_height_and_sigma(
                    outputs, sample, height_config, sigma_config, {}, torch.device("cpu")
                )
                self.assertTrue(receipt["gravity_valid"])
                self.assertIsNone(receipt["fallback"])
                self.assertAlmostEqual(float(estimated[0]), true_height, places=4)
                self.assertTrue(torch.allclose(normal[0], torch.from_numpy(gravity), atol=1.0e-6))
                self.assertGreaterEqual(float(sigma[0]), 0.0299)


if __name__ == "__main__":
    unittest.main()
