#!/usr/bin/env python3

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from scripts.research.assistive_geometry.run_ag_r2_hybrid_factor_student_to_ag_seam import (
    composed_student_log_depth,
    estimate_height,
    height_observations,
    scale_intrinsics,
    weighted_quantile,
)


class HybridFactorStudentSeamTest(unittest.TestCase):
    def test_weighted_quantile_respects_weight_mass(self) -> None:
        value = weighted_quantile(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([1.0, 10.0, 1.0]),
            0.5,
        )
        self.assertAlmostEqual(value, 2.0)

    def test_intrinsics_scaling_preserves_pixel_centers(self) -> None:
        source = np.asarray(
            [[400.0, 0.0, 319.5], [0.0, 400.0, 239.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        scaled = scale_intrinsics(source, (480, 640), (120, 160))
        np.testing.assert_allclose(
            scaled,
            np.asarray(
                [[100.0, 0.0, 79.5], [0.0, 100.0, 59.5], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
        )

    def test_horizontal_plane_recovers_camera_height(self) -> None:
        height, width = 16, 16
        intrinsics = np.asarray(
            [[100.0, 0.0, 7.5], [0.0, 5.0, 7.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        gravity = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
        rows, _ = np.indices((height, width), dtype=np.float64)
        ray_y = (rows - intrinsics[1, 2]) / intrinsics[1, 1]
        # Only the lower half sees a horizontal floor 1.2 m below the camera.
        depth = np.full((height, width), 1.0, dtype=np.float64)
        lower = ray_y > 0.0
        depth[lower] = 1.2 / ray_y[lower]
        support = lower.astype(np.float64)
        valid_probability = np.ones((height, width), dtype=np.float64)
        observations, weights = height_observations(
            depth,
            support,
            valid_probability,
            intrinsics,
            gravity,
            0.5,
        )
        estimated, sigma = estimate_height(
            observations,
            weights,
            "weighted_quantile",
            0.5,
        )
        self.assertGreaterEqual(observations.size, 64)
        self.assertAlmostEqual(estimated, 1.2, places=5)
        self.assertAlmostEqual(sigma, 0.01, places=5)

    def test_metric_student_output_is_not_recentered(self) -> None:
        base = torch.tensor([[[1.0, 2.0], [4.0, 8.0]]])
        correction = torch.tensor([[[[0.4, -0.2], [0.1, 0.3]]]])
        sample = SimpleNamespace(base_depth_feature=base)
        predicted, global_correction = composed_student_log_depth(
            sample,
            {
                "metric_depth_student_log_depth": base[None].log() + correction,
                "metric_depth_student_global_correction": torch.tensor([0.25]),
            },
        )
        self.assertAlmostEqual(global_correction, 0.25, places=6)
        self.assertAlmostEqual(
            float(predicted.mean()),
            float(base.log().mean() + correction.mean()),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
