#!/usr/bin/env python3

import unittest

import numpy as np

from run_ag_st_icl_pixel_boundary_canary import (
    canonical_camera_to_world,
    downsample_exact_depth,
    exact_mesh_boundary_target,
)


class IclPixelBoundaryCanaryTest(unittest.TestCase):
    def test_icl_y_up_maps_to_canonical_z_up(self) -> None:
        pose = np.eye(4, dtype=np.float64)
        canonical = canonical_camera_to_world(pose)
        gravity_camera = canonical[:3, :3].T @ np.asarray([0.0, 0.0, 1.0])
        np.testing.assert_allclose(gravity_camera, np.asarray([0.0, 1.0, 0.0]))
        self.assertAlmostEqual(1.0, np.linalg.det(canonical[:3, :3]), places=6)

    def test_exact_target_requires_mesh_transition_and_metric_gap(self) -> None:
        labels = np.asarray([[1, 2, 2], [1, 1, -1]], dtype=np.int32)
        points = np.zeros((2, 3, 3), dtype=np.float64)
        points[0, 0, 2] = 1.0
        points[0, 1, 2] = 1.2
        points[0, 2, 2] = 1.21
        points[1, :, 2] = 1.0
        valid = np.ones((2, 3), dtype=np.bool_)
        target, evaluable = exact_mesh_boundary_target(labels, points, valid)
        self.assertTrue(target[0, 0])
        self.assertTrue(target[0, 1])
        self.assertFalse(target[0, 2])
        self.assertTrue(evaluable[0, 0])
        self.assertFalse(evaluable[1, 2])

    def test_icl_negative_fy_is_materialized_as_vertical_raster_flip(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from PIL import Image

        raw = np.zeros((480, 640), dtype=np.uint16)
        raw[477, 2] = 5000
        with TemporaryDirectory() as directory:
            path = Path(directory) / "depth.png"
            Image.fromarray(raw).save(path)
            depth, valid, intrinsics = downsample_exact_depth(path)
        self.assertTrue(valid[0, 0])
        self.assertAlmostEqual(1.0, float(depth[0, 0]))
        self.assertGreater(float(intrinsics[1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
