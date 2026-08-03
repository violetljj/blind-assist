#!/usr/bin/env python3

import unittest

import cv2
import numpy as np

from calibrate_multizone_tof_rgb import calibrate


class CalibrateMultizoneTofRgbTest(unittest.TestCase):
    def test_recovers_known_rigid_transform(self) -> None:
        camera = np.asarray(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
        )
        vectors = []
        zones = []
        ranges = []
        for depth in (1.2, 2.0, 3.0):
            for index, (x, y) in enumerate(
                ((-0.3, -0.2), (0.0, -0.2), (0.3, -0.2), (-0.3, 0.2), (0.0, 0.2), (0.3, 0.2))
            ):
                ray = np.asarray([x, y, 1.0], dtype=np.float64)
                ray /= np.linalg.norm(ray)
                vectors.append(ray * depth)
                zones.append(str(index))
                ranges.append(depth)
        object_points = np.asarray(vectors)
        true_rotation_vector = np.asarray([0.01, -0.02, 0.015])
        true_translation = np.asarray([0.04, -0.01, 0.02])
        image_points, _ = cv2.projectPoints(
            object_points,
            true_rotation_vector,
            true_translation,
            camera,
            np.zeros(5),
        )
        transform, metrics, gates = calibrate(
            object_points,
            image_points.reshape(-1, 2),
            zones,
            ranges,
            camera,
            minimum_observations=12,
            minimum_zones=6,
            minimum_range_span_m=1.0,
            minimum_inlier_fraction=0.9,
            maximum_rmse_px=0.1,
            ransac_threshold_px=1.0,
        )
        self.assertTrue(all(gates.values()))
        self.assertLess(metrics["reprojection_rmse_px"], 1e-4)
        self.assertTrue(np.allclose(transform[:3, 3], true_translation, atol=1e-4))

    def test_coverage_gates_reject_weak_fixture(self) -> None:
        points = np.asarray([[0.0, 0.0, float(index + 1)] for index in range(6)])
        image = np.asarray([[10.0, 10.0]] * 6)
        _, _, gates = calibrate(
            points,
            image,
            ["one"] * 6,
            [1.0] * 6,
            np.eye(3),
            minimum_observations=12,
            minimum_zones=3,
            minimum_range_span_m=1.0,
            minimum_inlier_fraction=0.8,
            maximum_rmse_px=1.0,
            ransac_threshold_px=1.0,
        )
        self.assertFalse(gates["observation_count"])
        self.assertFalse(gates["zone_coverage"])
        self.assertFalse(gates["range_span"])


if __name__ == "__main__":
    unittest.main()
