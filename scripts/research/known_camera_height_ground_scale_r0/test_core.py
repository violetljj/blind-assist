import unittest

import numpy as np

from core import (
    CameraHeightReceipt,
    causal_median_scale,
    fit_relative_ground_plane,
    recover_metric_scale,
    relative_depth_to_points,
)


class KnownCameraHeightGroundScaleR0Test(unittest.TestCase):
    def test_causal_median_uses_only_fixed_trailing_window(self):
        self.assertEqual(causal_median_scale([100.0, 1.0, 3.0, 2.0], 3), 2.0)

    def test_causal_median_rejects_invalid_history(self):
        with self.assertRaises(ValueError):
            causal_median_scale([])
        with self.assertRaises(ValueError):
            causal_median_scale([1.0, float("nan")])

    def setUp(self) -> None:
        self.height = 120
        self.width = 160
        self.intrinsics = np.asarray(
            [[120.0, 0.0, 79.5], [0.0, 120.0, 59.5], [0.0, 0.0, 1.0]]
        )
        self.receipt = CameraHeightReceipt("camera-a", "mount-a", 1.2, 0.03)

    def planar_depth(self, camera_height: float, scale: float = 1.0) -> np.ndarray:
        rows = np.arange(self.height, dtype=np.float64)[:, None]
        denominator = (rows - self.intrinsics[1, 2]) / self.intrinsics[1, 1]
        depth = np.zeros((self.height, self.width), dtype=np.float64)
        valid = denominator[:, 0] > 0.0
        depth[valid, :] = camera_height / denominator[valid]
        return depth / scale

    def test_recovers_known_scale(self) -> None:
        relative = self.planar_depth(1.2, scale=2.5)
        result = recover_metric_scale(
            relative, self.intrinsics, self.receipt, "camera-a", "mount-a"
        )
        self.assertEqual("VALID", result["status"])
        self.assertAlmostEqual(2.5, result["scale"], places=6)
        np.testing.assert_allclose(result["metric_depth"], self.planar_depth(1.2))

    def test_fit_is_scale_equivariant_and_deterministic(self) -> None:
        depth = self.planar_depth(1.2)
        points, pixels = relative_depth_to_points(depth, self.intrinsics)
        first, reason = fit_relative_ground_plane(points, pixels, self.height)
        self.assertIsNone(reason)
        self.assertIsNotNone(first)
        scaled, scaled_pixels = relative_depth_to_points(depth * 7.0, self.intrinsics)
        second, second_reason = fit_relative_ground_plane(
            scaled, scaled_pixels, self.height
        )
        self.assertIsNone(second_reason)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        self.assertAlmostEqual(first.relative_height * 7.0, second.relative_height, places=6)
        self.assertAlmostEqual(
            first.normalized_median_residual,
            second.normalized_median_residual,
            places=12,
        )

    def test_unknown_on_profile_mismatch(self) -> None:
        result = recover_metric_scale(
            self.planar_depth(1.2),
            self.intrinsics,
            self.receipt,
            "another-camera",
            "mount-a",
        )
        self.assertEqual(
            {"status": "UNKNOWN", "reason": "HEIGHT_PROFILE_IDENTITY_MISMATCH"},
            result,
        )

    def test_unknown_without_ground_support(self) -> None:
        result = recover_metric_scale(
            np.zeros((self.height, self.width)),
            self.intrinsics,
            self.receipt,
            "camera-a",
            "mount-a",
        )
        self.assertEqual("UNKNOWN", result["status"])
        self.assertEqual("INSUFFICIENT_GROUND_CANDIDATES", result["reason"])

    def test_unknown_when_scale_is_out_of_range(self) -> None:
        result = recover_metric_scale(
            self.planar_depth(1.2, scale=10.0),
            self.intrinsics,
            self.receipt,
            "camera-a",
            "mount-a",
        )
        self.assertEqual("UNKNOWN", result["status"])
        self.assertEqual("SCALE_OUT_OF_RANGE", result["reason"])

    def test_height_uncertainty_produces_scale_interval(self) -> None:
        result = recover_metric_scale(
            self.planar_depth(1.2, scale=2.0),
            self.intrinsics,
            self.receipt,
            "camera-a",
            "mount-a",
        )
        self.assertEqual("VALID", result["status"])
        np.testing.assert_allclose(result["scale_interval"], [1.95, 2.05])


if __name__ == "__main__":
    unittest.main()
