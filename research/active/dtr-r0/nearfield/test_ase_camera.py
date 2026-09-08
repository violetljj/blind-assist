"""Offline mathematical consistency checks; no assertion of SDK parity."""
import unittest
import numpy as np
from ase_camera import AseCalibration


class AseCameraTests(unittest.TestCase):
    def test_axis_and_transform(self):
        calibration = AseCalibration()
        np.testing.assert_allclose(calibration.project([0, 0, 2]), calibration.principal_point)
        np.testing.assert_allclose(calibration.unproject(calibration.principal_point), [0, 0, 1], atol=1e-12)
        transform = calibration.T_device_camera
        np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(transform[:3, 3], [-.007530096566173914, -.010908549841580260, -.003598063315542823])

    def test_vectorized_roundtrip_and_norm(self):
        theta, phi = np.meshgrid(np.linspace(0, .99, 30), np.linspace(-np.pi, np.pi, 41))
        rays = np.stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)), axis=-1)
        for size in (704, 1408):
            calibration = AseCalibration(size)
            uv = calibration.project(rays)
            valid = np.isfinite(uv).all(axis=-1)
            self.assertGreater(valid.mean(), .95)
            # Some edge angles land beyond the asymmetric sensor boundary.
            uv = uv[valid]
            reconstructed = calibration.unproject(uv)
            np.testing.assert_allclose(reconstructed, rays[valid], atol=1e-10)
            np.testing.assert_allclose(np.linalg.norm(reconstructed, axis=-1), 1., atol=1e-12)
            np.testing.assert_allclose(calibration.project(reconstructed), uv, atol=1e-7)
            # Bisection requires a monotone polynomial over the admitted cone.
            self.assertTrue((np.diff(calibration._radial(np.linspace(0, 1, 1001))) > 0).all())

    def test_resized_pixel_center(self):
        small, large = AseCalibration(704), AseCalibration(1408)
        np.testing.assert_allclose(large.principal_point, 2*(small.principal_point+.5)-.5)
        ray = [.3, -.2, 1.]
        np.testing.assert_allclose(large.project(ray), 2*(small.project(ray)+.5)-.5)
        self.assertEqual(large.valid_radius, 707.5)

    def test_invalid_and_outside_cone(self):
        calibration = AseCalibration()
        points = [[0, 0, -1], [0, 0, 0], [np.nan, 0, 1], [np.sin(1.01), 0, np.cos(1.01)]]
        self.assertTrue(np.isnan(calibration.project(points)).all())
        self.assertTrue(np.isnan(calibration.unproject([[np.nan, 0], [-1, 0], [704, 350], [0, 0]])).all())
        # Within the sensor and valid-radius disk, but past the angle cap.
        self.assertTrue(np.isnan(calibration.unproject(calibration.principal_point+[-352, 0])).all())


if __name__ == '__main__':
    unittest.main()
