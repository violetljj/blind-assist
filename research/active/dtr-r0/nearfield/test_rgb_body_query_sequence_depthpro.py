"""Floating-point camera fixtures independent of any model or evaluator truth."""
import unittest
import numpy as np
from rgb_body_query_geometry import resize_intrinsics
from rgb_body_query_sequence_depthpro import centered_camera, coordinate_map, valid_cells


class CameraFixtures(unittest.TestCase):
    def setUp(self):
        self.k = np.array([[830., 0., 1100.7], [0., 812., 601.1], [0., 0., 1.]])
        self.shape = (1242, 2208)

    def test_full_pixel_cell_extent_and_center(self):
        ck, (h, w) = centered_camera(self.k, self.shape)
        self.assertEqual(ck[0, 0], ck[1, 1])
        self.assertEqual(ck[0, 2], (w-1)/2)
        self.assertEqual(ck[1, 2], (h-1)/2)
        for u, v in [(-.5, -.5), (2207.5, 1241.5)]:
            ray = np.linalg.solve(self.k, [u, v, 1.])
            pixel = ck @ ray
            self.assertTrue(-.5 <= pixel[0] <= w-.5)
            self.assertTrue(-.5 <= pixel[1] <= h-.5)

    def test_exact_roundtrip_rays_no_optical_z_change(self):
        ck, _ = centered_camera(self.k, self.shape)
        uv = np.array([[0., 0., 1.], [2207., 1241., 1.], [1134.2, 523.6, 1.]]).T
        rays = np.linalg.solve(self.k, uv)
        np.testing.assert_allclose(np.linalg.solve(ck, ck @ rays), rays, atol=1e-14)
        np.testing.assert_allclose(self.k @ np.linalg.solve(ck, ck @ rays), uv, atol=1e-12)
        self.assertTrue(np.all(rays[2] == 1))

    def test_low_focal_ratio_half_pixel_and_aspect(self):
        ck, shape = centered_camera(self.k, self.shape)
        low_shape = (round(128*shape[0]/shape[1]), 128)
        lk = resize_intrinsics(ck, shape, low_shape)
        self.assertAlmostEqual(lk[0, 0]/128, ck[0, 0]/shape[1])
        self.assertAlmostEqual(lk[0, 2], (128-1)/2)
        self.assertAlmostEqual(lk[1, 2], (low_shape[0]-1)/2)
        point = np.array([1500., 800., 1.])
        centered = ck @ np.linalg.solve(self.k, point)
        low = lk @ np.linalg.solve(self.k, point)
        expected = (centered[:2]+.5)*np.array([128/shape[1], low_shape[0]/shape[0]])-.5
        np.testing.assert_allclose(low[:2], expected, atol=1e-12)

    def test_common_grid_mapping_and_invalid_padding(self):
        ck, shape = centered_camera(self.k, self.shape)
        grid = resize_intrinsics(self.k, self.shape, (720, 1280))
        x, y = coordinate_map(ck, grid, (720, 1280))
        self.assertTrue(valid_cells(x, y, shape).all())
        sx, sy = coordinate_map(self.k, ck, shape)
        self.assertFalse(valid_cells(sx, sy, self.shape).all())
        ray = np.linalg.solve(grid, [123., 45., 1.])
        expected = ck @ ray
        np.testing.assert_allclose([x[45, 123], y[45, 123]], expected[:2], atol=.0002)


if __name__ == '__main__':
    unittest.main()
