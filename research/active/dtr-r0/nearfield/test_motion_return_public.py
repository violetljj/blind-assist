import unittest
import numpy as np
from motion_return_public import extract, projection_residual, sample_grid, CENTER


class MotionPublicTest(unittest.TestCase):
    def test_projection_ratio_and_center(self):
        p = np.array([[100., 80.], CENTER])
        q = CENTER+(p-CENTER)*1.5
        np.testing.assert_allclose(projection_residual(q, p, [2., 2.], [3., 3.], [True, True]), 0, atol=1e-12)

    def test_missing_range_and_explicit_validity(self):
        p = np.tile(CENTER, (4, 1))
        self.assertTrue(np.isnan(projection_residual(p, p, [2, 0, 2, 8], [2, 2, np.nan, 2], [False, True, True, True])).all())

    def test_grid_uses_original_native_pixel_centers(self):
        flat, xy = sample_grid()
        self.assertEqual(len(flat), 3072)
        self.assertEqual(flat[0], 0); self.assertEqual(flat[-1], 188*256+252)
        np.testing.assert_allclose(xy[0], [.25, -.25])

    def fixture(self):
        rng = np.random.default_rng(41)
        image = rng.integers(0, 256, (3, 180, 320), dtype=np.uint8)
        tof = np.zeros((64, 6)); tof[:, :2] = [2/8, 1]
        for i in range(64):
            y, x = divmod(i, 8)
            tof[i, 2:] = [.2+y*.6/8, .2+x*.6/8, .2+(y+1)*.6/8, .2+(x+1)*.6/8]
        return image, tof

    def test_stationary_texture_and_missing_ranges(self):
        image, tof = self.fixture(); out = extract(image, image, tof, tof)
        self.assertGreater(out['matched'].sum(), 900)
        self.assertTrue((out['fb_error'][out['tracked']] <= 1).all())
        np.testing.assert_allclose(out['residual'][out['matched']], 0, atol=1e-5)
        np.testing.assert_allclose(out['static_residual'][out['matched']], 0, atol=1e-5)
        self.assertTrue(np.isnan(out['residual'][~out['matched']]).all())
        broken = tof.copy(); broken[:, 1] = 0
        other = extract(image, image, tof, broken)
        self.assertFalse(other['matched'].any()); self.assertTrue(np.isnan(other['residual']).all())
        self.assertTrue((other['missing_reason'][other['tracked']] == 5).all())
        np.testing.assert_array_equal(out['current_zone'], other['current_zone'])


if __name__ == '__main__':
    unittest.main()
