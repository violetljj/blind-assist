"""Focused camera geometry checks, without models or research labels."""
import unittest
import numpy as np
from rgb_body_query_geometry import resize_intrinsics, visible_readout, camera_queries


class GeometryTests(unittest.TestCase):
    def test_optical_z_is_not_radial_range(self):
        # Pixel ray=(1,0,1); axial Z=2 gives x=2, radial R=2 gives x=sqrt(2).
        k = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        query = [dict(name='off_axis', low=[1.9, -.1, 1.9], high=[2.1, .1, 2.1])]
        axial = np.array([[np.nan, 2.]])
        radial_to_z = np.array([[np.nan, 2./np.sqrt(2)]])
        self.assertEqual(visible_readout(axial, k, (1, 2), query)['queries'][0]['visible_support_pixels'], 1)
        self.assertEqual(visible_readout(radial_to_z, k, (1, 2), query)['queries'][0]['state'], 'UNKNOWN')

    def test_half_pixel_K_resize_preserves_ray(self):
        k = np.array([[500., 0., 321.2], [0., 490., 181.4], [0., 0., 1.]])
        resized = resize_intrinsics(k, (360, 640), (72, 128))
        uv = np.array([103., 70.])
        low_uv = (uv+.5)*.2-.5
        np.testing.assert_allclose(np.linalg.inv(k)@[*uv, 1.], np.linalg.inv(resized)@[*low_uv, 1.], atol=1e-14)
        self.assertAlmostEqual(resized[0, 2], (321.2+.5)*.2-.5)

    def test_missing_and_nonpositive_are_UNKNOWN(self):
        depth = np.array([[np.nan, np.inf, -1., 0.]])
        result = visible_readout(depth, np.eye(3), depth.shape)
        self.assertEqual(result['invalid_depth_pixels'], 4)
        self.assertTrue(all(q['state'] == 'UNKNOWN' for q in result['queries']))

    def test_closed_shared_depth_boundary(self):
        result = visible_readout(np.array([[3.]]), np.eye(3), (1, 1))
        supported = [q['name'] for q in result['queries'] if q['state'] == 'ESTIMATED_VISIBLE_SUPPORT']
        self.assertEqual(supported, ['center_near', 'center_far'])
        self.assertTrue(all(q['visible_support_pixels'] >= 0 for q in result['queries']))

    def test_closed_xyz_bounds_and_outside(self):
        # Four pixels at x=-.3,0,.3,.6; closed center has first three only.
        k = np.array([[10., 0., 1.], [0., 10., 0.], [0., 0., 1.]])
        depth = np.full((1, 4), 3.)
        q = [dict(name='center', low=[-.3, 0., 3.], high=[.3, 0., 3.])]
        result = visible_readout(depth, k, depth.shape, q)
        self.assertEqual(result['queries'][0]['visible_support_pixels'], 3)

    def test_shape_and_threshold_contract(self):
        with self.assertRaises(ValueError):
            visible_readout(np.ones((2, 2)), np.eye(3), (1, 4))
        with self.assertRaises(ValueError):
            visible_readout(np.ones((1, 1)), np.eye(3), (1, 1), min_support=0)
        self.assertEqual(len(camera_queries()), 6)


if __name__ == '__main__':
    unittest.main()
