"""Focused CPU checks of geometry, distance semantics and frozen partitions."""
import unittest
import numpy as np
from ase_body_query_geometry import count_query_points, horizontal_frame, query_boxes, ray_depth_coverage


class GeometryTests(unittest.TestCase):
    def test_ray_distance_is_not_axial_z(self):
        ray = np.array([[0., .6, .8]])
        result = ray_depth_coverage(ray, [2.], np.eye(4), [0., 1., 0.])
        np.testing.assert_allclose(result['points_proxy'], [[1.6, 0., .5]])
        self.assertEqual(int(result['raw_counts'].sum()), 0)
        # Treating ray distance as axial Z would instead give X=2, height=.2.
        self.assertNotAlmostEqual(result['points_proxy'][0, 0], 2.)

    def test_rigid_world_rotation_and_translation_invariant(self):
        rays = np.array([[0., 0., 1.], [.1, .2, 1.]])
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        original = ray_depth_coverage(rays, [2., 2.], np.eye(4), [0., 1., 0.])
        angle = .71
        rot = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        transform = np.eye(4); transform[:3, :3] = rot; transform[:3, 3] = [10., -50., 3.]
        moved = ray_depth_coverage(rays, [2., 2.], transform, rot @ [0., 1., 0.])
        np.testing.assert_allclose(original['points_proxy'], moved['points_proxy'], atol=1e-12)
        np.testing.assert_array_equal(original['raw_counts'], moved['raw_counts'])

    def test_pitch_preserved_in_points_but_heading_horizontal(self):
        a = np.pi / 6
        transform = np.eye(4)
        transform[:3, :3] = [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
        result = ray_depth_coverage([[0, 0, 1]], [2.], transform, [0, 1, 0])
        np.testing.assert_allclose(result['points_proxy'], [[np.sqrt(3), 0, 2.7]], atol=1e-12)
        np.testing.assert_allclose(result['frame_world'][:, 0], [0, 0, 1], atol=1e-12)

    def test_internal_boundaries_and_shared_height(self):
        boxes = query_boxes()
        points = np.array([[boxes[0, 1, 0], boxes[0, 1, 1], 1.0],
                           [3.18, .28, .65], [.13, -.18, 1.85],
                           [1., 0., 1.4], [3.180001, 0., 1.0]])
        result = count_query_points(points)
        membership = result['membership']
        self.assertTrue(membership[4, 0])
        self.assertTrue(membership[5, 1])
        self.assertTrue(membership[6, 2])
        np.testing.assert_array_equal(membership.sum(axis=0), [1, 1, 1, 2, 0])
        self.assertTrue((membership[:6].sum(axis=0) <= 1).all())
        self.assertTrue((membership[6:].sum(axis=0) <= 1).all())

    def test_count_cap_near_and_unknown(self):
        points = np.array([[1., -.2, 1.], [1., 0., 1.], [1., .2, 1.]] + [[2., 0., 1.7]] * 5 + [[np.nan, 0, 0]])
        result = count_query_points(points)
        self.assertEqual(result['four_way'], 'BOTH')
        np.testing.assert_array_equal(result['near'], [True, True])
        self.assertEqual(result['raw_counts'].sum(), 8)
        self.assertEqual(result['capped_counts'].sum(), 6)
        self.assertEqual(result['unknown_count'], 1)
        self.assertEqual(sum(result['four_way_flags'].values()), 1)

    def test_invalid_depth_and_rays_remain_unknown(self):
        rays = np.tile([0., 0., 1.], (7, 1)); rays[5] = 0; rays[6] = [0, 0, 2]
        result = ray_depth_coverage(rays, [0, -1, np.nan, np.inf, 100, 2, 2], np.eye(4), [0, 1, 0])
        self.assertEqual(result['unknown_count'], 7)
        self.assertEqual(result['valid_count'], 0)
        self.assertEqual(result['four_way'], 'NEITHER_DETECTED')
        self.assertTrue(np.isnan(result['points_proxy']).all())

    def test_degenerate_frame_rejected(self):
        with self.assertRaises(ValueError):
            horizontal_frame(np.eye(4), [0, 0, -1])
        with self.assertRaises(ValueError):
            horizontal_frame(np.eye(4), [0, 0, 0])
        transform = np.eye(4); transform[0, 0] = 2
        with self.assertRaises(ValueError):
            horizontal_frame(transform, [0, 1, 0])


if __name__ == '__main__':
    unittest.main()
