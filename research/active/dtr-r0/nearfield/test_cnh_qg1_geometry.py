import unittest
import numpy as np

from cnh_qg1_geometry import (project_boxes, roi_grids, roi_valid,
                             query_descriptors, ray_coordinates, tof_query_weights)


class QG1GeometryTest(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[100., 0, 49.5], [0, 100., 39.5], [0, 0, 1.]])

    def test_known_corner_projection(self):
        box = np.array([[[-.2, -.1, 1], [.4, .3, 2]]])
        np.testing.assert_allclose(project_boxes(self.K, (100, 80), box),
                                   [[29.5, 29.5, 89.5, 69.5]], atol=1e-6)

    def test_near_plane_image_clip_and_empty(self):
        boxes = np.array([[[-1., -1, -1], [1, 1, 1]],
                          [[-1, -1, -2], [1, 1, -1]],
                          [[100, 100, 1], [101, 101, 2]]])
        bounds = project_boxes(self.K, (100, 80), boxes)
        np.testing.assert_array_equal(bounds[0], [-.5, -.5, 99.5, 79.5])
        np.testing.assert_array_equal(roi_valid(bounds), [True, False, False])
        np.testing.assert_array_equal(roi_grids(bounds, 80, 100)[1:], 2)

    def test_full_image_bin_centres_and_coordinates(self):
        bounds = np.array([[-.5, -.5, 99.5, 79.5]])
        grid = roi_grids(bounds, 80, 100)
        np.testing.assert_allclose(grid[0, 0, :, 0], [-.75, -.25, .25, .75])
        np.testing.assert_allclose(grid[0, :, 0, 1], [-.75, -.25, .25, .75])
        rays = ray_coordinates(self.K, 80, 100)
        np.testing.assert_allclose(rays[:, 0, 0], [-.495, -.395])
        np.testing.assert_allclose(rays[:, -1, -1], [.495, .395])

    def test_query_identity_and_old_mask_removed(self):
        desc = query_descriptors()
        self.assertEqual(desc.shape, (6, 10))
        self.assertEqual(len(np.unique(desc, axis=0)), 6)
        weights = tof_query_weights()
        self.assertEqual(weights.shape, (6, 64, 16))
        # centre_HEAD's actual box intersects all 64 zones, not old middle 16.
        self.assertEqual(int((weights[2].sum(-1) > 0).sum()), 64)
        self.assertTrue(np.all(weights >= 0))

    def test_grid_sample_preserves_spatial_bins(self):
        import torch
        from torch.nn.functional import grid_sample
        rays = ray_coordinates(self.K, 80, 100)
        grid = roi_grids(np.array([[-.5, -.5, 99.5, 79.5]]), 80, 100)
        sampled = grid_sample(torch.from_numpy(rays[None]), torch.from_numpy(grid),
                              align_corners=False).numpy()[0]
        np.testing.assert_allclose(sampled[0, 0], [-.375, -.125, .125, .375], atol=1e-6)
        np.testing.assert_allclose(sampled[1, :, 0], [-.3, -.1, .1, .3], atol=1e-6)

    def test_invalid_intrinsics_rejected(self):
        bad = self.K.copy()
        bad[0, 1] = 1
        with self.assertRaises(ValueError):
            project_boxes(bad)


if __name__ == '__main__':
    unittest.main()
