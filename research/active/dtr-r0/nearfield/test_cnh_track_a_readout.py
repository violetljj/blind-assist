"""Synthetic engineering tests only; no development datasets read."""
import unittest
import numpy as np
from cnh_route_sensor import (angular_rays, SensorParameters, synthesize_response,
                              derive_readout, H3)
from cnh_track_a_readout import (accumulate, transport, cell_points, query_weights,
    fit_calib_bias, subtract_bias, noisy_poses, WIDTH, SHAPE)


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260925)
        self.hist = self.rng.normal(size=(6,)+SHAPE)
        self.poses = np.repeat(np.eye(4)[None], 6, axis=0)

    def test_identity(self):
        actual, w = accumulate(self.hist, self.poses, 1)
        np.testing.assert_array_equal(actual, self.hist)
        np.testing.assert_array_equal(w, np.ones_like(w))
        hist = np.repeat(self.hist[:1], 6, axis=0)
        actual, w = accumulate(hist, self.poses)
        np.testing.assert_array_equal(actual, hist)
        np.testing.assert_array_equal(w[:, 0, 0, 0], [1, 2, 3, 4, 4, 4])
        actual, _ = accumulate(self.hist, self.poses)
        for i in range(6):
            np.testing.assert_allclose(actual[i], self.hist[max(0, i-3):i+1].mean(0), atol=1e-14)

    def test_rotation_permutation(self):
        t = np.eye(4)
        t[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        actual, w = transport(self.hist[0], t)
        np.testing.assert_allclose(actual, np.rot90(self.hist[0], -1, axes=(0, 1)), atol=1e-10)
        np.testing.assert_array_equal(w, np.ones(SHAPE))

    def test_translation_locations_mass(self):
        # Synthetic fixed count mass, not rho*cos/r^2 scene expectation.
        hist = np.zeros(SHAPE)
        hist[3, 3, 8] = 16
        t = np.eye(4)
        t[2, 3] = -WIDTH
        actual, _ = transport(hist, t)
        expected = np.zeros(SHAPE)
        points = cell_points().reshape(8, 8, 16, 16, 3)[3, 3, 8]+t[:3, 3]
        edge = np.tan(np.deg2rad(22.5))
        for x, y, z in points:
            row = int(np.floor((y/z+edge)/(2*edge)*8))
            col = int(np.floor((x/z+edge)/(2*edge)*8))
            b = int(np.floor(np.linalg.norm([x, y, z])/WIDTH))
            expected[row, col, b] += 1
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        self.assertAlmostEqual(actual.sum(), 16, places=10)
        t[2, 3] = -100
        actual, w = transport(hist, t)
        self.assertEqual(actual.sum(), 0)
        self.assertEqual(w.sum(), 0)

    def test_causality_and_noisy_pose_causality(self):
        self.poses[:, 2, 3] = np.arange(6)*.1
        first, w = accumulate(self.hist, self.poses)
        changed, p = self.hist.copy(), self.poses.copy()
        changed[4:] *= 100
        p[4:, 0, 3] += 2
        second, w2 = accumulate(changed, p)
        np.testing.assert_array_equal(first[:4], second[:4])
        np.testing.assert_array_equal(w[:4], w2[:4])
        np.testing.assert_array_equal(noisy_poses(self.poses, 5)[:4], noisy_poses(p, 5)[:4])
        np.testing.assert_array_equal(noisy_poses(self.poses[:4], 5), noisy_poses(self.poses, 5)[:4])

    def test_partial_fov_weights(self):
        angle = np.deg2rad(15)
        t = np.eye(4)
        t[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]]
        _, cover = transport(np.zeros(SHAPE), t)
        self.assertTrue(np.any(cover == 0))
        self.assertTrue(np.any((cover > 0) & (cover < 1)))
        self.assertTrue(np.any(cover == 1))
        poses = np.array([t, np.eye(4)])
        _, w = accumulate(np.zeros((2,)+SHAPE), poses)
        np.testing.assert_array_equal(w[1], 1+cover)

    def test_noiseless_wall_integrated_photon_mass(self):
        rays, area = angular_rays(16)
        cosine = rays[..., 2]
        ranges = 2/cosine
        params = SensorParameters(noise_scale=0, pulse_sigma_bins=0, tail_mass=0,
                                  neighbour_leak=0, crosstalk_fraction=0)
        response = synthesize_response(ranges, .5, cosine, area, params=params)
        hist = derive_readout(response, H3)['histogram']
        expected = (params.signal_counts*.5*cosine/ranges**2
                    *area/area.sum(-1, keepdims=True)).sum(-1)*params.output_gain
        np.testing.assert_allclose(hist.sum(-1), expected, rtol=1e-12, atol=1e-10)

    def test_h3_only_and_calib_bias(self):
        with self.assertRaises(ValueError):
            accumulate(np.zeros((6, 8, 8, 128)), self.poses)
        splits = np.array(['train', 'calib', 'calib', 'audit', 'train', 'audit'])
        bias = fit_calib_bias(self.hist, splits)
        np.testing.assert_array_equal(bias, np.median(self.hist[1:3], axis=0))
        changed = self.hist.copy()
        changed[splits != 'calib'] *= 100
        np.testing.assert_array_equal(bias, fit_calib_bias(changed, splits))
        with self.assertRaises(ValueError):
            fit_calib_bias(self.hist, splits, np.ones(6, dtype=bool))
        np.testing.assert_array_equal(subtract_bias(self.hist, bias), self.hist-bias)

    def test_projection_shape_and_slabs(self):
        w = query_weights(np.eye(4))
        self.assertEqual(w.shape, (6, 64, 16))
        self.assertTrue(np.isfinite(w).all())
        self.assertTrue(((w >= 0) & (w <= 1)).all())
        # BODY cannot occupy upper-facing rays for the identity origin.
        np.testing.assert_array_equal(w[[1, 3, 5], :32], 0)
        np.testing.assert_array_equal(w[:, :, 11:], 0)
        moved = np.eye(4)
        moved[0, 3] = 100
        np.testing.assert_array_equal(query_weights(moved), 0)


if __name__ == '__main__':
    unittest.main()
