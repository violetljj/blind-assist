"""Artificial geometry only; no source frames, labels, or model predictions."""
import math
import unittest

import numpy as np
import torch

from body_query_tof_coverage import footprints
from tof_body_support import support as aligned_support
from tof_jitter_geometry import footprint, support, trajectory


class JitterGeometryTests(unittest.TestCase):
    def test_frozen_trajectory(self):
        rows = trajectory()
        self.assertEqual(rows, trajectory())
        self.assertEqual(len(rows), 25)
        self.assertEqual((rows[0]['time_s'], rows[-1]['time_s']), (0., 2.))
        for row in rows:
            t = row['time_s']
            self.assertLessEqual(abs(row['pitch_deg']-2*math.sin(4*math.pi*t)), .25)
            self.assertLessEqual(abs(row['yaw_deg']-math.sin(2*math.pi*t)), .25)

    def test_zero_pose_matches_static_and_signs(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        actual, expected = footprint(device, 0., 0.), footprints(device)[15]
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0., atol=0.)
        mask = footprint(device, 2., 1.)[0]
        y, x = mask.nonzero().double().mean(0).tolist()
        self.assertLess(y, 179.5)
        self.assertGreater(x, 319.5)
        for distance in (.3, 1., 1.5, 2., 3.5):
            a = support(distance, True, .1, 0., 0.)
            b = aligned_support(distance, True, 15., .1)
            self.assertEqual(a['supported'], b['supported'])
            np.testing.assert_allclose(a['bounds'], b['bounds'], atol=1e-14)
        with self.assertRaises(ValueError):
            footprint(device, 85., 0.)

    def test_all_shell_samples_inside_rotated_bounds(self):
        slope = math.tan(math.radians(7.5))/math.sqrt(2.)
        y, z = np.meshgrid(np.linspace(-slope, slope, 31), np.linspace(-slope, slope, 31))
        rays = np.stack((np.ones(y.size), y.ravel(), z.ravel()), axis=1)
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        for row in trajectory():
            p, w = math.radians(row['pitch_deg']), math.radians(row['yaw_deg'])
            # Independent sequential rotations, without importing rotation().
            pitched = rays.copy()
            pitched[:, 0] = math.cos(p)*rays[:, 0]-math.sin(p)*rays[:, 2]
            pitched[:, 2] = math.sin(p)*rays[:, 0]+math.cos(p)*rays[:, 2]
            rotated = pitched.copy()
            rotated[:, 0] = math.cos(w)*pitched[:, 0]-math.sin(w)*pitched[:, 1]
            rotated[:, 1] = math.sin(w)*pitched[:, 0]+math.cos(w)*pitched[:, 1]
            for distance in (.15, 1., 2.5, 4.):
                result = support(distance, True, .1, row['pitch_deg'], row['yaw_deg'])
                low, high = np.asarray(result['bounds'])
                for radius in (max(0., distance-.1), distance+.1):
                    points = radius*rotated + [0., 0., 1.7]
                    self.assertTrue(np.all(points >= low-1e-12))
                    self.assertTrue(np.all(points <= high+1e-12))
        self.assertNotEqual(support(1., True, .1, 2., 1.)['bounds'],
                            support(1., True, .1, 0., 0.)['bounds'])
        self.assertEqual(support(None, False, .1, 2., 1.)['status'], 'UNKNOWN')


if __name__ == '__main__':
    unittest.main()
