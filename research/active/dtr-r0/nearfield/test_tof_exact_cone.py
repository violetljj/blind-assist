"""Independent scalar optimization: TASK_NOT_GPU_SUITABLE; no source outcomes."""
import math
import unittest

import numpy as np
from scipy.optimize import minimize

from tof_body_support import support as aligned_support
from tof_exact_cone import directional_extrema, support


SLOPE = math.tan(math.radians(7.5))/math.sqrt(2.)


def numerical_extrema(coefficients, slope):
    coefficients = np.asarray(coefficients)

    def value(uv):
        ray = np.array([1., uv[0], uv[1]])
        return np.dot(coefficients, ray)/np.linalg.norm(ray)

    candidates = []
    for u in (-slope, 0., slope):
        for v in (-slope, 0., slope):
            candidates.append(value((u, v)))
            for sign in (-1., 1.):
                fitted = minimize(lambda uv: sign*value(uv), [u, v],
                                  method='L-BFGS-B', bounds=[(-slope, slope)]*2,
                                  options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 150})
                candidates.append(value(fitted.x))
    return min(candidates), max(candidates)


def independent_rotation(pitch, yaw):
    p, y = np.deg2rad([pitch, yaw])
    pitch_matrix = np.array([[np.cos(p), 0., -np.sin(p)],
                             [0., 1., 0.], [np.sin(p), 0., np.cos(p)]])
    yaw_matrix = np.array([[np.cos(y), -np.sin(y), 0.],
                           [np.sin(y), np.cos(y), 0.], [0., 0., 1.]])
    return yaw_matrix @ pitch_matrix


class ExactConeTests(unittest.TestCase):
    def test_interior_edge_corner_and_degenerate_extrema(self):
        cases = [(1., .02, -.03), (-1., -.02, .03),  # interior max/min
                 (1., 2., .02), (-1., -2., -.02),  # edge stationary
                 (0., 1., 1.), (0., -1., -1.),    # corners
                 (0., 0., 0.), (0., 1., 0.), (0., 0., -1.),
                 (-SLOPE, 1., 0.), (SLOPE, 0., 1.)]  # zero edge denominator
        for coefficients in cases:
            with self.subTest(coefficients=coefficients):
                np.testing.assert_allclose(directional_extrema(coefficients, SLOPE),
                                           numerical_extrema(coefficients, SLOPE),
                                           rtol=0., atol=2e-9)
        norm = np.linalg.norm([1., .02, -.03])
        self.assertAlmostEqual(directional_extrema((1., .02, -.03), SLOPE)[1], norm, places=13)
        self.assertAlmostEqual(directional_extrema((-1., -.02, .03), SLOPE)[0], -norm, places=13)

    def test_positive_negative_orientations_and_radial_shell(self):
        for pitch, yaw in ((-2.25, -1.25), (-2.25, 1.25), (2.25, -1.25),
                           (2.25, 1.25), (0., 0.), (45., -60.), (100., 120.), (-170., -95.)):
            rows = independent_rotation(pitch, yaw)
            numerical = np.array([numerical_extrema(row, SLOPE) for row in rows])
            for row, expected in zip(rows, numerical):
                np.testing.assert_allclose(directional_extrema(row, SLOPE), expected, atol=2e-9, rtol=0.)
            for distance, uncertainty in ((1., .1), (.05, .1), (3., 0.)):
                low, high = max(0., distance-uncertainty), distance+uncertainty
                extrema = np.stack((low*numerical, high*numerical))
                expected_low = extrema.min(axis=(0, 2)) + [0., 0., 1.7]
                expected_high = extrema.max(axis=(0, 2)) + [0., 0., 1.7]
                result = support(distance, True, uncertainty, pitch, yaw)
                np.testing.assert_allclose(result['bounds'], [expected_low, expected_high], rtol=0., atol=1e-8)

    def test_zero_pose_and_invalid(self):
        for distance in np.linspace(.15, 4., 50):
            exact = support(float(distance), True, .1, 0., 0.)
            old = aligned_support(float(distance), True, 15., .1)
            self.assertEqual(exact['supported'], old['supported'])
            np.testing.assert_allclose(exact['bounds'], old['bounds'], atol=1.01e-12, rtol=0.)
        self.assertEqual(support(None, False, .1, 0., 0.),
                         dict(status='UNKNOWN', supported=[], bounds=None))


if __name__ == '__main__':
    unittest.main()
