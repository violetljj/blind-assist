import unittest

import numpy as np

from mz87_imu_realism_falsifier import (
    configs,
    orientation_to_delta,
    perturb_deltas,
    sample_with_offset,
)


class ImuStressTests(unittest.TestCase):
    def test_grid_is_frozen_and_not_factorial(self):
        rows = configs()
        self.assertEqual(len(rows), 36)
        self.assertEqual(sum(row['factor'] == 'combination' for row in rows), 3)

    def test_orientation_delta_round_trip(self):
        ids = np.asarray(['a', 'a', 'a', 'b', 'b'])
        orientation = np.asarray([1.0, 3.0, 2.0, -2.0, -1.0])
        delta = orientation_to_delta(orientation, ids)
        np.testing.assert_allclose(delta, [1.0, 2.0, -1.0, -2.0, 1.0])

    def test_zero_perturbation_preserves_orientation(self):
        ids = np.asarray(['a', 'a', 'b', 'b'])
        yaw = np.asarray([0.0, 2.0, 0.0, -3.0])
        pitch = yaw * 0.35
        stressed_yaw, stressed_pitch, _, _ = perturb_deltas(yaw, pitch, ids, {})
        np.testing.assert_allclose(stressed_yaw, yaw)
        np.testing.assert_allclose(stressed_pitch, pitch)
        np.testing.assert_allclose(sample_with_offset(yaw, ids, 0.0), yaw)


if __name__ == '__main__':
    unittest.main()
