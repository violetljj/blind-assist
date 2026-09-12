import unittest

import numpy as np

from mz84_bidirectional_complementarity import build_source
from mz85_rotation_compensated_state import (
    compensated_tof_expert,
    flatten_metadata,
    imu_forward_model,
    integrate_orientation,
    rotate_ray,
)


class RotationCompensationTests(unittest.TestCase):
    def test_gyro_integration_resets_at_episode_boundary(self):
        ids = np.asarray(['a', 'a', 'b', 'b'])
        increments = np.asarray([2.0, 3.0, -4.0, 1.0])
        np.testing.assert_allclose(integrate_orientation(increments, ids), [2.0, 5.0, -4.0, -3.0])

    def test_three_dimensional_rotation_changes_ray_direction(self):
        horizontal, vertical = rotate_ray(0.0, 18.0, 6.0)
        self.assertGreater(horizontal, 17.0)
        self.assertLess(vertical, -5.0)

    def test_only_head_motion_tof_predictions_change(self):
        source = build_source()
        episode_ids, families, _ = flatten_metadata(source)
        delta_yaw, delta_pitch = imu_forward_model(source)
        compensated, known, _ = compensated_tof_expert(
            source, episode_ids, delta_yaw, delta_pitch)
        from mz84_bidirectional_complementarity import tof_expert
        baseline, baseline_known, _ = tof_expert(source)
        np.testing.assert_array_equal(known, baseline_known)
        np.testing.assert_array_equal(compensated[families != 'head_motion'],
                                      baseline[families != 'head_motion'])
        self.assertEqual(int(baseline[families == 'head_motion'].sum()), 8)
        self.assertEqual(int(compensated[families == 'head_motion'].sum()), 0)


if __name__ == '__main__':
    unittest.main()
