import unittest

import numpy as np

from mz88_uncertainty_aware_association import (
    apply_authority,
    crossing_state,
    point_state,
)


class Mz88UncertaintyAssociationTest(unittest.TestCase):
    def test_point_state_has_three_regions(self):
        self.assertEqual(point_state(4.0, 3.0, 2.0), 'CERTAIN_IN')
        self.assertEqual(point_state(12.0, 3.0, 2.0), 'UNCERTAIN')
        self.assertEqual(point_state(18.0, 3.0, 2.0), 'CERTAIN_OUT')

    def test_crossing_interval_distinguishes_confident_and_uncertain(self):
        self.assertEqual(crossing_state(18.0, -2.0, 1.0, 2.0), 'CERTAIN_IN')
        self.assertEqual(crossing_state(18.0, 11.5, 1.0, 2.0), 'UNCERTAIN')
        self.assertEqual(crossing_state(19.0, 17.0, 1.0, 1.0), 'CERTAIN_OUT')

    def test_hold_is_one_frame_and_cannot_reseed(self):
        state = np.asarray(['CERTAIN_IN', 'UNCERTAIN', 'UNCERTAIN', 'UNCERTAIN'])
        known = np.ones(4, dtype=bool)
        radar = np.zeros(4, dtype=bool)
        episode = np.asarray(['a', 'a', 'a', 'a'])
        result = apply_authority(state, known, radar, episode)
        np.testing.assert_array_equal(result['full'], [True, True, False, False])
        np.testing.assert_array_equal(result['held'], [False, True, False, False])

    def test_certain_out_suppresses_radar(self):
        state = np.asarray(['CERTAIN_OUT'])
        result = apply_authority(
            state, np.ones(1, dtype=bool), np.ones(1, dtype=bool), np.asarray(['a']))
        self.assertFalse(result['full'][0])


if __name__ == '__main__':
    unittest.main()
