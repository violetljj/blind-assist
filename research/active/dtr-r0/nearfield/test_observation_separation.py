import unittest
import numpy as np
from observation_separation import rgb_matrices_numpy, tof_matrices, separation


class ObservationSeparationTests(unittest.TestCase):
    def test_small_local_signal_not_diluted_by_background(self):
        images = np.zeros((2, 3, 180, 320), np.uint8)
        images[1, :, :20, :20] = 255
        m = rgb_matrices_numpy(images)
        self.assertAlmostEqual(float(m[0, 0, 1]), 1/144, places=6)
        self.assertEqual(m[1, 0, 1], 1)
        np.testing.assert_array_equal(m, m.transpose(0, 2, 1))

    def test_missing_is_not_a_distant_or_clear_return(self):
        tokens = np.zeros((2, 64, 6), np.float32)
        tokens[1, :, 0] = 7/8
        tokens[1, :, 1] = 1
        m, n = tof_matrices(tokens)
        self.assertEqual(n[0, 1], 0)
        self.assertTrue(np.isnan(m[0, 0, 1]))
        self.assertEqual(m[2, 0, 1], 1)

    def test_known_noise_units(self):
        tokens = np.zeros((2, 64, 6), np.float32)
        tokens[:, :, 1] = 1
        tokens[0, :, 0] = 2/8
        tokens[1, :, 0] = 3/8
        m, n = tof_matrices(tokens)
        self.assertEqual(m[0, 0, 1], 1)
        self.assertAlmostEqual(m[1, 0, 1], 1/np.sqrt(.05**2+.07**2))
        self.assertEqual(n[0, 1], 64)

    def test_separation_and_ties(self):
        x = np.array([0., .1, 2., 2.1])
        m = np.abs(x[:, None]-x[None, :])
        result = separation(m, 2)
        self.assertTrue(result['separated'])
        self.assertEqual(result['auc'], 1)
        ties = separation(np.zeros((4, 4)), 2)
        self.assertFalse(ties['separated'])
        self.assertEqual(ties['auc'], .5)
        self.assertIsNone(ties['ratio'])

    def test_missing_comparison_prevents_separation(self):
        m = np.ones((4, 4))*4
        m[:2, :2] = m[2:, 2:] = .1
        m[0, 2] = np.nan
        result = separation(m, 2)
        self.assertEqual(result['missing'], 1)
        self.assertFalse(result['separated'])

    def test_arrival_is_excluded_on_both_sides(self):
        m = np.zeros((10, 10))
        m[:5, 5:] = m[5:, :5] = 2
        r = separation(m, 5, True)
        self.assertEqual((r['within_count'], r['cross_count']), (12, 16))
        self.assertTrue(r['separated'])


if __name__ == '__main__':
    unittest.main()
