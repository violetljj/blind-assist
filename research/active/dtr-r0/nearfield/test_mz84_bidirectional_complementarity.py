import unittest

import numpy as np

from mz84_bidirectional_complementarity import (
    FAMILIES, FRAMES, build_source, confusion, episodes, f1, hysteresis,
)


class BidirectionalComplementarityTests(unittest.TestCase):
    def test_source_is_fixed_six_family_factorial(self):
        source = build_source()
        self.assertEqual(len(source), 24)
        self.assertEqual({episode['family'] for episode in source}, set(FAMILIES))
        self.assertTrue(all(len(episode['frames']) == FRAMES for episode in source))

    def test_hysteresis_resets_between_episodes(self):
        raw = np.array([True, True, True, True])
        ids = np.array(['a', 'a', 'b', 'b'])
        np.testing.assert_array_equal(hysteresis(raw, ids), [False, True, False, True])

    def test_metrics(self):
        truth = np.array([True, True, False, False])
        alert = np.array([True, False, True, False])
        metrics = confusion(truth, alert)
        self.assertEqual(metrics, {'TP': 1, 'FP': 1, 'FN': 1, 'TN': 1})
        self.assertAlmostEqual(f1(metrics), 0.5)
        self.assertEqual(episodes(alert), 2)


if __name__ == '__main__':
    unittest.main()
