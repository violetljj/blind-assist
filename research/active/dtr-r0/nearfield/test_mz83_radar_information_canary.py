import unittest

import numpy as np

from mz83_radar_information_canary import confusion, episodes, expert, raw_clusters


class RadarInformationCanaryTests(unittest.TestCase):
    def test_forward_model_uses_surface_support_without_identity(self):
        depth = np.full((100, 200), 9.0, np.float32)
        depth[20:70, 95:105] = 2.0
        clusters = raw_clusters(depth, 100.0)
        center = len(clusters) // 2
        self.assertTrue(any(abs(distance - 2.0) < 0.2 for distance, _ in clusters[center]))

    def test_expert_is_causal_and_resets_at_clip_boundary(self):
        observations = {
            'range_m': np.array([[2.0], [1.9], [1.8], [1.7], [1.6]]),
            'radial_velocity_mps': np.full((5, 1), -1.0),
            'azimuth_deg': np.zeros((5, 1)), 'valid': np.ones((5, 1), bool)}
        alert, _, raw = expert(observations, np.array(['a', 'a', 'a', 'b', 'b']))
        np.testing.assert_array_equal(raw, np.ones(5, bool))
        np.testing.assert_array_equal(alert, [False, True, True, False, True])

    def test_confusion_and_episode_accounting(self):
        truth = np.array([False, True, True, True, False])
        alert = np.array([False, True, False, True, True])
        self.assertEqual(confusion(truth, np.ones(5, bool), alert),
                         {'TP': 2, 'FP': 1, 'FN': 1, 'TN': 1})
        self.assertEqual(episodes(alert), 2)


if __name__ == '__main__':
    unittest.main()
