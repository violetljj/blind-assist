import math
import unittest

import numpy as np

from mz82_temporal_visual_canary import oracle, temporal_scores


class TemporalVisualCanaryTests(unittest.TestCase):
    def test_history_does_not_cross_clip_or_use_future(self):
        clips = np.array(['a', 'a', 'a', 'b', 'b'])
        flow = np.array([[0, 0], [1, 1], [100, 100], [2, 2], [200, 200]], float)
        expansion = flow / 100.0
        ttc = np.full((5, 2), 2.0)
        _, smooth = temporal_scores(flow, expansion, ttc, clips)
        self.assertEqual(smooth[1, 0, 0], 0.5)
        self.assertEqual(smooth[3, 0, 0], 2.0)

    def test_oracle_respects_budget(self):
        scores = np.array([[4.0], [3.0], [2.0], [1.0]])
        truth = np.array([[True], [False], [True], [False]])
        known = np.ones_like(truth)
        result = oracle(scores, np.ones_like(truth), truth, known)
        self.assertEqual(result['0']['rescued_tp'], 1)
        self.assertEqual(result['0']['added_fp'], 0)
        self.assertEqual(result['5']['rescued_tp'], 2)

    def test_missing_ttc_is_finite_low_score(self):
        clips = np.array(['a'])
        scores, _ = temporal_scores(np.zeros((1, 2)), np.zeros((1, 2)),
                                    np.full((1, 2), math.inf), clips)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue((scores < -10).all())


if __name__ == '__main__':
    unittest.main()
