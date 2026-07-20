import unittest

import numpy as np

import run_public_video_marker_relation_pair_ranking_probe as subject


class MarkerRelationPairRankingProbeTest(unittest.TestCase):
    def test_nearest_time_pairs_stay_within_source(self) -> None:
        active = np.asarray([False, True, False, True, False, True])
        sources = np.asarray(["a", "a", "a", "a", "b", "b"])
        timestamps = np.asarray([0, 1000, 3000, 5000, 0, 1000])
        pairs = subject.nearest_time_pairs(active, sources, timestamps)
        self.assertEqual(3, len(pairs))
        self.assertEqual(0, pairs[0]["negative_timestamp_ms"])
        self.assertEqual(3000, pairs[1]["negative_timestamp_ms"])
        self.assertTrue(all(row["source_id"] in {"a", "b"} for row in pairs))

    def test_signed_pair_ridge_learns_positive_direction(self) -> None:
        deltas = np.asarray([[1.0, 0.0], [2.0, 0.1], [1.5, -0.1]])
        sources = np.asarray(["a", "b", "c"])
        model = subject.fit_signed_pair_ridge(deltas, sources, 1.0)
        self.assertTrue(np.all(subject.pair_projection(model, deltas) > 0.0))


if __name__ == "__main__":
    unittest.main()
