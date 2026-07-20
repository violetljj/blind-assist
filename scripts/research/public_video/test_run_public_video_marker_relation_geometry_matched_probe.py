import unittest

import numpy as np

import run_public_video_marker_relation_geometry_matched_probe as subject


class MarkerRelationGeometryMatchedProbeTest(unittest.TestCase):
    def test_matcher_prefers_geometry_over_time(self) -> None:
        x = np.zeros((3, 4), dtype=np.float64)
        x[:, -3:] = np.asarray([[0.1, 0.5, 0.5], [0.2, 0.5, 0.5], [0.101, 0.5, 0.5]])
        active = np.asarray([True, False, False])
        pairs = subject.geometry_matched_pairs(x, active, np.asarray(["a", "a", "a"]),
                                               np.asarray([0, 1000, 100000]), 1.0, 1.0)
        self.assertEqual(2, pairs[0]["negative_index"])

    def test_matcher_tie_breaks_by_time_gap(self) -> None:
        x = np.zeros((3, 4), dtype=np.float64)
        x[:, -3:] = np.asarray([[0.1, 0.5, 0.5], [0.1, 0.5, 0.5], [0.1, 0.5, 0.5]])
        pairs = subject.geometry_matched_pairs(x, np.asarray([True, False, False]), np.asarray(["a", "a", "a"]),
                                               np.asarray([5000, 0, 4000]), 1.0, 1.0)
        self.assertEqual(2, pairs[0]["negative_index"])


if __name__ == "__main__":
    unittest.main()
