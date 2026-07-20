import unittest

import numpy as np

import run_public_video_marker_relation_pair_ranking_probe as pair_probe


class MarkerRelationMajorityHorizonProbeTest(unittest.TestCase):
    def test_majority_target_has_fixed_semantics(self) -> None:
        values = np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        np.testing.assert_array_equal([False, False, True, True], values >= 2.0 / 3.0)

    def test_nearest_pairs_accept_majority_target(self) -> None:
        strong = np.asarray([False, True, False, True])
        pairs = pair_probe.nearest_time_pairs(strong, np.asarray(["a"] * 4), np.asarray([0, 1000, 3000, 5000]))
        self.assertEqual(2, len(pairs))


if __name__ == "__main__":
    unittest.main()
