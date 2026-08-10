#!/usr/bin/env python3

import unittest

import numpy as np

from materialize_ag_st_icl_boundary_consensus import consensus_positive


class IclBoundaryConsensusMaterializerTest(unittest.TestCase):
    def test_only_exact_targets_with_nearby_seed_are_retained(self) -> None:
        exact = np.zeros((7, 7), dtype=np.bool_)
        exact[1, 1] = True
        exact[6, 6] = True
        seed = np.zeros_like(exact)
        seed[2, 3] = True
        positive = consensus_positive(exact, seed)
        self.assertTrue(positive[1, 1])
        self.assertFalse(positive[6, 6])
        self.assertEqual(1, int(np.sum(positive)))


if __name__ == "__main__":
    unittest.main()
