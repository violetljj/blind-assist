#!/usr/bin/env python3

import unittest

from evaluate_unidepth_confidence_stratified_occupancy_a0 import confidence_bin


class UniDepthConfidenceStratifiedOccupancyA0Test(unittest.TestCase):
    def test_confidence_bin_has_four_ordered_strata(self) -> None:
        edges = [1.0, 2.0, 3.0]
        self.assertEqual(confidence_bin(0.5, edges), 0)
        self.assertEqual(confidence_bin(1.0, edges), 1)
        self.assertEqual(confidence_bin(2.5, edges), 2)
        self.assertEqual(confidence_bin(4.0, edges), 3)


if __name__ == "__main__":
    unittest.main()
