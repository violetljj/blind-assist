#!/usr/bin/env python3

import unittest

import numpy as np

from evaluate_ag_st_rgb_boundary_gate import rgb_edge_map


class RgbBoundaryGateTest(unittest.TestCase):
    def test_sobel_gate_is_sparse_and_detects_step(self) -> None:
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[:, 16:] = 255
        gradient, edge, threshold = rgb_edge_map(rgb)
        self.assertGreater(threshold, 0.0)
        self.assertGreater(float(np.max(gradient[:, 14:18])), threshold)
        self.assertGreater(int(np.sum(edge[:, 14:18])), 0)
        self.assertLess(float(np.mean(edge)), 0.20)


if __name__ == "__main__":
    unittest.main()
