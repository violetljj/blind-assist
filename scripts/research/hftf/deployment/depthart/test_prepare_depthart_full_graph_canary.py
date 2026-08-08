from __future__ import annotations

import unittest

import numpy as np

from scripts.research.hftf.deployment.depthart.prepare_depthart_full_graph_canary import (
    procedural_bgr,
)


class PrepareDepthArtFullGraphCanaryTest(unittest.TestCase):
    def test_procedural_image_is_deterministic_and_spatially_varied(self) -> None:
        first = procedural_bgr(64)
        second = procedural_bgr(64)
        self.assertEqual(first.shape, (64, 64, 3))
        self.assertEqual(first.dtype, np.uint8)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(len(np.unique(first.reshape(-1, 3), axis=0)), 100)


if __name__ == "__main__":
    unittest.main()
