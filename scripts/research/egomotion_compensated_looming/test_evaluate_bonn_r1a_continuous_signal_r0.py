#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


try:
    import numpy as np
except ImportError:  # pragma: no cover - dependency-free test runtime
    np = None

if np is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import evaluate_bonn_r1a_continuous_signal_r0 as subject


@unittest.skipIf(np is None, "numpy absent in dependency-free runtime")
class BonnR1AContinuousSignalEvaluationTest(unittest.TestCase):
    def test_average_ranks_handles_ties(self) -> None:
        ranks = subject.average_ranks(np.asarray([2.0, 1.0, 2.0]))
        np.testing.assert_allclose(ranks, [1.5, 0.0, 1.5])

    def test_spearman_sign(self) -> None:
        values = np.arange(10, dtype=np.float64)
        self.assertAlmostEqual(subject.spearman(values, values), 1.0)
        self.assertAlmostEqual(subject.spearman(values, -values), -1.0)

    def test_theil_sen_linear_slope(self) -> None:
        x = np.arange(8, dtype=np.float64)
        y = 3.0 * x + 2.0
        self.assertAlmostEqual(subject.theil_sen_slope(x, y), 3.0)


if __name__ == "__main__":
    unittest.main()
