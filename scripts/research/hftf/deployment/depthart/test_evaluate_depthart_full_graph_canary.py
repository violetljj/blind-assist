from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.research.hftf.deployment.depthart.evaluate_depthart_full_graph_canary import (
    compare,
)


class EvaluateDepthArtFullGraphCanaryTest(unittest.TestCase):
    def test_compare_reports_exact_and_non_exact_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.raw"
            exact = root / "exact.raw"
            changed = root / "changed.raw"
            value = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
            value.tofile(left)
            value.tofile(exact)
            (value + np.asarray([0.0, 0.0, 0.1], dtype=np.float32)).tofile(changed)
            self.assertTrue(compare(left, exact)["bit_exact"])
            self.assertTrue(compare(left, exact)["allclose"])
            self.assertFalse(compare(left, changed)["bit_exact"])
            self.assertFalse(compare(left, changed)["allclose"])


if __name__ == "__main__":
    unittest.main()
