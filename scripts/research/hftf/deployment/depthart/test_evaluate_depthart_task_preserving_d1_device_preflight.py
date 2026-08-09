from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.research.hftf.deployment.depthart.evaluate_depthart_task_preserving_d1_device_preflight import (
    EXPECTED_ELEMENTS,
    compare,
)


class EvaluateDepthArtTaskPreservingD1DevicePreflightTest(unittest.TestCase):
    def test_compare_reports_bit_exact_and_diagnostic_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.raw"
            exact = root / "exact.raw"
            changed = root / "changed.raw"
            value = np.linspace(1.0, 2.0, EXPECTED_ELEMENTS, dtype=np.float32)
            value.tofile(left)
            value.tofile(exact)
            modified = value.copy()
            modified[0] += np.float32(0.1)
            modified.tofile(changed)
            self.assertTrue(compare(left, exact)["bit_exact"])
            self.assertFalse(compare(left, changed)["bit_exact"])
            self.assertFalse(compare(left, changed)["allclose_rtol_3e_5_atol_3e_6"])


if __name__ == "__main__":
    unittest.main()
