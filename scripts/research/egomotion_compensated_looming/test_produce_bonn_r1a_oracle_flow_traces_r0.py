#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


try:
    import cv2  # noqa: F401
    import numpy as np
    import PIL  # noqa: F401
except ImportError:  # pragma: no cover - dependency-free test runtime
    np = None

if np is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import produce_bonn_r1a_base_flow_traces_r0 as base
    import produce_bonn_r1a_oracle_flow_traces_r0 as subject


@unittest.skipIf(np is None, "numpy/OpenCV/Pillow absent")
class BonnR1AOracleFlowTraceTest(unittest.TestCase):
    def test_identity_rotation_predicts_zero_flow(self) -> None:
        flow, valid = subject.rotational_flow(
            np.eye(3), base.spatial_arrays()
        )
        np.testing.assert_allclose(flow, 0.0, atol=1e-6)
        self.assertTrue(valid[:-1, :-1].all())
        self.assertFalse(valid[-1, :].any())
        self.assertFalse(valid[:, -1].any())

    def test_closest_rotation_removes_uniform_scale(self) -> None:
        rotation = subject.closest_rotation(1.2 * np.eye(3))
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)

    def test_nearest_abstains_over_hard_cap(self) -> None:
        rows = [["1.0", "x"]]
        self.assertIsNone(subject.nearest(rows, [1.0], 1.05, 0.04))


if __name__ == "__main__":
    unittest.main()
