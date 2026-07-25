#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


try:
    import cv2  # noqa: F401
    import numpy as np
except ImportError:  # pragma: no cover - dependency-free test runtime
    np = None

if np is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import freeze_r1a_oracle_signal_contract_r0 as contract_source
    import produce_bonn_r1a_base_flow_traces_r0 as subject


@unittest.skipIf(np is None, "numpy/OpenCV absent in dependency-free runtime")
class BonnR1ABaseFlowTraceTest(unittest.TestCase):
    def test_average_hash_is_stable(self) -> None:
        gray = np.arange(64, dtype=np.uint8).reshape(8, 8)
        self.assertEqual(subject.average_hash(gray), "00000000ffffffff")

    def test_zero_flow_has_full_support_and_zero_signal(self) -> None:
        flow = np.zeros((480, 640, 2), dtype=np.float32)
        summary = subject.summarize_pair(
            flow,
            flow,
            1.0 / 30.0,
            contract_source.build(),
            subject.spatial_arrays(),
        )
        self.assertTrue(summary["evaluated"])
        self.assertEqual(
            summary["RAW_FLOW_ENERGY"][
                "q90_flow_magnitude_pixels_per_second"
            ],
            0.0,
        )
        self.assertEqual(
            summary["UNCOMPENSATED_LOCAL_RADIAL_EXPANSION"][
                "q90_positive_radial_rate_per_second"
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
