import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_external_rgb_metric_track_sidecar import d44_predict, relative_position


class ExternalRgbMetricTrackSidecarTest(unittest.TestCase):
    def test_relative_position_uses_calibrated_pinhole_geometry(self) -> None:
        value = relative_position([10, 20, 30, 40], [100, 100, 20, 30], 2.0)
        np.testing.assert_allclose(value, [2.0, 0.0, 0.0])

    def test_d44_recovers_constant_velocity(self) -> None:
        history = []
        for index in range(7):
            history.append(
                {
                    "timestamp_ns": index * 100_000_000,
                    "depth_m": 3.0 - 0.1 * index,
                    "torso_roi_xyxy_px": [10, 20, 30, 40],
                    "intrinsics_fx_fy_cx_cy": [100, 100, 20, 30],
                }
            )
        prediction = d44_predict(history, 1_600_000_000)
        np.testing.assert_allclose(prediction, [1.4, 0.0, 0.0], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
