#!/usr/bin/env python3

import unittest

from metric_scale_anchor import MetricScaleAnchor, MetricScaleTracker
from run_external_rgb_clearance_sidecar import calibrated_field


class RunExternalRgbClearanceSidecarTest(unittest.TestCase):
    def test_field_is_unknown_until_anchor_exists(self) -> None:
        tracker = MetricScaleTracker(100)
        raw = {
            "status": "VALID",
            "bands": {
                band: {"clearance_m": 2.0}
                for band in ("left", "center", "right")
            },
        }
        self.assertEqual(
            calibrated_field(raw, tracker, 10)["status"],
            "UNKNOWN_NO_METRIC_SCALE_ANCHOR",
        )

    def test_field_applies_scale_and_recomputes_horizons(self) -> None:
        tracker = MetricScaleTracker(100)
        tracker.update(MetricScaleAnchor(10, 0.5, 3, 0.0, "tof"))
        raw = {
            "status": "VALID",
            "bands": {
                "left": {"clearance_m": 2.0},
                "center": {"clearance_m": 3.2},
                "right": {"clearance_m": None},
            },
        }
        result = calibrated_field(raw, tracker, 20)
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(result["bands"]["left"]["clearance_m"], 1.0)
        self.assertTrue(result["bands"]["center"]["occupied_by_horizon"]["2.0"])
        self.assertFalse(result["bands"]["right"]["occupied_by_horizon"]["2.0"])


if __name__ == "__main__":
    unittest.main()
