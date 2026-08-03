#!/usr/bin/env python3

import unittest

from metric_scale_anchor import (
    MetricScaleAnchor,
    MetricScaleTracker,
    estimate_scale_anchor,
)


class MetricScaleAnchorTest(unittest.TestCase):
    def test_estimate_uses_shared_robust_band_scale(self) -> None:
        candidate = [{"left": 2.0, "center": 4.0, "right": 6.0}]
        metric = [{"left": 1.0, "center": 2.0, "right": 3.0}]
        anchor = estimate_scale_anchor(10, candidate, metric, "tof-3-zone")
        self.assertEqual(anchor.scale, 0.5)
        self.assertEqual(anchor.pair_count, 3)
        self.assertEqual(anchor.median_abs_ratio_residual, 0.0)

    def test_tracker_fails_closed_without_or_after_anchor(self) -> None:
        tracker = MetricScaleTracker(max_age_ns=100)
        self.assertEqual(
            tracker.apply(10, {"left": 1.0})["status"],
            "UNKNOWN_NO_METRIC_SCALE_ANCHOR",
        )
        tracker.update(MetricScaleAnchor(10, 0.5, 3, 0.0, "tof"))
        self.assertEqual(
            tracker.apply(111, {"left": 1.0})["status"],
            "UNKNOWN_STALE_METRIC_SCALE_ANCHOR",
        )

    def test_tracker_applies_valid_anchor_and_rejects_reorder(self) -> None:
        tracker = MetricScaleTracker(max_age_ns=100)
        anchor = MetricScaleAnchor(10, 0.5, 3, 0.0, "tof")
        tracker.update(anchor)
        output = tracker.apply(
            20, {"left": 2.0, "center": 4.0, "right": None}
        )
        self.assertEqual(output["status"], "VALID")
        self.assertEqual(
            output["bands_m"], {"left": 1.0, "center": 2.0, "right": None}
        )
        with self.assertRaises(ValueError):
            tracker.update(anchor)


if __name__ == "__main__":
    unittest.main()
