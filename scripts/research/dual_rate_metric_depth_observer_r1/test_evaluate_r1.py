#!/usr/bin/env python3

import unittest

from evaluate_r1 import (
    busy_intervals,
    corrected_field,
    replay_d_arm,
    robust_affine_fit,
)

CONFIG = {
    "anchor_request_period_frames": 5,
    "anchor_history_count": 3,
    "minimum_fit_pairs": 3,
    "minimum_pair_delta_m": 0.05,
    "minimum_fast_span_m": 0.25,
    "slope_bounds": [0.25, 4.0],
    "maximum_fit_median_absolute_residual_m": 0.25,
    "maximum_anchor_source_age_s": 1.0,
}


def field(values, status="VALID"):
    if status != "VALID":
        return {"status": status}
    return {
        "status": "VALID",
        "camera_height_m": 1.5,
        "bands": {
            band: {
                "clearance_m": value,
                "occupied_by_horizon": {
                    "1.0": value <= 1.0,
                    "1.5": value <= 1.5,
                    "2.0": value <= 2.0,
                },
            }
            for band, value in zip(("left", "center", "right"), values, strict=True)
        },
    }


def report(metric: bool, count=30):
    frames = []
    for index in range(count):
        fast_values = [0.8 + 0.02 * index, 1.2 + 0.01 * index, 1.8 - 0.01 * index]
        candidate_values = (
            [1.2 * value + 0.1 for value in fast_values] if metric else fast_values
        )
        sensor = field([1.2 * value + 0.1 for value in fast_values])
        frames.append(
            {
                "sequence_root": "fixture",
                "sequence_id": "fixture-0",
                "timestamp": 100.0 + index * 0.1,
                "frame_path": f"frame-{index:03d}.png",
                "latency_ms": 10.0,
                "sensor": sensor,
                "candidate": field(candidate_values),
            }
        )
    return {"frames": frames}


class DualRateMetricDepthObserverR1Test(unittest.TestCase):
    def test_theil_sen_recovers_affine_with_one_outlier(self):
        pairs = [(0.5, 1.5), (1.0, 2.5), (1.5, 3.5), (2.0, 4.5), (2.5, 99.0)]
        result = robust_affine_fit([{"pairs": pairs}], CONFIG)
        self.assertEqual(result["status"], "VALID")
        self.assertAlmostEqual(result["slope"], 2.0)
        self.assertAlmostEqual(result["intercept_m"], 0.5)

    def test_fit_fails_closed_without_span(self):
        result = robust_affine_fit(
            [{"pairs": [(1.0, 1.1), (1.05, 1.15), (1.1, 1.2)]}], CONFIG
        )
        self.assertEqual(result["status"], "UNKNOWN_INSUFFICIENT_FAST_SPAN")

    def test_corrected_field_recomputes_horizons(self):
        result = corrected_field(
            field([1.0, 1.5, 2.0]),
            {
                "status": "VALID",
                "slope": 1.2,
                "intercept_m": 0.1,
                "pair_count": 3,
                "median_absolute_residual_m": 0.0,
            },
            0.2,
        )
        self.assertAlmostEqual(result["bands"]["left"]["clearance_m"], 1.3)
        self.assertTrue(result["bands"]["center"]["occupied_by_horizon"]["2.0"])
        self.assertFalse(result["bands"]["right"]["occupied_by_horizon"]["2.0"])

    def test_invalid_fast_field_cannot_become_valid_from_valid_fit(self):
        result = corrected_field(
            {"status": "UNKNOWN_SOURCE"},
            {
                "status": "VALID",
                "slope": 1.2,
                "intercept_m": 0.1,
                "pair_count": 3,
                "median_absolute_residual_m": 0.0,
            },
            0.2,
        )
        self.assertEqual(result["status"], "UNKNOWN_FAST_FIELD")

    def test_async_replay_never_uses_uncompleted_anchor(self):
        result = replay_d_arm(report(True), report(False), CONFIG, 142.33)
        rows = result["trace"]["rows"]
        self.assertEqual(rows[0]["candidate"]["status"], "UNKNOWN_ANCHOR_STARTUP")
        self.assertEqual(rows[1]["candidate"]["status"], "UNKNOWN_ANCHOR_STARTUP")
        self.assertEqual(rows[2]["candidate"]["status"], "VALID")
        for row in rows:
            completion = row["latest_anchor_completion_timestamp"]
            if row["candidate"].get("status") == "VALID":
                self.assertLessEqual(completion, row["timestamp"])
        self.assertEqual(result["summary"]["causality_violations"], 0)

    def test_slow_anchor_is_stale_on_first_completion(self):
        result = replay_d_arm(report(True), report(False), CONFIG, 1500.794)
        self.assertEqual(result["summary"]["known_output_frames"], 0)
        self.assertGreater(
            result["summary"]["unknown_reason_counts"]["UNKNOWN_STALE_ANCHOR"], 0
        )

    def test_busy_scheduler_drops_due_requests_until_worker_is_free(self):
        frames = report(False, count=30)["frames"]
        intervals = busy_intervals(frames, period=5, service_time_s=1.500794)
        self.assertEqual(len(intervals), 2)
        self.assertGreater(intervals[1][0], intervals[0][1])


if __name__ == "__main__":
    unittest.main()
