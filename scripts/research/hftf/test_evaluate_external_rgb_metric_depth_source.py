#!/usr/bin/env python3

from __future__ import annotations

import math
import unittest

from evaluate_external_rgb_metric_depth_source import summarize


def observation(
    model_id: str,
    sequence_id: str,
    frame_index: int,
    truth_depth_m: float,
    predicted_depth_m: float,
    *,
    scenario: str,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "sequence_id": sequence_id,
        "frame_index": frame_index,
        "timestamp_ns": frame_index * 100_000_000,
        "scenario": scenario,
        "camera_motion": "static",
        "truth_depth_m": truth_depth_m,
        "predicted_depth_m": predicted_depth_m,
        "latency_ms": 10 + frame_index,
    }


class ExternalRgbMetricDepthSourceTest(unittest.TestCase):
    def test_depth_error_availability_jitter_and_direction(self) -> None:
        rows = []
        for frame in range(8):
            rows.append(observation("good", "static-2m", frame, 2.0, 2.0 + (0.01 if frame % 2 else -0.01), scenario="static"))
            truth = 4.0 - frame * 0.1
            rows.append(observation("good", "approach", frame, truth, truth + 0.02, scenario="approach"))
        report = summarize(rows)["models"]["good"]
        self.assertAlmostEqual(report["mean_absolute_error_m"], 0.015)
        self.assertEqual(report["seven_frame_availability"], 1.0)
        self.assertEqual(report["direction_accuracy"], 1.0)
        self.assertAlmostEqual(report["median_static_mad_jitter_m"], 0.01)
        self.assertEqual(report["p95_latency_ms"], 17.0)

    def test_nonfinite_prediction_reduces_availability(self) -> None:
        rows = [
            observation("arm", "static", frame, 2.0, math.nan if frame == 3 else 2.0, scenario="static")
            for frame in range(7)
        ]
        report = summarize(rows)["models"]["arm"]
        self.assertEqual(report["valid_observations"], 6)
        self.assertEqual(report["seven_frame_availability"], 0.0)


if __name__ == "__main__":
    unittest.main()
