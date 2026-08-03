#!/usr/bin/env python3

import unittest

from evaluate_camera_scale_calibrated_clearance_r0 import evaluate


def field(clearance: float) -> dict:
    return {
        "status": "VALID",
        "bands": {
            band: {"clearance_m": clearance}
            for band in ("left", "center", "right")
        },
    }


class EvaluateCameraScaleCalibratedClearanceR0Test(unittest.TestCase):
    def test_prefix_scale_is_applied_only_to_later_frames(self) -> None:
        frames = []
        for index in range(30):
            frames.append(
                {
                    "sequence_id": "s0",
                    "timestamp": float(index),
                    "candidate": field(2.0 if index < 10 else 4.0),
                    "sensor": field(1.0 if index < 10 else 2.0),
                }
            )
        result = evaluate({"status": "source", "frames": frames})
        self.assertEqual(result["scales"], {"s0": 0.5})
        self.assertAlmostEqual(result["clearance_mae_m"], 0.0)
        self.assertEqual(
            result["status"], "CAMERA_SCALE_CALIBRATED_CLEARANCE_DEVELOPMENT_PASS"
        )

    def test_missing_calibration_pairs_fail_closed(self) -> None:
        frames = [
            {
                "sequence_id": "s0",
                "timestamp": float(index),
                "candidate": {"status": "UNKNOWN"},
                "sensor": field(1.0),
            }
            for index in range(30)
        ]
        with self.assertRaises(ValueError):
            evaluate({"frames": frames})

    def test_global_scale_uses_only_first_sequence_prefix(self) -> None:
        frames = []
        for sequence in ("a", "b"):
            for index in range(30):
                frames.append(
                    {
                        "sequence_id": sequence,
                        "timestamp": float(index),
                        "candidate": field(2.0),
                        "sensor": field(1.0),
                    }
                )
        result = evaluate({"frames": frames}, "global_first_sequence")
        self.assertEqual(result["protocol"]["global_calibration_sequence"], "a")
        self.assertEqual(result["scales"], {"a": 0.5, "b": 0.5})
        self.assertEqual(result["evaluation_frames"], 50)


if __name__ == "__main__":
    unittest.main()
