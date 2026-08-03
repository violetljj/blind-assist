#!/usr/bin/env python3

import unittest

from evaluate_dual_frequency_clearance_replay_r0 import adjusted_field


class DualFrequencyClearanceReplayR0Test(unittest.TestCase):
    def test_adjustment_is_metric_and_recomputes_horizons(self) -> None:
        field = {
            "status": "VALID",
            "camera_height_m": 1.0,
            "bands": {
                band: {
                    "clearance_m": 2.0,
                    "occupied_by_horizon": {
                        "1.0": False,
                        "1.5": False,
                        "2.0": True,
                    },
                }
                for band in ("left", "center", "right")
            },
        }
        result = adjusted_field(
            field,
            {"left": -0.75, "center": 0.25, "right": 0.0},
            0.2,
        )
        self.assertEqual(result["bands"]["left"]["clearance_m"], 1.25)
        self.assertTrue(
            result["bands"]["left"]["occupied_by_horizon"]["1.5"]
        )
        self.assertEqual(result["camera_height_m"], 1.2)

    def test_missing_anchor_fails_closed(self) -> None:
        self.assertEqual(
            adjusted_field({"status": "VALID"}, None, None)["status"],
            "UNKNOWN_METRIC_ANCHOR",
        )

    def test_missing_band_anchor_fails_closed_per_band(self) -> None:
        field = {
            "status": "VALID",
            "camera_height_m": 1.0,
            "bands": {
                band: {
                    "clearance_m": 2.0,
                    "occupied_by_horizon": {
                        str(horizon): False
                        for horizon in (1.0, 1.5, 2.0)
                    },
                }
                for band in ("left", "center", "right")
            },
        }
        result = adjusted_field(field, {"left": -0.5}, None)
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(result["bands"]["left"]["clearance_m"], 1.5)
        self.assertIsNone(result["bands"]["center"]["clearance_m"])
        self.assertIsNone(
            result["bands"]["center"]["occupied_by_horizon"]["1.0"]
        )


if __name__ == "__main__":
    unittest.main()
