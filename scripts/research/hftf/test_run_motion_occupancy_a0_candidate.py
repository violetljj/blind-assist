import json
import unittest
from pathlib import Path

import numpy as np

from run_motion_occupancy_a0_candidate import predict_field


class RunMotionOccupancyA0CandidateTest(unittest.TestCase):
    def test_predicts_all_bands_and_horizons(self) -> None:
        model = json.loads(
            Path(__file__).with_name(
                "MOTION_CONDITIONED_OCCUPANCY_A0_1_FROZEN_MODEL.json"
            ).read_text(encoding="utf-8")
        )
        band = {
            "clearance_m": 1.5,
            "clearance_log1p_confidence": 0.6,
            "obstacle_points": 1000,
        }
        field = {
            "status": "VALID",
            "ground_plane_median_residual_m": 0.01,
            "bands": {"left": band, "center": band, "right": band},
        }
        result = predict_field(field, np.zeros(10), model)
        self.assertEqual(result["status"], "VALID")
        for value in result["bands"].values():
            self.assertEqual(
                set(value["occupancy_probability_by_horizon_m"]),
                {"1.0", "1.5", "2.0"},
            )


if __name__ == "__main__":
    unittest.main()
