from __future__ import annotations

import unittest

import numpy as np

from scripts.research.assistive_geometry.observe_b1_a0_development import (
    ROLE,
    band_observation,
    flatten_manifest,
)


class DevelopmentObservationTests(unittest.TestCase):
    def test_right_censored_prediction_maps_to_task_horizon(self) -> None:
        predicted = {
            "clearance_m": None,
            "occupied_by_horizon": {"1.0": False, "1.5": False, "2.0": False},
        }
        result = band_observation(
            "center",
            1.2,
            True,
            np.asarray([True, True, True]),
            np.asarray([True, True, True]),
            predicted,
        )
        self.assertTrue(result["predicted_clearance_valid"])
        self.assertEqual(result["predicted_clearance_m"], 2.0)
        self.assertTrue(all(cell["predicted_state"] == "CLEAR_OBSERVED" for cell in result["cells"]))

    def test_unknown_postprocess_does_not_fabricate_clearance(self) -> None:
        result = band_observation(
            "left",
            0.0,
            False,
            np.zeros(3, dtype=bool),
            np.zeros(3, dtype=bool),
            None,
        )
        self.assertFalse(result["predicted_clearance_valid"])
        self.assertIsNone(result["predicted_clearance_m"])
        self.assertTrue(all(cell["truth_state"] == "UNKNOWN" for cell in result["cells"]))

    def test_manifest_firewall_preserves_exact_selection_order(self) -> None:
        expected = [{"visit_id": str(index), "video_id": f"v{index}"} for index in range(4)]
        videos = []
        for index in range(4):
            videos.append(
                {
                    "evaluation_role": ROLE,
                    "visit_id": str(index),
                    "video_id": f"v{index}",
                    "frames": [
                        {
                            "frame_index": frame,
                            "frame_stem": f"{frame}",
                            "orientation_family": "portrait",
                        }
                        for frame in range(300)
                    ],
                }
            )
        manifest = {
            "schema": "blindassist_assistive_geometry_b1_development_target_manifest_v1",
            "data_role": ROLE,
            "development_content_opened": True,
            "development_calibration_content_opened": False,
            "confirmation_content_opened": False,
            "videos": videos,
        }
        frames = flatten_manifest(manifest, expected)
        self.assertEqual(len(frames), 1200)
        self.assertEqual(frames[0]["video_id"], "v0")
        self.assertEqual(frames[-1]["video_id"], "v3")


if __name__ == "__main__":
    unittest.main()
