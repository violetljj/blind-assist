import math
import unittest

from evaluate_stage_c_d41_jrdb_causal_future_box_field import (
    NOT_SUPPORTED_STATUS,
    SUPPORTED_STATUS,
    determine_terminal,
    project_box,
)


class D41FutureBoxFieldTest(unittest.TestCase):
    def test_projection_extrapolates_translation_and_scale(self):
        rows = []
        for frame in range(7):
            center_x = 100.0 + 2.0 * frame
            center_y = 80.0 + 1.0 * frame
            width = 20.0 * math.exp(0.01 * frame)
            height = 40.0 * math.exp(0.02 * frame)
            rows.append(
                {
                    "frame_index": frame,
                    "timestamp_ns": frame * 100_000_000,
                    "bbox_xyxy": [
                        center_x - width / 2,
                        center_y - height / 2,
                        center_x + width / 2,
                        center_y + height / 2,
                    ],
                }
            )
        projected = project_box(rows, 1_000_000_000)
        center_x = (projected[0] + projected[2]) / 2
        center_y = (projected[1] + projected[3]) / 2
        self.assertAlmostEqual(center_x, 120.0, places=5)
        self.assertAlmostEqual(center_y, 90.0, places=5)
        self.assertAlmostEqual(
            projected[2] - projected[0],
            20.0 * math.exp(0.10),
            places=5,
        )
        self.assertAlmostEqual(
            projected[3] - projected[1],
            40.0 * math.exp(0.20),
            places=5,
        )

    def test_exact_supported_status_requires_all_gates(self):
        pooled = {
            "opportunities": 500,
            "distinct_native_identities": 20,
            "mean_iou_delta": 0.03,
            "median_iou_delta": 0.025,
            "candidate_iou_better_fraction": 0.60,
            "center_error_relative_reduction": 0.15,
            "mean_absolute_log_area_error_delta": -0.01,
        }
        by_sequence = [
            {
                "sequence": str(index),
                "opportunities": 100,
                "mean_iou_delta": 0.01,
            }
            for index in range(4)
        ]
        _, _, status = determine_terminal(pooled, by_sequence, 480)
        self.assertEqual(status, SUPPORTED_STATUS)
        pooled["median_iou_delta"] = 0.0
        _, _, status = determine_terminal(pooled, by_sequence, 480)
        self.assertEqual(status, NOT_SUPPORTED_STATUS)


if __name__ == "__main__":
    unittest.main()
