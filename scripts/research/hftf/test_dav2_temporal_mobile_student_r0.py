from __future__ import annotations

import unittest

import torch
from dav2_temporal_mobile_student_r0 import (
    TemporalMobileDepthStudent,
    parameter_count,
)
from train_dav2_temporal_mobile_student_a3_r0 import combined_loss, temporal_pairs


class TemporalMobileDepthStudentTest(unittest.TestCase):
    def test_shape_positive_depth_and_parameter_cap(self) -> None:
        model = TemporalMobileDepthStudent(pretrained=False).eval()
        with torch.inference_mode():
            depth = model(torch.zeros((1, 3, 294, 392)), (48, 64))
        self.assertEqual(tuple(depth.shape), (1, 48, 64))
        self.assertTrue(torch.all(depth > 0.0))
        self.assertLessEqual(parameter_count(model), 1_600_000)

    def test_identity_pair_has_zero_combined_loss(self) -> None:
        depth = torch.ones((4, 8, 8))
        per_frame = {
            "log_depth_smooth_l1_beta": 0.05,
            "log_depth_gradient_l1_weight": 0.5,
            "median_log_scale_weight": 0.25,
            "depth_clamp_m": [0.1, 20.0],
        }
        temporal = {
            "log_depth_delta_smooth_l1_weight": 0.75,
            "log_depth_delta_smooth_l1_beta": 0.03,
        }
        loss, components = combined_loss(depth, depth, per_frame, temporal)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(components["temporal"], 0.0)

    def test_pair_timestamp_is_parsed_from_bound_frame_id(self) -> None:
        records = [
            {
                "frame_id": "41048097_1.000",
                "video_id": "41048097",
                "role": "train",
            },
            {
                "frame_id": "41048097_1.334",
                "video_id": "41048097",
                "role": "train",
            },
        ]
        self.assertEqual(temporal_pairs(records, "train"), [(0, 1)])


if __name__ == "__main__":
    unittest.main()
