from __future__ import annotations

import unittest

from dtr_m1_point_velocity_oracle import NativeBox
from dtr_m2_extent_gap_audit import (
    _footprint_geometry,
    _minimum_segment_box_distance,
    _point_geometry,
    classify_geometry,
)


def box(*, forward_m: float, left_m: float, length_m: float, width_m: float) -> NativeBox:
    return NativeBox(
        frame=1,
        time_s=0.0,
        label_id="pedestrian:1",
        center_forward_m=forward_m,
        center_left_m=left_m,
        center_z_m=0.0,
        length_m=length_m,
        width_m=width_m,
        height_m=1.8,
        yaw_ego_rad=0.0,
        ego_x_m=0.0,
        ego_y_m=0.0,
        ego_yaw_rad=0.0,
    )


class DTRM2ExtentGapAuditTest(unittest.TestCase):
    def test_point_can_miss_while_body_hits(self) -> None:
        point = _point_geometry(3.0, 1.0, -1.0, 0.0)
        footprint = _footprint_geometry(
            box(forward_m=3.0, left_m=1.0, length_m=0.8, width_m=0.8),
            -1.0,
            0.0,
        )
        self.assertFalse(point["hit"])
        self.assertTrue(footprint["hit"])
        self.assertEqual(
            classify_geometry(point_hit=False, footprint_hit=True, truth_positive=True),
            "POINT_MISS_FOOTPRINT_HIT",
        )

    def test_minimum_segment_box_distance_is_continuous(self) -> None:
        distance, fraction = _minimum_segment_box_distance(
            start_x=0.0,
            start_y=0.0,
            end_x=4.0,
            end_y=0.0,
            box_x=2.0,
            box_y=1.0,
            box_yaw=0.0,
            length_m=1.0,
            width_m=1.0,
        )
        self.assertAlmostEqual(distance, 0.5)
        self.assertGreaterEqual(fraction, 0.375)
        self.assertLessEqual(fraction, 0.625)

    def test_truth_negative_footprint_hit_is_not_a_recovery(self) -> None:
        self.assertEqual(
            classify_geometry(point_hit=True, footprint_hit=True, truth_positive=False),
            "FOOTPRINT_HIT_TRUTH_NEGATIVE",
        )


if __name__ == "__main__":
    unittest.main()
