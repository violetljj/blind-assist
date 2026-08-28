from __future__ import annotations

import unittest

import numpy as np

from dtr_m1_point_velocity_oracle import (
    NativeBox,
    _box_history,
    _rigid_world_velocity,
    aggregate_frame,
    gate,
)


def box(
    frame: int,
    time_s: float,
    center_forward_m: float,
    *,
    label_id: str = "pedestrian:1",
) -> NativeBox:
    return NativeBox(
        frame=frame,
        time_s=time_s,
        label_id=label_id,
        center_forward_m=center_forward_m,
        center_left_m=0.0,
        center_z_m=0.0,
        length_m=1.0,
        width_m=1.0,
        height_m=2.0,
        yaw_ego_rad=0.0,
        ego_x_m=0.0,
        ego_y_m=0.0,
        ego_yaw_rad=0.0,
    )


class DTRM1PointVelocityOracleTest(unittest.TestCase):
    def test_history_reuses_frozen_r7_temporal_support(self) -> None:
        values = {
            115: [box(115, 0.00, 1.0)],
            116: [box(116, 0.10, 1.1)],
            117: [box(117, 0.20, 1.2)],
            118: [box(118, 0.35, 1.35)],
        }
        history = _box_history(values)
        self.assertEqual(history[(118, "pedestrian:1")].frame, 115)
        self.assertNotIn((116, "pedestrian:1"), history)

    def test_piecewise_rigid_velocity_is_point_wise(self) -> None:
        previous = box(115, 0.0, 1.0)
        current = box(118, 0.4, 1.4)
        velocity = _rigid_world_velocity(
            np.asarray([[1.4, 0.0], [1.6, 0.2]], dtype=np.float64),
            current,
            previous,
        )
        np.testing.assert_allclose(velocity, [[1.0, 0.0], [1.0, 0.0]], atol=1e-12)

    def test_aggregation_removes_static_and_keeps_direct_motion(self) -> None:
        pose = {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}
        points = np.asarray(
            [[1.36, 0.00, 0.0], [1.37, 0.01, 0.1], [1.44, 0.02, -0.1]],
            dtype=np.float64,
        )
        current = box(118, 0.35, 1.40)
        moving_history = {(118, current.label_id): box(115, 0.0, 1.05)}
        row, diagnostics = aggregate_frame(
            points, pose, [current], moving_history, {current.label_id: 0}
        )
        self.assertGreater(len(row["forward"]), 0)
        self.assertTrue(np.all(row["support"] == 1.0))
        self.assertEqual(diagnostics["assigned_points"], 3)

        static_history = {(118, current.label_id): box(115, 0.0, 1.39)}
        static_row, _ = aggregate_frame(
            points, pose, [current], static_history, {current.label_id: 0}
        )
        self.assertEqual(len(static_row["forward"]), 0)

    def test_oracle_gate_never_opens_r8(self) -> None:
        baseline = {
            "critical_event_recall": 1.0,
            "false_alert_segments": 12,
        }
        r7 = {
            "original_cohort": {
                "r2": baseline,
                "r7_p_occupancy_flow": {"event_detection_f1": 0.22},
            }
        }
        original = {"critical_event_recall": 1.0, "event_detection_f1": 0.30}
        stress = {
            "0.2": {"trials": 3, "occupancy_flow": {"recovered_track_only_window_misses": 3}},
            "0.4": {"trials": 3, "occupancy_flow": {"recovered_track_only_window_misses": 3}},
            "0.8": {"trials": 3, "occupancy_flow": {"recovered_track_only_window_misses": 3}},
        }
        result = gate(
            r7,
            original,
            stress,
            {"false_segments": 12},
            {"flow_induced_or_modified_false_segments": 2},
            {"segments_with_surviving_point_velocity_risk": 2},
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["m1_t_authorized"])
        self.assertFalse(result["r8_authorized"])
        self.assertFalse(result["route_conditioned_forecasting_authorized"])


if __name__ == "__main__":
    unittest.main()
