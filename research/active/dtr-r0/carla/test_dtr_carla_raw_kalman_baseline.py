from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_raw_kalman_baseline as baseline
import dtr_carla_x24_plan_adherent_predictor as x24
import dtr_carla_x24_plan_route_core as route


def measurement(x: float, y: float = 0.0) -> x24.Measurement:
    return x24.Measurement(0, "person", 0.9, (0.1, 0.1, 0.2, 0.3), np.asarray([x, y]), 10)


def selection(*, changed: bool = False, receipt: str = "r1") -> route.RouteSelection:
    return route.RouteSelection(
        mode=route.ROUTE_MODE_ISSUED_PLAN,
        authority=route.AUTHORITY_VALID,
        receipt_valid=True,
        receipt_sha256=receipt,
        plan_position_residual_m=0.0,
        plan_velocity_direction_error_deg=0.0,
        fallback_reason=None,
        mode_changed=changed,
    )


class RawKalmanBaselineTest(unittest.TestCase):
    def test_tracker_learns_motion_and_bounds_prediction_age(self) -> None:
        tracker = baseline.RawKalmanTracker()
        first = tracker.update([measurement(5.0)], 0.0)
        self.assertEqual([], tracker.emitted(0.0, first))
        second = tracker.update([measurement(4.8)], 0.1)
        emitted = tracker.emitted(0.1, second)
        self.assertEqual(1, len(emitted))
        self.assertLess(emitted[0]["velocity_forward_mps"], 0.0)
        missing = tracker.update([], 0.5)
        held = tracker.emitted(0.5, missing)
        self.assertEqual("PREDICTED_HOLD", held[0]["disposition"])
        tracker.update([], 0.71)
        self.assertEqual([], tracker.emitted(0.71, set()))

    def test_event_hold_is_bounded_and_route_identity_qualified(self) -> None:
        hold = baseline.EventHold()
        self.assertEqual((True, 1.2), hold.update(raw_risk=True, raw_entry_s=1.2, now_s=1.0, selection=selection()))
        self.assertEqual((True, 0.8), hold.update(raw_risk=False, raw_entry_s=None, now_s=1.4, selection=selection()))
        self.assertEqual((False, None), hold.update(raw_risk=False, raw_entry_s=None, now_s=1.61, selection=selection()))

        hold.update(raw_risk=True, raw_entry_s=1.0, now_s=2.0, selection=selection())
        self.assertEqual(
            (False, None),
            hold.update(raw_risk=False, raw_entry_s=None, now_s=2.1, selection=selection(changed=True, receipt="r2")),
        )


if __name__ == "__main__":
    unittest.main()
