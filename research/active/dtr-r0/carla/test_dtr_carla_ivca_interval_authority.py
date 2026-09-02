from __future__ import annotations

import unittest

import dtr_carla_ivca_interval_authority as ivca
import dtr_carla_ivca_interval_evaluator as evaluator
import dtr_carla_x24_plan_route_core as route


class IvcaIntervalAuthorityTest(unittest.TestCase):
    def test_recovers_narrow_continuous_contact_interval(self) -> None:
        segments = (
            route.RouteSegment(
                start_offset_s=0.0,
                end_offset_s=2.0,
                start_position_xy=(0.0, 0.0),
                velocity_xy=(2.0, 0.0),
            ),
        )
        result = ivca.transported_interval_set(
            ((1.95, -0.05), (2.05, -0.05), (2.05, 0.05), (1.95, 0.05)),
            (0.0, 0.0),
            segments,
            tube_radius_m=0.01,
        )
        self.assertEqual(1, len(result.components))
        self.assertLess(result.components[0].overlap_duration_s, 0.06)
        self.assertAlmostEqual(0.97, result.components[0].entry_s, places=4)
        self.assertAlmostEqual(1.03, result.components[0].exit_s, places=4)

    def test_preserves_disjoint_reentry_components(self) -> None:
        segments = (
            route.RouteSegment(0.0, 1.0, (0.0, 0.0), (2.0, 0.0)),
            route.RouteSegment(1.0, 3.0, (1.0, 1.0), (0.0, -1.0)),
        )
        result = ivca.transported_interval_set(
            ((0.9, -0.1), (1.1, -0.1), (1.1, 0.1), (0.9, 0.1)),
            (0.0, 0.0),
            segments,
            tube_radius_m=0.05,
        )
        self.assertEqual(2, len(result.components))
        self.assertLess(result.components[0].exit_s, result.components[1].entry_s)

    def test_transport_cannot_birth_but_can_renew_same_receipt_and_parent(self) -> None:
        intervals = ivca.CollisionIntervalSet(
            components=(ivca.CollisionInterval(0.2, 0.8, 0.6, -0.1, 0.5),),
            total_overlap_duration_s=0.6,
            earliest_entry_s=0.2,
        )
        receipt = ivca.measured_receipt(
            plan_receipt_sha256="plan-a",
            parent_id="parent-a",
            carrier_id="carrier-a",
            representation="RIGID_FOOTPRINT",
            interval_set=intervals,
            carrier_authorized=True,
        )
        self.assertIsNotNone(receipt)
        arm = {
            "x94_one_frame_full_dropout_continuity_used": True,
            "plan_receipt_sha256": "plan-a",
            "x94_continuity_parent_ids": ["parent-a"],
        }
        active, born, reason = ivca.authorize_frame(
            previous_receipt=None,
            current_measured_receipt=None,
            x94_arm=arm,
        )
        self.assertFalse(active)
        self.assertIsNone(born)
        self.assertEqual("TRANSPORT_CANNOT_BIRTH", reason)

        active, renewed, reason = ivca.authorize_frame(
            previous_receipt=receipt,
            current_measured_receipt=None,
            x94_arm=arm,
        )
        self.assertTrue(active)
        self.assertIsNotNone(renewed)
        self.assertEqual(ivca.TRANSPORT_RENEWAL, reason)

        active, reseeded, reason = ivca.authorize_frame(
            previous_receipt=renewed,
            current_measured_receipt=None,
            x94_arm=arm,
        )
        self.assertFalse(active)
        self.assertIsNone(reseeded)
        self.assertEqual("TRANSPORT_CANNOT_RESEED", reason)

        active, renewed, reason = ivca.authorize_frame(
            previous_receipt=receipt,
            current_measured_receipt=None,
            x94_arm={**arm, "plan_receipt_sha256": "plan-b"},
        )
        self.assertFalse(active)
        self.assertIsNone(renewed)
        self.assertEqual("PLAN_RECEIPT_CHANGED", reason)

    def test_evaluator_keeps_entry_and_exit_interval_censored(self) -> None:
        rows = []
        for ordinal, (time_s, contact, distance) in enumerate(
            (
                (0.8, False, 0.9),
                (0.9, False, 0.7),
                (1.0, True, 0.5),
                (1.1, True, 0.4),
                (1.2, False, 0.8),
            )
        ):
            rows.append(
                {
                    "sample_index": ordinal,
                    "time_s": time_s,
                    "truth": {
                        "current_contact": contact,
                        "minimum_distance_m": distance,
                    },
                }
            )
        truth = evaluator.realized_interval_truth(
            rows,
            current_index=0,
            horizon_s=0.4,
            wearer_radius_m=0.65,
        )
        self.assertEqual(1, len(truth.components))
        component = truth.components[0]
        self.assertAlmostEqual(0.1, component.entry_lower_s)
        self.assertAlmostEqual(0.2, component.entry_upper_s)
        self.assertAlmostEqual(0.3, component.exit_lower_s)
        self.assertAlmostEqual(0.4, component.exit_upper_s)

        prediction = ivca.CollisionIntervalSet(
            components=(ivca.CollisionInterval(0.15, 0.35, 0.2, -0.25, 0.3),),
            total_overlap_duration_s=0.2,
            earliest_entry_s=0.15,
        )
        metrics = evaluator.score_interval_prediction(prediction, truth)
        self.assertAlmostEqual(1.0, metrics["midpoint_interval_iou"])
        self.assertEqual(0.0, metrics["entry_censored_error_s"])
        self.assertEqual(0.0, metrics["exit_censored_error_s"])
        self.assertFalse(metrics["false_onset_birth"])


if __name__ == "__main__":
    unittest.main()
